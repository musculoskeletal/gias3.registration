"""
FILE: shapemodel.py
LAST MODIFIED: 31-7-2017 
DESCRIPTION: Shape model-based non-rigid registration.

===============================================================================
This file is part of GIAS2. (https://bitbucket.org/jangle/gias2)

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
===============================================================================
"""
import logging
from typing import List, Optional, Union, Callable, Tuple

import numpy as np
import sys
from scipy.optimize import leastsq, least_squares
from scipy.spatial import cKDTree

from gias2.common import transform3D
from gias2.learning.PCA import PrincipalComponents

log = logging.getLogger(__name__)


# =============================================================================#
# utility functions
# =============================================================================#
def _sample_data(data: np.ndarray, n: int) -> np.ndarray:
    """
    Pick N evenly spaced rows from data
    """

    if n < 1:
        raise ValueError('N must be > 1')
    elif n > len(data):
        return data
    else:
        i = np.linspace(0, len(data) - 1, n).astype(int)
        return data[i, :]


def mahalanobis(x: np.ndarray) -> float:
    """
    Calculate the mahalanobis distance on an array of standard deviations
    """
    return np.sqrt(np.multiply(x, x).sum())


# =============================================================================#
# reconstruction functions
# =============================================================================#
def r2c13(x_recon: np.ndarray) -> np.ndarray:
    """
    Reconstruction in to nx3 array of coordinates
    """
    return x_recon.reshape((-1, 3))


def r2c31(x_recon: np.ndarray) -> np.ndarray:
    """
    Reconstruction in to 3xn array of coordinates
    """
    return x_recon.reshape((3, -1))


# =============================================================================#
# Main registration function
# =============================================================================#
def fitSSMTo3DPoints(
        data: np.ndarray,
        ssm: PrincipalComponents,
        fit_comps: List[int],
        fit_mode: str,
        fit_inds: Optional[Union[list, np.ndarray]] = None,
        mw: float = 0.0,
        init_t: Optional[np.ndarray] = None,
        fit_scale: bool = False,
        ftol: float = 1e-6,
        sample: Optional[int] = None,
        ldmk_targs: Optional[np.ndarray] = None,
        ldmk_evaluator: Optional[List[Callable]] = None,
        ldmk_weights: Optional[np.ndarray] = None,
        recon2coords: Optional[Callable] = None,
        verbose: bool = False,
        n_jobs: int = 1,
        f_scale: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, Tuple[np.ndarray, float, float]]:
    """
    Fit a shape model to a set of non-correspondent points by optimising
    translation, rotation, and PCA scores.

    Rigid and optionally scaling registration should be performed before
    running this.
    
    arguments
    ---------
    data: nx3 array of target point coordinates.
    ssm: a gias2.learning.PCA.PrincipalComponents object
    fit_comps: a list of PC modes to fit, e.g. [0,1,2] to fit the 1st 3 modes.
    fit_mode: {'st'|'ts'|'2way'|'corr'} source to target, target to source, 
        2 way, or corresponding fitting. Use st if target datacloud covers more
        of the object than the shape model. Use ts if the target data cloud
        covers less of the object than the shape model. Use 2 way if neither st
        ts produces good results. Use corr if the target datacloud is
        correspondent with the points in the shape model.
    fit_inds: [list of ints] restrict fitting to certain points in the
        shape model
    mw: [float] weighting on the mahalanobis fitting penalty. Reasonable value
        is 0.1 to 1.0
    init_t: [list] initial [tx,ty,tz,rx,ry,rz,[s]]
    fit_scale: [bool] fit for scaling, default is False
    ftol: [float] relative error desired in sum of squared error
    sample: [int] number of points to sample from the target data
    ldmk_targs: [mx3 array] additional target landmark points to use during
        fitting
    ldmk_evaluator: functional that evaluates corresponding landmark
        coordinates from reconstructed data points. Should take as input a nx3
        array of reconstructed coordinates from the shape model and output a
        mx3 array of landmark coordinates.
    ldmk_weights: [mx1 float array] the fitting weight for each landmark.
    recon2coords: A function for reconstructing point coordinates from shape
        model data. e.g. r2c13 and r2c31 in this module.
    verbose: [bool] extra info during fit
    n_jobs: number of threads to use when calculating closes neighbours using cKDTree.query
    f_scale: f_scale parameter for least_squares trf solver. If None, uses the
        leastsq solver. Else should be something like 5.0 to regard points more
        than 5.0 mm distant was outliers.
    
    Returns
    -------
    xOpt: array of optimised parameters. First 3 elements are translation, 
        then 3 for rotation, and the rest are PC scores in terms of standard
        deviations.
    recon_data_opt: fitted shape model points
    (err_opt, dist_opt_rms, mdist_opt): final error, final rms distance, final
        mahalanobis distance
    """

    if recon2coords is None:
        # Function to convert ssm data into point coordinates. Default is for
        # nx3 point clouds.
        recon2coords = r2c13

    log.debug('fitting ssm to points')
    if init_t is None:
        init_t = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        if fit_scale:
            init_t = np.hstack([init_t, 1.0])
    else:
        init_t = np.array(init_t)

    if fit_scale:
        assert (len(init_t) == 7)
    else:
        assert (len(init_t) == 6)

    if fit_inds is None:
        log.debug('fit_inds shape: None')
    else:
        log.debug('fit_inds shape:', fit_inds.shape)

    if sample is not None:
        data = _sample_data(data, sample)

    # -------------------------------------------------------------------------#
    # define reconstruction functions
    # -------------------------------------------------------------------------#
    def _recon_no_scale(X):
        recon = ssm.reconstruct(
            ssm.getWeightsBySD(fit_comps, X[6:]), fit_comps
        )
        # reconstruct rigid transform
        recon_pts = transform3D.transformRigid3DAboutCoM(
            recon2coords(recon), X[:6]
        )
        mahalanobis_dist = mahalanobis(X[6:])
        return recon_pts, mahalanobis_dist

    def _recon_scale(X):
        recon = ssm.reconstruct(
            ssm.getWeightsBySD(fit_comps, X[7:]), fit_comps
        )
        # reconstruct rigid transform
        recon_pts = transform3D.transformRigidScale3DAboutCoM(
            recon2coords(recon), X[:7]
        )
        mahalanobis_dist = mahalanobis(X[7:])
        return recon_pts, mahalanobis_dist

    if fit_scale:
        _recon = _recon_scale
    else:
        _recon = _recon_no_scale

    # -------------------------------------------------------------------------#
    # define distance error functions
    # -------------------------------------------------------------------------#
    targ_tree = cKDTree(data)

    def _dist_sptp(recon_pts, m):
        return targ_tree.query(recon_pts, eps=1e-9, n_jobs=n_jobs)[0] + mw * m

    def _dist_tpsp(recon_pts, m):
        recon_tree = cKDTree(recon_pts)
        return recon_tree.query(data, eps=1e-9, n_jobs=n_jobs)[0] + mw * m

    def _dist_2way(recon_pts, m):
        recon_tree = cKDTree(recon_pts)
        d_sptp = targ_tree.query(recon_pts, eps=1e-9, n_jobs=n_jobs)[0]
        d_tpsp = recon_tree.query(data, eps=1e-9, n_jobs=n_jobs)[0]
        dm = mw * m
        return np.hstack([d_sptp, d_tpsp]) + dm

    def _dist_corr(recon_pts, m):
        return np.sqrt(((data - recon_pts) ** 2.0).sum(1))

    fit_modes_map = {
        'st': _dist_sptp,
        'ts': _dist_tpsp,
        '2way': _dist_2way,
        'corr': _dist_corr,
    }

    try:
        _dist = fit_modes_map[fit_mode]
    except KeyError:
        raise ValueError('invalid fit mode {}'.format(fit_mode))

    # -------------------------------------------------------------------------#
    # define objective functions
    # -------------------------------------------------------------------------#
    def _obj_no_ldmks(X):
        # reconstruct data points
        recon_data, mdist = _recon(X)

        # select the fitting points
        if fit_inds is not None:
            recon_data = recon_data[fit_inds, :]

        # calc error
        err = _dist(recon_data, mdist)

        if verbose:
            sys.stdout.write('\robj rms:' + str(np.sqrt(err.mean())))
            sys.stdout.flush()

        return err

    def _obj_ldmks(X):
        # reconstruct data points
        recon_data, mdist = _recon(X)
        recon_ldmks = ldmk_evaluator(recon_data.T.ravel())

        # calc error
        err_data = _dist(recon_data, mdist)

        err_ldmks = ((ldmk_targs - recon_ldmks) ** 2.0).sum(1) * ldmk_weights
        err = np.hstack([err_data, err_ldmks])

        if verbose:
            sys.stdout.write(
                '\rPC fit rmse: %6.3f (data: %6.3f) (landmarks: %6.3f)' % \
                (np.sqrt(err.mean()), np.sqrt(err_data.mean()), np.sqrt(err_ldmks.mean()))
            )
            sys.stdout.flush()

        return err

    if ldmk_targs is None:
        _obj = _obj_no_ldmks
    else:
        _obj = _obj_ldmks

    # -------------------------------------------------------------------------#
    # fit
    # -------------------------------------------------------------------------#
    x0 = np.hstack([init_t, np.zeros(len(fit_comps), dtype=float)])

    if verbose:
        recon_data_init, mdist_init = _recon(x0)
        err_init = _obj(x0)
        dist_init_rms = np.sqrt((_dist(recon_data_init, 0.0) ** 2.0).mean())
        log.debug('\ninitial rms distance: {}'.format(dist_init_rms))

    if f_scale is None:
        x_opt = leastsq(_obj, x0, ftol=ftol)[0]
    else:
        x_opt = least_squares(
            _obj, x0, method='trf', loss='soft_l1', f_scale=f_scale, ftol=ftol, verbose=0
        )['x']

    recon_data_opt, mdist_opt = _recon(x_opt)
    err_opt = _obj(x_opt)
    dist_opt_rms = np.sqrt((_dist(recon_data_opt, 0.0) ** 2.0).mean())

    if verbose:
        log.debug('\nfinal rms distance: {}'.format(dist_opt_rms))

    return x_opt, recon_data_opt, (err_opt, dist_opt_rms, mdist_opt)
