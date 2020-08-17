"""
FILE: alignment_fitting.py
LAST MODIFIED: 24-12-2015 
DESCRIPTION: Iterative alignment of points.

===============================================================================
This file is part of GIAS2. (https://bitbucket.org/jangle/gias2)

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
===============================================================================
"""
import logging
from typing import Optional, Union, Tuple

import numpy as np
from scipy.linalg import lstsq
from scipy.optimize import leastsq, fmin
from scipy.spatial import cKDTree

from gias2.common import transform3D

log = logging.getLogger(__name__)


def _sampleData(data: np.ndarray, n_points: int) -> np.ndarray:
    """
    Pick N evenly spaced points from data
    """

    if n_points < 1:
        raise ValueError('N must be > 1')
    elif n_points > len(data):
        return data
    else:
        i = np.linspace(0, len(data) - 1, n_points).astype(int)
        return data[i, :]


# ======================================================================#
# correspondent fitting data fitting                                   #
# ======================================================================#
def fitAffine(
        data: np.ndarray,
        target: np.ndarray,
        xtol: float = 1e-5,
        maxfev: int = 0,
        sample: Optional[int] = None,
        verbose: int = 0,
        outputErrors: int = 0) -> Union[Tuple[np.ndarray, np.ndarray, Tuple[float, float]],
                                        Tuple[np.ndarray, np.ndarray]]:
    if len(data) != len(target):
        raise ValueError('data and target points must have same number of points')

    rms0 = np.sqrt(((data - target) ** 2.0).sum(1).mean())

    if sample is not None:
        D = _sampleData(data, sample)
        T = _sampleData(target, sample)
    else:
        D = data
        T = target

    t = transform3D.directAffine(D, T)
    data_fitted = transform3D.transformAffine(data, t)
    rms_opt = np.sqrt(((data_fitted - target) ** 2.0).sum(1).mean())
    if verbose:
        log.info('initial & final RMS: %s, %s', rms0, rms_opt)

    if outputErrors:
        return t, data_fitted, (rms0, rms_opt)
    else:
        return t, data_fitted


def fitTranslation(
        data: np.ndarray,
        target: np.ndarray,
        xtol: float = 1e-5,
        maxfev: int = 0,
        sample: Optional[int] = None,
        verbose: int = 0,
        outputErrors: int = 0) -> Union[Tuple[np.ndarray, np.ndarray, Tuple[float, float]],
                                        Tuple[np.ndarray, np.ndarray]]:
    """ fits for tx,ty for transforms points in data to points
    in target. Points in data and target are assumed to correspond by
    order
    """

    if sample is not None:
        D = _sampleData(data, sample)
        T = _sampleData(target, sample)
    else:
        D = data
        T = target

    def obj(x):
        dt = D + x
        d = ((dt - T) ** 2.0).sum(1)
        return d

    t0 = target.mean(0) - data.mean(0)

    rms0 = np.sqrt(obj(t0).mean())
    if verbose:
        log.info('initial RMS: %s', rms0)

    x_opt = leastsq(obj, t0, xtol=xtol, maxfev=maxfev)[0]

    rms_opt = np.sqrt(obj(x_opt).mean())
    if verbose:
        log.info('final RMS: %s', rms_opt)

    data_fitted = data + x_opt
    if outputErrors:
        return x_opt, data_fitted, (rms0, rms_opt)
    else:
        return x_opt, data_fitted


def fitRigid(
        data: np.ndarray,
        target: np.ndarray,
        t0: Optional[np.ndarray] = None,
        xtol: float = 1e-3,
        rotcentre: Optional[np.ndarray] = None,
        maxfev: int = None,
        maxfun: int = None,
        sample: int = None,
        verbose: bool = False,
        epsfcn: float = 0,
        outputErrors: float = 0) -> Union[Tuple[np.ndarray, np.ndarray, Tuple[float, float]],
                                          Tuple[np.ndarray, np.ndarray]]:
    """ fits for tx,ty,tz,rx,ry,rz to transform points in data to points
    in target. Points in data and target are assumed to correspond by
    order
    """

    if sample is not None:
        D = _sampleData(data, sample)
        T = _sampleData(target, sample)
    else:
        D = data
        T = target

    if t0 is None:
        t0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    else:
        t0 = np.array(t0)

    if rotcentre is None:
        rotcentre = D.mean(0)

    if data.shape[0] >= t0.shape[0]:
        def obj(x):
            dt = transform3D.transformRigid3DAboutP(D, x, rotcentre)
            d = ((dt - T) ** 2.0).sum(1)
            return d
    else:
        def obj(x):
            dt = transform3D.transformRigid3DAboutP(D, x, rotcentre)
            d = ((dt - T) ** 2.0).sum(1)
            return d.sum()

    t0 = np.array(t0)
    rms0 = np.sqrt(obj(t0).mean())
    if verbose:
        log.info('initial RMS: %s', rms0)

    if data.shape[0] >= t0.shape[0]:
        if maxfev is None:
            maxfev = 0
        x_opt = leastsq(obj, t0, xtol=xtol, maxfev=maxfev, epsfcn=epsfcn)[0]
    else:
        x_opt = fmin(obj, t0, xtol=xtol, maxiter=maxfev, maxfun=maxfun, disp=verbose)

    rms_opt = np.sqrt(obj(x_opt).mean())
    if verbose:
        log.info('final RMS: %s', rms_opt)

    data_fitted = transform3D.transformRigid3DAboutP(data, x_opt, rotcentre)
    if outputErrors:
        return x_opt, data_fitted, (rms0, rms_opt)
    else:
        return x_opt, data_fitted


def fitRigidFMin(
        data: np.ndarray,
        target: np.ndarray,
        t0: np.ndarray = None,
        xtol: float = 1e-3,
        maxfev: int = 0,
        sample: int = None,
        verbose: int = 0,
        outputErrors: int = 0) -> Union[Tuple[np.ndarray, np.ndarray, Tuple[float, float]],
                                        Tuple[np.ndarray, np.ndarray]]:
    """ fits for tx,ty,tz,rx,ry,rz to transform points in data to points
    in target. Points in data and target are assumed to correspond by
    order
    """

    if sample is not None:
        D = _sampleData(data, sample)
        T = _sampleData(target, sample)
    else:
        D = data
        T = target

    if t0 is None:
        t0 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    def obj(x):
        dt = transform3D.transformRigid3DAboutCoM(D, x)
        d = ((dt - T) ** 2.0).sum(1)
        rmsd = np.sqrt(d.mean())
        return rmsd

    t0 = np.array(t0)
    rms0 = np.sqrt(obj(t0).mean())
    if verbose:
        log.info('initial RMS: %s', rms0)

    x_opt = fmin(obj, t0, xtol=xtol, maxiter=maxfev)

    rms_opt = np.sqrt(obj(x_opt).mean())
    if verbose:
        log.info('final RMS: %s', rms_opt)

    data_fitted = transform3D.transformRigid3DAboutCoM(data, x_opt)
    if outputErrors:
        return x_opt, data_fitted, (rms0, rms_opt)
    else:
        return x_opt, data_fitted


def fitRigidScale(
        data: np.ndarray,
        target: np.ndarray,
        t0: np.ndarray = None,
        xtol: float = 1e-3,
        maxfev: int = 0,
        sample: int = None,
        verbose: int = 0,
        outputErrors: int = 0) -> Union[Tuple[np.ndarray, np.ndarray, Tuple[float, float]],
                                        Tuple[np.ndarray, np.ndarray]]:
    """ fits for tx,ty,tz,rx,ry,rz,s to transform points in data to points
    in target. Points in data and target are assumed to correspond by
    order
    """

    if sample is not None:
        D = _sampleData(data, sample)
        T = _sampleData(target, sample)
    else:
        D = data
        T = target

    if t0 is None:
        t0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    else:
        t0 = np.array(t0)

    if data.shape[0] >= t0.shape[0]:
        def obj(x):
            dt = transform3D.transformRigidScale3DAboutCoM(D, x)
            d = ((dt - T) ** 2.0).sum(1)
            return d
    else:
        def obj(x):
            dt = transform3D.transformRigidScale3DAboutCoM(D, x)
            d = ((dt - T) ** 2.0).sum(1)
            return d.sum()

    t0 = np.array(t0)
    rms0 = np.sqrt(obj(t0).mean())
    if verbose:
        log.info('initial RMS: %s', rms0)

    if data.shape[0] >= t0.shape[0]:
        if maxfev is None:
            maxfev = 0
        x_opt = leastsq(obj, t0, xtol=xtol, maxfev=maxfev)[0]
    else:
        x_opt = fmin(obj, t0, xtol=xtol, maxiter=maxfev)

    rms_opt = np.sqrt(obj(x_opt).mean())
    if verbose:
        log.info('final RMS: %s', rms_opt)

    data_fitted = transform3D.transformRigidScale3DAboutCoM(data, x_opt)
    if outputErrors:
        return x_opt, data_fitted, (rms0, rms_opt)
    else:
        return x_opt, data_fitted


# ======================================================================#
# Non correspondent fitting data fitting                                #
# ======================================================================#
def fitDataRigidEPDP(
        data: np.ndarray,
        target: np.ndarray,
        xtol: float = 1e-3,
        maxfev: int = 0,
        t0: np.ndarray = None,
        sample: int = None,
        outputErrors: int = 0) -> Union[Tuple[np.ndarray, np.ndarray, Tuple[float, float]],
                                        Tuple[np.ndarray, np.ndarray]]:
    """ fit list of points data to list of points target by minimising
    least squares distance between each point in data and closest neighbour
    in target.

    Note that the resulting rotations are applied about the centre of mass of the `data` point cloud.
    The centre of mass is calculated simply as the euclidean mean of all the points in `data`.

    If you want to apply the resulting transformation `t` on another point cloud `p` that has a different centre of
    mass, you must use `transform3D.transformRigid3DAboutP(p, t, c)` where c is the centre of mass of `data`.
    """

    if sample is not None:
        D = _sampleData(data, sample)
        T = _sampleData(target, sample)
    else:
        D = data
        T = target

    if t0 is None:
        t0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    t_tree = cKDTree(T)
    D = np.array(D)

    def obj(t):
        dt = transform3D.transformRigid3DAboutCoM(D, t)
        d = t_tree.query(dt)[0]
        return d * d

    initial_rmse = np.sqrt(obj(t0).mean())
    t_opt = leastsq(obj, t0, xtol=xtol, maxfev=maxfev)[0]
    data_fitted = transform3D.transformRigid3DAboutCoM(data, t_opt)
    final_rmse = np.sqrt(obj(t_opt).mean())

    if outputErrors:
        return t_opt, data_fitted, (initial_rmse, final_rmse)
    else:
        return t_opt, data_fitted


def fitDataTranslateEPDP(
        data: np.ndarray,
        target: np.ndarray,
        xtol: float = 1e-3,
        maxfev: int = 0,
        t0: np.ndarray = None,
        sample: int = None,
        outputErrors: int = 0) -> Union[Tuple[np.ndarray, np.ndarray, Tuple[float, float]],
                                        Tuple[np.ndarray, np.ndarray]]:
    """ fit list of points data to list of points target by minimising
    least squares distance between each point in data and closest neighbour
    in target
    """

    if sample is not None:
        D = _sampleData(data, sample)
        T = _sampleData(target, sample)
    else:
        D = data
        T = target

    if t0 is None:
        t0 = np.array([0.0, 0.0, 0.0])

    t_tree = cKDTree(T)
    D = np.array(D)

    def obj(t):
        dt = transform3D.transformRigid3DAboutCoM(D, np.hstack((t, [0.0, 0.0, 0.0])))
        d = t_tree.query(list(dt))[0]
        return d * d

    initial_rmse = np.sqrt(obj(t0).mean())
    t_opt = leastsq(obj, t0, xtol=xtol, maxfev=maxfev)[0]
    data_fitted = transform3D.transformRigid3DAboutCoM(data, np.hstack((t_opt, [0.0, 0.0, 0.0])))
    final_rmse = np.sqrt(obj(t_opt).mean())

    if outputErrors:
        return t_opt, data_fitted, (initial_rmse, final_rmse)
    else:
        return t_opt, data_fitted


def fitDataRigidDPEP(
        data: np.ndarray,
        target: np.ndarray,
        xtol: float = 1e-3,
        maxfev: int = 0,
        t0: np.ndarray = None,
        sample: int = None,
        outputErrors: int = 0) -> Union[Tuple[np.ndarray, np.ndarray, Tuple[float, float]],
                                        Tuple[np.ndarray, np.ndarray]]:
    """ fit list of points data to list of points target by minimising
    least squares distance between each point in target and closest neighbour
    in data

    Note that the resulting rotations are applied about the centre of mass of the `data` point cloud.
    The centre of mass is calculated simply as the euclidean mean of all the points in `data`.

    If you want to apply the resulting transformation `t` on another point cloud `p` that has a different centre of
    mass, you must use `transform3D.transformRigid3DAboutP(p, t, c)` where c is the centre of mass of `data`.
    """

    if sample is not None:
        D = _sampleData(data, sample)
        T = _sampleData(target, sample)
    else:
        D = data
        T = target

    if t0 is None:
        t0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    D = np.array(D)

    def obj(t):
        dt = transform3D.transformRigid3DAboutCoM(D, t)
        dt_tree = cKDTree(dt)
        d = dt_tree.query(list(T))[0]
        return d * d

    initial_rmse = np.sqrt(obj(t0).mean())
    t_opt = leastsq(obj, t0, xtol=xtol, maxfev=maxfev)[0]
    data_fitted = transform3D.transformRigid3DAboutCoM(data, t_opt)
    final_rmse = np.sqrt(obj(t_opt).mean())

    if outputErrors:
        return t_opt, data_fitted, (initial_rmse, final_rmse)
    else:
        return t_opt, data_fitted


def fitDataRigidScaleEPDP(
        data: np.ndarray,
        target: np.ndarray,
        xtol: float = 1e-3,
        maxfev: int = 0,
        t0: np.ndarray = None,
        sample: int = None,
        outputErrors: int = 0,
        scaleThreshold: Optional[float] = None) -> Union[Tuple[np.ndarray, np.ndarray, Tuple[float, float]],
                                                         Tuple[np.ndarray, np.ndarray]]:
    """ fit list of points data to list of points target by minimising
    least squares distance between each point in data and closest neighbour
    in target

    Note that the resulting rotations and scaling are applied about the centre of mass of the `data` point cloud.
    The centre of mass is calculated simply as the euclidean mean of all the points in `data`.

    If you want to apply the resulting transformation `t` on another point cloud `p` that has a different centre of
    mass, you must use `transform3D.transformRigid3DAboutP(p, t, c)` where c is the centre of mass of `data`.
    """

    if sample is not None:
        D = _sampleData(data, sample)
        T = _sampleData(target, sample)
    else:
        D = data
        T = target

    if t0 is None:
        t0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])

    t_tree = cKDTree(T)
    D = np.array(D)

    if scaleThreshold is not None:
        def obj(t):
            dt = transform3D.transformRigidScale3DAboutCoM(D, t)
            d = t_tree.query(list(dt))[0]
            s = max(t[-1], 1.0 / t[-1])
            if s > scaleThreshold:
                sw = 1000.0 * s
            else:
                sw = 1.0
            return d * d + sw
    else:
        def obj(t):
            dt = transform3D.transformRigidScale3DAboutCoM(D, t)
            d = t_tree.query(list(dt))[0]
            return d * d

    initial_rmse = np.sqrt(obj(t0).mean())
    t_opt = leastsq(obj, t0, xtol=xtol, maxfev=maxfev)[0]
    data_fitted = transform3D.transformRigidScale3DAboutCoM(data, t_opt)
    final_rmse = np.sqrt(obj(t_opt).mean())

    if outputErrors:
        return t_opt, data_fitted, (initial_rmse, final_rmse)
    else:
        return t_opt, data_fitted


def fitDataRigidScaleDPEP(
        data: np.ndarray,
        target: np.ndarray,
        xtol: float = 1e-3,
        maxfev: int = 0,
        t0: np.ndarray = None,
        sample: int = None,
        outputErrors: int = 0,
        scaleThreshold: Optional[float] = None) -> Union[Tuple[np.ndarray, np.ndarray, Tuple[float, float]],
                                                         Tuple[np.ndarray, np.ndarray]]:
    """ fit list of points data to list of points target by minimising
    least squares distance between each point in target and closest neighbour
    in data

    Note that the resulting rotations and scaling are applied about the centre of mass of the `data` point cloud.
    The centre of mass is calculated simply as the euclidean mean of all the points in `data`.

    If you want to apply the resulting transformation `t` on another point cloud `p` that has a different centre of
    mass, you must use `transform3D.transformRigid3DAboutP(p, t, c)` where c is the centre of mass of `data`.
    """

    if sample is not None:
        D = _sampleData(data, sample)
        T = _sampleData(target, sample)
    else:
        D = data
        T = target

    if t0 is None:
        t0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])

    D = np.array(D)

    if scaleThreshold is not None:
        def obj(t):
            DT = transform3D.transformRigidScale3DAboutCoM(D, t)
            DTTree = cKDTree(DT)
            d = DTTree.query(T)[0]
            s = t[6]
            if s > scaleThreshold:
                sw = 1000.0 * s
            else:
                sw = 1.0
            return d * d + sw
    else:
        def obj(t):
            DT = transform3D.transformRigidScale3DAboutCoM(D, t)
            DTTree = cKDTree(DT)
            d = DTTree.query(T)[0]
            return d * d

    initial_rmse = np.sqrt(obj(t0).mean())
    t_opt = leastsq(obj, t0, xtol=xtol, maxfev=maxfev)[0]
    data_fitted = transform3D.transformRigidScale3DAboutCoM(data, t_opt)
    final_rmse = np.sqrt(obj(t_opt).mean())

    if outputErrors:
        return t_opt, data_fitted, (initial_rmse, final_rmse)
    else:
        return t_opt, data_fitted


# ===========================================================================#


def fitSphere(pts: np.ndarray) -> Tuple[Tuple[float, float, float], float]:
    """
    least squares fits the sphere centre and radius to a cloud of points X
    """
    b_mat = (pts ** 2.0).sum(1)
    a_mat = np.hstack([2.0 * pts, np.ones(pts.shape[0])[:, np.newaxis]])
    x, res, rank, s = lstsq(a_mat, b_mat)

    a, b, c, m = x
    r = np.sqrt(m + a * a + b * b + c * c)
    return (a, b, c), r
