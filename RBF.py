"""
FILE: RBF.py
LAST MODIFIED: 27-7-2017 
DESCRIPTION: Radial basis function non-rigid registration

===============================================================================
This file is part of GIAS2. (https://bitbucket.org/jangle/gias2)

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
===============================================================================
"""
import logging

import numpy as np
import sys
from scipy.linalg import lstsq, solve, inv, qr
from scipy.optimize import leastsq
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist, cdist, squareform

from gias2.registration import alignment_fitting as af

if sys.version_info.major < 3:
    import cPickle as pickle
else:
    import pickle

log = logging.getLogger(__name__)


# =============================================================================#
# Basis functions
# =============================================================================#
def polyCubic3D(x, y, z):
    X0 = np.ones(x.shape[0])
    X1 = x
    X2 = y
    X3 = z
    X4 = x * x
    X5 = y * y
    X6 = z * z
    X7 = x * y
    X8 = x * z
    X9 = y * z
    X10 = X4 * x  # x^3
    X11 = X5 * y  # y^3
    X12 = X6 * z  # z^3
    X13 = X4 * y  # x^2y
    X14 = X5 * x  # xy^2
    X15 = X4 * z  # x^2z
    X16 = X6 * x  # xz^2
    X17 = X5 * z  # y^2z
    X18 = X6 * y  # yz^2
    X19 = x * y * z

    return np.array([X0, X1, X2, X3, X4, X5, X6, X7, X8, X9, X10, X11, X12, X13, X14, X15, X16, X17, X19])


def polyLinear3D(x, y, z):
    X0 = np.ones(x.shape[0])
    X1 = x
    X2 = y
    X3 = z

    return np.array([X0, X1, X2, X3])


def polyConst3D(x, y, z):
    X0 = np.ones(x.shape[0])

    return np.array([X0, ])


def makeBasisBiharmonic():
    def b(r):
        return r

    return b


def makeBasisGaussian(s):
    """
    Gaussian basis function, normalised to integrate to 1
    """

    def b(r):
        # ~ return 1.0/np.sqrt(2.0*np.pi*s) * np.exp( -0.5*r*r/(s*s) )
        return 1.0 / np.sqrt(2.0 * np.pi * s) * np.exp(-0.5 * r * r / (s * s)) + 1.0

    return b


def makeBasisGaussianNonUniformWidth(s, scaling):
    """
    Gaussian basis function, normalised to integrate to 1
    """
    # print 'making gaussian non-uniform width RBF'
    S = s * scaling

    def b(r):
        return np.exp(-r * r / (S * S))

    return b


def makeBasisNormalisedGaussianNonUniformWidth(s, scaling):
    """
    Gaussian basis function, normalised to integrate to 1
    """
    # print 'making normalised gaussian non-uniform width RBF'

    S = s * scaling

    def b(r):
        return 1.0 / np.sqrt(2.0 * np.pi * S) * np.exp(-0.5 * r * r / (S * S)) + 1.0

    return b


# ~ def makeBasisGaussianBilinear( s ):
# ~
# ~ def b( r, x, y ):
# ~ g = 1.0/np.sqrt(2.0*np.pi*s) * np.exp( -0.5*r*r/(s*s) ) +1.0
# ~ return g, x, y
# ~
# ~ def makeBasisGaussianBasis( s ):
# ~
# ~ def b( r, x, y, z ):
# ~ g = 1.0/np.sqrt(2.0*np.pi*s) * np.exp( -0.5*r*r/(s*s) ) +1.0
# ~ return g, x, y, z

def makeBasisMultiQuadric(c):
    def b(r):
        return np.sqrt(c * c + r * r)

    return b


def makeBasisTPS():
    def b(r):
        x = r * r * np.log(r)
        return np.where(np.isfinite(x), x, 0.0)

    return b


def makeBasisWendlandsC31NonUniformWidth(s, scaling):
    """
    see Fornefett, Rohr, Steihl (1999) elastic registration of medical
    images using RBFs with compact support
    
    3-dimensional, c2 continuous
    """

    def b(r):
        rNorm = r / (s * scaling)
        return np.where(rNorm < 1.0, (4.0 * rNorm + 1) * (1 - rNorm) ** 4.0, 0.0)
        # ~ return (4.0*rNorm+1)*(1-rNorm)**4.0

    return b


def makeBasisWendlandsC32NonUniformWidth(s, scaling):
    """
    see Fornefett, Rohr, Steihl (1999) elastic registration of medical
    images using RBFs with compact support
    
    3-dimensional, c4 continuous
    """

    def b(r):
        rNorm = r / (s * scaling)
        return np.where(rNorm < 1.0, (35.0 * rNorm ** 2.0 + 18.0 * rNorm + 3.0) * (1.0 - rNorm) ** 6.0, 0.0)
        # ~ return (4.0*rNorm+1)*(1-rNorm)**4.0

    return b


RBFBases = {'gaussian': makeBasisGaussian,
            'gaussianNonUniformWidth': makeBasisGaussianNonUniformWidth,
            'normalisedgaussianNonUniformWidth': makeBasisNormalisedGaussianNonUniformWidth,
            'TPS': makeBasisTPS,
            'multiquadric': makeBasisMultiQuadric,
            'biharmonic': makeBasisBiharmonic,
            'WC31NonUniformWidth': makeBasisWendlandsC31NonUniformWidth,
            'WC32NonUniformWidth': makeBasisWendlandsC32NonUniformWidth,
            }


# =============================================================================#
# Util functions
# =============================================================================#
def estimateNonUniformWidth(X, k=2):
    XTree = cKDTree(X)
    XNNd = XTree.query(list(X), k=k + 1)[0][:, 1:]  # 1st NN is always point itself

    s = np.sqrt((XNNd ** 2.0).sum(1)) / k
    return s


def xDist(x, X):
    """ calculates the distance of x to each point in X
    """
    v = (X - x)
    return np.sqrt((v * v).sum(1))


def xDist1D(x, X):
    return X - x


def fitData(C, basis, dataX, dataU, verbose=True):
    if verbose:
        log.debug('fitting {} knots to {} data points'.format(len(C), len(dataX)))
    # split distance matrix calculation in groups to save memory
    groupSize = 5000
    # ~ pdb.set_trace()
    if len(dataX) > groupSize:
        A = np.zeros((len(dataX), len(C)))
        for i in range(0, len(dataX), groupSize):
            # log.debug(str(i)+' - '+str(i+groupSize))
            A[i:i + groupSize, :] = basis(cdist(dataX[i:i + groupSize, :], C))
    else:
        A = basis(cdist(dataX, C))

    # ~ pdb.set_trace()
    # ~ A = np.array( [ basis( xDist( d, C ) ) for d in dataX ] )
    W, residual, rank, singVal = lstsq(A, dataU.copy(), overwrite_a=True, overwrite_b=True)
    W = W.T

    return W, (residual, rank, singVal)


def fitDataPoly3D(C, basis, dataX, dataU, poly, verbose=True):
    if verbose:
        log.debug('fitting {} knots to {} data points'.format(len(C), len(dataX)))
    # split distance matrix calculation in groups to save memory
    groupSize = 5000
    # ~ pdb.set_trace()
    if len(dataX) > groupSize:
        A = np.zeros((len(dataX), len(C)))
        for i in range(0, len(dataX), groupSize):
            # log.debug(str(i)+' - '+str(i+groupSize))
            A[i:i + groupSize, :] = basis(cdist(dataX[i:i + groupSize, :], C))
    else:
        A = basis(cdist(dataX, C))

    # append polynomial
    # polynomial terms in 2nd dim, points in 1st dim
    P = poly(dataX[:, 0], dataX[:, 1], dataX[:, 2]).T

    AAUpper = np.hstack([A, P])

    # ~ pdb.set_trace()
    # ~ AALower = np.hstack( [P.T, np.zeros([P.shape[1], AAUpper.shape[1]-P.shape[0]])] )
    # ~ AA = np.vstack([AAUpper, AALower])

    AA = AAUpper

    W, residual, rank, singVal = lstsq(AA, dataU.copy(), overwrite_a=True, overwrite_b=True)
    W = W.T

    rbfW = W[:C.shape[0]]
    polyCoeff = W[C.shape[0]:]

    return rbfW, polyCoeff, (residual, rank, singVal)


def fitDataQR(C, basis, dataX, dataU, verbose=True):
    if verbose:
        log.debug('fitting {} knots to {} data points'.format(len(C), len(dataX)))
    # split distance matrix calculation in groups to save memory
    groupSize = 5000
    # ~ pdb.set_trace()
    if len(dataX) > groupSize:
        A = np.zeros((len(dataX), len(C)))
        for i in range(0, len(dataX), groupSize):
            # print str(i)+' - '+str(i+groupSize)
            A[i:i + groupSize, :] = basis(cdist(dataX[i:i + groupSize, :], C))
    else:
        A = basis(cdist(dataX, C))

    Q, R = qr(A)
    P = np.dot(Q.T, dataU.copy())
    W = np.dot(inv(R[:R.shape[1], :]), P)  # not working, need a mapping matrix to match matrix shapes
    residual = dataU - np.dot(A, W)
    rank = -1
    singVal = -1
    W = W.T

    return W, (residual, rank, singVal)


def fitComponentFieldKnotCoordWidth(RBFF, dataX, dataU, fullOutput=False, xtol=1e-3, maxfev=0):
    """
    nonlinear least squares fit of knot coordinates and width
    """

    nKnots = len(RBFF.C)

    def obj(X):

        knotCoords = X[:(nKnots * RBFF.nComponents)].reshape((nKnots, RBFF.nComponents))
        knotWidths = X[(nKnots * RBFF.nComponents):]
        RBFF.setCentres(knotCoords, width=knotWidths)
        W, (res, rank, singVal) = RBFF.fitData(dataX, dataU, fullOutput=True)
        sys.stdout.write('rbf fit rmse: %6.4f\r' % (np.sqrt((res ** 2.0).mean())))
        sys.stdout.flush()
        return res

    x0 = np.hstack([RBFF.C.ravel(), RBFF.CWidth])

    xOpt, cov_x, fitInfo, mesg, ier = leastsq(obj, x0, xtol=xtol, maxfev=maxfev, full_output=1)
    knotCoords = xOpt[:(nKnots * RBFF.nComponents)].reshape((nKnots, RBFF.nComponents))
    knotWidths = xOpt[(nKnots * RBFF.nComponents):]
    RBFF.setCentres(knotCoords, width=knotWidths)

    fitRMSE = np.sqrt((fitInfo['fvec'] ** 2.0).mean())

    if not fullOutput:
        return RBFF
    else:
        return RBFF, fitRMSE, fitInfo


def fitComponentFieldKnotWidth(RBFF, dataX, dataU, fullOutput=False, xtol=1e-3, maxfev=0):
    """
    nonlinear least squares fit of knot width
    """

    nKnots = len(RBFF.C)
    knotCoords = np.array(RBFF.C)

    def obj(widths):
        RBFF.setCentres(knotCoords, width=widths)
        W, (res, rank, singVal) = RBFF.fitData(dataX, dataU, fullOutput=True)
        err = np.sqrt(((RBFF.evalMany(dataX).T - dataU) ** 2.0).sum(1))
        sys.stdout.write('rbf fit rmse: %6.4f\r' % (np.sqrt((err ** 2.0).mean())))
        sys.stdout.flush()
        return err

    x0 = np.array(RBFF.CWidth)
    xOpt, cov_x, fitInfo, mesg, ier = leastsq(obj, x0, xtol=xtol, maxfev=maxfev, full_output=1)


    RBFF.setCentres(knotCoords, width=xOpt)
    fitRMSE = np.sqrt((fitInfo['fvec'] ** 2.0).mean())

    if not fullOutput:
        return RBFF
    else:
        return RBFF, fitRMSE, fitInfo


polynomials = {0: polyConst3D,
               1: polyLinear3D,
               3: polyCubic3D,
               }


# =============================================================================#
# Main RBF Classes
# =============================================================================#

class RBFComponentsField(object):
    """
    Multivarite RBF field

    The values of the field has nComponent number of components.
    """

    CWidthNN = 5

    def __init__(self, nComponents):
        self.W = None
        self.C = None
        self.U = None
        self.basis = None
        self.M = None
        self.nComponents = nComponents
        self.basisType = None
        self.basisArgs = {}
        self.CWidth = None
        self.polyOrder = None
        self.polyCoeffs = None
        self.poly = None
        self.verbose = True

    def save(self, filename):

        d = {'CWidthNN': self.CWidthNN,
             'C': self.C,
             'U': self.U,
             'basisType': self.basisType,
             'basisArgs': self.basisArgs,
             'W': self.W,
             'nComponents': self.nComponents,
             'polyCoeffs': self.polyCoeffs,
             'polyOrder': self.polyCoeffs,
             }

        with open(filename, 'w') as f:
            pickle.dump(d, f, protocol=2)

    def load(self, filename):
        with open(filename, 'r') as f:
            d = pickle.load(f)

        self.CWidthNN = d['CWidthNN']
        self.nComponents = d['nComponents']
        self.W = d['W']
        self.C = d['C']
        self.U = d['U']
        self.basisType = d['basisType']
        self.basisArgs = d['basisArgs']
        if self.basisArgs != None:
            if isinstance(self.basisArgs, list) and self.basisType == 'gaussian':
                self.basisArgs = {'s': self.basisArgs[0]}

            self.setBasis(RBFBases[self.basisType](**self.basisArgs))
        else:
            self.setBasis(RBFBases[self.basisType]())

        self.setCentres(self.C)

        try:
            self.polyOrder = d['polyOrder']
            self.polyCoeffs = d['polyCoeffs']
        except KeyError:
            self.polyCoeffs = None
            self.polyOrder = None
        else:
            if self.polyOrder != None:
                self.setPolynomial(self.polyOrder)

    def setCentres(self, C, width=None):
        self.C = C
        M = squareform(pdist(self.C))
        if width != None:
            self.CWidth = width
        else:
            self.CWidth = estimateNonUniformWidth(C, k=self.CWidthNN)

        if 'NonUniformWidth' in self.basisType:
            self.basisArgs['s'] = self.CWidth
            self.makeBasis(self.basisType, self.basisArgs)

        self.M = self.basis(M)

    def setPolynomial(self, polyOrder):
        self.poly = polynomials[polyOrder]
        self.usePoly = 1

    def setCentreValues(self, U):
        if U.shape[0] != self.nComponents:
            raise ValueError('incorrect number of components')
        else:
            self.U = U

    def setBasis(self, phi):
        self.basis = phi

    def makeBasis(self, basisType, basisArgs):
        self.basisType = basisType
        self.basisArgs = basisArgs

        if self.basisArgs != None:
            self.setBasis(RBFBases[self.basisType](**self.basisArgs))
        else:
            self.setBasis(RBFBases[self.basisType]())

    def calcWeights(self):
        # ~ print 'solving system...'
        # ~ LUA = lu_factor(A)
        # ~ self.W = np.array([lu_solve( LUA, u ) for u in self.U])

        self.W = solve(self.M, self.U.T).T

    def eval(self, x):
        """
        evaluate the value of the field at point x
        """
        r = xDist(x, self.C)
        B = self.basis(r)
        y = (self.W * B).sum(1)
        return y

    def evalMany(self, x):
        """
        evaluate the value of the field at points x
        """

        # ~ # calculate distance from each x to each rbf centre
        # ~ # row i of D contains the distances of each rbf centre to point
        # ~ # x[i]
        # ~ D = cdist( x, self.C, 'euclidean' )
        # ~ B = self.basis( D )
        # ~ y = np.dot(self.W, B.T)
        # ~ return y
        if self.polyOrder != None:
            return self.evalManyPoly3D(x)

        if self.verbose:
            log.debug('evaluating at ' + str(len(x)) + ' points')
        # calculate distance from each x to each rbf centre
        # row i of D contains the distances of each rbf centre to point
        # x[i] 

        # split distance matrix calculation in groups to save memory
        groupSize = 10000
        if len(x) > groupSize:
            y = np.zeros((self.nComponents, len(x)), dtype=float)
            for i in range(0, len(x), groupSize):
                # ~ print str(i)+' - '+str(i+groupSize)
                r = cdist(x[i:i + groupSize, :], self.C, 'euclidean')
                B = self.basis(r)
                y[:, i:i + groupSize] = np.dot(self.W, B.T)

        else:
            r = cdist(x, self.C, 'euclidean')
            B = self.basis(r)
            y = np.dot(self.W, B.T)

        return y

    def evalManyPoly3D(self, x):
        """
        evaluate the value of the field at points x
        """
        if self.verbose:
            log.debug('evaluating at ' + str(len(x)) + ' points poly')
        # calculate distance from each x to each rbf centre
        # row i of D contains the distances of each rbf centre to point
        # x[i] 

        # split distance matrix calculation in groups to save memory
        groupSize = 10000
        if len(x) > groupSize:
            y = np.zeros((self.nComponents, len(x)), dtype=float)
            for i in range(0, len(x), groupSize):
                # ~ print str(i)+' - '+str(i+groupSize)
                xGroup = x[i:i + groupSize, :]
                r = cdist(xGroup, self.C, 'euclidean')
                B = self.basis(r)
                y[:, i:i + groupSize] = np.dot(self.W, B.T) + np.dot(
                    self.poly(xGroup[:, 0], xGroup[:, 1], xGroup[:, 2]).T, self.polyCoeffs)
        else:
            r = cdist(x, self.C, 'euclidean')
            B = self.basis(r)
            # ~ pdb.set_trace()
            y = np.dot(self.W, B.T) + np.dot(self.poly(x[:, 0], x[:, 1], x[:, 2]).T, self.polyCoeffs)

        return y

    def fitData(self, dataX, dataU, fullOutput=True):
        """
        calculate weights for self.C to fit to field sampled at dataX
        with field values dataU
        """
        if self.polyOrder != None:
            return self.fitDataPoly3D(dataX, dataU, fullOutput)

        if dataU.shape[1] != self.nComponents:
            # ~ pdb.set_trace()
            raise ValueError('incorrect number of components in data')

        else:
            # print 'fitting data...'
            self.W, extraInfo = fitData(
                self.C, self.basis, dataX, dataU, verbose=self.verbose
            )
            if fullOutput:
                return self.W, extraInfo
            else:
                return self.W

    def fitDataPoly3D(self, dataX, dataU, fullOutput=False):
        """
        calculate weights for self.C to fit to field sampled at dataX
        with field values dataU
        """
        if dataU.shape[1] != self.nComponents:
            # ~ pdb.set_trace()
            raise ValueError('incorrect number of components in data')

        else:
            # print 'fitting data poly...'
            self.W, self.polyCoeffs, extraInfo = fitDataPoly3D(
                self.C, self.basis, dataX, dataU, self.poly, verbose=self.verbose
            )
            # ~ self.W, extraInfo = fitDataQR( self.C, self.basis, dataX, dataU )
            if fullOutput:
                return self.W, self.polyCoeffs, extraInfo
            else:
                return self.W, self.polyCoeffs

            # =============================================================================#


# Fitting function
# =============================================================================#
def _generateBBoxPointsGrid(points, spacing=None, padding=None):
    """
    generate a grid of points internal to the bounding box of a set of points
    with spacing specified by tuple spacing.
    """
    points = np.array(points)
    # sample surface to get bounding box
    if padding is None:
        bboxMin = points.min(0)
        bboxMax = points.max(0)
    else:
        bboxMin = points.min(0) - padding
        bboxMax = points.max(0) + padding

    if spacing is None:
        spacing = (bboxMax - bboxMin) / 3.0

    N = ((bboxMax - bboxMin) / spacing).astype(int) + 1
    for ni, n in enumerate(N):
        if n < 2:
            N[ni] = 2

    # generate grid of points in bounding box
    PAll = np.array([[x, y, z] for z in np.linspace(bboxMin[2], bboxMax[2], N[2]) \
                     for y in np.linspace(bboxMin[1], bboxMax[1], N[1]) \
                     for x in np.linspace(bboxMin[0], bboxMax[0], N[0])
                     ])

    return PAll


def rbfreg(knots, source, target, basistype, basisargs, disttype, verbose=True):
    """
    Perform a single iteration of RBF registration from source to target
    data points.

    
    """
    # create a RBF Coordinates field
    rcf = RBFComponentsField(3)
    rcf.verbose = verbose

    # set radial basis function
    rcf.makeBasis(basistype, basisargs)

    # create a knot at every 10 data point
    rcf.setCentres(knots)

    # find correspondence
    if disttype == 'st':
        # find closest target point to each source point
        # log.debug('using ts')
        targetTree = cKDTree(target)
        closestInds = targetTree.query(source)[1]
        X = source
        Y = target[closestInds]
    elif disttype == 'ts':
        # find closest source point to each target point
        # log.debug('using ts')
        sourceTree = cKDTree(source)
        closestInds = sourceTree.query(target)[1]
        X = source[closestInds]
        Y = target
    else:
        raise ValueError("disttype must be 'st' or 'ts'")

    # fit knot weights
    rcf.fitData(X, Y)

    # evaluate registered source points
    sourceReg = rcf.evalMany(source).T

    # calculated RMS distance
    d = np.sqrt(((X - Y) ** 2.0).sum(1))
    dRms = np.sqrt((d * d).mean())
    if verbose:
        log.debug('RMS distance: {}'.format(dRms))

    return sourceReg, dRms, rcf, d


def _checkTermination(it, cost1, cost0, nknots, xtol, max_it, max_knots, verbose=True):
    if it > max_it:
        if verbose:
            log.debug('terminating because max iterations reached')
        return True

    if nknots > max_knots:
        if verbose:
            log.debug('terminating because max knots reached')
        return True

    if (abs(cost1 - cost0) / cost0) < xtol:
        log.debug(abs(cost1 - cost0) / cost0)
        if verbose:
            log.debug('terminating because xtol reached')
        return True

    return False


def rbfRegIterative(source, target, distmode='ts', knots=None,
                    basisType='gaussianNonUniformWidth', basisArgs=None, xtol=1e-3,
                    minKnotDist=5.0, maxIt=50, maxKnots=500, maxKnotsPerIt=20, verbose=True
                    ):
    """Iterative RBF registration with greedy knots adding per iteration.

    New knots are placed on registered source points.

    inputs
    ------
    source: list of source point coordinates.
    target: list of target point coordinates.
    distmode: how source to target distance is calculated.
        'st': source to target - between each source point and closest target
              point.
        'ts': target to source - between each target point and closest source
              point.
        'alt': alternate between st and ts each iteration
    knots: list of knot coordinates.
    basisType: Radial basis type.
    basisArgs: dictionary of arguments fro the basis type.
    xtol: relative change in error for termination.
    maxIt: max number of iterations.
    minKnotDist: minimum distance between knots.
    maxKnotsPerIt: max number of knots to add per iteration.

    returns
    -------
    sourceCurrent: final morphed source point coordinates.
    rms: final RMS distance between morphed source and target points.
    rcf: final RBF deformation field.
    history: fitting results from each iteration. Dict containing
        'rms': rms error at each iteration,
        'ssdist': sum of squared distance at each iteration,
        'nknots': number knots at each iteration.
    """

    if basisArgs is None:
        basisArgs = {'s': 1.0, 'scaling': 500.0}

    if knots is None:
        knots = _generateBBoxPointsGrid(source, padding=10.0)

    terminate = False
    it = 0
    sourceCurrent = source
    ssdistCurrent = 9999999999999
    history = {
        'rms': [],
        'ssdist': [],
        'nknots': [],
    }
    distmodes = ['ts', 'st']
    while not terminate:

        # perform fit
        if distmode == 'alt':
            _distmode = distmodes[it % 2]
        else:
            _distmode = distmode

        sourceNew, rms, rcf, dist = rbfreg(
            knots,
            sourceCurrent,
            target,
            basisType,
            basisArgs,
            _distmode,
            verbose=verbose
        )
        ssdistNew = (dist * dist).sum()

        # check if should terminate
        if distmode == 'alt':
            if not it % 2:
                terminate = _checkTermination(
                    it, ssdistNew, ssdistCurrent, knots.shape[0],
                    xtol, maxIt, maxKnots, verbose=verbose
                )
        else:
            terminate = _checkTermination(
                it, ssdistNew, ssdistCurrent, knots.shape[0],
                xtol, maxIt, maxKnots, verbose=verbose
            )

        # add knot
        if not terminate:
            if verbose:
                log.debug('\niteration {}'.format(it))
            # find source locations with highest errors
            sourceTree = cKDTree(sourceNew)
            tsDist, tsInds = sourceTree.query(target, k=1)
            sourceMaxDistInds = tsInds[np.argsort(tsDist)[::-1]]

            # go through source points from highest error and find
            # first one that is more than min_knot_dist from an
            # existing knot
            nKnotsAdded = 0
            for maxInd in sourceMaxDistInds:
                knotsTree = cKDTree(knots)
                closestKnotDist = knotsTree.query(sourceNew[maxInd])[0]
                if closestKnotDist > minKnotDist:
                    knots = np.vstack([knots, sourceNew[maxInd]])
                    nKnotsAdded += 1

                if nKnotsAdded == maxKnotsPerIt:
                    break

            if nKnotsAdded == 0:
                terminate = True
                if verbose:
                    log.debug('terminating because no new knots can be added')

        sourceCurrent = sourceNew
        ssdistCurrent = ssdistNew
        history['rms'].append(rms)
        history['ssdist'].append(ssdistCurrent)
        history['nknots'].append(len(knots))
        it += 1

    return sourceCurrent, rms, rcf, history


def rbfRegIterative2(source, target, distmode='ts', knots=None,
                     basisType='gaussianNonUniformWidth', basisArgs=None, xtol=1e-3,
                     minKnotDist=5.0, maxIt=50, maxKnots=500, maxKnotsPerIt=20):
    """Iterative RBF registration with greedy knots adding per iteration.

    New knots are placed on source points.

    inputs
    ------
    source: list of source point coordinates.
    target: list of target point coordinates.
    distmode: how source to target distance is calculated.
        'st': source to target - between each source point and closest target
              point.
        'ts': target to source - between each target point and closest source
              point.
        'alt': alternate between st and ts each iteration
    knots: list of knot coordinates.
    basisType: Radial basis type.
    basisArgs: dictionary of arguments fro the basis type.
    xtol: relative change in error for termination.
    maxIt: max number of iterations.
    minKnotDist: minimum distance between knots.
    maxKnotsPerIt: max number of knots to add per iteration.

    returns
    -------
    sourceCurrent: final morphed source point coordinates.
    rms: final RMS distance between morphed source and target points.
    rcf: final RBF deformation field.
    history: fitting results from each iteration. Dict containing
        'rms': rms error at each iteration,
        'ssdist': sum of squared distance at each iteration,
        'nknots': number knots at each iteration.
    """

    if basisArgs is None:
        basisArgs = {'s': 1.0, 'scaling': 500.0}

    if knots is None:
        knots = _generateBBoxPointsGrid(source, padding=10.0)

    terminate = False
    it = 0
    sourceCurrent = source
    ssdistCurrent = 9999999999999
    history = {
        'rms': [],
        'ssdist': [],
        'nknots': [],
    }
    distmodes = ['ts', 'st']
    while not terminate:

        # perform fit
        if distmode == 'alt':
            _distmode = distmodes[it % 2]
        else:
            _distmode = distmode

        sourceNew, rms, rcf, dist = rbfreg(
            knots,
            sourceCurrent,
            target,
            basisType,
            basisArgs,
            _distmode,
        )
        ssdistNew = (dist * dist).sum()

        # check if should terminate
        if distmode == 'alt':
            if not it % 2:
                terminate = _checkTermination(
                    it, ssdistNew, ssdistCurrent, knots.shape[0],
                    xtol, maxIt, maxKnots,
                )
        else:
            terminate = _checkTermination(
                it, ssdistNew, ssdistCurrent, knots.shape[0],
                xtol, maxIt, maxKnots,
            )

        # add knot
        if not terminate:
            log.debug('\niteration {}'.format(it))
            # find source locations with highest errors
            sourceTree = cKDTree(sourceNew)
            tsDist, tsInds = sourceTree.query(target, k=1)
            sourceMaxDistInds = tsInds[np.argsort(tsDist)[::-1]]

            # go through source points from highest error and find
            # first one that is more than min_knot_dist from an
            # existing knot
            nKnotsAdded = 0
            for maxInd in sourceMaxDistInds:
                knotsTree = cKDTree(knots)
                closestKnotDist = knotsTree.query(source[maxInd])[0]
                if closestKnotDist > minKnotDist:
                    knots = np.vstack([knots, source[maxInd]])
                    nKnotsAdded += 1

                if nKnotsAdded == maxKnotsPerIt:
                    break

            if nKnotsAdded == 0:
                terminate = True
                log.debug('terminating because no new knots can be added')

        sourceCurrent = sourceNew
        ssdistCurrent = ssdistNew
        history['rms'].append(rms)
        history['ssdist'].append(ssdistCurrent)
        history['nknots'].append(len(knots))
        it += 1

    return sourceCurrent, rms, rcf, history


def rbfRegIterative3(source, target, distmode='ts', knots=None,
                     basisType='gaussianNonUniformWidth', basisArgs=None, xtol=1e-3,
                     minKnotDist=5.0, maxIt=50, maxKnots=500, maxKnotsPerIt=20):
    """Iterative RBF registration with greedy knots adding per iteration.

    Each iteration only contains it 0 knots plus those added in the last
    iteration.

    inputs
    ------
    source: list of source point coordinates.
    target: list of target point coordinates.
    distmode: how source to target distance is calculated.
        'st': source to target - between each source point and closest target
              point.
        'ts': target to source - between each target point and closest source
              point.
        'alt': alternate between st and ts each iteration
    knots: list of knot coordinates.
    basisType: Radial basis type.
    basisArgs: dictionary of arguments fro the basis type.
    xtol: relative change in error for termination.
    maxIt: max number of iterations.
    minKnotDist: minimum distance between knots.
    maxKnotsPerIt: max number of knots to add per iteration.

    returns
    -------
    sourceCurrent: final morphed source point coordinates.
    rms: final RMS distance between morphed source and target points.
    rcf: final RBF deformation field.
    history: fitting results from each iteration. Dict containing
        'rms': rms error at each iteration,
        'ssdist': sum of squared distance at each iteration,
        'nknots': number knots at each iteration.
    """

    if basisArgs is None:
        basisArgs = {'s': 1.0, 'scaling': 500.0}

    if knots is None:
        knots0 = _generateBBoxPointsGrid(source, padding=10.0)
    else:
        knots0 = np.array(knots)

    # knots for 1st iteration
    knots = np.array(knots0)

    terminate = False
    it = 0
    sourceCurrent = source
    ssdistCurrent = 9999999999999
    history = {
        'rms': [],
        'ssdist': [],
        'nknots': [],
    }
    distmodes = ['ts', 'st']
    while not terminate:

        # perform fit
        if distmode == 'alt':
            _distmode = distmodes[it % 2]
        else:
            _distmode = distmode

        sourceNew, rms, rcf, dist = rbfreg(
            knots,
            sourceCurrent,
            target,
            basisType,
            basisArgs,
            _distmode,
        )
        ssdistNew = (dist * dist).sum()

        # check if should terminate
        if distmode == 'alt':
            if not it % 2:
                terminate = _checkTermination(
                    it, ssdistNew, ssdistCurrent, knots.shape[0],
                    xtol, maxIt, maxKnots,
                )
        else:
            terminate = _checkTermination(
                it, ssdistNew, ssdistCurrent, knots.shape[0],
                xtol, maxIt, maxKnots,
            )

        # add knot
        if not terminate:
            log.debug('\niteration {}'.format(it))
            # find source locations with highest errors
            sourceTree = cKDTree(sourceNew)
            tsDist, tsInds = sourceTree.query(target, k=1)
            sourceMaxDistInds = tsInds[np.argsort(tsDist)[::-1]]

            # go through source points from highest error and find
            # first one that is more than min_knot_dist from an
            # existing knot
            nKnotsAdded = 0
            knots = np.array(knots0)
            for maxInd in sourceMaxDistInds:
                knotsTree = cKDTree(knots)
                closestKnotDist = knotsTree.query(source[maxInd])[0]
                if closestKnotDist > minKnotDist:
                    knots = np.vstack([knots, source[maxInd]])
                    nKnotsAdded += 1

                if nKnotsAdded == maxKnotsPerIt:
                    break

            if nKnotsAdded == 0:
                terminate = True
                log.debug('terminating because no new knots can be added')

        sourceCurrent = sourceNew
        ssdistCurrent = ssdistNew
        history['rms'].append(rms)
        history['ssdist'].append(ssdistCurrent)
        history['nknots'].append(len(knots))
        it += 1

    return sourceCurrent, rms, rcf, history


def rbfRegNPass(source, target, init_rot=(0, 0, 0), rbfargs=None, verbose=False):
    """
    Multi-pass RBF fitting from source to target point clouds.

    Each pass is run with the parameters contained in one item in rbfargs.

    Inputs
    ------
    source : np.ndarray
        nx3 array of source point coordinates
    target : np.ndarray
        mx3 array of target point coordinates
    init_rot : tuple
        3 rotation angles (radians) for initial rotation to apply
        to source before fitting
    rbfargs : list of dicts
        should be a list of dicts containing the rbf fitting args
        for each fitting pass. Default is 2 pass with these args:

        DEFAULT_PARAMS = [
            {
                'basisType': 'gaussianNonUniformWidth',
                'basisArgs': {'s':1.0, 'scaling':1000.0},
                'distmode': 'alt',
                'xtol': 1e-1,
                'maxIt': 20,
                'maxKnots': 500,
                'minKnotDist': 20.0,
                'maxKnotsPerIt': 20,
            },
            {
                'basisType': 'gaussianNonUniformWidth',
                'basisArgs': {'s':1.0, 'scaling':10.0},
                'distmode': 'alt',
                'xtol': 1e-3,
                'maxIt': 20,
                'maxKnots': 1000,
                'minKnotDist': 2.5,
                'maxKnotsPerIt': 20,
            }
        ]
    verbose : bool
        print extra outputs

    Returns
    -------
    _source : np.ndarray
        nx3 array of the fitted source point coordinates
    (rms_i, rcf_i) : Tuple[float, RbfCoordinateField]
        The rms error and coordinate field for each fitting pass
    """

    if rbfargs is None:
        rbfargs = [
            {
                'basisType': 'gaussianNonUniformWidth',
                'basisArgs': {'s': 1.0, 'scaling': 1000.0},
                'distmode': 'alt',
                'xtol': 1e-1,
                'maxIt': 20,
                'maxKnots': 500,
                'minKnotDist': 20.0,
                'maxKnotsPerIt': 20,
            },
            {
                'basisType': 'gaussianNonUniformWidth',
                'basisArgs': {'s': 1.0, 'scaling': 10.0},
                'distmode': 'alt',
                'xtol': 1e-3,
                'maxIt': 20,
                'maxKnots': 1000,
                'minKnotDist': 2.5,
                'maxKnotsPerIt': 20,
            }
        ]

    init_rot = np.deg2rad(init_rot)
    n_iterations = len(rbfargs)

    # =============================================================#
    # rigidly register source points to target points
    init_trans = target.mean(0) - source.mean(0)
    t0 = np.hstack([init_trans, init_rot])
    reg1_T, source_reg1, reg1_errors = af.fitDataRigidDPEP(
        source,
        target,
        xtol=1e-6,
        sample=1000,
        t0=t0,
        outputErrors=1,
    )

    # add isotropic scaling to rigid registration
    reg2_T, source_reg2, reg2_errors = af.fitDataRigidScaleDPEP(
        source,
        target,
        xtol=1e-6,
        sample=1000,
        t0=np.hstack([reg1_T, 1.0]),
        outputErrors=1,
    )

    # =============================================================#

    _source = source_reg2
    for it, rbfargs_i in enumerate(rbfargs):
        if verbose:
            log.debug('RBF registration pass {}'.format(it + 1))

        _source, rms_i, rcf_i, regHist = rbfRegIterative(
            _source, target, verbose=verbose, **rbfargs_i
        )

    if verbose:
        log.debug('RBF registration final rms: {}'.format(rms_i))

    return _source, (rms_i, rcf_i)
