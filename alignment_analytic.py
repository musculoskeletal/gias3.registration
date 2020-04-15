"""
FILE: alignment_analytic.py
LAST MODIFIED: 24-12-2015 
DESCRIPTION: Analytical alignment of points.

===============================================================================
This file is part of GIAS2. (https://bitbucket.org/jangle/gias2)

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
===============================================================================
"""

import numpy as np

from gias2.common import math
from gias2.common import transform3D


def norm(v):
    m = np.sqrt((v.astype(float) ** 2.0).sum())
    return v / m


def calcAffine(old, new):
    """ calc affine matrix to transform old(origin, pAxes) to new(origin, pAxes)
    where pAxes = [X, Y, Z] where X, Y and Z are column vectors
    """

    dataLandmarks = np.array([old[0],
                              np.add(old[0], old[1][:, 0]),
                              np.add(old[0], old[1][:, 1]),
                              np.add(old[0], old[1][:, 2])])

    targetLandmarks = np.array([new[0],
                                np.add(new[0], new[1][:, 0]),
                                np.add(new[0], new[1][:, 1]),
                                np.add(new[0], new[1][:, 2])])

    affineMatrix = transform3D.directAffine(dataLandmarks, targetLandmarks)

    return affineMatrix


def scaleAlignScan(scan, s):
    """ affine transform scan to line up its principal axes with global
    x ,y, z at the centre of mass, and scales by 3-tuple s
    
    returns pointer to scan object, and [CoM, pAxes] after scaling,
    before alignment.
    """

    scale_matrix = np.array([[1 / s[0], 0.0, 0.0],
                            [0.0, 1 / s[1], 0.0],
                            [0.0, 0.0, 1 / s[2]]])

    scan.affine(scale_matrix, order=3)
    old_com = list(scan.CoM)
    old_p_axes = np.array(scan.pAxes)
    # ~ scan.crop( 10 )

    target_landmarks = [old_com, np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])]
    data_landmarks = [old_com, old_p_axes]
    align_matrix = calcAffine(target_landmarks, data_landmarks)  # affine is the otherway round

    scan.affine(align_matrix[:3, :3], offset=align_matrix[:3, -1], order=3)
    scan.crop(10)

    return scan, [old_com, old_p_axes]


def alignAffinePoints(X, u, ut):
    """
    based on 4 pairs of points in u and ut, calculate and apply an affine
    transform to X
    """
    t = transform3D.directAffine(u, ut)
    return transform3D.transformAffine(X, t)


def calcAffineRigid3Points(p1, p2):
    """ calculates rigid affine matrix that transforms
    p1 to p2
    """
    cs1 = calcOrthogCS(p1[0], p1[1], p1[2])
    cs2 = calcOrthogCS(p2[0], p2[1], p2[2])
    U = np.array([cs1[0],
                  cs1[0] + cs1[1],
                  cs1[0] + cs1[2],
                  cs1[0] + cs1[3],
                  ])
    UT = np.array([cs2[0],
                   cs2[0] + cs2[1],
                   cs2[0] + cs2[2],
                   cs2[0] + cs2[3],
                   ])
    return transform3D.directAffine(U, UT)


def alignRigid3Points(X, u, ut):
    """
    based on 3 pairs of points in u and ut, calculate and apply a rigid
    transform to X
    """
    t = calcAffineRigid3Points(u, ut)
    return transform3D.transformAffine(X, t)


def norm4Points(x):
    mat = np.zeros((4, 3))
    mat[0] = x[0]
    mat[1] = x[0] + math.norm(x[1] - x[0])
    mat[2] = x[0] + math.norm(x[2] - x[0])
    mat[3] = x[0] + math.norm(x[3] - x[0])
    return mat


def calcOrthogCS(x1, x2, x3):
    """
    Calculates an orthogonal CS using 3 points.
    Origin is midpoint between x1 and x2
    Z is cross product of ox1 and ox3
    Y is cross product of Z and ox3
    Z is cross product of X and Y
    """
    o = 0.5 * (x1 + x2)
    z = np.cross(x1 - o, x3 - o)
    y = np.cross(z, x1 - o)
    x = np.cross(y, z)
    x = math.norm(x)
    y = math.norm(y)
    z = math.norm(z)
    return o, x, y, z
