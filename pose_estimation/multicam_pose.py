"""Implements the multi-camera pose estimation algorithm proposed in:

@article{Lee2015,
    author = {Gim Hee Lee and Bo Li and Marc Pollefeys and Friedrich Fraundorfer},
    title ={Minimal solutions for the multi-camera pose estimation problem},
    journal = {The International Journal of Robotics Research},
    volume = {34},
    number = {7},
    pages = {837-848},
    year = {2015},
    doi = {10.1177/0278364914557969},
    URL = {https://doi.org/10.1177/0278364914557969}
}

Open access: https://folia.unifr.ch/global/documents/234169
Early draft: https://people.inf.ethz.ch/pomarc/pubs/LeeISRR13.pdf
"""
from typing import Tuple

import numpy as np

def threepoint_abs_pose(x1: np.ndarray, x2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Finds a rigid transform (R, t) that aligns two constellations of 3 points.

    Args:
        x1 (np.ndarray): constellation 1 as a 3 x 3 array, points are columns
        x2 (np.ndarray): constellation 2

    Returns:
        Tuple[np.ndarray, np.ndarray]: the sought rotation/traslation pair (R, t)
            such that x2 = R * x1 + t.

    Algorithm:
        1) Place the origin at the first point in each set.
        2) Find combinations of rotations around y- and z-axes that position the 
           second point in each set on the x-axis in some intermediate frame.
        3) Using the third point from each set, find a rotation around the x-axis 
           that aligns the points in the intermediate frames.
        4) Use the results from 2 & 3 to compose R, t. 

    While this is a minimal solution to the abolute pose problem, it has an edge case 
    if points in x2 have the same z-coordinates, resulting in NaNs due to division by zero.
    Use np.isnan() to filter out such instances.
    """

    assert x1.shape == (3, 3) and x1.shape == x2.shape

    p1 = x1 - x1[:, 0, np.newaxis]
    p2 = x2 - x2[:, 0, np.newaxis]

    def xalign_rot(p):
        assert p.size == 3
        mag = np.linalg.norm(p)
        d = -p[2] / mag
        c = np.sqrt(1 - d*d)
        e = p[0] / (mag * c)
        f = p[1] / (mag * c)

        return np.array([[c*e, -f, d*e], [c*f, e, d*f], [-d, 0.0, c]])

    R1 = xalign_rot(p1[:, 1])
    assert np.allclose(R1 @ np.array([np.linalg.norm(p1[:, 1]), 0, 0]), p1[:, 1])
    R2 = xalign_rot(p2[:, 1])
    assert np.allclose(R2 @ np.array([np.linalg.norm(p2[:, 1]), 0, 0]), p2[:, 1])

    u = R1.T @ p1[:, 2]
    v = R2.T @ p2[:, 2]
    sx = (u[1] * v[1] + u[2] * v[2]) / (v[1] * v[1] + v[2] * v[2])
    cx = (v[1] * sx - u[1]) / v[2]

    Rx = np.array([[1, 0, 0], [0, sx, -cx], [0, cx, sx]])

    # The formulas below differ from the ones quoted in the paper!
    # This is because we seek the inverse of what is calculated in the paper.
    # Note that the original formula for R in Eq. (16) in the paper accepts point set M 
    # (equivalent to p1 above) on the right and N (equivalent to p2) on the left, 
    # which does not agree with the definition of R in Eq. (7)! 
    R = R2 @ Rx.T @ R1.T
    t = -R @ x1[:, 0] + x2[:, 0]

    return R, t