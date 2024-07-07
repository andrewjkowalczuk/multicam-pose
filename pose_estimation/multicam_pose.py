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
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from .solver import solve

@dataclass
class PluckerLine:
    """Plucker parameterization of a line in 3D (direction-moment)"""

    d: np.ndarray
    m: np.ndarray

    @staticmethod
    def from_point_and_direction(p: np.ndarray, d: np.ndarray) -> 'PluckerLine':
        return PluckerLine(d=d, m=np.cross(p, d))

def multicam_abs_pose(
        camera_centers: np.ndarray,
        camera_rays: np.ndarray,
        world_points: np.ndarray
    ) -> List[Tuple[np.ndarray, Tuple[float, float, float]]]:
    """Estimates the pose of a multi-camera system given observations of 3 points.
    
    This implementation consumes camera centers and rays pointing towards 
    observations, making it independent of camera models. Any setup of 3 cameras, 
    not pinhole cameras in general and not necessarily sharing the formulation
    of the camera model, can be interfaced with this routine.

    Args:
        camera_centers (np.ndarray): 3 x 3 matrix of camera centers (points are columns)
        camera_rays (np.ndarray): 3 x 3 matrix of camera rays
        world_points (np.ndarray): 3 x 3 matrix of observed world points

    Returns:
        List[Tuple[np.ndarray, Tuple[float, float, float]]]: a list of 
            (rig pose, depth triplet) pairs, where depths define the position of
            observed points along cameras rays, i.e., pt = center + d * ray
    """
    print(camera_centers.shape, camera_rays.shape, world_points.shape)
    assert camera_centers.shape == (3, 3) and camera_centers.shape == camera_rays.shape \
        and camera_centers.shape == world_points.shape

    # Represent camera rays using Plucker lines
    lines: List[PluckerLine] = []
    for i in range(3):
        lines.append(
            PluckerLine.from_point_and_direction(camera_centers[:, i], camera_rays[:, i]))

    # Form key equations and call the solver, need pair-wise point distances
    distances = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            if i != j:
                diff = world_points[0:3, i] - world_points[0:3, j]
                distances[i, j] = np.dot(diff, diff)

    def make_constraint(i: int, j: int):
        # To arrive at the coefficients k1, ..., k6, rewrite the LHS of Eq. (3) as
        # \lVert d_i \times m_i + \lambda_i d_i - (d_j \times m_j + \lambda_j d_j) \rVert^2  =
        # (\lambda_i d_i^T - \lambda_j d_j^T + c_{ij}^T)(\lambda_i d_i - \lambda_j d_j + c_{ij})
        # where c_{ij} = d_i \times m_i - d_j \times m_j, the multiply and group terms
        c = np.cross(lines[i].d, lines[i].m) - np.cross(lines[j].d, lines[j].m)

        k1 = np.dot(lines[i].d, lines[i].d)
        k2 = -2 * np.dot(lines[i].d, lines[j].d)
        k3 = 2 * np.dot(lines[i].d, c)
        k4 = np.dot(lines[j].d, lines[j].d)
        k5 = -2 * np.dot(lines[j].d, c)
        k6 = np.dot(c, c) - distances[i, j]

        return [k1, k2, k3, k4, k5, k6]

    K = np.zeros((3, 6))
    K[0, :] = make_constraint(0, 1)
    K[1, :] = make_constraint(0, 2)
    K[2, :] = make_constraint(1, 2)

    lambdas = solve(K)

    # Recover points form Plucker depths, work out pose transforms
    solutions = []
    for (d1, d2, d3) in lambdas:
        solved_points = np.zeros((3, 3))
        solved_points[:, 0] = np.cross(lines[0].d, lines[0].m) + d1 * lines[0].d
        solved_points[:, 1] = np.cross(lines[1].d, lines[1].m) + d2 * lines[1].d
        solved_points[:, 2] = np.cross(lines[2].d, lines[2].m) + d3 * lines[2].d

        rig_from_world = np.eye(4)
        rig_from_world[0:3, 0:3], rig_from_world[0:3, 3] = \
            threepoint_abs_pose(world_points[0:3, :], solved_points)

        if np.any(np.isnan(rig_from_world)):
            continue

        # Translate Plucker depths to depths along camera rays
        depths = [np.dot(camera_rays[:, i], solved_points[:, i] - camera_centers[:, i]) for i in range(3)]

        solutions.append((rig_from_world, depths))

    return solutions

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