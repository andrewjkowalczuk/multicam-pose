from dataclasses import dataclass
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from drawing import draw_camera, draw_pose
from pose_estimation.multicam_pose import multicam_abs_pose

@dataclass
class Camera:
    """Represents a pinhole camera"""

    pinhole: np.ndarray
    transform: np.ndarray # camera from rig

def main():
    # Experimental setup: 
    # 3 pinhole cameras, each mapping a 90 x 67.5 deg. FOV to a 1440 x 1080px 
    # image space, observing a planar grid of points, e.g., a checkerboard.
    # The first camera serves as the origin of the camera rig.
    cameras = [
        Camera(
            pinhole=np.array([
                [720.0000,   0.0000, 720.0000],
                [  0.0000, 808.1671, 540.0000],
                [  0.0000,   0.0000,   1.0000]
            ]),
            transform=np.eye(4)) for i in range(3)
    ]

    cameras[1].transform[0:3, 3] = [ 0.56845614, -0.00416491,  0.17729697]
    cameras[1].transform[0:3, 0:3] = Rotation.from_quat([ 0.15215065, -0.25592707, -0.4938431,  0.81698868]).as_matrix()
    cameras[2].transform[0:3, 3] = [ 0.20033441, -0.46549121,  0.13791876]
    cameras[2].transform[0:3, 0:3] = Rotation.from_quat([-0.17897584,  0.19216407,  0.9085472,  0.32493477]).as_matrix()

    rig_from_world = np.array([
        [-0.98467744,  0.14666741, -0.09433452,  0.        ],
        [-0.17126545, -0.91521966,  0.36474801,  0.        ],
        [-0.03284016,  0.37531538,  0.92631522,  1.        ],
        [ 0.        ,  0.        ,  0.        ,  1.        ]
    ])

    checker_size = 0.1 # in world units (meters)
    grid_shape = (7, 11)
    world_points = np.array(
        [[i * checker_size, j * checker_size, 0., 1.] for i in range(grid_shape[1]) for j in range(grid_shape[0])]
    ).T
    world_points[0, :] -= (grid_shape[1] - 1) * 0.5 * checker_size
    world_points[1, :] -= (grid_shape[0] - 1) * 0.5 * checker_size

    image_points = np.zeros((3, world_points.shape[1], 3))
    for i, camera in enumerate(cameras):
        camera_points = camera.transform @ rig_from_world @ world_points
        image_points[:, :, i] = camera.pinhole @ camera_points[0:3, :]
        image_points[:, :, i] /= image_points[2, :, i]

    noise_sigma = 0.25
    np.random.seed(54547)
    image_points[0:2, :, :] += noise_sigma * np.random.randn(*image_points[0:2, :, :].shape)

    point_indices = [50, 17, 54] # 

    three_world_points = world_points[:, point_indices]

    three_image_points = np.zeros((3, 3))
    three_image_points[:, 0] = image_points[:, point_indices[0], 0]
    three_image_points[:, 1] = image_points[:, point_indices[1], 1]
    three_image_points[:, 2] = image_points[:, point_indices[2], 2]

    # Calculate camera centers, convert observations to rays, run pose estimation
    camera_centers = np.zeros((3, 3))
    camera_rays = np.zeros((3, 3))
    for i, camera in enumerate(cameras):
        camera_rays[:, i] = camera.transform[0:3, 0:3].T @ np.linalg.inv(camera.pinhole) @ three_image_points[:, i]
        camera_rays[:, i] /= np.linalg.norm(camera_rays[:, i])

        camera_centers[:, i] = -camera.transform[0:3, 0:3].T @ camera.transform[0:3, 3]

    solutions = multicam_abs_pose(camera_centers, camera_rays, three_world_points[0:3, :])

    assert solutions

    # Pick the solution with the lowest reprojection error
    def evaluate(
        rig_from_world: np.ndarray,
        cameras: List[Camera],
        world_points: np.ndarray,
        image_points: np.ndarray) -> float:
        """Calculates reprojection errors w.r.t. a given rig-from-world transform"""

        # signed_errors = image_points - project(R * world_points + t)
        predictions = np.zeros((3, world_points.shape[1], len(cameras)))
        for i, camera in enumerate(cameras):
            camera_points = camera.transform @ rig_from_world @ world_points
            predictions[:, :, i] = camera.pinhole @ camera_points[0:3, :]
            predictions[:, :, i] /= predictions[2, :, i]

        return image_points[0:2, :, :] - predictions[0:2, :, :]

    mean_errors = []
    for transform, _ in solutions:
        signed_errors = evaluate(transform, cameras, world_points, image_points)
        mean_errors.append(
            np.sqrt(signed_errors[0, :, :] ** 2 + signed_errors[1, :, :] ** 2).mean())

    pose, _ = solutions[mean_errors.index(min(mean_errors))]

    # Refine rig pose estimate by minimizing reprojection error.
    # Rotation.from_matrix(...).as_rotvec() is effectively so(3) log().
    params = np.hstack((Rotation.from_matrix(pose[0:3, 0:3]).as_rotvec(), pose[0:3, 3]))

    def cost_func(x):
        pose = np.eye(4)
        pose[0:3, 0:3] = Rotation.from_rotvec(x[0:3]).as_matrix()
        pose[0:3, 3] = x[3:]

        return evaluate(pose, cameras, world_points, image_points).flatten()

    result = least_squares(cost_func, params)

    optimized_pose = np.eye(4)
    optimized_pose[0:3, 0:3] = Rotation.from_rotvec(result.x[0:3]).as_matrix()
    optimized_pose[0:3, 3] = result.x[3:]

    # How close did we land?
    mean_err = lambda signed_errors: np.sqrt(signed_errors[0, :, :] ** 2 + signed_errors[1, :, :] ** 2).mean()
    initial_error = mean_err(evaluate(pose, cameras, world_points, image_points))
    final_error = mean_err(evaluate(optimized_pose, cameras, world_points, image_points))
    pose_diff = np.linalg.inv(optimized_pose) @ rig_from_world

    print(result)
    print()
    print(f"   initial error: {initial_error:.6f}px")
    print(f"     final error: {final_error:.6f}px")
    print(f" rotational diff: {1000 * np.linalg.norm(Rotation.from_matrix(pose_diff[0:3, 0:3]).as_rotvec()):.6f}mrad")
    print(f" positional diff: {np.linalg.norm(pose_diff[0:3, 3]):.6f}m")

    f = plt.figure()
    ax = f.add_subplot(projection='3d')

    colors = np.linspace(0.0, 1.0, image_points.shape[1])
    ax.scatter(world_points[0, :], world_points[1, :], world_points[2, :], s=2.0, c=colors, cmap='Spectral')
    draw_pose(np.eye(4), scale=0.1)

    for i, camera in enumerate(cameras):
        world_from_camera = np.linalg.inv(camera.transform @ rig_from_world)
        ax.plot(
            [world_from_camera[0, 3], three_world_points[0, i]],
            [world_from_camera[1, 3], three_world_points[1, i]],
            [world_from_camera[2, 3], three_world_points[2, i]],
            '--',
            color='gray',
            alpha=0.5)
        draw_camera(camera.transform @ rig_from_world, (90.0, 67.5), scale=0.05, label=f"CAM{i}")

    ax.set_aspect('equal')

    f, ax = plt.subplots(1, 3, figsize=(12, 3.5))
    for i, camera in enumerate(cameras):
        def project(world_pt):
            camera_pt = camera.transform @ optimized_pose @ world_pt
            camera_pt = camera_pt[0:3] / camera_pt[2]
            return camera.pinhole @ camera_pt

        s = 0.1
        origin = project(np.array([0, 0, 0, 1]))
        x_axis = project(np.array([s, 0, 0, 1]))
        y_axis = project(np.array([0, s, 0, 1]))
        z_axis = project(np.array([0, 0, s, 1]))

        ax[i].plot([origin[0], x_axis[0]], [origin[1], x_axis[1]], 'r-')
        ax[i].plot([origin[0], y_axis[0]], [origin[1], y_axis[1]], 'g-')
        ax[i].plot([origin[0], z_axis[0]], [origin[1], z_axis[1]], 'b-')

        ax[i].scatter(image_points[0, :, i], image_points[1, :, i], s=1.0, c=colors, cmap='Spectral')

        ax[i].set_xlim(0, 1440)
        ax[i].set_ylim(0, 1080)
        ax[i].set_facecolor('black')
        ax[i].set_aspect('equal')
        ax[i].xaxis.set_ticks([])
        ax[i].yaxis.set_ticks([])
        ax[i].invert_yaxis()
        ax[i].set_title(f"CAM{i}")

        plt.tight_layout()

    plt.show()

if __name__ == "__main__":
    main()