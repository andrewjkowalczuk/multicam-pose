from typing import Sequence, Union

import matplotlib.pyplot as plt
import numpy as np

def draw_pose(local_from_ref: np.ndarray, scale: float = 1.0, label: str = ''):
    """Draws axes of the coordinate system described by the given transform.

    X-axis is drawn in red, Y in green, Z in blue.

    Args:
        local_from_ref (np.ndarray): a 4 x 4 transformation that takes points
            expressed in the reference coordinate system to the coordinate system
            of interest (local-from-reference).
        scale (float, optional): Scale of the axes drawn, e.g., 0.01 produces
            axes with a length of 0.01 world units (meters) = 1cm. Defaults to 1.0.
        label (str, optional): Label text. Defaults to ''.
    """

    ref_from_local = np.eye(4)
    ref_from_local[0:3, 0:3] = local_from_ref[0:3, 0:3].T
    ref_from_local[0:3, 3] = -ref_from_local[0:3, 0:3] @ local_from_ref[0:3, 3] # -R' * t idiom

    R, t = ref_from_local[0:3, 0:3], ref_from_local[0:3, 3]

    origin = R @ np.array([0, 0, 0]) + t

    x_axis = R @ (scale * np.array([1, 0, 0])) + t
    y_axis = R @ (scale * np.array([0, 1, 0])) + t
    z_axis = R @ (scale * np.array([0, 0, 1])) + t

    plt.plot([origin[0], x_axis[0]], [origin[1], x_axis[1]], [origin[2], x_axis[2]], 'r-')
    plt.plot([origin[0], y_axis[0]], [origin[1], y_axis[1]], [origin[2], y_axis[2]], 'g-')
    plt.plot([origin[0], z_axis[0]], [origin[1], z_axis[1]], [origin[2], z_axis[2]], 'b-')

    if label:
        plt.gca().text3D(origin[0], origin[1], origin[2], label, color='black')

def draw_camera(
        camera_from_world: np.ndarray,
        fov: Sequence = (90., 60.), # in degrees
        scale: float = 1.0,
        color: Union[str, Sequence] = 'gray',
        label: str = ''):
    """Draws a camera coordinate frame and viewing volume.

    Args:
        camera_from_world (np.ndarray): 4 x 4 camera-from-world transformation matrix
        fov (Sequence): Field of view in degrees.
        scale (float, optional): Scale for drawing. Defaults to 1.0.
        color(Union[str, Sequence], optional): Line color; string or RGB(A) sequence.
            Defaults to 'grey'.
        label (str, optional): Label text, e.g., camera name. Defaults to ''.
    """

    rect = np.tan(np.deg2rad(0.5 * np.array([-fov[0], fov[0], -fov[1], fov[1]])))

    world_from_camera = np.eye(4)
    world_from_camera[0:3, 0:3] = camera_from_world[0:3, 0:3].T
    world_from_camera[0:3, 3] = -world_from_camera[0:3, 0:3] @ camera_from_world[0:3, 3] # -R' * t idiom

    R, t = world_from_camera[0:3, 0:3], world_from_camera[0:3, 3]

    z = scale
    xmin = z * rect[0]
    xmax = z * rect[1]
    ymin = z * rect[2]
    ymax = z * rect[3]

    # L, T, R, B
    faces = np.array([ \
        [0, xmin, xmin, 0, xmin, xmax, 0, xmax, xmax, 0, xmax, xmin, 0], \
        [0, ymin, ymax, 0, ymax, ymax, 0, ymax, ymin, 0, ymin, ymin, 0], \
        [0,    z,    z, 0,    z,    z, 0,    z,    z, 0,    z,    z, 0]  \
    ])

    faces = R @ faces + t[:, np.newaxis] # need np.newaxis for broadcasting

    draw_pose(camera_from_world, scale=scale, label=label)

    plt.plot(faces[0, :], faces[1, :], faces[2, :], color=color)
