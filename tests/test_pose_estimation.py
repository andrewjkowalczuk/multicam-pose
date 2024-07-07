import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from pose_estimation.multicam_pose import threepoint_abs_pose

# cd multicam-pose
# python -m tests.test_pose_estimation
class PoseEstimationTests(unittest.TestCase):

    def test_threepoint_abs_pose(self):

        rng = np.random.default_rng(54541)
        for i in range(50):
            points = np.array([
                [-0.05,  0.0 , 0.01 , 1],
                [ 0.1 ,  0.2 , 0.025, 1],
                [ 0.2 , -0.05, 0.02 , 1]
            ]).transpose()

            transform = np.eye(4)
            transform[0:3, 0:3] = Rotation.random(random_state=rng).as_matrix()
            transform[2, 3] = 1

            other_points = transform @ points

            estimated_transform = np.eye(4)
            estimated_transform[0:3, 0:3], estimated_transform[0:3, 3] = \
                threepoint_abs_pose(points[0:3, :], other_points[0:3, :])

            self.assertTrue(np.allclose(transform, estimated_transform))

if __name__ == "__main__":
    unittest.main()
