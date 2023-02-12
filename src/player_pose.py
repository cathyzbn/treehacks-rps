import math
import numpy as np
from helpers import landmark_vector_dir as dir

RIGHT_ARM_ANGLE_THRESHOLD = math.pi / 2

class PlayerPose:
    def __init__(self):
        self.pose_archive = [False for i in range(3)]
        self.move = False

    def _right_arm_angle(self, shoulder, elbow, wrist):
        upper_arm = dir(shoulder, elbow)
        forearm = dir(wrist, elbow)
        arm_angle = np.arccos(np.clip(np.dot(upper_arm, forearm), -1.0, 1.0))
        return arm_angle

    def _is_play(self, shoulder, elbow, wrist):
        return self._right_arm_angle(shoulder, elbow, wrist) > RIGHT_ARM_ANGLE_THRESHOLD

    def update_pose(self, shoulder, elbow, wrist):
        status = self._is_play(shoulder, elbow, wrist)
        self.pose_archive = self.pose_archive[1:] + [status]
        self.move = all(self.pose_archive)

    def is_move(self):
        return self.move
