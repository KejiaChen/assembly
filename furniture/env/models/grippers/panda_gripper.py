"""
Gripper for Franka's Panda (has two fingers).
"""
import numpy as np

from ...mjcf_utils import xml_path_completion
from .gripper import Gripper


class PandaGripperBase(Gripper):
    """
    Gripper for Franka's Panda (has two fingers).
    """

    def __init__(self, idn=0):
        super().__init__(xml_path_completion("grippers/panda_gripper.xml"), idn)

    def format_action(self, action):
        return action

    @property
    def init_qpos(self):
        return np.array([0.04, -0.04])

    @property
    def joints(self):
        return [self.prefix + "finger_joint1", 
                self.prefix + "finger_joint2"]
        
    @property
    def sensors(self):
        return [self.prefix + "force_ee",
                self.prefix + "torque_ee"]

    @property
    def dof(self):
        return 2

    @property
    def visualization_sites(self):
        return [self.prefix + "grip_site", 
                self.prefix + "grip_site_cylinder"]

    @property
    def contact_geoms(self):
        return [
            self.prefix + "hand_collision",
            self.prefix + "finger1_collision",
            self.prefix + "finger2_collision",
            self.prefix + "finger1_tip_collision",
            self.prefix + "finger2_tip_collision",
        ]

    @property
    def left_finger_geoms(self):
        return [
            self.prefix + "finger1_tip_collision",
        ]

    @property
    def right_finger_geoms(self):
        return [ 
            self.prefix + "finger2_tip_collision",
        ]


class PandaGripper(PandaGripperBase):
    """
    Modifies PandaGripperBase to only take one action.
    """

    def format_action(self, action):
        """
        Args:
            action: 1 => open, -1 => closed
        """
        assert len(action) == 1
        return np.array([-1 * action[0], 1 * action[0]])

    @property
    def dof(self):
        return 1
