import numpy as np

from ...mjcf_utils import array_to_string, xml_path_completion
from .robot import Robot


class Panda(Robot):
    """Panda is a sensitive single-arm robot designed by Franka."""

    def __init__(
        self,
        use_torque=False,
        xml_path="robots/panda/robot.xml",
        idn=0,
    ):
        if use_torque:
            xml_path = "robots/panda/robot_torque.xml"
        super().__init__(xml_path_completion(xml_path), idn=idn)

        self.bottom_offset = np.array([0, 0, -0.913])
        self.set_joint_damping()
        # TODO: should model name also be changed?
        self._model_name = "panda"
        # Careful of init_qpos -- certain init poses cause ik controller to go unstable (e.g: pi/4 instead of -pi/4
        # for the final joint angle)
        self._init_qpos = np.array([0, np.pi / 16.0, 0.00, -np.pi / 2.0 - np.pi / 3.0, 0.00, np.pi - 0.2, -np.pi/4])

    def set_base_xpos(self, pos):
        """Places the robot on position @pos."""
        node = self.worldbody.find("./body[@name='{}']".format(self.name+"_link0"))
        node.set("pos", array_to_string(pos - self.bottom_offset))

    def set_base_xquat(self, quat):
        """Places the robot on position @quat."""
        node = self.worldbody.find("./body[@name='{}']".format(self.name+"_link0"))
        node.set("quat", array_to_string(quat))

    def set_joint_damping(self, damping=np.array((0.1, 0.1, 0.1, 0.1, 0.1, 0.01, 0.01))):
        """Set joint damping """
        body = self._base_body
        for i in range(len(self._link_body)):
            body = body.find("./body[@name='{}']".format(self._link_body[i]))
            joint = body.find("./joint[@name='{}']".format(self._joints[i]))
            joint.set("damping", array_to_string(np.array([damping[i]])))

    def set_joint_frictionloss(self, friction=np.array((0.1, 0.1, 0.1, 0.1, 0.1, 0.01, 0.01))):
        """Set joint friction loss (static friction)"""
        body = self._base_body
        for i in range(len(self._link_body)):
            body = body.find("./body[@name='{}']".format(self._link_body[i]))
            joint = body.find("./joint[@name='{}']".format(self._joints[i]))
            joint.set("frictionloss", array_to_string(np.array([friction[i]])))

    @property
    def dof(self):
        return 7

    @property
    def joints(self):
        return self._joints

    @property
    def init_qpos(self):
        return self._init_qpos

    # @init_qpos.setter
    # def init_qpos(self, init_qpos):
    #     self._init_qpos = init_qpos

    @property
    def contact_geoms(self):
        return [self.name+"link{}_collision".format(x) for x in range(1, 8)]

    @property
    def _base_body(self):
        # alternatively
        # node = self._elements["root_body"]
        node = self.worldbody.find("./body[@name='{}']".format(self.name+"_link0"))
        return node

    @property
    def _link_body(self):
        return [self.name+"_link{}".format(x) for x in range(1, 8)]

    @property
    def _joints(self):
        return [self.name+"_joint{}".format(x) for x in range(1, 8)]
    
    def get_base_xpos(self):
        node = self.worldbody.find("./body[@name='{}']".format(self.name+"_link0"))
        pos_string = node.attrib['pos']
        poses = pos_string.split(" ")
        return poses
