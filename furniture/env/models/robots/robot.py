from collections import OrderedDict
from doctest import FAIL_FAST

from ..base import MujocoXML, XMLError
from ....env.mjcf_utils import add_prefix, sort_elements, _element_filter

def exclude_prefix(element):
    exclude_root = ["default", "asset"]
    exclude_attrib = ["class"] # "material"
    
    exclude = False
    if element in exclude_root or element in exclude_attrib:
        # print ("omit", element)
        exclude =  True
    return exclude


class Robot(MujocoXML):
    """Base class for all robot models."""

    def __init__(self, fname, idn=0):
        """Initializes from file @fname."""
        super().__init__(fname)
        # key: gripper name and value: gripper model
        self.grippers = OrderedDict()
        self.idn = idn
        self.name = self.name + str(self.idn)
        self.prefix = self.name + "_"

        # Define filter method to automatically add a default name to visual / collision geoms if encountered
        group_mapping = {
            None: "col",
            "0": "col",
            "1": "vis",
        }
        ctr_mapping = {
            "col": 0,
            "vis": 0,
        }

        def _add_default_name_filter(element, parent):
            # Run default filter
            filter_key = _element_filter(element=element, parent=parent)
            # Also additionally modify element if it is (a) a geom and (b) has no name
            if element.tag == "geom" and element.get("name") is None:
                group = group_mapping[element.get("group")]
                element.set("name", f"g{ctr_mapping[group]}_{group}")
                ctr_mapping[group] += 1
            # Return default filter key
            return filter_key

        self._elements = sort_elements(root=self.root, element_filter=_add_default_name_filter)
        assert (
                len(self._elements["root_body"]) == 1
        ), "Invalid number of root bodies found for robot model. Expected 1," "got {}".format(
            len(self._elements["root_body"])
        )
        self._elements["root_body"] = self._elements["root_body"][0]

        # from robosuite: rename each instance to avoid conflicts
        add_prefix(root=self.root, prefix=self.prefix, exclude=exclude_prefix)

    def add_gripper(self, arm_name, gripper):
        """
        Mounts gripper to arm.

        Throws error if robot already has a gripper or gripper type is incorrect.

        Args:
            arm_name (str): name of arm mount
            gripper (MujocoGripper instance): gripper MJCF model
        """
        if arm_name in self.grippers:
            raise ValueError("Attempts to add multiple grippers to one body")

        arm_subtree = self.worldbody.find(".//body[@name='{}']".format(arm_name))

        for actuator in gripper.actuator:

            if actuator.get("name") is None:
                raise XMLError("Actuator has no name")

            # if not actuator.get("name").startswith("gripper"):
            #     raise XMLError(
            #         "Actuator name {} does not have prefix 'gripper'".format(
            #             actuator.get("name")
            #         )
            #     )

        for body in gripper.worldbody:
            arm_subtree.append(body)

        self.merge(gripper, merge_body=False)
        self.grippers[arm_name] = gripper

    def is_robot_part(self, geom_name):
        """
        Checks if @geom_name is part of robot.
        """
        is_robot_geom = False

        # check geoms of robot.
        if geom_name in self.contact_geoms:
            is_robot_geom = True

        # check geoms of grippers.
        for gripper in self.grippers.values():
            if geom_name in gripper.contact_geoms:
                is_robot_geom = True

        return is_robot_geom

    @property
    def dof(self):
        """Returns the number of DOF of the robot, not including gripper."""
        raise NotImplementedError

    @property
    def joints(self):
        """Returns a list of joint names of the robot."""
        raise NotImplementedError

    @property
    def init_qpos(self):
        """Returns default qpos."""
        raise NotImplementedError

