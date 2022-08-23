import numpy as np

from ..base import MujocoXML
from ...mjcf_utils import xml_path_completion
from ...mjcf_utils import array_to_string, string_to_array
from ...mjcf_utils import new_geom, new_body, new_joint


class Arena(MujocoXML):
    """Base arena class."""

    def set_origin(self, offset):
        """Applies a constant offset to all objects."""
        offset = np.array(offset)
        for node in self.worldbody.findall("./*[@pos]"):
            cur_pos = string_to_array(node.get("pos"))
            new_pos = cur_pos + offset
            node.set("pos", array_to_string(new_pos))

    def add_pos_indicator(self):
        """Adds a new position indicator."""
        body = new_body(name="pos_indicator")
        body.append(
            new_geom(
                "sphere",
                [0.03],
                rgba=[1, 0, 0, 0.5],
                group=1,
                contype="0",
                conaffinity="0",
            )
        )
        body.append(new_joint(type="free", name="pos_indicator"))
        self.worldbody.append(body)


class TableArena(Arena):
    """Workspace that contains an empty table."""

    def __init__(
        self, table_full_size=(0.8, 0.8, 0.8), table_friction=(1, 0.005, 0.0001)
    ):
        """
        Args:
            table_full_size: full dimensions of the table
            friction: friction parameters of the table
        """
        super().__init__(xml_path_completion("arenas/table_arena.xml"))

        self.table_full_size = np.array(table_full_size)
        self.table_half_size = self.table_full_size / 2
        self.table_friction = table_friction

        self.floor = self.worldbody.find("./geom[@name='FLOOR']")
        self.table_body = self.worldbody.find("./body[@name='TABLE']")
        self.table_collision = self.table_body.find("./geom[@name='TABLE_collision']")
        self.table_visual = self.table_body.find("./geom[@name='TABLE_visual']")
        self.table_top = self.table_body.find("./site[@name='TABLE_top']")

        self.configure_location()

    def configure_location(self):
        # bottom of the robot is (approx)
        self.bottom_pos = np.array([0, 0, -0.677])
        self.floor.set("pos", array_to_string(self.bottom_pos))

        self.center_pos = self.bottom_pos + np.array([0, 0, self.table_half_size[2]])
        self.table_body.set("pos", array_to_string(self.center_pos))
        self.table_collision.set("size", array_to_string(self.table_half_size))
        self.table_collision.set("friction", array_to_string(self.table_friction))
        self.table_visual.set("size", array_to_string(self.table_half_size))

        self.table_top.set(
            "pos", array_to_string(np.array([0, 0, self.table_half_size[2]]))
        )

    @property
    def table_top_abs(self):
        """Returns the absolute position of table top"""
        table_height = np.array([0, 0, self.table_full_size[2]])
        return string_to_array(self.floor.get("pos")) + table_height


class FloorArena(Arena):
    """Workspace that contains an empty floor."""

    def __init__(
        self, floor_pos=[0, 0], floor_full_size=(1., 1.), floor_friction=(1, 0.005, 0.0001)
    ):
        """
        Args:
            floor_full_size: full dimensions of the floor
            friction: friction parameters of the floor
        """
        super().__init__(xml_path_completion("arenas/floor_arena.xml"))
        height = 0.1

        self.floor_full_size = np.array([floor_full_size[0], floor_full_size[1], 0.125])
        self.floor_half_size = self.floor_full_size / 2
        self.table_full_size = np.array([floor_full_size[0], floor_full_size[1], 0.05])
        self.table_half_size = self.table_full_size/2
        self.leg_full_size = np.array([0.05, 0.677-self.table_full_size[2]])
        self.leg_half_size = self.leg_full_size/2
        self.hori_prof_full_size = np.array([0.05, 0.05, 1.2])
        self.hori_prof_half_size = self.hori_prof_full_size/2
        self.vert_prof_full_size = np.array([floor_full_size[0], 0.05, 0.05])
        self.vert_prof_half_size = self.vert_prof_full_size / 2
        self.floor_friction = floor_friction
        self.floor_pos = np.array([floor_pos[0], floor_pos[1], -0.025])

        self.floor = self.worldbody.find("./geom[@name='FLOOR']")
        self.floor.set("pos", array_to_string(np.array([floor_pos[0], floor_pos[1], 0])))
        self.floor.set("size", array_to_string(self.floor_half_size))
        self.floor.set("friction", array_to_string(self.floor_friction))

        # self.floor = self.worldbody.find(".//geom[@name='floor']")
        # self.floor.set("pos", array_to_string(self.floor_pos))
        # self.floor.set("size", array_to_string(self.table_half_size))
        # self.floor.set("friction", array_to_string(self.floor_friction))

        self.table_visual = self.worldbody.find(".//geom[@name='floor_visual']")
        self.table_visual.set("pos", array_to_string(self.floor_pos))
        self.table_visual.set("size", array_to_string(self.table_half_size))

        self.leg1 = self.worldbody.find(".//geom[@name='table_leg1_visual']")
        # self.leg1_pos = np.array([self.table_half_size[0], self.table_half_size[1], -0.677+self.leg_half_size[1]])
        self.leg1_pos = np.array([self.floor_pos[0]-(self.table_half_size[0]-3*self.leg_half_size[0]),
                                  self.floor_pos[1]+self.table_half_size[1]-3*self.leg_half_size[0],
                                  self.floor_pos[2]-0.677 + self.leg_half_size[1]])
        self.leg1.set("pos", array_to_string(self.leg1_pos))
        self.leg1.set("size", array_to_string(self.leg_half_size))

        self.leg2 = self.worldbody.find(".//geom[@name='table_leg2_visual']")
        # self.leg1_pos = np.array([self.table_half_size[0], self.table_half_size[1], -0.677+self.leg_half_size[1]])
        self.leg2_pos = np.array([self.floor_pos[0]-(self.table_half_size[0] - 3 * self.leg_half_size[0]),
                                  self.floor_pos[1]-(self.table_half_size[1] - 3 * self.leg_half_size[0]),
                                  self.floor_pos[2]-0.677 + self.leg_half_size[1]])
        self.leg2.set("pos", array_to_string(self.leg2_pos))
        self.leg2.set("size", array_to_string(self.leg_half_size))

        self.leg3 = self.worldbody.find(".//geom[@name='table_leg3_visual']")
        # self.leg1_pos = np.array([self.table_half_size[0], self.table_half_size[1], -0.677+self.leg_half_size[1]])
        self.leg3_pos = np.array([self.floor_pos[0]+self.table_half_size[0] - 3 * self.leg_half_size[0],
                                  self.floor_pos[1]+self.table_half_size[1] - 3 * self.leg_half_size[0],
                                  self.floor_pos[2]-0.677 + self.leg_half_size[1]])
        self.leg3.set("pos", array_to_string(self.leg3_pos))
        self.leg3.set("size", array_to_string(self.leg_half_size))

        self.leg4 = self.worldbody.find(".//geom[@name='table_leg4_visual']")
        # self.leg1_pos = np.array([self.table_half_size[0], self.table_half_size[1], -0.677+self.leg_half_size[1]])
        self.leg4_pos = np.array([self.floor_pos[0]+self.table_half_size[0] - 3 * self.leg_half_size[0],
                                  self.floor_pos[1]-(self.table_half_size[1] - 3 * self.leg_half_size[0]),
                                  self.floor_pos[2]-0.677 + self.leg_half_size[1]])
        self.leg4.set("pos", array_to_string(self.leg4_pos))
        self.leg4.set("size", array_to_string(self.leg_half_size))

        self.prof1 = self.worldbody.find(".//geom[@name='shelf_profile1_visual']")
        if self.prof1 is not None:
            # self.leg1_pos = np.array([self.table_half_size[0], self.table_half_size[1], -0.677+self.leg_half_size[1]])
            self.prof1_pos = np.array([self.floor_pos[0] + (self.table_half_size[0] - self.hori_prof_half_size[0]),
                                      self.floor_pos[1] + self.table_half_size[1]/2 - self.hori_prof_half_size[1],
                                      self.floor_pos[2] + self.hori_prof_half_size[2]])
            self.prof1.set("pos", array_to_string(self.prof1_pos))
            self.prof1.set("size", array_to_string(self.hori_prof_half_size))

        self.prof2 = self.worldbody.find(".//geom[@name='shelf_profile2_visual']")
        if self.prof2 is not None:
            # self.leg1_pos = np.array([self.table_half_size[0], self.table_half_size[1], -0.677+self.leg_half_size[1]])
            self.prof2_pos = np.array([self.floor_pos[0] - (self.table_half_size[0] - self.hori_prof_half_size[0]),
                                       self.floor_pos[1] + self.table_half_size[1] / 2 - self.hori_prof_half_size[1],
                                       self.floor_pos[2] + self.hori_prof_half_size[2]])
            self.prof2.set("pos", array_to_string(self.prof2_pos))
            self.prof2.set("size", array_to_string(self.hori_prof_half_size))

        self.prof3 = self.worldbody.find(".//geom[@name='shelf_profile3_visual']")
        if self.prof3 is not None:
            # self.leg1_pos = np.array([self.table_half_size[0], self.table_half_size[1], -0.677+self.leg_half_size[1]])
            self.prof3_pos = np.array([self.floor_pos[0],
                                       self.floor_pos[1] + self.table_half_size[1] / 2 - self.hori_prof_half_size[1],
                                       self.floor_pos[2] + self.hori_prof_full_size[2] - self.vert_prof_half_size[2]])
            self.prof3.set("pos", array_to_string(self.prof3_pos))
            self.prof3.set("size", array_to_string(self.vert_prof_half_size))