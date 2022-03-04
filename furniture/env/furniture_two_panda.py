""" Define Panda environment class FurniturePandaEnv. """

from collections import OrderedDict

import numpy as np
import gym.spaces

import furniture.env.transform_utils as T
from .furniture_two_arm import FurnitureTwoEnv
from furniture.util.logger import logger


class FurnitureTwoPandaEnv(FurnitureTwoEnv):
    """
    Panda environment.
    """

    def __init__(self, config):
        """
        Args:
            config: configurations for the environment.
        """
        config.agent_type = ["Panda", "Panda"]

        super().__init__(config)

    @property
    def observation_space(self):
        """
        Returns the observation space.
        """
        ob_space = super().observation_space

        if self._robot_ob:
            if self._control_type in ["impedance", "torque"]:
                ob_space.spaces["robot_ob"] = gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(
                        7 + 7 + 2 + 3 + 4 + 3 + 3,
                    ),  # qpos, qvel, gripper, eefp, eefq, velp, velr
                )
            elif self._control_type in ["ik", "ik_quaternion"]:
                ob_space.spaces["robot_ob"] = gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(3 + 4 + 3 + 3 + 1,),  # pos, quat, velp, velr, gripper
                )

        return ob_space

    @property
    def dof(self):
        """
        Returns the DoF of the robot.
        """
        dof = 0  # 'No' Agent
        if self._control_type in ["impedance", "torque"]:
            dof = 7 + 2  # 7 joints, select, connect
        elif self._control_type == "ik":
            dof = 3 + 3 + 1 + 1  # move, rotate, select, connect
        elif self._control_type == "ik_quaternion":
            dof = 3 + 4 + 1 + 1  # move, rotate, select, connect
        return dof

    def _step(self, a):
        """
        Takes a simulation step with @a and computes reward.
        """

        # TODO: check from sawyer: discretize gripper action
        # applied_action = a.copy()
        # if self._discrete_grip:
        #     applied_action[-2] = -1 if a[-2] < 0 else 1

        ob, _, done, _ = super()._step(a)

        # TODO: reward depends only on first panda
        reward, _done, info = self._compute_reward(a[0])
        done = done or _done

        if self._success:
            logger.info("Success!")

        return ob, reward, done, info

    def _reset(self, furniture_id=None, background=None):
        """
        Resets simulation.

        Args:
            furniture_id: ID of the furniture model to reset.
            background: name of the background scene to reset.
        """
        super()._reset(furniture_id, background)

        # set two bodies for picking or assemblying
        id1 = self.sim.model.eq_obj1id[0]
        id2 = self.sim.model.eq_obj2id[0]
        self._target_body1 = self.sim.model.body_id2name(id1)
        self._target_body2 = self.sim.model.body_id2name(id2)

    def _get_obs(self, include_qpos=False):
        """
        Returns the current observation.
        """
        state = super()._get_obs()

        state["robot_ob"] = [None for _ in range(self.num_robots)]

        # TODO: include_qpos?
        # proprioceptive features
        if self._robot_ob:
            for idx in range(self.num_robots):
                robot_states = OrderedDict()
                if self._control_type in ["impedance", "torque"] or include_qpos:
                    robot_states["joint_pos"] = np.array(
                        [
                            self.sim.data.qpos[x]
                            for x in self._ref_joint_pos_indexes[idx]["right"]
                        ]
                    )
                    robot_states["joint_vel"] = np.array(
                        [
                            self.sim.data.qvel[x]
                            for x in self._ref_joint_vel_indexes[idx]["right"]
                        ]
                    )
                    robot_states["gripper_qpos"] = np.array(
                        [
                            self.sim.data.qpos[x]
                            for x in self._ref_gripper_joint_pos_indexes[idx]["right"]
                        ]
                    )
                    robot_states["eef_pos"] = np.array(
                        self.sim.data.site_xpos[self.eef_site_id[idx]["right"]]
                    )
                    robot_states["eef_quat"] = T.convert_quat(
                        self.sim.data.get_body_xquat(self.mujoco_robots[idx].prefix + "right_hand"), to="xyzw"
                    )
                    robot_states["eef_velp"] = np.array(
                        self.sim.data.site_xvelp[self.eef_site_id[idx]["right"]]
                    )  # 3-dim
                    robot_states["eef_velr"] = self.sim.data.site_xvelr[
                        self.eef_site_id[idx]["right"]
                    ]  # 3-dim

                else:
                    gripper_qpos = [
                        self.sim.data.qpos[x]
                        for x in self._ref_gripper_joint_pos_indexes[idx]["right"]
                    ]
                    robot_states["gripper_dis"] = np.array(
                        [(gripper_qpos[0] + 0.0115) - (gripper_qpos[1] - 0.0115)]
                    )  # range of grippers are [-0.0115, 0.0208] and [-0.0208, 0.0115]
                    robot_states["eef_pos"] = np.array(
                        self.sim.data.site_xpos[self.eef_site_id[idx]["right"]]
                    )
                    robot_states["eef_quat"] = T.convert_quat(
                        self.sim.data.get_body_xquat(self.mujoco_robots[idx].prefix + "right_hand"), to="xyzw"
                    )
                    robot_states["eef_velp"] = np.array(
                        self.sim.data.site_xvelp[self.eef_site_id[idx]["right"]]
                    )  # 3-dim
                    robot_states["eef_velr"] = self.sim.data.site_xvelr[
                        self.eef_site_id[idx]["right"]
                    ]  # 3-dim

                state["robot_ob"][idx] = np.concatenate(
                    [x.ravel() for _, x in robot_states.items()]
                )

        return state

    def _get_single_reference(self, idx=0):

        self.l_finger_geom_ids[idx] = {
            "right": [
                self.sim.model.geom_name2id(x)
                for x in self.grippers[idx]["right"].left_finger_geoms
            ]
        }
        self.r_finger_geom_ids[idx] = {
            "right": [
                self.sim.model.geom_name2id(x)
                for x in self.grippers[idx]["right"].right_finger_geoms
            ]
        }

        # indices for joints in qpos, qvel
        self.robot_joints[idx] = list(self.mujoco_robots[idx].joints)
        self._ref_joint_pos_indexes_all[idx] = [
            self.sim.model.get_joint_qpos_addr(x) for x in self.robot_joints[idx]
        ]
        self._ref_joint_vel_indexes_all[idx] = [
            self.sim.model.get_joint_qvel_addr(x) for x in self.robot_joints[idx]
        ]
        self._ref_joint_pos_indexes[idx] = {
            "right": self._ref_joint_pos_indexes_all[idx],
            "left": [],
        }
        self._ref_joint_vel_indexes[idx] = {
            "right": self._ref_joint_vel_indexes_all[idx],
            "left": [],
        }

        # indices for grippers in qpos, qvel
        self.gripper_joints[idx] = list(self.grippers[idx]["right"].joints)
        self._ref_gripper_joint_pos_indexes_all[idx] = [
            self.sim.model.get_joint_qpos_addr(x) for x in self.gripper_joints[idx]
        ]
        self._ref_gripper_joint_vel_indexes_all[idx] = [
            self.sim.model.get_joint_qvel_addr(x) for x in self.gripper_joints[idx]
        ]
        self._ref_gripper_joint_pos_indexes[idx] = {
            "right": self._ref_gripper_joint_pos_indexes_all[idx]
        }
        self._ref_gripper_joint_vel_indexes[idx] = {
            "right": self._ref_gripper_joint_vel_indexes_all[idx]
        }

        # IDs of sites for gripper visualization
        [grip_site_name, grip_site_cylinder_name] = self.grippers[idx]['right'].visualization_sites
        self.eef_site_id[idx] = {"right": self.sim.model.site_name2id(grip_site_name)}
        self.eef_cylinder_id[idx] = {
            "right": self.sim.model.site_name2id(grip_site_cylinder_name)
        }

    def _get_reference(self):
        """
        Sets up references to robot joints and objects.
        """
        super()._get_reference()

        # print("set objects")

        # initialization as list
        self.l_finger_geom_ids = [None for _ in range(self.num_robots)]
        self.r_finger_geom_ids = [None for _ in range(self.num_robots)]

        # indices for joints in qpos, qvel
        self.robot_joints = [None for _ in range(self.num_robots)]
        self._ref_joint_pos_indexes_all = [None for _ in range(self.num_robots)]
        self._ref_joint_vel_indexes_all = [None for _ in range(self.num_robots)]
        self._ref_joint_pos_indexes = [None for _ in range(self.num_robots)]
        self._ref_joint_vel_indexes = [None for _ in range(self.num_robots)]
        # indices for grippers in qpos, qvel
        self.gripper_joints = [None for _ in range(self.num_robots)]
        self._ref_gripper_joint_pos_indexes_all = [None for _ in range(self.num_robots)]
        self._ref_gripper_joint_vel_indexes_all = [None for _ in range(self.num_robots)]
        self._ref_gripper_joint_pos_indexes = [None for _ in range(self.num_robots)]
        self._ref_gripper_joint_vel_indexes = [None for _ in range(self.num_robots)]

        # IDs of sites for gripper visualization
        self.eef_site_id = [None for _ in range(self.num_robots)]
        self.eef_cylinder_id = [None for _ in range(self.num_robots)]

        for idx in range(self.num_robots):
            self._get_single_reference(idx)

    def _compute_reward(self, ac):
        """
        Computes reward of the current state.
        """
        return super()._compute_reward(ac)

    # TODO: change idx
    def _finger_contact(self, obj, idx=0):
        """
        Returns if left, right fingers contact with obj
        """
        touch_left_finger = False
        touch_right_finger = False
        for j in range(self.sim.data.ncon):
            c = self.sim.data.contact[j]
            body1 = self.sim.model.geom_bodyid[c.geom1]
            body2 = self.sim.model.geom_bodyid[c.geom2]
            body1_name = self.sim.model.body_id2name(body1)
            body2_name = self.sim.model.body_id2name(body2)

            if c.geom1 in self.l_finger_geom_ids[idx]["right"] and body2_name == obj:
                touch_left_finger = True
            if c.geom2 in self.l_finger_geom_ids[idx]["right"] and body1_name == obj:
                touch_left_finger = True

            if c.geom1 in self.r_finger_geom_ids[idx]["right"] and body2_name == obj:
                touch_right_finger = True
            if c.geom2 in self.r_finger_geom_ids[idx]["right"] and body1_name == obj:
                touch_right_finger = True

        return touch_left_finger, touch_right_finger


def main():
    from ..config import create_parser

    parser = create_parser(env="FurniturePandaEnv")
    parser.set_defaults(max_episode_steps=2000)
    parser.add_argument(
        "--run_mode", type=str, default="manual", choices=["manual", "vr", "demo"]
    )
    config, unparsed = parser.parse_known_args()
    if len(unparsed):
        logger.error("Unparsed argument is detected:\n%s", unparsed)
        return

    # create an environment and run manual control of Panda environment
    env = FurnitureTwoPandaEnv(config)
    if config.run_mode == "manual":
        env.run_manual(config)
    elif config.run_mode == "vr":
        env.run_vr(config)
    elif config.run_mode == "demo":
        env.run_demo_actions(config)


if __name__ == "__main__":
    main()
