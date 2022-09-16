import yaml
import numpy as np
from tqdm import tqdm
import copy
import gym.spaces

import os

# # Print the current working directory
# print("Current working directory: {0}".format(os.getcwd()))
#
# os.chdir('/home/kejia/Documents/assembly')
#
# # Print the current working directory
# print("Current working directory: {0}".format(os.getcwd()))

# import sys
# sys.path.append('/home/kejia/Documents/assembly')

import furniture.env.transform_utils as T
from furniture.env.furniture_two_panda_dense import FurnitureTwoPandaDenseRewardEnv
from furniture.env.models import background_names, furniture_name2id, furniture_xmls
from furniture.util.logger import logger


class FurnitureTwoPandaGenEnv(FurnitureTwoPandaDenseRewardEnv):
    """
    Panda environment for assemblying furniture programmatically.
    """

    def __init__(self, config):
        """
        Args:
            config: configurations for the environment.

        Abbreviations:
            grip ~ gripper
            g ~ gripped body
            t ~ target body
            conn ~ connection

        Phases Descrption:
            0. init_grip:
                    move gripper to grip_init_pos (for the first part)
            1. xy_move_g:
                    move to xy-pos of gripper with gbody
            2. align_g:
                    rotate gripper fingers vector to gbody_gripsites vector (xy plane only),
                    rotate up-vector of gripper to down vector (0,0,-1)
            3. z_move_g:
                    move gripper down to z-pos of gbody
                    *has special check to ensure gripper doesn't reactipeatedly hit ground/table
            4. move_waypoints:
                    grip gbody then move up to the waypoints
            5. xy_move_t:
                    move gripper to tbody xy-pos
            6. align_conn:
                    rotate gbody conn_site up-vector to tbody conn_site up-vector
            7. xy_move_conn:
                    move to xy-pos of gbody conn_site w.r.t. tbody conn_site
            8. z_move_conn:
                    move to xyz-pos of gbody conn_site w.r.t. tbody conn_site
            9. align_conn_fine:
                    finely rotate up-vector of gbody conn_site to up-vector of tbody conn_site
            10. z_move_conn_fine:
                    finely move to xyz position of gbody conn_site w.r.t. tbody conn_site,
                    then try connecting
            11. move_nogrip_safepos:
                    release gripper and move up to nogrip_safepos
            12. part_done:
                    set part_done = True, and part is connected
        """
        config.record_demo = True
        super().__init__(config)

        # control scaling factor
        self.trans_scaling = 0.5
        self.rot_scaling = 0.1

        self._phase = None
        self._num_connected_prev = 0
        self._part_success = False
        self.skill_list = ["insertion, "]
        insertion_phases_gen = [
            "init_grip",
            "xy_move_g",
            "align_g",

            "z_move_g_1",
            "z_move_g_2",
            # "buffer",
            "move_waypoints",
            # "xy_move_t",
            "align_conn",
            # "straighten",
            "xy_move_conn",
            # "align_conn",
            "z_move_reach",
            # "z_move_conn",
            "align_conn_fine",
            # "follow",
            "z_move_conn_fine",
            "move_nogrip_safepos",
            "straighten",
            "align_g1",
            "xy_move_wire",
            "xy_fit_wire",
            "part_done",
        ]

        fit_phases_gen = [
            "straighten",
            "align_g1",
            "xy_move_wire",
            "xy_fit_wire",
        ]

        self.skill_library = {"insertion": insertion_phases_gen}
        # self._phases_to_phase_i = {
        #     None: 0,
        #     "init_grip": 0,
        #     "xy_move_g": 1,
        #     "align_g": 1,
        #     "z_move_g": 2,
        #     "move_waypoints": 4,
        #     "align_conn": 5,
        #     "xy_move_conn": 6,
        #     "z_move_conn": 6,
        #     "align_conn_fine": 7,
        #     "z_move_conn_fine": 7,
        #     "move_nogrip_safepos": 0,
        #     "part_done": 0,
        # }

        self._phase_noise = {
            #   phase      : (min_val, max_val, dimensions)
            "init_grip": (0, 0, 3),
            "xy_move_g": (0, 0, 2),
            "xy_move_t": (-self._config.furn_xyz_rand, self._config.furn_xyz_rand, 2),
            "move_waypoints": (0, 2 * self._config.furn_xyz_rand, 3),
            "straighten": (0, 2 * self._config.furn_xyz_rand, 3),
            "move_nogrip_safepos": (0, 2 * self._config.furn_xyz_rand, 3),
        }
        self.reset()


    def _read_sensor(self, obj_site):
        """
        Returns if left, right fingers contact with obj
        """
        self.sensor_data = self.sim.data.sensordata
        obj_sensor_idx = self.mujoco_model.sensor_dict[obj_site]
        data = self.sensor_data[obj_sensor_idx]
        return data

    def _get_random_noise(self):
        noise = {}
        for phase, val in self._phase_noise.items():
            minimum, maximum, size = val
            # noise[phase] = self._rng.uniform(low=minimum, high=maximum, size=size)
            # TODO: debug with no noise
            noise[phase] = None
        return noise

    def _norm_rot_action(self, action, cap=1):
        if "fine" in self._phase:
            for a in range(3, 7):
                if 0 < abs(action[a]) < self.min_rot_act_fine:
                    action[a] *= self.min_rot_act_fine / abs(action[a])
        else:
            for a in range(3, 7):
                if 0 < abs(action[a]) < self.min_rot_act:
                    action[a] *= self.min_rot_act / abs(action[a])
        return action

    def _step(self, a):
        """
        Takes a simulation step with @a and computes reward.
        """
        ob, reward, done, info = super()._step(a)
        if self._num_connected > self._num_connected_prev:
            self._part_success = True
            self._num_connected_prev = self._num_connected

        if (
            self._num_connected == self._success_num_conn
            and len(self._object_names) > 1
        ):
            self._success = True
            done = True
        return ob, reward, done, info

    def get_bodyiterator(self, bodyname):
        for body in self.mujoco_objects[bodyname].root.find("worldbody"):
            if "name" in body.attrib and bodyname == body.attrib["name"]:
                return body.getiterator()
        return None

    def _get_groupname(self, bodyname):
        bodyiterator = self.get_bodyiterator(bodyname)
        for child in bodyiterator:
            if child.tag == "site":
                if "name" in child.attrib and "conn_site" in child.attrib["name"]:
                    return child.attrib["name"].split("-")[0]
        return None

    def get_conn_sites(self, gbody_name, tbody_name):
        gripbody_conn_site, tbody_conn_site = [], []
        group1 = self._get_groupname(gbody_name)
        group2 = self._get_groupname(tbody_name)
        iter1 = self.get_bodyiterator(gbody_name)
        iter2 = self.get_bodyiterator(tbody_name)
        griptag = group1 + "-" + group2
        tgttag = group2 + "-" + group1
        for child in iter1:
            if child.tag == "site":
                if (
                    "name" in child.attrib
                    and "conn_site" in child.attrib["name"]
                    and griptag in child.attrib["name"]
                    and child.attrib["name"] not in self._used_sites
                ):
                    gripbody_conn_site.append(child.attrib["name"])
        for child in iter2:
            if child.tag == "site":
                if (
                    "name" in child.attrib
                    and "conn_site" in child.attrib["name"]
                    and tgttag in child.attrib["name"]
                    and child.attrib["name"] not in self._used_sites
                ):
                    tbody_conn_site.append(child.attrib["name"])
        return gripbody_conn_site, tbody_conn_site

    def get_furthest_conn_site(self, conn_sites, gripper_pos):
        furthest = None
        max_dist = None
        for name in conn_sites:
            pos = self.sim.data.get_site_xpos(name)
            dist = T.l2_dist(gripper_pos, pos)
            if furthest is None:
                furthest = name
                max_dist = dist
            else:
                if dist > max_dist:
                    furthest = name
                    max_dist = dist
        return furthest

    def get_closest_conn_site(self, conn_sites, gripper_pos):
        closest = None
        min_dist = None
        for name in conn_sites:
            pos = self.sim.data.get_site_xpos(name)
            dist = T.l2_dist(gripper_pos, pos)
            if closest is None:
                closest = name
                min_dist = dist
            else:
                if dist < min_dist:
                    closest = name
                    min_dist = dist
        return closest

    def align_gripsites(self, gripvec, gbodyvec, epsilon):
        if T.angle_between(-gripvec, gbodyvec) < T.angle_between(gripvec, gbodyvec):
            gripvec = -gripvec
        xyaction = T.angle_between2D(gripvec, gbodyvec)
        if abs(xyaction) < epsilon:
            # print("xyaction", xyaction)
            # print("gripfwdvec", gripvec)
            xyaction = 0
        return xyaction

    def get_closest_xy_fwd(self, allowed_angles, gconn, tconn):
        # return tconn forward vector with most similar xy-plane angle to gconn vector
        if len(allowed_angles) == 0:
            # no need for xy-alignment, all angles are acceptable
            return self._get_forward_vector(gconn)[0:2]
        # get closest forward vector
        gfwd = self._get_forward_vector(gconn)[0:2]
        tfwd = self._get_forward_vector(tconn)[0:2]
        min_angle = min(
            abs(T.angle_between2D(gfwd, tfwd)),
            abs((2 * np.pi) + T.angle_between2D(gfwd, tfwd)),
        )
        min_all_angle = 0
        min_tfwd = tfwd
        for angle in allowed_angles:
            tfwd_rotated = T.rotate_vector2D(tfwd, angle * (np.pi / 180))
            xy_angle = T.angle_between2D(gfwd, tfwd_rotated)
            if np.pi <= xy_angle < 2 * np.pi:
                xy_angle = 2 * np.pi - xy_angle
            elif -(2 * np.pi) <= xy_angle < -np.pi:
                xy_angle = 2 * np.pi + xy_angle
            if abs(xy_angle) < min_angle:
                min_angle = abs(xy_angle)
                min_tfwd = tfwd_rotated
                min_all_angle = angle
        return min_tfwd

    def align2D(self, vec, targetvec, epsilon):
        """
        Returns a scalar corresponding to a rotation action in a single 2D-dimension
        Gives a rotation action to align vec with targetvec
        epsilon ~ threshold at which to set action=0
        """
        if abs(vec[0]) + abs(vec[1]) < 0.5:
            # unlikely current orientation allows for helpful rotation action due to gimbal lock
            return 0
        angle = T.angle_between2D(vec, targetvec)
        # move in direction that gets closer to closest of (-2pi, 0, or 2pi)
        if -(2 * np.pi) < angle <= -np.pi:
            action = -(2 * np.pi + angle)
        if -np.pi < angle <= 0:
            action = -angle
        if 0 < angle <= np.pi:
            action = -angle
        if np.pi < angle <= 2 * np.pi:
            action = 2 * np.pi - angle
        if abs(action) < epsilon:
            action = 0
        return action

    def move_xy(self, cur_pos, target_pos, epsilon, noise=None):
        """
        Returns a vector corresponding to action[0:2] (xy action)
        Move from current position to target position in xy dimensions
        epsilon ~ threshold at which to set action=0
        """
        d = target_pos - cur_pos
        if noise is not None:
            d += noise
        if abs(d[0]) > epsilon or abs(d[1]) > epsilon:
            if abs(d[0]) < epsilon:
                d[0] = 0
            if abs(d[1]) < epsilon:
                d[1] = 0
        else:
            # print("current distance", d)
            # self._phase_num += 1
            # print("the next phase", self._phases_gen[self._phase_num])
            d[0:2] = 0
        # TODO: normalize -0.3 to -1 is meaningless!
        if abs(d[0]) > 0.04:
            d[0] = (d[0]/abs(d[0]))*0.04
        if abs(d[1]) > 0.04:
            d[1] = (d[1]/abs(d[1]))*0.04
        return d

    def move_xyz(self, cur_pos, target_pos, epsilon, noise=None):
        """
        Returns a vector corresponding to action[0:3] (xyz action)
        Move from current position to target position in xyz dimensions
        epsilon ~ threshold at which to set action=0
        """
        d = target_pos - cur_pos
        if noise is not None:
            d += noise
        if abs(d[0]) > epsilon or abs(d[1]) > epsilon or abs(d[2]) > epsilon:
            if abs(d[0]) < epsilon:
                d[0] = 0
            if abs(d[1]) < epsilon:
                d[1] = 0
            if abs(d[2]) < epsilon:
                d[2] = 0
        else:
            d[0:3] = 0
        if abs(d[0]) > 0.04:
            d[0] = (d[0] / abs(d[0])) * 0.04
        if abs(d[1]) > 0.04:
            d[1] = (d[1] / abs(d[1])) * 0.04
        if abs(d[2]) > 0.04:
            d[2] = (d[2] / abs(d[2])) * 0.04
        return d

    def move_z(self, cur_pos, target_pos, epsilon, conn_dist, noise=None, fine=None):
        """
        Returns a vector corresponding to action[0:3] (xyz action)
        Move from current position to target position in xyz dimensions
        epsilon ~ threshold at which to set action=0
        conn_dist ~ a scalar YAML configurable variable to make connection easier/harder
            closer to 0 -> harder, more than 0 -> easier
        fine ~ a YAML configurable variable to reduce scale of movement,
            useful for phase 'z_move_conn_fine' to get consistent connection behavior

        """
        target_pos = target_pos + [0, 0, conn_dist]
        d = target_pos - cur_pos
        if noise is not None:
            d += noise
        # print("fine z dist", d)
        if abs(d[0]) < epsilon:
            d[0] = 0
        if abs(d[1]) < epsilon:
            d[1] = 0
        if abs(d[2]) < epsilon:
            d[2] = 0
        if fine:
            d /= fine
            d = np.clip(d, -0.02, 0.02)
        if abs(d[0]) > 0.04:
            d[0] = (d[0] / abs(d[0])) * 0.04
        if abs(d[1]) > 0.04:
            d[1] = (d[1] / abs(d[1])) * 0.04
        if abs(d[2]) > 0.04:
            d[2] = (d[2] / abs(d[2])) * 0.04
        return d

    def rotation_matrix_2d(self, angle):
        return np.array([[np.cos(angle), -np.sin(angle)],[np.sin(angle), np.cos(angle)]])

    def follow_rotation_xy(self, grip_site, xy_ac, eps):
        grip_pos_xy = self._get_pos(grip_site[0])[0:2]
        grip_pos_xy2 = self._get_pos(grip_site[1])[0:2]
        rotation_matrix_xy = self.rotation_matrix_2d(xy_ac)
        rotation_vector = np.reshape(grip_pos_xy2 - grip_pos_xy, (2, 1))
        rotated_grip_pos_xy2 = grip_pos_xy + np.matmul(rotation_matrix_xy, rotation_vector).flatten()
        return self.move_xy(grip_pos_xy2, rotated_grip_pos_xy2, eps)

    def generate_demos(self, n_demos):
        """
        Issues:
            1. Only downward gripping works
            2. Once any collision occurs, unlikely to recover
            3. fine adjustment phase sometimes very challenging
        """
        # recipe: describe the start and target parts in each phrase
        p = self._recipe
        trans_dist_limit = 0.42
        align_dist_limit = 0.25
        height_add = 0.15

        n_successful_demos = 0
        n_failed_demos = 0
        safepos_idx = 0
        safepos = []
        pbar = tqdm(total=n_demos)
        num = 0
        # two_finger gripper sites, as defined in gripper xml
        # TODO: for panda gripper
        # griptip_site = "griptip_site"
        # gripbase_site = "hand_collision"
        # grip_site = "grip_site"

        # for two finger gripper
        griptip_site = [None for _ in range(self.num_robots)]
        gripbase_site = [None for _ in range(self.num_robots)]
        grip_site = [None for _ in range(self.num_robots)]
        for idx in range(self.num_robots):
            griptip_site[idx] = self.grippers[idx]["right"].prefix + "griptip_site"
            gripbase_site[idx] = self.grippers[idx]["right"].prefix +"right_gripper_base_collision"
            grip_site[idx] = self.grippers[idx]["right"].prefix + "grip_site"

        # define assembly order and furniture specific variables
        grip_angles = None
        if "grip_angles" in p:
            grip_angles = p["grip_angles"]
        if self._config.max_episode_steps is None:
            self._config.max_episode_steps = p["max_success_steps"]
        # align_g target vector
        align_g_tgt = np.array([0, -1])
        # background specific, only tested on --background Industrial
        ground_offset = 0.0001
        self.min_rot_act = p["min_rot_act"]
        self.min_rot_act_fine = p["min_rot_act_fine"]

        while n_successful_demos < n_demos:
            ob = self.reset()
            self._used_sites = set()
            noise = self._get_random_noise()
            max_griptip_height = 0

            for j in range(len(self._config.preassembled), len(p["recipe"])):
                self._phase_num = 0
                t_fwd = None
                z_move_g_prev = [None for _ in range(self.num_robots)]

                safepos_idx = 0
                safepos.clear()
                self.idx = 0
                if "grip_init_pos" in p and p["grip_init_pos"][j] is not None:
                    gripbase_pos = self._get_pos(gripbase_site)
                    # gripbase_pos = [0.05866653 0.26087148 0.17194385]
                    for pos in p["grip_init_pos"][j]:
                        init_pos = gripbase_pos + pos[:3]
                        if len(pos) == 4:
                            init_pos[2] = pos[3]
                        safepos.append(init_pos)
                        print("grip init", init_pos)
                else:
                    self._phase_num = 1

                # self._phase = self._phases_gen[self._phase_num]

                gbody_name, tbody_name, task_name = p["recipe"][j]
                self._phases_gen = self.skill_library[task_name]
                self._phase = self._phases_gen[self._phase_num]

                # use conn_sites in site_recipe, other dynamically get closest/furthest conn_site from gripper
                if "site_recipe" in p:
                    gconn, tconn = p["site_recipe"][j][:2]
                    print("hole_site", self._get_pos(tconn))
                    # gconn_inverse = 'leg-table-inverse,0,90,180,270,conn_site2'
                else:
                    gconn_names, tconn_names = self.get_conn_sites(
                        gbody_name, tbody_name
                    )
                    # gconn_inverse = 'leg-table-inverse,0,90,180,270,conn_site2'
                    grip_pos = self._get_pos(grip_site[0])
                    if p["use_closest"]:
                        gconn = self.get_closest_conn_site(gconn_names, grip_pos)
                    else:
                        gconn = self.get_furthest_conn_site(gconn_names, grip_pos)
                greach, treach = p["reach_recipe"][j][:2]
                g_pos = self._get_pos(gbody_name)
                allowed_angles = [float(x) for x in gconn.split(",")[1:-1] if x]
                z_conn_dist = p["z_conn_dist"]
                if isinstance(z_conn_dist, list):
                    z_conn_dist = z_conn_dist[j]

                if self._config.render:
                    self._get_viewer()
                    self.render()
                    # # TODO: debugging
                    while True:
                        zero_action = np.zeros((8,))
                        ob, reward, _, info = self.step([zero_action, zero_action])
                        self.render()
                        # print("hole_pos", self._get_pos('hole-wire,0,90,180,270,conn_site1'))
                    # # TODO: debugging
                if self._config.record_vid:
                    self.vid_rec.capture_frame(self.render("rgb_array")[0])

                # initiate phases for single-part assembly
                while self._phase != "part_done":
                    # TODO: why 8
                    action = np.zeros((8,))
                    action_robot2 = np.zeros((8,))
                    max_griptip_height = max(
                        max_griptip_height, self._get_pos(griptip_site[0])[2]
                    )
                    # logger.info(self._phase)
                    if self._phase == "init_grip":
                        action[6] = -1
                        gripbase_pos = self._get_pos(gripbase_site)
                        action[0:3] = self.trans_scaling*self.move_xyz(
                            gripbase_pos,
                            safepos[safepos_idx],
                            p["eps"],
                            noise=noise[self._phase],
                        )
                        if not np.any(action[0:3]):
                            safepos_idx += 1
                            if safepos_idx >= len(safepos):
                                safepos_idx = 0
                                safepos.clear()
                                self._phase_num += 1
                                print("the next phase", self._phases_gen[self._phase_num])

                    elif self._phase == "xy_move_g":
                        action[6] = -1
                        action_robot2[6] = -1
                        # grip_xy_pos = self._get_pos(grip_site[0])[0:2]
                        # g_xy_pos = self._get_leg_grasp_pos()[0:2]
                        # action[0:2] = self.move_xy(
                        #     grip_xy_pos, g_xy_pos, p["eps"], noise=noise[self._phase]
                        # )

                        grip_xyz_pos = self._get_pos(grip_site[0])
                        # TODO: integrate get leg grasp pos
                        g_xyz_pos = self._get_leg_grasp_pos()
                        pre_xyz_pos = g_xyz_pos + np.array([0, 0, 0.1])
                        action[0:3] = self.move_xyz(
                            grip_xyz_pos, pre_xyz_pos, p["eps"], noise=np.array([0,0,0])
                        )

                        grip_xyz_pos2 = self._get_pos(grip_site[1])
                        g_xyz_pos2 = self._get_leg_follow_grasp_pos()
                        pre_xyz_pos2 = g_xyz_pos2 + np.array([0, 0, 0.1])
                        action_robot2[0:3] = self.move_xyz(
                            grip_xyz_pos2, pre_xyz_pos2, p["eps"], noise=np.array([0, 0, 0])
                        )

                        # print("action 2", action_robot2)

                        if all(action[0:3]==0) and all(action_robot2[0:3]==0):
                            self._phase_num += 1
                            print("the next phase", self._phases_gen[self._phase_num])

                    elif self._phase == "align_g":
                        action[6] = -1
                        action_robot2[6] = -1
                        rot_action = []
                        if grip_angles is None or grip_angles[j] is not None:
                            for idx in range(self.num_robots):
                                # align gripper fingers with grip sites
                                gripfwd_xy = self._get_forward_vector(grip_site[idx])[0:2]
                                # TODO: use the next line for panda gripper to correct xy rotation
                                # gvec_xy = np.array([self._get_leg_grasp_vector()[0], self._get_leg_grasp_vector()[2]])
                                if idx == 0:
                                    gvec_xy = self._get_leg_grasp_vector()[0:2]
                                elif idx == 1:
                                    gvec_xy = self._get_leg_follow_grasp_vector()[0:2]
                                # xy_ac: rotation angle around z axis
                                # print("gvec_xy", gvec_xy)
                                xy_ac = self.align_gripsites(
                                    gripfwd_xy, gvec_xy, p["rot_eps"]
                                )
                                # xy_ac = self.align2D(gripfwd_xy, gvec_xy, p['rot_eps'])
                                # point gripper z downwards
                                gripvec = self._get_up_vector(grip_site[idx])
                                # up vector should in z direction
                                yz_ac = self.align2D(
                                    gripvec[1:3], align_g_tgt, p["rot_eps"]
                                )
                                xz_ac = self.align2D(
                                    gripvec[0::2], align_g_tgt, p["rot_eps"]
                                )
                                rot_action = [xy_ac, yz_ac, xz_ac]
                                # if xy_ac == 0:
                                if rot_action == [0, 0, 0]:
                                    grip_pos = self._get_pos(grip_site[idx])[0:2]
                                    if idx == 0:
                                        g_pos = self._get_leg_grasp_pos()
                                        action[0:2] = self.move_xy(
                                            grip_pos, g_pos[0:2], p["eps"]
                                        )
                                    elif idx == 1:
                                        g_pos = self._get_leg_follow_grasp_pos()
                                        action_robot2[0:2] = self.move_xy(
                                            grip_pos, g_pos[0:2], p["eps"]
                                        )
                                    # if all(action[0:2]==0) and all(action_robot2[0:2]==0):
                                    #     self._phase_num += 1
                                    #     print("the next phase", self._phases_gen[self._phase_num])

                                    # grip_xyz_pos = self._get_pos(grip_site)
                                    # g_xyz_pos = self._get_leg_grasp_pos()
                                    # pre_xyz_pos = g_xyz_pos + np.array([0, 0, 0.05])
                                    # action[0:3] = self.move_xyz(
                                    #     grip_xyz_pos, pre_xyz_pos, p["eps"]
                                    # )

                                else:
                                    # panda: smaller angular velocity
                                    if idx == 0:
                                        action[3:6] = [xy_ac, yz_ac, xz_ac]
                                    elif idx == 1:
                                        action_robot2[3:6] = [xy_ac, yz_ac, xz_ac]
                                    # action[3:6] = rot_action

                            if all(action[0:6] == 0) and all(action_robot2[0:6] == 0):
                                self._phase_num += 1
                                print("the next phase", self._phases_gen[self._phase_num])
                        else:
                            self._phase_num += 1
                            print("the next phase", self._phases_gen[self._phase_num])

                    elif self._phase == "z_move_g_1":
                        action[6] = -1
                        action_robot2[6] = -1
                        for idx in [int(0)]:
                            grip_pos = self._get_pos(grip_site[idx])
                            grip_tip = self._get_pos(griptip_site[idx])
                            if idx == 0:
                                g_pos = self._get_leg_grasp_pos()
                            elif idx == 1:
                                g_pos = self._get_leg_follow_grasp_pos()
                            d = (g_pos) - grip_pos
                            if z_move_g_prev[idx] is None:
                                # TODO: what is the meaning of z_move_g_prev?
                                z_move_g_prev[idx] = grip_tip[2] + ground_offset

                            if abs(d[2]) > p["eps"] and grip_tip[2] < z_move_g_prev[idx]:
                            # if abs(d[2]) > p["eps"] and grip_tip[2] > 0:
                                # distance is too large to grasp
                                # keep moving in xyz
                                if idx == 0:
                                    action[0:3] = d
                                elif idx == 1:
                                    action_robot2[0:3] = d
                                z_move_g_prev[idx] = grip_tip[2] - ground_offset
                            else:
                                print("distance", d)
                                # distance is small enough
                                # grasp the object
                                if idx == 0:
                                    action[6] = 1
                                elif idx == 1:
                                    action_robot2[6] = 1
                                # self._phase_num += 1
                                # print("the next phase", self._phases_gen[self._phase_num])
                        if action[6] == 1:
                        # if action[6] == 1:
                            self._phase_num += 1
                            print("the next phase", self._phases_gen[self._phase_num])

                    elif self._phase == "z_move_g_2":
                        action[6] = 1
                        action_robot2[6] = -1
                        for idx in [int(1)]:
                            grip_pos = self._get_pos(grip_site[idx])
                            grip_tip = self._get_pos(griptip_site[idx])
                            if idx == 0:
                                g_pos = self._get_leg_grasp_pos()
                            elif idx == 1:
                                g_pos = self._get_leg_follow_grasp_pos()
                            d = (g_pos) - grip_pos
                            if z_move_g_prev[idx] is None:
                                # TODO: what is the meaning of z_move_g_prev?
                                z_move_g_prev[idx] = grip_tip[2] + ground_offset

                            if abs(d[2]) > p["eps_fine"] and grip_tip[2] < z_move_g_prev[idx]:
                            # if abs(d[2]) > p["eps"] and grip_tip[2] > 0:
                                # distance is too large to grasp
                                # keep moving in xyz
                                if idx == 0:
                                    action[0:3] = d
                                elif idx == 1:
                                    action_robot2[0:3] = d
                                z_move_g_prev[idx] = grip_tip[2] - ground_offset
                            else:
                                print("distance", d)
                                # distance is small enough
                                # grasp the object
                                if idx == 0:
                                    action[6] = 1
                                elif idx == 1:
                                    action_robot2[6] = 1
                                # self._phase_num += 1
                                # print("the next phase", self._phases_gen[self._phase_num])

                        if action[6] == 1 and action_robot2[6] == 1:
                        # if action[6] == 1:
                            self._phase_num += 1
                            self.count = 0
                            print("the next phase", self._phases_gen[self._phase_num])
                            if p["waypoints"][j] is not None:
                                gripbase_pos = self._get_pos(gripbase_site[0])
                                gripbase_pos2 = self._get_pos(gripbase_site[1])
                                # grip_pos = self._get_pos(grip_site[0])
                                # grip_pos2 = self._get_pos(grip_site[1])
                                # straighten_pos2 = np.array([trans_dist_limit, 0, 0]) + grip_pos

                                for pos in p["waypoints"][j]:
                                    align_dist = np.array([align_dist_limit, 0, 0.02])
                                    height_diff = np.array([0, 0, gripbase_pos[2] - gripbase_pos2[2]])
                                    # TODO: if follower is higher
                                    height_add_array = np.array([0, 0, height_add])
                                    safe_pos2 = gripbase_pos + pos + np.array([0.43, 0, height_add])
                                    # safepos.append([gripbase_pos + pos, gripbase_pos + pos + align_dist])
                                    # safepos.append([gripbase_pos + pos, gripbase_pos2 + pos + height_diff + height_add_array])
                                    safepos.append([gripbase_pos + pos, safe_pos2])
                                    print("pick up", safepos[-1])

                    elif self._phase == "buffer":
                        action[6] = 1
                        action_robot2[6] = 1
                        self.count += 1
                        if self.count > 5:
                            self._phase_num += 1
                            print("the next phase", self._phases_gen[self._phase_num])

                    elif self._phase == "move_waypoints":
                        action[6] = 1
                        action_robot2[6] = 1
                        if p["waypoints"][j] is None or (  # no available way points
                            p["waypoints"][j] and safepos_idx >= len(p["waypoints"][j])  # have reached way points
                        ):
                            safepos_idx = 0
                            safepos.clear()
                            self._phase_num += 1
                            print("the next phase", self._phases_gen[self._phase_num])
                            gconn_pos = self.sim.data.get_site_xpos(gconn)
                            if "site_recipe" not in p:
                                if p["use_closest"]:
                                    # if conn_site is not defined in recipe
                                    # choose the closest or the furthest
                                    tconn = self.get_closest_conn_site(
                                        tconn_names, gconn_pos
                                    )
                                else:
                                    tconn = self.get_furthest_conn_site(
                                        tconn_names, gconn_pos
                                    )
                            tconn_pos = self.sim.data.get_site_xpos(tconn)
                        else:
                            gripbase_pos = self._get_pos(gripbase_site[0])
                            action[0:3] = self.trans_scaling*self.move_xyz(
                                gripbase_pos,
                                safepos[safepos_idx][0],
                                p["eps"],
                                noise=noise[self._phase],
                            )
                            gripbase_pos2 = self._get_pos(gripbase_site[1])
                            action_robot2[0:3] = self.trans_scaling * self.move_xyz(
                                gripbase_pos2,
                                safepos[safepos_idx][1],
                                p["eps"],
                                noise=noise[self._phase],
                            )
                            if not np.any(action[0:3]) and not np.any(action_robot2[0:3]):
                                safepos_idx += 1

                    # elif self._phase == "xy_move_t":
                    #     action[6] = 1
                    #     grip_pos = self._get_pos(grip_site)
                    #     action[0:2] = self.move_xy(
                    #         grip_pos[0:2],
                    #         tconn_pos[0:2],
                    #         p["eps"],
                    #         noise=noise[self._phase],
                    #     )

                    elif self._phase == "align_conn":
                        action[6] = 1
                        action_robot2[6] = 1
                        g_up = self._get_up_vector(gconn)
                        # gripper_up = self._get_up_vector(gconn)
                        t_up = self._get_up_vector(tconn)
                        yz_ac = self.align2D(g_up[1:3], t_up[1:3], p["rot_eps"])
                        xz_ac = self.align2D(g_up[0::2], t_up[0::2], p["rot_eps"])
                        rot_action = [0, yz_ac, xz_ac]
                        if rot_action == [0, 0, 0]:
                            g_xy_fwd = self._get_forward_vector(gconn)
                            if t_fwd is None:
                                t_fwd = self.get_closest_xy_fwd(
                                    allowed_angles, gconn, tconn
                                )
                                t_xy_fwd = t_fwd[0:2]
                            xy_ac = self.align2D(g_xy_fwd, t_xy_fwd, p["rot_eps"])
                            # print("xy_ac", xy_ac)
                            # # TODO: contact face is a circular face regardless of xy_ac
                            xy_ac = 0
                            if xy_ac == 0:
                                t_fwd = None
                                self._phase_num += 1
                                print("the next phase", self._phases_gen[self._phase_num])
                                # next straighten pos
                                grip_pos = self._get_pos(grip_site[0])
                                grip_pos2 = self._get_pos(grip_site[1])
                                B1_l_site = 'B1_ltgt_site0'
                                B1_r_site = 'B1_rtgt_site0'
                                # grip_vec = -(self._get_pos(B1_l_site) - self._get_pos(B1_r_site))
                                grip_vec = (grip_pos2 - grip_pos) / np.linalg.norm(grip_pos2 - grip_pos)
                                self.straighten_pos = grip_pos + 0.44 * grip_vec
                            else:
                                action[3] = -xy_ac
                                # # TODO: follow rotation
                                grip_pos_xy = self._get_pos(grip_site[0])[0:2]
                                grip_pos_xy2 = self._get_pos(grip_site[1])[0:2]
                                rotation_matrix_xy = self.rotation_matrix_2d(-xy_ac)
                                rotation_vector = np.reshape(grip_pos_xy2 - grip_pos_xy, (2, 1))
                                rotated_grip_pos_xy2 = grip_pos_xy + np.matmul(rotation_matrix_xy,
                                                                               rotation_vector).flatten()
                                action_robot2[0:2] = self.move_xy(
                                    grip_pos_xy2, rotated_grip_pos_xy2, p["eps"]
                                )
                        else:
                            action[3:6] = rot_action

                        # follow translation:
                        # action_robot2[0:3] = copy.deepcopy(action[0:3])

                    # elif self._phase == "straighten":
                    #     action[6] = -1
                    #     action_robot2[6] = 1
                    #     # wire_end_1 = self._get_pos("wire1_end")
                    #     # wire_end_2 = self._get_pos("wire2_end")
                    #     # wire_end_dist = np.linalg.norm(wire_end_2 - wire_end_1)
                    #     # wire_end_straighten =
                    #     grip_pos = self._get_pos(grip_site[0])
                    #     grip_pos2 = self._get_pos(grip_site[1])
                    #     grip_vec = (grip_pos2 - grip_pos)/np.linalg.norm(grip_pos2 - grip_pos)
                    #     straighten_pos = grip_pos + 0.42*grip_vec
                    #     # TODO: keep two grippers within a certain distance (max=0.45)
                    #     grip_dist = np.linalg.norm(grip_pos2 - grip_pos)
                    #     # if grip_dist < 0.3:
                    #     if True:
                    #         # TODO: change straighten_pos2
                    #         straighten_pos2 = np.array([align_dist_limit, 0, 0]) + grip_pos
                    #         # gripbase_pos = self._get_pos(gripbase_site[0])
                    #         # gripbase_pos2 = self._get_pos(gripbase_site[1])
                    #         # straighten_pos2 = np.array([trans_dist_limit, 0, 0]) + gripbase_pos
                    #         action_robot2[0:3] = self.trans_scaling * self.move_xyz(
                    #             grip_pos2,
                    #             self.straighten_pos,
                    #             p["eps"],
                    #             noise=noise[self._phase],
                    #         )
                    #         print("action", action)
                    #         if not np.any(action_robot2[0:3]):
                    #             self._phase_num += 1
                    #             print("the next phase", self._phases_gen[self._phase_num])
                    #     else:
                    #         self._phase_num += 1
                    #         print("the next phase", self._phases_gen[self._phase_num])

                    elif self._phase == "xy_move_conn":
                        action[6] = 1
                        action_robot2[6] = 1
                        gconn_pos = self.sim.data.get_site_xpos(gconn)
                        tconn_pos = self.sim.data.get_site_xpos(tconn)
                        treach_pos = self.sim.data.get_site_xpos(treach)
                        # action[0:2] = self.move_xy(
                        #     gconn_pos[0:2], tconn_pos[0:2], p["eps"]
                        # )
                        # pre_tconn_pos = tconn_pos + np.array([0, 0, 0.05])
                        # add reach pos from xml
                        pre_tconn_pos = treach_pos + np.array([0, 0, 0.05])
                        action[0:3] = self.move_xyz(
                            gconn_pos, pre_tconn_pos, p["eps"]
                        )

                        # follow translation:
                        action_robot2[0:3] = copy.deepcopy(action[0:3])
                        if all(action[0:3] == 0):
                            # gripbase_pos = self._get_pos(gripbase_site[0])
                            # gripbase_pos2 = self._get_pos(gripbase_site[1])
                            # straighten_pos2 = np.array([trans_dist_limit, 0, 0]) + gripbase_pos
                            # action_robot2[0:3] = self.trans_scaling * self.move_xyz(
                            #     gripbase_pos2, straighten_pos2, p["eps"]
                            # )
                            if all(action_robot2[0:3] == 0):
                                self._phase_num += 1
                                print("the next phase", self._phases_gen[self._phase_num])

                        # if all(action[0:3] == 0) and all(action_robot2[0:3] == 0):
                        #     self._phase_num += 1
                        #     print("the next phase", self._phases_gen[self._phase_num])

                    elif self._phase == "z_move_reach":
                        action[6] = 1
                        action_robot2[6] = 1
                        gconn_pos = self.sim.data.get_site_xpos(gconn)
                        # gconn_inverse_pos = self.sim.data.get_site_xpos(gconn_inverse)
                        tconn_pos = self.sim.data.get_site_xpos(tconn)
                        treach_pos = self.sim.data.get_site_xpos(treach)
                        action[0:3] = self.move_z(
                            gconn_pos,
                            # gconn_inverse_pos,
                            treach_pos,
                            p["eps"],
                            z_conn_dist, # + p["z_finedist"],
                        )
                        # follow translation:
                        action_robot2[0:3] = copy.deepcopy(action[0:3])
                        if all(action[0:3] == 0):
                            grip_pos = self._get_pos(grip_site[0])
                            grip_pos2 = self._get_pos(grip_site[1])
                            # TODO: next step is align
                            # straighten_pos2 = np.array([0.4, 0.0, height_add]) + grip_pos
                            # # gripbase_pos = self._get_pos(gripbase_site[0])
                            # # gripbase_pos2 = self._get_pos(gripbase_site[1])
                            # # straighten_pos2 = np.array([trans_dist_limit, 0, 0]) + gripbase_pos
                            # action_robot2[0:3] = self.trans_scaling * self.move_xyz(
                            #     grip_pos2, straighten_pos2, p["eps"]
                            # )
                            if all(action_robot2[0:3] == 0):
                                self._phase_num += 1
                                print("the next phase", self._phases_gen[self._phase_num])

                        # if not np.any(action[0:3]) and not np.any(action_robot2[0:3]):
                        #     self._phase_num += 1
                        #     print("the next phase", self._phase)

                    elif self._phase == "z_move_conn":
                        action[6] = 1
                        action_robot2[6] = 1
                        gconn_pos = self.sim.data.get_site_xpos(gconn)
                        # gconn_inverse_pos = self.sim.data.get_site_xpos(gconn_inverse)
                        tconn_pos = self.sim.data.get_site_xpos(tconn)
                        action[0:3] = self.move_z(
                            gconn_pos,
                            # gconn_inverse_pos,
                            tconn_pos,
                            p["eps"],
                            z_conn_dist + p["z_finedist"],
                        )
                        # follow translation:
                        action_robot2[0:3] = copy.deepcopy(action[0:3])
                        if all(action[0:3] == 0):
                            grip_pos = self._get_pos(grip_site[0])
                            grip_pos2 = self._get_pos(grip_site[1])
                            # TODO: next step is align
                            # straighten_pos2 = np.array([0.4, 0.0, height_add]) + grip_pos
                            # # gripbase_pos = self._get_pos(gripbase_site[0])
                            # # gripbase_pos2 = self._get_pos(gripbase_site[1])
                            # # straighten_pos2 = np.array([trans_dist_limit, 0, 0]) + gripbase_pos
                            # action_robot2[0:3] = self.trans_scaling * self.move_xyz(
                            #     grip_pos2, straighten_pos2, p["eps"]
                            # )
                            if all(action_robot2[0:3] == 0):
                                self._phase_num += 1
                                print("the next phase", self._phases_gen[self._phase_num])

                        # if not np.any(action[0:3]) and not np.any(action_robot2[0:3]):
                        #     self._phase_num += 1
                        #     print("the next phase", self._phase)

                    elif self._phase == "align_conn_fine":
                        action[6] = 1
                        action_robot2[6] = 1
                        g_up = self._get_up_vector(gconn)
                        t_up = self._get_up_vector(tconn)
                        yz_ac = self.align2D(g_up[1:], t_up[1:], p["rot_eps_fine"])
                        xz_ac = self.align2D(g_up[0::2], t_up[0::2], p["rot_eps_fine"])
                        rot_action = [0, yz_ac, xz_ac]
                        # TODO: follow rotation xz_ac
                        # grip_pos_xz = np.array([self._get_pos(grip_site[0])[0], self._get_pos(grip_site[0])[2]])
                        # grip_pos_xz2 = np.array([self._get_pos(grip_site[1])[0], self._get_pos(grip_site[1])[2]])
                        # rotation_matrix_xz = self.rotation_matrix_2d(-xz_ac)
                        # # rotation_vector = np.reshape(grip_pos_xz2 - grip_pos_xz, (2, 1))
                        # rotation_vector = 0.075*(grip_pos_xz2 - grip_pos_xz)/np.linalg.norm(grip_pos_xz2 - grip_pos_xz)
                        # rotated_grip_pos_xz2 = grip_pos_xz + np.matmul(rotation_matrix_xz, rotation_vector).flatten()
                        # action_robot2[0] = self.move_xy(
                        #     grip_pos_xz2, rotated_grip_pos_xz2, p["eps"]
                        # )[0]
                        # action_robot2[2] = self.move_xy(
                        #     grip_pos_xz2, rotated_grip_pos_xz2, p["eps"]
                        # )[1]
                        # print("rot_action", rot_action)
                        if rot_action == [0, 0, 0]:
                            # g_xy_fwd = self._get_forward_vector(gconn)[0:2]
                            # if t_fwd is None:
                            #     t_fwd = self.get_closest_xy_fwd(
                            #         allowed_angles, gconn, tconn
                            #     )
                            #     t_xy_fwd = t_fwd[0:2]
                            # xy_ac = self.align2D(g_xy_fwd, t_xy_fwd, p["rot_eps_fine"])
                            xy_ac = 0
                            # print("xyac", xy_ac)
                            if xy_ac == 0:
                                # must be finely aligned rotationally and translationally to go to next phase
                                gconn_pos = self.sim.data.get_site_xpos(gconn)
                                # gconn_inverse_pos = self.sim.data.get_site_xpos(gconn_inverse)
                                tconn_pos = self.sim.data.get_site_xpos(tconn)
                                treach_pos = self.sim.data.get_site_xpos(treach)
                                action[0:3] = self.move_xyz(
                                    gconn_pos[0:3], treach_pos[0:3], p["eps_fine"]
                                )
                                # follow translation
                                action_robot2[0:3] = copy.deepcopy(action[0:3])
                                if all(action[0:3] == 0):
                                    # grip_pos = self._get_pos(grip_site[0])
                                    # grip_pos2 = self._get_pos(grip_site[1])
                                    # straighten_pos2 = np.array([trans_dist_limit, 0, 0]) + grip_pos
                                    # # gripbase_pos = self._get_pos(gripbase_site[0])
                                    # # gripbase_pos2 = self._get_pos(gripbase_site[1])
                                    # # straighten_pos2 = np.array([trans_dist_limit, 0, 0]) + gripbase_pos
                                    # action_robot2[0:3] = self.trans_scaling * self.move_xyz(
                                    #     grip_pos2, straighten_pos2, p["eps"]
                                    # )
                                    # if all(action_robot2[0:3] == 0):
                                    if True:
                                        self._phase_num += 1
                                        print("the next phase", self._phases_gen[self._phase_num])
                                        grip_pos2 = self._get_pos(grip_site[1])
                                        self.follow_pos = grip_pos2 + np.array([0, 0, -0.1])
                                # if not np.any(action[0:3]) and not np.any(action_robot2[0:3]):
                                #     self._phase_num += 1
                                #     print("the next phase", self._phase)
                            else:
                                action[3] = -xy_ac
                                # TODO: follow rotation
                                grip_pos_xy = self._get_pos(grip_site[0])[0:2]
                                grip_pos_xy2 = self._get_pos(grip_site[1])[0:2]
                                rotation_matrix_xy = self.rotation_matrix_2d(xy_ac)
                                rotation_vector = np.reshape(grip_pos_xy2-grip_pos_xy, (2, 1))
                                rotated_grip_pos_xy2 = grip_pos_xy + np.matmul(rotation_matrix_xy, rotation_vector).flatten()
                                action_robot2[0:2] = self.move_xy(
                                    grip_pos_xy2, rotated_grip_pos_xy2, p["eps"]
                                )
                                # rotation_radius = np.linalg.norm(grip_pos_xy2 - grip_pos_xy)
                                # rotation_dist = xy_ac * rotation_radius
                                # action_robot2[3] = copy.deepcopy(action[3])
                        else:
                            action[3:6] = rot_action

                    elif self._phase == "follow":
                        action[6] = 1
                        action_robot2[6] = 1
                        grip_pos2 = self._get_pos(grip_site[1])
                        action_robot2[0:3] = self.move_xyz(
                            grip_pos2, self.follow_pos, p["eps"]
                        )
                        if all(action_robot2[0:3] == 0):
                            self._phase_num += 1
                            print("the next phase", self._phases_gen[self._phase_num])

                    elif self._phase == "z_move_conn_fine":
                        action[6] = 1
                        action_robot2[6] = 1
                        gconn_pos = self.sim.data.get_site_xpos(gconn)
                        tconn_pos = self.sim.data.get_site_xpos(tconn)
                        # TODO: move xyz?
                        action[0:3] = self.move_z(
                            gconn_pos,
                            tconn_pos,
                            p["eps_fine"],
                            z_conn_dist,
                            fine=p["fine_magnitude"],
                        )
                        # follow
                        action_robot2[0:3] = copy.deepcopy(action[0:3])
                        # print("action", action[0:3])

                        g_up = self._get_up_vector(gconn)
                        t_up = self._get_up_vector(tconn)
                        yz_ac = self.align2D(g_up[1:], t_up[1:], p["rot_eps_fine"])
                        xz_ac = self.align2D(g_up[0::2], t_up[0::2], p["rot_eps_fine"])
                        xy_ac = 0
                        if yz_ac == 0 and xz_ac == 0:
                            g_xy_fwd = self._get_forward_vector(gconn)[0:2]
                            if t_fwd is None:
                                t_fwd = self.get_closest_xy_fwd(
                                    allowed_angles, gconn, tconn
                                )
                                t_xy_fwd = t_fwd[0:2]
                            xy_ac = 0
                            # xy_ac = self.align2D(g_xy_fwd, t_xy_fwd, p["rot_eps_fine"])
                            # if abs(xy_ac) < 0.05:
                            #     xy_ac = 0
                            # if abs(yz_ac) < 0.05:
                            #     yz_ac = 0
                            # if abs(yz_ac) < 0.05:
                            #     xz_ac = 0
                        # action[3:6] = [-xy_ac, yz_ac, xz_ac]
                        # TODO: does this make sense?
                        action[3:6] = [0, 0, 0]
                        if not np.any(action[0:3]):
                            # action [7]: connect
                            action[6] = -1
                            action[7] = 1
                            self._phase_num += 1
                            # # next straighten pos
                            grip_pos = self._get_pos(grip_site[0])
                            grip_pos2 = self._get_pos(grip_site[1])
                            grip_vec = (grip_pos2 - grip_pos) / np.linalg.norm(grip_pos2 - grip_pos)

                            wire1_end_site = "wire1_end"
                            wire1_end_pos_xy = self._get_pos(wire1_end_site)[0:2]
                            clip_center_site = "clip-wire,reach_site1"
                            clip_center_pos_xy = self._get_pos(clip_center_site)[0:2]
                            fit_vec = (clip_center_pos_xy - wire1_end_pos_xy) / np.linalg.norm(wire1_end_pos_xy - clip_center_pos_xy)

                            straighten_dir = np.array([fit_vec[0], fit_vec[1], 1.0])
                            straighten_vec = straighten_dir/np.linalg.norm(straighten_dir)
                            # self.straighten_pos = grip_pos + 0.5 * grip_vec
                            self.straighten_pos_xy = wire1_end_pos_xy + 0.44 * fit_vec
                            self.straighten_pos = self._get_pos(wire1_end_site) + 0.44 * straighten_vec
                            if self._config.reset_robot_after_attach:
                                self._phase_num += 1
                            else:
                                gripbase_pos = self._get_pos(gripbase_site[0])
                                safepos_idx = 0
                                safepos.clear()
                                if p["nogrip_safepos"][j] is not None:
                                    safepos.append(grip_pos + np.array([0, 0, 0]))
                                    # grip_pos = self._get_pos(grip_site[0])
                                    # grip_pos2 = self._get_pos(grip_site[1])
                                    # grip_vec = (grip_pos2 - grip_pos) / np.linalg.norm(grip_pos2 - grip_pos)
                                    # self.straighten_pos = grip_pos + 0.5 * grip_vec
                                    # for pos in p["nogrip_safepos"][j]:
                                    #     safepos.append(gripbase_pos + pos)
                                    #     print("after attach", safepos[-1])

                    elif self._phase == "move_nogrip_safepos":
                        action[6] = -1
                        action_robot2[6] = 1
                        if p["nogrip_safepos"][j] is None or (
                            p["nogrip_safepos"][j]
                            and safepos_idx >= len(p["nogrip_safepos"][j])
                        ):
                            safepos_idx = 0
                            safepos.clear()
                            self._phase_num += 1
                        else:
                            gripbase_pos = self._get_pos(gripbase_site[0])
                            grip_pos = self._get_pos(grip_site[0])

                            action[0:3] = self.trans_scaling*self.move_xyz(
                                grip_pos,
                                safepos[safepos_idx],
                                p["eps"],
                                noise=noise[self._phase],
                            )
                            # action_robot2[0:3] = copy.deepcopy(action[0:3])
                            # print(action[0:3])
                            if not np.any(action[0:3]):
                                # safepos_idx += 1
                                self._phase_num += 1
                                print("the next phase", self._phases_gen[self._phase_num])

                    elif self._phase == "straighten":
                        action[6] = 0.2
                        action_robot2[6] = 1
                        # wire_end_1 = self._get_pos("wire1_end")
                        # wire_end_2 = self._get_pos("wire2_end")
                        # wire_end_dist = np.linalg.norm(wire_end_2 - wire_end_1)
                        # wire_end_straighten =
                        g_pos = self._get_leg_grasp_pos()
                        grip_pos = self._get_pos(grip_site[0])
                        grip_pos2 = self._get_pos(grip_site[1])
                        grip_vec = (grip_pos2 - grip_pos)/np.linalg.norm(grip_pos2 - grip_pos)
                        straighten_pos = grip_pos + 0.4*grip_vec
                        # TODO: keep two grippers within a certain distance (max=0.45)
                        grip_dist = np.linalg.norm(grip_pos2 - grip_pos)
                        # if grip_dist < 0.3:
                        if True:
                            # TODO: change straighten_pos2
                            straighten_pos2 = np.array([align_dist_limit, 0, 0]) + grip_pos
                            # gripbase_pos = self._get_pos(gripbase_site[0])
                            # gripbase_pos2 = self._get_pos(gripbase_site[1])
                            # straighten_pos2 = np.array([trans_dist_limit, 0, 0]) + gripbase_pos
                            # action_robot2[0:3] = self.trans_scaling * self.move_xyz(
                            action_robot2[0:3] = self.move_xyz(
                                grip_pos2,
                                self.straighten_pos,
                                p["eps"],
                                noise=noise[self._phase],
                            )
                            # action_robot2[0:2] = self.move_xy(
                            #     grip_pos2[0:2], self.straighten_pos_xy, p["eps"]
                            # )
                            # print("action", action)
                            # print("action2", action_robot2)
                            if not np.any(action_robot2[0:3]):
                                self._phase_num += 1
                                print("the next phase", self._phases_gen[self._phase_num])
                        else:
                            self._phase_num += 1
                            print("the next phase", self._phases_gen[self._phase_num])

                    elif self._phase == "align_g1":
                        action[6] = 0.2
                        action_robot2[6] = 1
                        rot_action = []
                        if grip_angles is None or grip_angles[j] is not None:
                                idx = 0
                                # point gripper z downwards
                                g_l, g_r = f"{'B4'}_ltgt_site{0}", f"{'wire2'}_ltgt_site{0}"
                                vec_g = self._get_pos(g_r) - self._get_pos(g_l)

                                # align gripper fingers with grip sites
                                gripfwd_xy = self._get_forward_vector(grip_site[idx])[0:2]
                                gvec_xy = self._get_leg_grasp_vector()[0:2]
                                # xy_ac: rotation angle around z axis
                                # print("gvec_xy", gvec_xy)
                                xy_ac = self.align_gripsites(
                                    gripfwd_xy, gvec_xy, p["rot_eps"]
                                )
                                # xy_ac = self.align2D(gripfwd_xy, gvec_xy, p['rot_eps'])
                                yz_align_g = np.array([-vec_g[2], vec_g[1]])
                                xz_align_g = np.array([vec_g[2], -vec_g[0]])
                                gripvec = self._get_up_vector(grip_site[idx])
                                # up vector should in z direction
                                yz_ac = self.align2D(
                                    gripvec[1:3], align_g_tgt, p["rot_eps"]
                                )
                                xz_ac = self.align2D(
                                    gripvec[0::2], xz_align_g, p["rot_eps"]
                                )
                                rot_action = [xy_ac, yz_ac, xz_ac]
                                print("rotation", rot_action)
                                # if xy_ac == 0:
                                if rot_action == [0, 0, 0]:
                                    grip_pos = self._get_pos(grip_site[idx])[0:2]
                                    g_pos = self._get_leg_grasp_pos()
                                    # action[0:2] = self.move_xy(
                                    #     grip_pos, g_pos[0:2], p["eps"]
                                    # )
                                else:
                                    # panda: smaller angular velocity
                                    action[3:6] = [xy_ac, yz_ac, xz_ac]
                        if all(action[0:6] == 0):
                                self._phase_num += 1
                                print("the next phase", self._phases_gen[self._phase_num])

                    elif self._phase == "xy_move_wire":
                        action[6] = 0.2
                        action_robot2[6] = 1
                        # grip_xy_pos = self._get_pos(grip_site[0])[0:2]
                        # g_xy_pos = self._get_leg_grasp_pos()[0:2]
                        # action[0:2] = self.move_xy(
                        #     grip_xy_pos, g_xy_pos, p["eps"], noise=noise[self._phase]
                        # )

                        grip_xyz_pos = self._get_pos(grip_site[0])
                        # TODO: integrate get leg grasp pos
                        g_l, g_r = f"{'B4'}_ltgt_site{0}", f"{'B4'}_rtgt_site{0}"
                        g_xyz_pos = (self._get_pos(g_l) + self._get_pos(g_r)) / 2
                        pre_xyz_pos = g_xyz_pos + np.array([0, 0, 0.05])
                        action[0:3] = self.move_xyz(
                            grip_xyz_pos, pre_xyz_pos, p["eps"], noise=np.array([0, 0, 0])
                        )
                        # print("action", action)
                        if all(action[0:3] == 0):
                            self._phase_num += 1
                            print("the next phase", self._phases_gen[self._phase_num])

                    elif self._phase == "xy_fit_wire":
                        action[6] = 1
                        action_robot2[6] = 1
                        # grip_xy_pos = self._get_pos(grip_site[0])[0:2]
                        # g_xy_pos = self._get_leg_grasp_pos()[0:2]
                        # action[0:2] = self.move_xy(
                        #     grip_xy_pos, g_xy_pos, p["eps"], noise=noise[self._phase]
                        # )

                        grip_xyz_pos = self._get_pos(grip_site[0])
                        treach_pos = self.sim.data.get_site_xpos("clip-wire,reach_site1")
                        # TODO: integrate get leg grasp pos
                        g_l, g_r = f"{'B4'}_ltgt_site{0}", f"{'B4'}_rtgt_site{0}"
                        g_xyz_pos = (self._get_pos(g_l) + self._get_pos(g_r)) / 2
                        # pre_xyz_pos = g_xyz_pos + np.array([0, 0, 0.05])
                        # action[0:3] = self.move_xyz(
                        #     grip_xyz_pos, pre_xyz_pos, p["eps"], noise=np.array([0, 0, 0])
                        # )
                        action[0:2] = self.move_xy(
                            grip_xyz_pos[0:2],
                            # gconn_inverse_pos,
                            treach_pos[0:2],
                            p["eps"],
                            np.array([0, 0]),  # + p["z_finedist"],
                        )
                        # follow translation:
                        action_robot2[0:2] = 0.5*copy.deepcopy(action[0:2])
                        # print("action", action)
                        if all(action[0:2] == 0):
                            action[0:3] = self.move_xyz(
                                grip_xyz_pos[0:3],
                                # gconn_inverse_pos,
                                treach_pos[0:3],
                                p["eps"],
                                np.array([0, 0, 0]),  # + p["z_finedist"],
                            )
                            # follow translation:
                            action_robot2[0:3] = 0.5 * copy.deepcopy(action[0:3])
                            touch = self._read_sensor("clip1_touch")[0]
                            if touch > 0.2:
                                print("touched")
                            # if all(action[0:3] == 0):
                                self._phase_num += 1
                                print("the next phase", self._phases_gen[self._phase_num])

                    self._phase = self._phases_gen[self._phase_num]
                    action[0:3] = p["lat_magnitude"] * action[0:3]
                    action[3:6] = p["rot_magnitude"] * action[3:6]
                    action = self._norm_rot_action(action)
                    action[0:5] = np.clip(action[0:5], -0.4, 0.4) # [-1,1]

                    action_robot2[0:3] = p["lat_magnitude"] * action_robot2[0:3]
                    action_robot2[3:6] = p["rot_magnitude"] * action_robot2[3:6]
                    action_robot2 = self._norm_rot_action(action_robot2)
                    action_robot2[0:5] = np.clip(action_robot2[0:5], -0.4, 0.4)  # [-1,1]
                    # print(action)
                    # TODO: action should have 2*8 dims
                    # total_action = np.append(action, action_robot2)
                    # total_action = {'0': action,
                    #                 "1": action_robot2}
                    total_action = [action, action_robot2]
                    ob, reward, _, info = self.step(total_action)

                    if self._config.render:
                        self.render(mode="human")
                        # time buffer
                        # self.render()
                    if self._config.record_vid:
                        self.vid_rec.capture_frame(self.render("rgb_array")[0])

                    buffer = True

                    if self._episode_length > self._config.max_episode_steps:
                        logger.info(
                            "Time-limit exceeds %d/%d",
                            self._episode_length,
                            self._config.max_episode_steps,
                        )
                        break
                    if self._success and self._phase == 'part_done':
                        break

                if self._part_success:
                    self._used_sites.add(gconn)
                    self._used_sites.add(tconn)
                    self._part_success = False

                if self._success and self._phase == 'part_done':
                    logger.warn(
                        "assembled (%s) in %d steps!",
                        self._config.furniture_name,
                        self._episode_length,
                    )
                    if self._config.record_vid:
                        self.vid_rec.close()
                    if self._config.start_count is not None:
                        demo_count = self._config.start_count + n_successful_demos
                        self._demo.save(self.file_prefix, count=demo_count)
                    else:
                        self._demo.save(self.file_prefix)
                    pbar.update(1)
                    n_successful_demos += 1
                    print("Max griptip height = %f" % max_griptip_height)
                    break
                elif self._episode_length > self._config.max_episode_steps:
                    # failed
                    logger.warn("Failed to assemble!")
                    n_failed_demos += 1
                    if self._config.record_vid:
                        self.vid_rec.close(success=True)
                    break

        logger.info("n_failed_demos: %d", n_failed_demos)


def main():
    from furniture.config import create_parser

    parser = create_parser(env="IKEATwoPandaGen-v0", single_arm=False)
    parser.set_defaults(render=True)
    parser.set_defaults(start_count=0)
    parser.set_defaults(furniture_name="wire_insertion_parallel_v1")
    parser.set_defaults(n_demos=100)
    parser.set_defaults(camera_ids=[0])
    parser.set_defaults(debug=True)
    parser.set_defaults(init_base_position=[[0, 0.65, -0.7], [0.7, 0.65, -0.7]])
    parser.set_defaults(fix_init_parts=["wire1"])
    # parser.set_defaults(follow=True)
    parser.set_defaults(unity=False)
    parser.set_defaults(background='Simple')
    parser.set_defaults(record_demo=False)
    parser.set_defaults(record_vid=False)
    parser.set_defaults(obstacles=False)

    config, unparsed = parser.parse_known_args()
    if len(unparsed):
        logger.error("Unparsed argument is detected:\n%s", unparsed)
        return

    env = FurnitureTwoPandaGenEnv(config)
    env.generate_demos(config.n_demos)


if __name__ == "__main__":
    main()
