import os
import numpy as np
import mujoco_py
from mujoco_py import load_model_from_xml, MjSim, MjViewer


mj_path = mujoco_py.utils.discover_mujoco()
# XML_PATH = os.path.join('/home/kejia/Documents/assembly-mujoco/models/robots', '/', 'two_franka_panda.xml')
XML_PATH = '/home/kejia/Documents/assembly-mujoco/models/franka_sim/two_franka_panda.xml'
XML_PATH_ROPE = '/home/kejia/Documents/assembly-mujoco/models/franka_sim/assets/rope_expanded_short.xml'
XML_IKEA_TABLE = '/home/kejia/Documents/assembly/furniture/env/models/assets/objects/table_lack_0825.xml'
XML_PANDA_GRIPPER = '/home/kejia/Documents/assembly/furniture/env/models/assets/grippers/panda_gripper.xml'
XML_ROBOTIQ_GRIPPER = '/home/kejia/Documents/assembly/furniture/env/models/assets/grippers/robotiq_gripper.xml'
XML_SAWYER_GRIPPER = '/home/kejia/Documents/assembly/furniture/env/models/assets/grippers/two_finger_gripper.xml'
XML_WIRE = '/home/kejia/Documents/assembly/furniture/env/models/assets/objects/wire_insertion_parallel.xml'
XML_CHAIR = '/home/kejia/Documents/assembly/furniture/env/models/assets/objects/swivel_chair_0700.xml'
XML_PANDA = '/home/kejia/Documents/assembly/furniture/env/models/assets/robots/panda/robot.xml'
XML_ENV = '/home/kejia/Documents/assembly/furniture/env/models/assets/arenas/floor_arena.xml'
XML_ENV_DEFAULT = '/home/kejia/Documents/assembly/furniture/env/models/assets/arenas/floor_default.xml'
# CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
# XML_PATH = os.path.join(CURRENT_DIR, "models/objects/rope.xml")


def get_pos(sim, name):
    """
    Get the position of a site, body, or geom
    """
    if name in sim.model.body_names:
        return sim.data.get_body_xpos(name).copy()
    if name in sim.model.geom_names:
        return sim.data.get_geom_xpos(name).copy()
    if name in sim.model.site_names:
        return sim.data.get_site_xpos(name).copy()
    raise ValueError


def print_box_xpos(sim):
    print("box xpos:", sim.data.get_body_xpos("box"))


# Creating the rope
# model = load_model_from_xml(MODEL_XML)
model = mujoco_py.load_model_from_path(XML_ROBOTIQ_GRIPPER)
sim = MjSim(model)
viewer = MjViewer(sim)
viewer.vopt.geomgroup[0] = 0

states = [{'box:x': +0.8, 'box:y': +0.8},
          {'box:x': -0.8, 'box:y': +0.8},
          {'box:x': -0.8, 'box:y': -0.8},
          {'box:x': +0.8, 'box:y': -0.8},
          {'box:x': +0.0, 'box:y': +0.0}]

# MjModel.joint_name2id returns the index of a joint in
# MjData.qpos.
x_joint_i = sim.model.get_joint_qpos_addr("box:x")
y_joint_i = sim.model.get_joint_qpos_addr("box:y")

print_box_xpos(sim)

while True:
    viewer.render()
    sim.step()
    # print("hole", get_pos(sim, "hole-wire,conn_site1"))
    # print("clip", get_pos(sim, "clip-wire,0,90,180,270,conn_site1"))
    # for state in states:
    #     sim_state = sim.get_state()
        # sim_state.qpos[x_joint_i] = state["box:x"]
        # sim_state.qpos[y_joint_i] = state["box:y"]
        # sim.set_state(sim_state)
        # sim.forward()
        # print("updated state to", state)
        # print_box_xpos(sim)
        # viewer.render()

    # if os.getenv('TESTING') is not None:
    #     break