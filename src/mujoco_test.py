"""
Shows how to toss a capsule to a container.
"""
from mujoco_py import load_model_from_path, MjSim, MjViewer
import os
import numpy as np

# Load the model and environment from its xml file
# model = load_model_from_path("/home/soheil/Sync/unh/courses/hri/project/src/Soheil/mujoco210/model/control_pendulum/pendulum.xml")
model = load_model_from_path("/home/soheil/Sync/unh/courses/hri/project/src/" +
                             "Soheil/mujoco210/model/threejoint.xml")
sim = MjSim(model)

# the time for each episode of the simulation
sim_horizon = 2000

# initialize the simulation visualization
viewer = MjViewer(sim)

# get initial state of simulation
sim_state = sim.get_state()
sim_state.qpos[2] = sim_state.qpos[0] + .1 * np.random.rand()

# repeat indefinitely
while True:
    # set simulation to initial state
    sim.set_state(sim_state)

    # for the entire simulation horizon
    for i in range(sim_horizon):

        # trigger the lever within the 0 to 150 time period
        # if i < 150:
        #     sim.data.ctrl[:] = 0.0
        # else:
        #     sim.data.ctrl[:] = -1.0
        # import ipdb; ipdb.set_trace()
        states = sim.get_state()
        sim.data.ctrl[1] = -20 * (states.qpos[1] - 0) - 10 * states.qvel[1]
        sim.data.ctrl[2] = -20 * (states.qpos[2] - 0) - 10 * states.qvel[2]
        # move one time step forward in simulation
        sim.step()
        viewer.render()

    if os.getenv('TESTING') is not None:
        break
