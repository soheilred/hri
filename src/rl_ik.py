from sac import Agent
from mujoco_py import load_model_from_path, MjSim, MjViewer
import os
import numpy as np
import math
import numpy.linalg as LA

def get_reward(state, state_desired):
    l1 = 0.1
    l2 = 0.1
    l3 = 0.1
    x_e = l1 * math.cos(state[0]) + l2 * math.cos(state[1]) + l3 *
          math.cos(state[2]) 
    y_e = l1 * math.sin(state[0]) + l2 * math.sin(state[0] + state[1]) + l3 *
          math.sin(state[0] + state[1] + state[2]) 
    phi = state[0] + state[1] + state[2]
    reward = LA.norm(np.array([x_e, y_e, phi]) - state_desired, 2)
    return reward


# ENV_NAME = 'HumanoidBulletEnv-v0'
ENV_NAME = 'mujoco-threelink'
PATH_SAVE = "/home/soheil/Documents/tmp/model"
PATH_LOAD = None # "model/save_agent_humanoidbulletenv-v0_202203081546"
BATCH_SIZE = 64
MIN_SIZE_BUFFER = 100
BUFFER_CAPACITY = 1000000
CRITIC_HIDDEN_0 = 512
CRITIC_HIDDEN_1 = 256
ACTOR_HIDDEN_0 = 512
ACTOR_HIDDEN_1 = 256
ACTOR_LR = .0005
CRITIC_LR = .0005
GAMMA = .99
TAU = .05
THETA = 0.15
DT = 1e-1
REWARD_SCALE = 2
EPSILON = 1e-6
LOG_STD_MIN = -20
LOG_STD_MAX = 2
MAX_GAMES = 20000
SAVE_FREQUENCY = 1000

config = dict(
  learning_rate_actor = ACTOR_LR,
  learning_rate_critic = ACTOR_LR,
  batch_size = BATCH_SIZE,
  architecture = "SAC",
  infra = "Manjaro",
  env = ENV_NAME
)


# Load the model and environment from its xml file
file_name = "/home/soheil/Sync/unh/courses/hri/project/src/Soheil/mujoco210/model/threejoint.xml"
model = load_model_from_path(file_name)
sim = MjSim(model)

# the time for each episode of the simulation
sim_horizon = 4 * 1000

# initialize the simulation visualization
viewer = MjViewer(sim)

# get initial state of simulation
sim_state = sim.get_state()
# sim_state.qpos[0] = sim_state.qpos[0] + 1000 * np.random.rand()


# env = gym.make(ENV_NAME)
# agent = Agent(env)
agent = Agent(sim)
scores = []
evaluation = True
if PATH_LOAD is not None:
    print("loading weights")
    observation = sim.get_state() # env.reset()
    action, log_probs = agent.actor.get_action_log_probs(observation[None, :], False)
    agent.actor(observation[None, :])
    agent.critic_0(observation[None, :], action)
    agent.critic_1(observation[None, :], action)
    agent.critic_value(observation[None, :])
    agent.critic_target_value(observation[None, :])
    agent.load()
    print(agent.replay_buffer.buffer_counter)
    print(agent.replay_buffer.n_games)

for _ in tqdm(range(MAX_GAMES)):
    start_time = time.time()
    # states = env.reset()
    states = sim.get_state()
    done = False
    score = 0
    for i in range(sim_horizon):
        if i == sim_horizon - 1:
            done = True
        action = agent.get_action(states)
        sim.step()
        new_states = sim.get_state()
        # viewer.render()
        # new_states, reward, done, info = env.step(action)
        reward = get_reward(state)
        score += reward
        agent.add_to_replay_buffer(states, action, reward, new_states, done)
        agent.learn()
        states = new_states

    scores.append(score)
    agent.replay_buffer.update_n_games()
    if (_ + 1) % SAVE_FREQUENCY == 0:
        agent.save()


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
        sim.data.ctrl[0] = -10 * (states.qpos[0] - 0) - 1 * states.qvel[0]
        # move one time step forward in simulation
        sim.step()
        viewer.render()

    if os.getenv('TESTING') is not None:
        break
