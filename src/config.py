import os
#******************************
#******** Enviroment **********
#******************************

#ENV_NAME = 'LunarLanderContinuous-v2'
# ENV_NAME = 'HumanoidBulletEnv-v0'
# ENV_NAME = "FetchPickAndPlace-v0" # 'HumanoidBulletEnv-v0'  "FetchReach-v1"
ENV_NAME = "FetchReach-v1"
#ENV_NAME = 'InvertedPendulumBulletEnv-v0'
#ENV_NAME = 'AntBulletEnv-v0'

HOME =  os.path.expanduser("~") 
PATH_SAVE = HOME + "/Documents/tmp/model"
# PATH_LOAD = None
PATH_LOAD =  HOME + "/Documents/tmp/model/save_agent_202204111224"

#******************************
#****** Replay Buffer *********
#******************************

BATCH_SIZE = 64
MIN_SIZE_BUFFER = 100 # Minimum size of the buffer to start learning, until then random actions
BUFFER_CAPACITY = 1000000

#******************************
#******** Networks ************
#******************************

ACTOR_HIDDEN_0 = 512
ACTOR_HIDDEN_1 = 256
INIT_MINVAL = -0.05
INIT_MAXVAL = 0.05

CRITIC_HIDDEN_0 = 512
CRITIC_HIDDEN_1 = 256

LOG_STD_MIN = -20 # exp(-10) = 4.540e-05
LOG_STD_MAX = 2 # exp(2) = 7.389
EPSILON = 1e-6

#******************************
#********** Agent *************
#******************************

GAMMA = 0.99
ACTOR_LR = 0.0015
CRITIC_LR = 0.0015

TAU = 0.05 # For soft update the target network

REWARD_SCALE = 2 # Scale factor for rewards

# Parameters for Ornstein–Uhlenbeck process
THETA = 0.15
DT = 1e-1

#******************************
#********** Main **************
#******************************

MAX_GAMES = 00000
SAVE_FREQUENCY = 2000
EVALUATION_FREQUENCY = 80
RECORDED_EVERY = 4

# ENV_observation_space_shape = 3
# ENV_action_space_shape = 3
# ENV_action_space_high = 1
# ENV_action_space_low = -1
