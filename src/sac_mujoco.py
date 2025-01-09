import sys
import os
import gym
import json
import pybullet_envs
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense
import tensorflow_probability as tfp
import base64
import IPython
import imageio
from tensorflow.keras import optimizers as opt
import numpy as np
import random
import time
import wandb
from tqdm import tqdm

ENV_NAME = 'HumanoidBulletEnv-v0'
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

ENV_observation_space_shape = 3
ENV_action_space_shape = 3
ENV_action_space_high = 1
ENV_action_space_low = -1

class ReplayBuffer():
    def __init__(self, env, buffer_capacity=BUFFER_CAPACITY, batch_size=BATCH_SIZE, min_size_buffer=MIN_SIZE_BUFFER):
        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size
        self.min_size_buffer = min_size_buffer
        self.buffer_counter = 0
        self.n_games = 0
        self.states = np.zeros((self.buffer_capacity, ENV_observation_space_shape))
        self.actions = np.zeros((self.buffer_capacity, ENV_action_space_shape))
        self.rewards = np.zeros((self.buffer_capacity))
        self.next_states = np.zeros((self.buffer_capacity, ENV_observation_space_shape))
        self.dones = np.zeros((self.buffer_capacity), dtype=bool)
        
        
    def __len__(self):
        return self.buffer_counter


    def add_record(self, state, action, reward, next_state, done):
        # Set index to zero if counter = buffer_capacity and start again (1 % 100 = 1 and 101 % 100 = 1) so we substitute the older entries
        index = self.buffer_counter % self.buffer_capacity
        self.states[index] = state
        self.actions[index] = action
        self.rewards[index] = reward
        self.next_states[index] = next_state
        self.dones[index] = done
        
        # Update the counter when record something
        self.buffer_counter += 1
    

    def check_buffer_size(self):
        return self.buffer_counter >= self.batch_size and self.buffer_counter >= self.min_size_buffer
    

    def update_n_games(self):
        self.n_games += 1
    

    def get_minibatch(self):
        # If the counter is less than the capacity we don't want to take zeros records, 
        # if the cunter is higher we don't access the record using the counter 
        # because older records are deleted to make space for new one
        buffer_range = min(self.buffer_counter, self.buffer_capacity)
        
        batch_index = np.random.choice(buffer_range, self.batch_size, replace=False)
        # Convert to tensors
        state = self.states[batch_index]
        action = self.actions[batch_index]
        reward = self.rewards[batch_index]
        next_state = self.next_states[batch_index]
        done = self.dones[batch_index]
        return state, action, reward, next_state, done
    

    def save(self, folder_name):
        """
        Save the replay buffer
        """
        if not os.path.isdir(folder_name):
            os.mkdir(folder_name)
        np.save(folder_name + '/states.npy', self.states)
        np.save(folder_name + '/actions.npy', self.actions)
        np.save(folder_name + '/rewards.npy', self.rewards)
        np.save(folder_name + '/next_states.npy', self.next_states)
        np.save(folder_name + '/dones.npy', self.dones)
        
        dict_info = {"buffer_counter": self.buffer_counter, "n_games": self.n_games}
        
        with open(folder_name + '/dict_info.json', 'w') as f:
            json.dump(dict_info, f)


    def load(self, folder_name):
        """
        Load the replay buffer
        """
        self.states = np.load(folder_name + '/states.npy')
        self.actions = np.load(folder_name + '/actions.npy')
        self.rewards = np.load(folder_name + '/rewards.npy')
        self.next_states = np.load(folder_name + '/next_states.npy')
        self.dones = np.load(folder_name + '/dones.npy')
        
        with open(folder_name + '/dict_info.json', 'r') as f:
            dict_info = json.load(f)
        self.buffer_counter = dict_info["buffer_counter"]
        self.n_games = dict_info["n_games"]

class Critic(tf.keras.Model):
    def __init__(self, name, hidden_0=CRITIC_HIDDEN_0, hidden_1=CRITIC_HIDDEN_1):
        super(Critic, self).__init__()
        self.hidden_0 = hidden_0
        self.hidden_1 = hidden_1
        self.net_name = name
        self.dense_0 = Dense(self.hidden_0, activation='relu')
        self.dense_1 = Dense(self.hidden_1, activation='relu')
        self.q_value = Dense(1, activation=None)


    def call(self, state, action):
        state_action_value = self.dense_0(tf.concat([state, action], axis=1))
        state_action_value = self.dense_1(state_action_value)
        q_value = self.q_value(state_action_value)
        return q_value

class CriticValue(tf.keras.Model):
    def __init__(self, name, hidden_0=CRITIC_HIDDEN_0, hidden_1=CRITIC_HIDDEN_1):
        super(CriticValue, self).__init__()
        self.hidden_0 = hidden_0
        self.hidden_1 = hidden_1
        self.net_name = name
        self.dense_0 = Dense(self.hidden_0, activation='relu')
        self.dense_1 = Dense(self.hidden_1, activation='relu')
        self.value = Dense(1, activation=None)


    def call(self, state):
        value = self.dense_0(state)
        value = self.dense_1(value)
        value = self.value(value)
        return value

class Actor(tf.keras.Model):
    def __init__(self, name, upper_bound, actions_dim, hidden_0=CRITIC_HIDDEN_0, hidden_1=CRITIC_HIDDEN_1, epsilon=EPSILON, log_std_min=LOG_STD_MIN, log_std_max=LOG_STD_MAX):
        super(Actor, self).__init__()
        self.hidden_0 = hidden_0
        self.hidden_1 = hidden_1
        self.actions_dim = actions_dim
        self.net_name = name
        self.upper_bound = upper_bound
        self.epsilon = epsilon
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.dense_0 = Dense(self.hidden_0, activation='relu')
        self.dense_1 = Dense(self.hidden_1, activation='relu')
        self.mean = Dense(self.actions_dim, activation=None)
        self.log_std = Dense(self.actions_dim, activation=None)


    def call(self, state):
        policy = self.dense_0(state)
        policy = self.dense_1(policy)
        mean = self.mean(policy)
        log_std = self.log_std(policy)
        log_std = tf.clip_by_value(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std


    def get_action_log_probs(self, state, reparameterization_trick=True):
        mean, log_std = self.call(state)
        std = tf.exp(log_std)
        normal_distr = tfp.distributions.Normal(mean, std)
        # Reparameterization trick
        z = tf.random.normal(shape=mean.shape, mean=0., stddev=1.)
        if reparameterization_trick:
            actions = mean + std * z
        else:
            actions = normal_distr.sample()
        action = tf.math.tanh(actions) * self.upper_bound
        log_probs = normal_distr.log_prob(actions) - tf.math.log(1 - tf.math.pow(action,2) + self.epsilon)
        log_probs = tf.math.reduce_sum(log_probs, axis=1, keepdims=True)
        return action, log_probs

class Agent:
    def __init__(self, env, path_save=PATH_SAVE, path_load=PATH_LOAD, actor_lr=ACTOR_LR, critic_lr=CRITIC_LR, gamma=GAMMA, tau=TAU, reward_scale=REWARD_SCALE):
        self.gamma = gamma
        self.tau = tau
        self.replay_buffer = ReplayBuffer(env)
        self.actions_dim = ENV_action_space_shape
        self.upper_bound = ENV_action_space_high
        self.lower_bound = ENV_action_space_low
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.path_save = path_save
        self.path_load = path_load
        self.actor = Actor(actions_dim=self.actions_dim, name='actor', upper_bound=env.action_space.high)
        self.critic_0 = Critic(name='critic_0')
        self.critic_1 = Critic(name='critic_1')
        self.critic_value = CriticValue(name='value')
        self.critic_target_value = CriticValue(name='target_value')
        self.actor.compile(optimizer=opt.Adam(learning_rate=self.actor_lr))
        self.critic_0.compile(optimizer=opt.Adam(learning_rate=self.critic_lr))
        self.critic_1.compile(optimizer=opt.Adam(learning_rate=self.critic_lr))
        self.critic_value.compile(optimizer=opt.Adam(learning_rate=self.critic_lr))
        self.critic_target_value.compile(optimizer=opt.Adam(learning_rate=self.critic_lr))
        self.reward_scale = reward_scale
        self.critic_target_value.set_weights(self.critic_value.weights)

    def add_to_replay_buffer(self, state, action, reward, new_state, done):
        self.replay_buffer.add_record(state, action, reward, new_state, done)


    def save(self):
        date_now = time.strftime("%Y%m%d%H%M")
        if not os.path.isdir(f"{self.path_save}/save_agent_{ENV_NAME.lower()}_{date_now}"):
            os.makedirs(f"{self.path_save}/save_agent_{ENV_NAME.lower()}_{date_now}")
        self.actor.save_weights(f"{self.path_save}/save_agent_{ENV_NAME.lower()}_{date_now}/{self.actor.net_name}.h5")
        self.critic_0.save_weights(f"{self.path_save}/save_agent_{ENV_NAME.lower()}_{date_now}/{self.critic_0.net_name}.h5")
        self.critic_1.save_weights(f"{self.path_save}/save_agent_{ENV_NAME.lower()}_{date_now}/{self.critic_1.net_name}.h5")
        self.critic_value.save_weights(f"{self.path_save}/save_agent_{ENV_NAME.lower()}_{date_now}/{self.critic_value.net_name}.h5")
        self.critic_target_value.save_weights(f"{self.path_save}/save_agent_{ENV_NAME.lower()}_{date_now}/{self.critic_target_value.net_name}.h5")
        self.replay_buffer.save(f"{self.path_save}/save_agent_{ENV_NAME.lower()}_{date_now}")


    def load(self):
        self.actor.load_weights(f"{self.path_load}/{self.actor.net_name}.h5")
        self.critic_0.load_weights(f"{self.path_load}/{self.critic_0.net_name}.h5")
        self.critic_1.load_weights(f"{self.path_load}/{self.critic_1.net_name}.h5")
        self.critic_value.load_weights(f"{self.path_load}/{self.critic_value.net_name}.h5")
        self.critic_target_value.load_weights(f"{self.path_load}/{self.critic_target_value.net_name}.h5")
        self.replay_buffer.load(f"{self.path_load}")


    def get_action(self, observation):
        state = tf.convert_to_tensor([observation])
        actions, _ = self.actor.get_action_log_probs(state, reparameterization_trick=False)
        return actions[0]

    def learn(self):
        if self.replay_buffer.check_buffer_size() == False:
            return
        state, action, reward, new_state, done = self.replay_buffer.get_minibatch()
        states = tf.convert_to_tensor(state, dtype=tf.float32)
        new_states = tf.convert_to_tensor(new_state, dtype=tf.float32)
        rewards = tf.convert_to_tensor(reward, dtype=tf.float32)
        actions = tf.convert_to_tensor(action, dtype=tf.float32)

        # Critic value
        with tf.GradientTape() as tape:
            value = tf.squeeze(self.critic_value(states), 1)
            target_value = tf.squeeze(self.critic_target_value(new_states), 1)
            policy_actions, log_probs = self.actor.get_action_log_probs(states, reparameterization_trick=False)
            log_probs = tf.squeeze(log_probs,1)
            q_value_0 = self.critic_0(states, policy_actions)
            q_value_1 = self.critic_1(states, policy_actions)
            q_value = tf.squeeze(tf.math.minimum(q_value_0, q_value_1), 1)
            value_target = q_value - log_probs
            value_critic_loss = 0.5 * tf.keras.losses.MSE(value, value_target)
        value_critic_gradient = tape.gradient(value_critic_loss, self.critic_value.trainable_variables)
        self.critic_value.optimizer.apply_gradients(zip(value_critic_gradient, self.critic_value.trainable_variables))

        # Actor
        with tf.GradientTape() as tape:
            new_policy_actions, log_probs = self.actor.get_action_log_probs(states, reparameterization_trick=True)
            log_probs = tf.squeeze(log_probs, 1)
            new_q_value_0 = self.critic_0(states, new_policy_actions)
            new_q_value_1 = self.critic_1(states, new_policy_actions)
            new_q_value = tf.squeeze(tf.math.minimum(new_q_value_0, new_q_value_1), 1)

            actor_loss = log_probs - new_q_value
            actor_loss = tf.math.reduce_mean(actor_loss)
        actor_gradient = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(actor_gradient, self.actor.trainable_variables))

        # Critic Q value
        with tf.GradientTape(persistent=True) as tape:
            q_pred = self.reward_scale * reward + self.gamma * target_value * (1-done)
            old_q_value_0 = tf.squeeze(self.critic_0(state, action), 1)
            old_q_value_1 = tf.squeeze(self.critic_1(state, action), 1)
            critic_0_loss = 0.5 * tf.keras.losses.MSE(old_q_value_0, q_pred)
            critic_1_loss = 0.5 * tf.keras.losses.MSE(old_q_value_1, q_pred)

        critic_0_network_gradient = tape.gradient(critic_0_loss, self.critic_0.trainable_variables)
        critic_1_network_gradient = tape.gradient(critic_1_loss, self.critic_1.trainable_variables)

        self.critic_0.optimizer.apply_gradients(zip(critic_0_network_gradient, self.critic_0.trainable_variables))
        self.critic_1.optimizer.apply_gradients(zip(critic_1_network_gradient, self.critic_1.trainable_variables))

        # self.update_target_networks(tau=self.tau)
        self.replay_buffer.update_n_games()


    def update_target_network(self):
        """
        Update the target Q network
        """
        self.target_dqn.set_weights(self.DQN.get_weights())


config = dict(
  learning_rate_actor=ACTOR_LR,
  learning_rate_critic=ACTOR_LR,
  batch_size=BATCH_SIZE,
  architecture="SAC",
  infra="Colab",
  env=ENV_NAME
)


wandb.init(
  project=f"tensorflow2_sac_{ENV_NAME.lower()}",
  tags=["SAC", "FCL", "RL"],
  config=config,
)

env = gym.make(ENV_NAME)
agent = Agent(env)
scores = []
evaluation = True
if PATH_LOAD is not None:
    print("loading weights")
    observation = env.reset()
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
    states = env.reset()
    done = False
    score = 0
    while not done:
        action = agent.get_action(states)
        new_states, reward, done, info = env.step(action)
        score += reward
        agent.add_to_replay_buffer(states, action, reward, new_states, done)
        agent.learn()
        states = new_states

    scores.append(score)
    agent.replay_buffer.update_n_games()
    wandb.log({'Game number': agent.replay_buffer.n_games,
                '# Episodes': agent.replay_buffer.buffer_counter,
                "Average reward": round(np.mean(scores[-10:]), 2),
                "Time taken": round(time.time() - start_time, 2)})

    if (_ + 1) % SAVE_FREQUENCY == 0:
        # print("saving...")
        agent.save()
        # print("saved")



with imageio.get_writer("video/sac.mp4", fps=60) as video:
    terminal = True
    states = env.reset()
    for frame in tqdm(range(MAX_GAMES)):
        if terminal:
            env.reset()
            terminal = False

        # action = np.random.uniform(env.action_space.low[0],
        #                             env.action_space.high[0],
        #                             size=env.action_space.shape)
        action = agent.get_action(states)

        # Step action
        new_states, reward, terminal, info = env.step(action)

        video.append_data(env.render(mode='rgb_array'))
        states = new_states
