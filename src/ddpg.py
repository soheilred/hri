class ReplayBuffer():
    def __init__(self, env, buffer_capacity=BUFFER_CAPACITY, batch_size=BATCH_SIZE, min_size_buffer=MIN_SIZE_BUFFER):
        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size
        self.min_size_buffer = min_size_buffer
        self.buffer_counter = 0
        self.n_games = 0
        self.states = np.zeros((self.buffer_capacity, env.observation_space.shape[0]))
        self.actions = np.zeros((self.buffer_capacity, env.action_space.shape[0]))
        self.rewards = np.zeros((self.buffer_capacity))
        self.next_states = np.zeros((self.buffer_capacity, env.observation_space.shape[0]))
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

class Actor(tf.keras.Model):
    def __init__(self, name, actions_dim, upper_bound, hidden_0=CRITIC_HIDDEN_0, hidden_1=CRITIC_HIDDEN_1, init_minval=INIT_MINVAL, init_maxval=INIT_MAXVAL):
        super(Actor, self).__init__()
        self.hidden_0 = hidden_0
        self.hidden_1 = hidden_1
        self.actions_dim = actions_dim
        self.init_minval = init_minval
        self.init_maxval = init_maxval
        self.upper_bound = upper_bound
        
        self.net_name = name
        self.dense_0 = Dense(self.hidden_0, activation='relu')
        self.dense_1 = Dense(self.hidden_1, activation='relu')
        self.policy = Dense(self.actions_dim,
                            kernel_initializer=tf.keras.initializers.\
                                random_uniform(minval=self.init_minval,
                                               maxval=self.init_maxval),
                            activation='tanh')
                            minval=self.init_minval, maxval=self.init_maxval),
                            activation='tanh')


    def call(self, state):
        x = self.dense_0(state)
        policy = self.dense_1(x)
        policy = self.policy(policy)
        return policy * self.upper_bound


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

class Agent:
    def __init__(self, env, actor_lr=ACTOR_LR, critic_lr=CRITIC_LR, gamma=GAMMA, max_size=BUFFER_CAPACITY, tau=TAU, path_save=PATH_SAVE, path_load=PATH_LOAD):
        
        self.gamma = gamma
        self.tau = tau
        self.replay_buffer = ReplayBuffer(env, max_size)
        self.actions_dim = env.action_space.shape[0]
        self.upper_bound = env.action_space.high[0]
        self.lower_bound = env.action_space.low[0]
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
        self.critic_target_value.compile(optimizer=opt.Adam(learning_rate=
                                                            self.critic_lr))
        self.reward_scale = reward_scale
        self.critic_target_value.set_weights(self.critic_value.weights)

    def update_target_networks(self, tau):
        critic_value_weights = self.critic_value.weights
        critic_target_value_weights = self.critic_target_value.weights
        for index in range(len(critic_value_weights)):
            critic_target_value_weights[index] = tau * critic_value_weights[index] + (1 - tau) * critic_target_value_weights[index]

        self.critic_target_value.set_weights(critic_target_value_weights)
        
        # actor_weights = self.actor.weights
        # target_actor_weights = self.target_actor.weights
        # for index in range(len(actor_weights)):
        #     target_actor_weights[index] = (tau * actor_weights[index] +
        #                                 (1 - tau) * target_actor_weights[index])
        #     self.target_actor.set_weights(target_actor_weights)
        
        # critic_weights = self.critic.weights
        # target_critic_weights = self.target_critic.weights
    
        # for index in range(len(critic_weights)):
        #     target_critic_weights[index] = (tau * critic_weights[index] +
        #                                 (1 - tau) * target_critic_weights[index])
        #     self.target_critic.set_weights(target_critic_weights)

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

    def _ornstein_uhlenbeck_process(self, x, theta=THETA, mu=0, dt=DT, std=0.2):
        """
        Ornstein–Uhlenbeck process
        """
        return x + theta * (mu-x) * dt + std * np.sqrt(dt) *
                                np.random.normal(size=self.actions_dim)


    def get_action(self, observation, noise, evaluation=False):
        state = tf.convert_to_tensor([observation], dtype=tf.float32)
        actions = self.actor(state)
        if not evaluation:
            self.noise = self._ornstein_uhlenbeck_process(noise)
            actions += self.noise
            actions = tf.clip_by_value(actions, self.lower_bound, self.upper_bound)
            return actions[0]

    def learn(self):
        if self.replay_buffer.check_buffer_size() == False:
            return

        state, action, reward, new_state, done = self.replay_buffer.get_minibatch()
        states = tf.convert_to_tensor(state, dtype=tf.float32)
        new_states = tf.convert_to_tensor(new_state, dtype=tf.float32)
        rewards = tf.convert_to_tensor(reward, dtype=tf.float32)
        actions = tf.convert_to_tensor(action, dtype=tf.float32)

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

        self.update_target_networks(tau=self.tau)
        
        self.replay_buffer.update_n_games()

        with tf.GradientTape() as tape:
            target_actions = self.target_actor(new_states)
            target_critic_values = tf.squeeze(self.target_critic(new_states,
                                                        target_actions), 1)
            critic_value = tf.squeeze(self.critic(states, actions), 1)
            target = reward + self.gamma * target_critic_values * (1-done)
            critic_loss = tf.keras.losses.MSE(target, critic_value)
            critic_gradient = tape.gradient(critic_loss,
                                            self.critic.trainable_variables)

        self.critic.optimizer.apply_gradients(zip(critic_gradient,
                                                self.critic.trainable_variables))

        with tf.GradientTape() as tape:
            policy_actions = self.actor(states)
            actor_loss = -self.critic(states, policy_actions)
            actor_loss = tf.math.reduce_mean(actor_loss)
            actor_gradient = tape.gradient(actor_loss, 
                                    self.actor.trainable_variables)

        self.actor.optimizer.apply_gradients(zip(actor_gradient,
                                                self.actor.trainable_variables))
        self.update_target_networks(self.tau)

config = dict(
  learning_rate_actor=ACTOR_LR,
  learning_rate_critic=ACTOR_LR,
  batch_size=BATCH_SIZE,
  architecture="DDPG",
  infra="Manjaro",
  env=ENV_NAME
)

wandb.init(
  project=f"tensorflow2_ddpg_{ENV_NAME.lower()}",
  tags=["DDPG", "FCL", "RL"],
  config=config,
)

env = gym.make(ENV_NAME)
agent = Agent(env)
scores = []
evaluation = True

if PATH_LOAD is not None:
    print("loading weights")
    observation = env.reset()
    action = agent.actor(observation[None, :])
    agent.target_actor(observation[None, :])
    agent.critic(observation[None, :], action)
    agent.target_critic(observation[None, :], action)
    agent.load()
    print(agent.replay_buffer.buffer_counter)
    print(agent.replay_buffer.n_games)
    print(agent.noise)

for _ in tqdm(range(MAX_GAMES)):
    start_time = time.time()
    states = env.reset()
    done = False
    score = 0
    while not done:
        action = agent.get_action(states, evaluation)
        new_states, reward, done, info = env.step(action)
        score += reward
        agent.add_to_replay_buffer(states, action, reward, new_states, done)
        agent.learn()
        states = new_states
        
    agent.replay_buffer.update_n_games()
    
    scores.append(score)
    wandb.log({'Game number': agent.replay_buffer.n_games,
                   '# Episodes': agent.replay_buffer.buffer_counter, 
                   "Average reward": round(np.mean(scores[-10:]), 2),
                    "Time taken": round(time.time() - start_time, 2)})
    if (_ + 1) % EVALUATION_FREQUENCY == 0:
        evaluation = True
        states = env.reset()
        done = False
        score = 0
        while not done:
            action = agent.get_action(states, evaluation)
            new_states, reward, done, info = env.step(action)
            score += reward
            states = new_states
        wandb.log({'Game number': agent.replay_buffer.n_games, 
                   '# Episodes': agent.replay_buffer.buffer_counter, 
                   'Evaluation score': score})
        evaluation = False
     
    if (_ + 1) % SAVE_FREQUENCY == 0:
        print("saving...")
        agent.save()
        print("saved")


