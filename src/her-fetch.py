import gym
import numpy as np
import imageio

from stable_baselines3 import HerReplayBuffer, SAC, DDPG, TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv


class CustomWrapper(gym.Wrapper):
    """
    :param env: (gym.Env) Gym environment that will be wrapped
    """
    def __init__(self, env, max_steps=100):
    # Call the parent constructor, so we can access self.env later
        super(CustomWrapper, self).__init__(env)
        self.max_steps = max_steps
        self.current_step = 0

    def reset(self):
        obs = self.env.reset()
        self.current_step = 0
        obs['desired_goal'] = np.array([1.3, .7, .5], dtype=float) # + 0.001 * np.ones(3) * self.current_step
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, info

    def step_to_goal(self, action):
        self.current_step += 1
        T = 50
        obs, reward, done, info = self.env.step(action)
        # read the human arm (LRSP) pose based on time
        obs['desired_goal'] += 0.1 * np.array([np.cos(3 / T * self.current_step),
                                                 np.sin(3 / T * self.current_step), 0])
        if self.current_step >= self.max_steps:
            done = True
            info['time_limit_reached'] = True
        return obs, reward, done, info


def main():
    env = gym.make("FetchReach-v1")
    # wrap the environment
    env = CustomWrapper(env)
    # check_env(env)

    # Create 4 artificial transitions per real transition
    n_sampled_goal = 4

    # The noise objects for DDPG
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    # DDPG hyperparams:
    model = DDPG( "MultiInputPolicy",
                env,
                replay_buffer_class=HerReplayBuffer,
                replay_buffer_kwargs=dict(
                    n_sampled_goal=n_sampled_goal,
                    goal_selection_strategy="future",
                    max_episode_length=100,
                    online_sampling=True,
                ),
                verbose=1,
                buffer_size=int(1e6),
                learning_rate=1e-3,
                gamma=0.99,
                batch_size=256,
                policy_kwargs=dict(net_arch=[256, 256, 256]),
                action_noise=action_noise,
        )

    # model.learn(int(2e5))
    # model.save("her_ddpg_fetchreach")

    # Evaluate the model every 1000 steps on 5 test episodes
    # and save the evaluation to the "logs/" folder
    # model.learn(6000, eval_freq=1000, n_eval_episodes=5, eval_log_path="./logs/")


    # load saved model
    # Because it needs access to `env.compute_reward()`
    # HER must be loaded with the env
    model = DDPG.load("her_ddpg_fetchreach", env=env)

    # train again
    # model.learn(int(1e4))

    # loaded_model = SAC.load("sac_pendulum")
    # print(f"The loaded_model has {loaded_model.replay_buffer.size()} transitions in its buffer")

    # now save the replay buffer too
    # model.save_replay_buffer("ddpg_replay_buffer")

    # load it into the loaded_model
    # loaded_model.load_replay_buffer("ddpg_replay_buffer")

    # now the loaded replay is not empty anymore
    # print(f"The loaded_model has {loaded_model.replay_buffer.size()} transitions in its buffer")

    # Save the policy independently from the model
    # Note: if you don't save the complete model with `model.save()`
    # you cannot continue training afterward
    # policy = model.policy

    obs = env.reset()

    # Evaluate the agent
    episode_reward = 0
    for _ in range(1000):
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step_to_goal(action)
        # import ipdb; ipdb.set_trace()
        # print(obs['achieved_goal'])
        # print(obs['desired_goal'])
        # print(action)
        env.render()
        # episode_reward += reward
        # if done or info.get("is_success", False):
        #     print("Reward:", episode_reward, "Success?",
        #             info.get("is_success", False))
        #     episode_reward = 0.0
        #     obs = env.reset()


    # env_id = "FetchReach-v1"
    # video_folder = 'videos/'
    # video_length = 100

    # env = DummyVecEnv([lambda: gym.make(env_id)])
    # # Record the video starting at the first step
    # env = VecVideoRecorder(env, video_folder,
    #                        record_video_trigger=lambda x: x == 0,
    #                        video_length=video_length,
    #                        name_prefix=f"her-ddpg-{env_id}")
    # obs = env.reset()
    # for _ in range(video_length + 1):
    #     action = [model.predict(obs)]
    #     obs, _, _, _ = env.step(action)
    # # Save the video
    # env.close()

    # images = []
    # obs = model.env.reset()
    # img = model.env.render(mode='rgb_array')
    # for i in range(350):
    #     images.append(img)
    #     action, _ = model.predict(obs)
    #     obs, _, _, _ = model.env.step(action)
    #     img = model.env.render(mode='rgb_array')

    # imageio.mimsave('fetchreach.gif', [np.array(img) for i, img in enumerate(images) if i%2 == 0], fps=29)

if __name__ == "__main__":
    main()
