from stable_baselines3 import PPO
from gymnasium.wrappers.flatten_observation import FlattenObservation
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from carl.envs import CARLLunarLander
from gymnasium import Wrapper
import pandas as pd


class CurriculumGravityChangeWrapper(Wrapper):
    def __init__(self, env, gravity_change_interval=10000, gravity_change_kind=None, progress=0., initial_gravity=-1):
        super().__init__(env)
        self.n_steps = 0
        self.n_total_steps = 0
        self.n_switches = 0
        self.gravity_change_interval = gravity_change_interval
        self.gravity_change_kind = gravity_change_kind
        self.progress = progress
        self.initial_gravity = initial_gravity

    @property
    def observation_space(self):
        return self.env.observation_space["obs"]

    def step(self, action):
        self.n_steps += 1
        state, reward, terminated, truncated, info = self.env.step(action)
        if self.n_steps >= self.gravity_change_interval:
            truncated = True
        return state["obs"], reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        self.n_total_steps += self.n_steps
        self.n_steps = 0
        
        if self.n_total_steps  // self.gravity_change_interval > self.n_switches:
            progress = min(1, self.progress)
            gravity = (-10 - self.initial_gravity) * progress + self.initial_gravity
            gravity = max(-10, gravity)
            print("Progress:", self.progress)
            print("Gravity:", gravity)
        
            self.env.contexts[0] = {"GRAVITY_Y": gravity}
            self.env.context["GRAVITY_Y"] = gravity
            self.n_switches += 1
            self.env._update_context()
        obs, info = self.env.reset()
        return obs["obs"], info


def make_continual_rl_env(gravity_change=None, gravity_change_interval=10000, initial_gravity=-1):
    contexts = {0: {"GRAVITY_Y": initial_gravity}}
    env = CARLLunarLander(contexts=contexts)
    env = CurriculumGravityChangeWrapper(env, gravity_change_kind=gravity_change, gravity_change_interval=gravity_change_interval, initial_gravity=initial_gravity)
    return env


class RewardTrackerCallback(BaseCallback):
    def __init__(self, env, total_timesteps, verbose=0):
        super(RewardTrackerCallback, self).__init__(verbose)
        self.env = env
        self.total_timesteps = total_timesteps
        self.episode_rewards = []
        self.gravities = []

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        rewards = [ep_info['r'] for ep_info in self.model.ep_info_buffer]
        if rewards:
            mean_reward = np.mean(rewards)
            self.episode_rewards.append(mean_reward)
        self.gravities.append(self.env.context["GRAVITY_Y"])

        # Update the progress of the environment
        self.env.progress = self.num_timesteps / self.total_timesteps * 1.5


def gravity_change(initial_gravity):
    TOTAL_TIMESTEPS = 1e4
    GRAVITY_CHANGE_INTERVAL = 1e3

    env = make_continual_rl_env(gravity_change="random", gravity_change_interval=GRAVITY_CHANGE_INTERVAL, initial_gravity=initial_gravity)
    callback = RewardTrackerCallback(env=env, total_timesteps=TOTAL_TIMESTEPS)

    env = FlattenObservation(env)

    model = PPO("MlpPolicy", env)
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback)
    return callback.episode_rewards, callback.gravities


def main():
    dataframes = []
    for seed in range(2):
        for initial_gravity, name in zip([-1, -10], ["Default", "Curriculum"]):
            episode_rewards, gravities = gravity_change(initial_gravity)

            df = pd.DataFrame({
                "Avg. Episode Return": episode_rewards,
                "Gravity": gravities,
                "Seed": [seed] * len(gravities),
                "Name": [name] * len(gravities),
            })

            dataframes += [df]
    
    result = pd.concat(dataframes)
    result.to_csv("./w8_results/result.csv", index=False)


if __name__ == "__main__":
    main()