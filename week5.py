from stable_baselines3 import PPO
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize
import gymnasium
import torch
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

class RewardTrackerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(RewardTrackerCallback, self).__init__(verbose)
        self.episode_rewards = []

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        # Compute and track the mean episode reward
        mean_reward = np.mean(self.model.ep_info_buffer['r'])
        self.episode_rewards.append(mean_reward)


PAPER_HPS = {
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "n_epochs": 5,
    "learning_rate": 1e-4,
    "batch_size": 64,
    "n_steps": 2048,
    "ent_coef": 0.0,
    "clip_range": 0.2,
}

SB_ZOO_HPS = {
    "gamma": 0.99,
    "clip_range": 0.2,
    "ent_coef": 0.00229519,
    "gae_lambda": 0.99,
    "n_epochs": 5,
    "learning_rate": 9.80828e-05,
    "max_grad_norm": 0.7,
    "batch_size": 32,
    "n_steps": 512,
    "vf_coef": 0.835671
    
}


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def make_env():
      return gymnasium.make("Walker2d-v3")

    env = DummyVecEnv([make_env])
    env = VecNormalize(env)

    model = PPO(
        "MlpPolicy",
        env,
        device=device
    )

    model.learn(total_timesteps=1000, callback=RewardTrackerCallback())