import gymnasium
import numpy as np
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
import imageio
import os
import torch
import csv


class ImageExtrationWrapper(gymnasium.Wrapper):
    def __init__(self, env: gymnasium.Env):
        super(ImageExtrationWrapper, self).__init__(env)
        if not isinstance(env.observation_space, gymnasium.spaces.Dict):
            raise ValueError(f"Only dict spaces are supported. Got {type(env.observation_space)}.")
        self.observation_space = env.observation_space["image"]

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs["image"], info
    
    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)

        return obs["image"], reward, term, trunc, info
       

class CountBasedIntrinsicRewardWrapper(gymnasium.Wrapper):
    def __init__(self, env: gymnasium.Env, intrinsic_reward_coeff=0.01):
        super(CountBasedIntrinsicRewardWrapper, self).__init__(env)
        if not isinstance(env.observation_space, gymnasium.spaces.Discrete):
            raise ValueError(f"Only discrete observation spaces are supported. Got {type(env.observation_space)}.")
        
        self.state_visitation_counts = np.zeros(env.observation_space.n)
        self.intrinsic_reward_coeff = intrinsic_reward_coeff

    def reset(self, **kwargs):
        self.state_visitation_counts = np.zeros(self.env.observation_space.n)
        return self.env.reset(**kwargs)

    def step(self, action):
        state, reward, term, trunc, info = self.env.step(action)
        self.state_visitation_counts[state] += 1

        # pseudo count-based as mentioned in RL slides
        intrinsic_reward = self.intrinsic_reward_coeff / np.sqrt((self.state_visitation_counts[state] + 0.01))

        return state, reward + intrinsic_reward, term, trunc, info
    

class EvalRenderCallback(BaseCallback):
    """
    Callback for evaluating and rendering one episode during training.
    """

    def __init__(self, eval_freq: int = 1000, eval_env: gymnasium.Env | None = None, gif_path: str = "evaluation.gif", verbose=0):
        super(EvalRenderCallback, self).__init__(verbose)
        self.eval_freq = eval_freq
        self.eval_env = eval_env
        self.gif_path = gif_path
        self.episode_rewards = []

    def _on_step(self) -> bool:
        """
        This method will be called after each training step.
        """
        if self.n_calls % self.eval_freq == 0:
            # Evaluate policy
            images = []
            obs, _ = self.eval_env.reset()
            done = False
            episode_reward = 0.0
            while not done:
                action, _ = self.model.predict(obs)
                obs, reward, term, trunc, _ = self.eval_env.step(action)
                image = self.eval_env.render(mode='rgb_array')
                images.append(image)
                done = term or trunc
                episode_reward += float(reward)
            self.episode_rewards += [episode_reward]

            imageio.mimsave(os.path.join(self.gif_path, f"eval_{self.n_calls}.gif"), [np.array(img) for img in images])
            self.logger.record("eval/episode_reward", episode_reward)

        return True
    
    def save_rewards_to_csv(self, csv_path: str):
        with open(csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Episode", "Reward"])
            for i, reward in enumerate(self.episode_rewards):
                writer.writerow([i + 1, reward])
                

if __name__ == "__main__":
    INTRINSIC_REWARD_COEFF = 0.0

    env = gymnasium.make("Taxi-v3", render_mode="human")
    #env = ImageExtrationWrapper(env)
    env = CountBasedIntrinsicRewardWrapper(env, intrinsic_reward_coeff=INTRINSIC_REWARD_COEFF)

    eval_env = gymnasium.make("Taxi-v3", render_mode="human")
    #eval_env = ImageExtrationWrapper(eval_env)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DQN("MlpPolicy", env, verbose=1, device=device)

    eval_callback = EvalRenderCallback(eval_freq=1000, eval_env=eval_env, gif_path=f"./w3_results/{INTRINSIC_REWARD_COEFF}/", verbose=1)

    model.learn(total_timesteps=100_000)

    eval_callback.save_rewards_to_csv(f"./w3_results/{INTRINSIC_REWARD_COEFF}/evaluation_rewards.csv")
