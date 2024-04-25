from pytorch_sac.train import train
import gymnasium
from adrl import make_continual_rl_env


SAC_ARGS = {
    "gamma": 0.99,
    "tau": 0.005,
    "lr": 0.0003,
    "alpha": 0.2,
    "policy": "Gaussian",
    "target_update_interval": 1,
    "automatic_entropy_tuning": False,
    "cuda": False,
    "hidden_size": 256,
}

    
if __name__ == "__main__":
    SEEDS = range(5)
    BUFFER_ALPHAS = [0., 0.25, 0.5, 0.75, 1.]
    STEPS = 1e6

    # Default LunarLander
    for seed in SEEDS:
        for buffer_alpha in BUFFER_ALPHAS:
            env = gymnasium.make("LunarLander-v2", continuous=True)
            eval_env = gymnasium.make("LunarLander-v2", continuous=True)
            result = train(env, eval_env, SAC_ARGS, num_steps=STEPS, buffer_alpha=0., seed=seed)
            result.to_csv(f"./w2_results/Gym_SAC_{buffer_alpha}_{seed}.csv")

    # CARL LunarLander
    ENV_KWARGS = { "continuous": True } 
    for seed in SEEDS:
        for buffer_alpha in BUFFER_ALPHAS:
            env = make_continual_rl_env(env_kwargs=ENV_KWARGS)
            eval_env = make_continual_rl_env(env_kwargs=ENV_KWARGS, eval_mode=True)
            result = train(env, eval_env, SAC_ARGS, num_steps=STEPS, buffer_alpha=0., seed=seed)
            result.to_csv(f"./w2_results/CARL_SAC_{buffer_alpha}_{seed}.csv")
