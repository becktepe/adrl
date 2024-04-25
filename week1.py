from ppo_pytorch.train_test import train
from ppo_pytorch.PPO import PPO
from ppo_pytorch.policies import DefaultAC, ConcatAC, InsertAC
from adrl import make_continual_rl_env
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import gymnasium


def plotting(seeds, policies):
    aggregated_stats = {}  # Dictionary to store the aggregated stats DataFrames for each policy

    for policy_cls in policies:
        df_list = []  # List to store DataFrames for each seed for the current policy
        for seed in seeds:
            # Read the CSV file for the current policy and seed
            df = pd.read_csv(f"./w1_results/{policy_cls.__name__}_{seed}.csv")
            df_list.append(df)
        
        # Concatenate all DataFrames for the current policy
        concatenated_df = pd.concat(df_list)
        
        # Group by 'step' and calculate mean and std for 'reward' and 'ep_reward'
        grouped = concatenated_df.groupby('step').agg({'reward': ['mean', 'std'], 'avg_ep_reward': ['mean', 'std']})
        
        # Rename columns for clarity
        grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]
        
        # Reset index to make 'step' a column again
        final_df = grouped.reset_index()
        
        # Store in dictionary
        aggregated_stats[policy_cls.__name__] = final_df

    combined_df = pd.DataFrame()
    for policy_name, df in aggregated_stats.items():
        df['policy'] = policy_name
        combined_df = combined_df.append(df, ignore_index=True)

    plt.figure(figsize=(10, 6))
    for policy in combined_df['policy'].unique():
        policy_df = combined_df[combined_df['policy'] == policy]
        sns.lineplot(data=policy_df, x='step', y='avg_ep_reward_mean', label=policy)
        plt.fill_between(policy_df['step'], policy_df['avg_ep_reward_mean'] - policy_df['avg_ep_reward_std'], 
                         policy_df['avg_ep_reward_mean'] + policy_df['avg_ep_reward_std'], alpha=0.5)
    plt.title('Mean Episode Return by Step for Each Policy')
    plt.xlabel('Step')
    plt.ylabel('Mean Reward')
    plt.legend(title='Policy')
    plt.tight_layout()
    plt.show()

    # Plot Mean Reward by Step for Each Policy with std deviation area
    plt.figure(figsize=(10, 6))
    for policy in combined_df['policy'].unique():
        policy_df = combined_df[combined_df['policy'] == policy]
        sns.lineplot(data=policy_df, x='step', y='reward_mean', label=policy)
        plt.fill_between(policy_df['step'], policy_df['reward_mean'] - policy_df['reward_std'], 
                         policy_df['reward_mean'] + policy_df['reward_std'], alpha=0.5)
    plt.title('Mean Reward by Step for Each Policy')
    plt.xlabel('Step')
    plt.ylabel('Mean Reward')
    plt.legend(title='Policy')
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    SEEDS = range(5)
    POLICIES = [DefaultAC, ConcatAC, InsertAC]
    TOTAL_STEPS = 1e5 + 200

    update_timestep = 1000
    K_epochs = 80               # update policy for K epochs in one PPO update

    eps_clip = 0.2          # clip parameter for PPO
    gamma = 0.99            # discount factor
    gae_lambda = 0.95

    action_std = 0.6    
    lr_actor = 0.0003       # learning rate for actor network
    lr_critic = 0.001       # learning rate for critic networ

    for seed in SEEDS:
        for policy_cls in POLICIES:

            env = make_continual_rl_env()
            # state space dimension
            state_dim = env.observation_space["obs"].shape[0]

            # action space dimension
            if isinstance(env.action_space, gymnasium.spaces.Discrete):
                has_continuous_action_space = False
                action_dim = env.action_space.shape[0]
            else:
                has_continuous_action_space = True
                action_dim = env.action_space.n

            ppo_agent = PPO(policy_cls, state_dim, 1, action_dim, lr_actor, lr_critic, gamma, gae_lambda, K_epochs, eps_clip, has_continuous_action_space, action_std)
            _, stats = train(ppo_agent, env, TOTAL_STEPS, seed, update_timestep, gif_path=f"./w1_results/gifs/{policy_cls.__name__}_{seed}")
            stats.to_csv(f"./w1_results/{policy_cls.__name__}_{seed}.csv", index=False)


    
