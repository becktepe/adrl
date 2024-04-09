from ppo_pytorch.train_test import train
from ppo_pytorch.policies import DefaultAC, ConcatAC, InsertAC
from adrl import make_continual_rl_env
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def training(seeds, policies, total_steps):
    for seed in seeds:
        for policy_cls in policies:

            env = make_continual_rl_env()
            _, stats = train(policy_cls, env, total_steps, False, seed, gif_path=f"./w1_results/gifs/{policy_cls.__name__}_{seed}")
            stats.to_csv(f"./w1_results/{policy_cls.__name__}_{seed}.csv", index=False)

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

    #training(SEEDS, POLICIES, TOTAL_STEPS)
    plotting(SEEDS, POLICIES)
    # plot 


    
