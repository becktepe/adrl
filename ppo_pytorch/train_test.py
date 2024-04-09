import time
from datetime import datetime

from collections import deque
import imageio

import torch
import numpy as np
import pandas as pd
import cv2 as cv

from .PPO import PPO

FLIP_INTERVAL = 10000
PRE_RECORD_INTERVAL = 50
POST_RECORD_INTERVAL = 100

################################### Training ###################################
def train(policy_cls, env, max_training_timesteps, has_continuous_action_space, random_seed, gif_path=None):
    print("============================================================================================")

    ################ PPO hyperparameters ################
    update_timestep = 1000      # update policy every n timesteps
    K_epochs = 80               # update policy for K epochs in one PPO update

    eps_clip = 0.2          # clip parameter for PPO
    gamma = 0.99            # discount factor

    action_std = 0.6    
    lr_actor = 0.0003       # learning rate for actor network
    lr_critic = 0.001       # learning rate for critic network
    #####################################################

    # state space dimension
    state_dim = env.observation_space["obs"].shape[0]

    # action space dimension
    if has_continuous_action_space:
        action_dim = env.action_space.shape[0]
    else:
        action_dim = env.action_space.n

    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    #####################################################

    print("============================================================================================")

    ################# training procedure ################

    # initialize a PPO agent
    ppo_agent = PPO(policy_cls, state_dim, 1, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std)

    # track total training time
    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)

    print("============================================================================================")

    # printing and logging variables
    running_reward = 0
    ep_reward = 0
    ep_rewards = np.zeros(10)
    
    stats = []

    time_step = 0
    i_episode = 0

    # gifs of gravity flips
    env.unwrapped.render_mode = "rgb_array"

    # Define font and position for annotation
    font = cv.FONT_HERSHEY_SIMPLEX
    org = (50, 50)  # Position for annotation (adjust as needed)
    fontScale = 1
    color = (255, 255, 255)  # White color
    thickness = 2

    recording = False
    frames = []  # This will store frames for the current special step sequence
    gif_counter = 1  # Counter to keep track of GIFs created

    # training loop
    while time_step <= max_training_timesteps:

        state, _ = env.reset()
        obs = state["obs"]
        
        done = False
        while not done:
            # select action with policy
            action = ppo_agent.select_action(obs, env.env.context["GRAVITY_Y"])
            state, reward, term, trunc, _ = env.step(action)
            obs = state["obs"]
            done = term or trunc

            if gif_path:
                if (time_step + 1) % FLIP_INTERVAL == FLIP_INTERVAL - PRE_RECORD_INTERVAL:
                    recording = True
                
                if recording:
                    frame = env.render()
                    frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)

                    # Add annotation
                    gravity_y_text = f"Gravity_Y: {round(env.env.context['GRAVITY_Y'], 2)}, step: {time_step}"
                    cv.putText(frame, gravity_y_text, org, font, fontScale, color, thickness, cv.LINE_AA)

                    # Convert frame back to RGB format for GIF
                    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

                    frames.append(frame)

                if recording and (time_step + 1) % FLIP_INTERVAL == POST_RECORD_INTERVAL:
                    recording = False

                    # save gif
                    output_file = f"{gif_path}_{gif_counter * FLIP_INTERVAL}.gif"
                    imageio.mimsave(output_file, frames, fps=30) 
                    frames = []
                    gif_counter += 1

            # saving reward and is_terminals
            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(done)

            time_step +=1
            running_reward += reward
            ep_reward += reward

            # update PPO agent
            if time_step % update_timestep == 0:
                avg_reward = running_reward / 1000
                running_reward = 0
                avg_ep_reward = ep_rewards.mean()

                print(f"{time_step}, step: {round(avg_reward, 2)}, ep: {round(avg_ep_reward, 2)}, gravity: {round(env.env.context['GRAVITY_Y'], 2)}")
                stats += [{
                    "step": time_step,
                    "reward": avg_reward,
                    "avg_ep_reward": avg_ep_reward,
                    "gravity": env.env.context["GRAVITY_Y"]
                }]

                ppo_agent.update()

            # break; if the episode is over
            if time_step > max_training_timesteps or done:
                break

        ep_rewards[i_episode % len(ep_rewards)] = ep_reward
        ep_reward = 0
        i_episode += 1

    # print total training time
    print("============================================================================================")
    end_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", end_time - start_time)
    print("============================================================================================")
    return ppo_agent, pd.DataFrame(stats)

#################################### Testing ###################################
def test(ppo_agent, env, test_timesteps):
    print("============================================================================================")   
    render = True              # render environment on screen
    frame_delay = 0             # if required; add delay b/w frames

    print("--------------------------------------------------------------------------------------------")

    test_running_reward = 0
    time_step = 0

    while time_step <= test_timesteps:
        state, _ = env.reset()
        obs = state["obs"]
        
        done = False
        while not done:
            action = ppo_agent.select_action(obs, env.env.context["GRAVITY_Y"])
            state, reward, term, trunc, _ = env.step(action)
            obs = state["obs"]  

            done = term or trunc
            test_running_reward += reward

            if render:
                env.render()
                time.sleep(frame_delay)

            if done:
                break

        # clear buffer
        ppo_agent.buffer.clear()

    env.close()

    print("============================================================================================")

    print("average test reward : " + str(round(test_running_reward, 2)))

    print("============================================================================================")
    return test_running_reward
    
    
    