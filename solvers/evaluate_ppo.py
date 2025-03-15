# code from previous one to evaluate the PPO model

import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import gym
import numpy as np
import os
import argparse
from stable_baselines3 import DDPG, PPO
import time
import matplotlib.pyplot as plt

from charging_station.environment.env import ChargingEnv
env = ChargingEnv()

models_dir = 'models/1705321963/'
model_path= f"{models_dir}/2140000.zip"

model = PPO.load(model_path, env=env)

episodes=100
final_reward_PPO=[0]*episodes
for ep in range(episodes):
    rewards_list_PPO = []
    obs, info = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, reward_PPO, done, finish, info = env.step(action)
        rewards_list_PPO.append(reward_PPO)
    final_reward_PPO[ep] = sum(rewards_list_PPO)
env.close

Mean_reward_PPO=np.mean(final_reward_PPO)
