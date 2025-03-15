# for testing purposes only

import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from charging_station.environment.env import ChargingEnv
from stable_baselines3 import PPO, A2C, DQN, DDPG

env = ChargingEnv()
model = PPO("MlpPolicy", env, verbose=1).learn(10000)
