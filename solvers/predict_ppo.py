# use this code to predict outputs

import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import os
from stable_baselines3 import DDPG, PPO

from charging_station.environment.env import ChargingEnv
env = ChargingEnv()

models_dir = 'models/1705321963/'
model_path= f"{models_dir}/2140000.zip"
model = PPO.load(model_path, env=env)

rewards_list_PPO = []
obs, info = env.reset()
action, _states = model.predict(obs)
env.close
