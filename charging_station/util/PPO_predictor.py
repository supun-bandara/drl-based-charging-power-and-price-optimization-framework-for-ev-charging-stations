import sys
import os
from stable_baselines3 import PPO
from charging_station.environment.env import ChargingEnv

class ppo_predictor:
    def __init__(self, models_dir='models/1705321963/', model_file='2140000.zip'):

        self.models_dir = models_dir
        self.model_path = os.path.join(models_dir, model_file)
        self.env = ChargingEnv()
        self.model = PPO.load(self.model_path, env=self.env)

    def predict(self, obs):

        action, _states = self.model.predict(obs)
        return action

