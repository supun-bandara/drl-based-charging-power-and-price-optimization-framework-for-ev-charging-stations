# check the environment
# runs for one termination

import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from charging_station.environment.env import ChargingEnv
env = ChargingEnv()

episodes = 1
for episode in range(episodes):
	done = False
	obs = env.reset()
	#print("reset")
	while not done:
		random_action = env.action_space.sample()
		#print("random_action", random_action)
		#print("step")
		obs, reward, done, finish, info = env.step(random_action)
		#print("obs", obs, "reward", reward, "done", done, "finish", finish, "info", info)