# default code to check the environment

import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from stable_baselines3.common.env_checker import check_env
from charging_station.environment.env import ChargingEnv
env = ChargingEnv()
check_env(env)