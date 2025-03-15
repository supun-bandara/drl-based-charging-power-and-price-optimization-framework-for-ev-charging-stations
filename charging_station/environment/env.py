# This file contains the environment class for the charging station environment
# can be used separately

import sys
import os
import pandas as pd
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os
import sys
import pathlib
from gym.utils import seeding

from charging_station.util.get_data import get_data
from charging_station.util.simulate_actions import simulate_actions
from charging_station.util.simulate_station import simulate_station
from charging_station.util.write_to_csv import write_to_csv
from panda_power.pandapower_handler import PandapowerHandler

class ChargingEnv(gym.Env):
    def __init__(self):
        super(ChargingEnv, self).__init__()
        self.number_of_chargers = 10        
        self.maximum_possible_demand = 500
        self.maximum_grid_demand = 0
        self.done = False
        low = np.array(np.zeros(2*self.number_of_chargers+17), dtype=np.float32)
        high = np.array(np.ones(2*self.number_of_chargers+17), dtype=np.float32)
        self.action_space = spaces.Box(low=0, high=1, shape=(self.number_of_chargers+1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.seed

    def step(self, actions):
        self.timestep = self.timestep + 1
        [reward, reward_1 ,reward_4, cost_2, cost_3, grid, self.res_charging_demand, self.res_parking_time] = simulate_actions(self, actions)
        pandapower_handler = PandapowerHandler(self.timestep)
        self.maximum_grid_demand = pandapower_handler.calculate_max_grid_demand()
        conditions = self.get_obs()
        results = {'timestep': self.timestep, 'price': self.grid_price, 'maximum_grid_demand': self.maximum_grid_demand, 'new_evs': self.new_evs, 'actions': actions, 'res_charging_demand': self.res_charging_demand, 'res_parking_time': self.res_parking_time, 'leave': self.leave, 'reward': -reward, 'reward_1': reward_1, 'reward_4': reward_4, 'cost_2': cost_2, 'cont_3': cost_3, 'grid': grid}
        write_to_csv(results)
        if (self.timestep == 500) or (np.count_nonzero(self.chargers == 0) == 1): # (0 not in self.chargers): #96 3072 # change this according to the number of timesteps
            self.done = True    
            self.timestep = 0
        self.info = {}
        # check if all values in chargers numpy are zero
        '''if np.all(self.chargers == 0):
            self.done = True'''
        return conditions, reward, self.done, False, self.info

    def reset(self, seed=None, reset_flag=1):
        super().reset(seed=seed)
        self.timestep = 1
        self.chargers = np.zeros(self.number_of_chargers)
        self.res_parking_time = np.zeros(self.number_of_chargers)
        self.res_charging_demand = np.zeros(self.number_of_chargers)
        self.done = False
        #self.df_station = pd.read_csv('charging_station\data\EVCS-day.csv')
        #self.df_price = pd.read_csv('charging_station\data\Price-day.csv')
        #self.df_ev_forecast = pd.read_csv('charging_station\data\EV_demand_forecast-day.csv')
        self.df_station = pd.read_csv('charging_station\data\EVCS-month.csv')
        self.df_price = pd.read_csv('charging_station\data\Price-month.csv')
        self.df_ev_forecast = pd.read_csv('charging_station\data\EV_demand_forecast-month.csv')
        self.info = {}
        return self.get_obs(), self.info

    def get_obs(self):
        [self.grid_price, self.new_evs, self.ev_forecast] = get_data(self)
        [self.leave, self.res_time, self.chargers, self.res_parking_time, self.res_charging_demand] = simulate_station(self)
        states = np.concatenate((np.array(self.grid_price+160)/2160, np.array([self.maximum_grid_demand/self.maximum_possible_demand]), np.array(self.ev_forecast)/120, np.array(self.res_charging_demand)/100, np.array(self.res_parking_time)/3000),axis=None)
        observations = np.concatenate((states),axis=None)
        observations = observations.astype(np.float32)
        return observations

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def close(self):
        return 0
