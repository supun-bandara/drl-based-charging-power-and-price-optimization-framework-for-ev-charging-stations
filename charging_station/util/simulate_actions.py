# contains actions function that is called in the step function of the environment class
# contains the reward functions

import numpy as np
import time
from charging_station.util.predict_price_train import predict_price

def simulate_actions(self,actions):
    #print('simulate_actions - simulate_clever_control - actions', actions)
    #print()
    chargers = self.chargers
    res_parking_time = self.res_parking_time
    res_charging_demand = self.res_charging_demand
    leave=self.leave
    res_time = self.res_time

    P_charging=np.zeros(self.number_of_chargers)

    for car in range(self.number_of_chargers): 
        # in case action=[-100,100] P_charging[car] = actions[car]/100*max_charging_energy otherwise if action=[-1,1] P_charging[car] = 100*actions[car]/100*max_charging_energy

        if chargers[car] == 1:
            max_charging_energy = min([2, res_charging_demand[car]]) 
            P_charging[car] = actions[car]*max_charging_energy 
            res_charging_demand[car] = res_charging_demand[car] - P_charging[car] 
            res_parking_time[car] = res_parking_time[car] - 1
        else:
            P_charging[car] = 0

    # Calculation of energy coming from Grid
    total_charging = sum(P_charging)

    drl_model_constant = actions[-1]
    revenue = 0
    for car in range(self.number_of_chargers):
        if res_charging_demand[car] > 0:
            charging_price = predict_price(self, car, drl_model_constant, res_charging_demand[car], res_parking_time[car])
        else:
            charging_price = 0
        revenue += P_charging[car] * charging_price

    # First cost index
    #energy_from_grid = max([total_charging, 0])
    energy_from_grid = total_charging
    reward_1 = revenue - energy_from_grid*self.grid_price[0] 

    # Penalty for exceeding the maximum demand of the station
    if total_charging > self.maximum_grid_demand:
        cost_2 = (total_charging - self.maximum_grid_demand)*1000
    else:
        cost_2 = 0

    # Penalty of not fully charging the cars that leave
    cost_EV =[]
    for ii in range(len(leave)):
        cost_EV.append((ii*2)*100)
    cost_3 = sum(cost_EV)

    # Charging quickly
    charging_cost = []
    for ii in range(len(res_time)):
        charging_cost.append((ii*2)*100)
    reward_4 = sum(charging_cost)

    reward = reward_1 + reward_4 - (cost_2 + cost_3)

    return reward, reward_1, reward_4, cost_2, cost_3, energy_from_grid, res_charging_demand, res_parking_time