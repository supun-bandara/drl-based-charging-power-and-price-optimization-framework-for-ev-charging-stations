# charging price prediction for drl model

import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import numpy as np

def predict_price(self, charger_id, drl_model_constant, res_parking_time, res_charging_demand):
    #print("predict_price - charger_id: ", charger_id, " current_time: ", current_time, " res_charging_demand: ", res_charging_demand, " res_parking_time: ", res_parking_time)
    
    service_fee = 5

    if charger_id in [0,1,2,3]:
        installation_cost = 30
    else:
        installation_cost = 2

    if res_parking_time == 0:
        res_parking_time = 1
    ev_constant = (res_charging_demand/res_parking_time)
    grid_price_constant = self.grid_price[0] * 1.1
    installation_cost = installation_cost

    charging_price = grid_price_constant + ev_constant + installation_cost + service_fee + drl_model_constant
    #print("predict_price - charging_price: ", charging_price, "grid_price_constant: ", grid_price_constant, "grid_price_forecast_constant: ", grid_price_forecast_constant, "ev_demand_forecast_constant: ", ev_demand_forecast_constant, "ev_constant: ", ev_constant, "installation_cost: ", installation_cost, "service_fee: ", service_fee)
    charging_price = charging_price/100
    return (charging_price * res_charging_demand)
