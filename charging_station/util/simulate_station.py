# contains states transition functions (add EVs, remove EVs)

import numpy as np

def simulate_station(self):
    #print('Simulate_Station')
    #print()
    chargers = self.chargers
    res_parking_time = self.res_parking_time
    res_charging_demand = self.res_charging_demand
    leave = []
    res_time = []
    new_evs = self.new_evs

    # add new EVs to the list of chargers
    for new_ev in new_evs:
        charger_index = np.where(chargers == 0)[0][0]
        chargers[charger_index] = 1
        res_parking_time[charger_index] = new_ev[0]
        res_charging_demand[charger_index] = new_ev[1]
    
    # remove EVs that are leaving
    for i in range(len(chargers)):
        if res_parking_time[i] == 0 and (chargers[i] != 0 or res_charging_demand[i] != 0):
            leave.append(res_charging_demand[i])
            res_charging_demand[i] = 0
            chargers[i] = 0
        if res_charging_demand[i] == 0 and (chargers[i] != 0 or res_parking_time[i] != 0): ################# remove this condition
            res_time.append(res_parking_time[i])
            res_parking_time[i] = 0
            chargers[i] = 0

    #print('Simulate_Station - leave', leave, 'res_parking_time', res_parking_time, 'res_charging_demand', res_charging_demand)
    #print()
    return leave, res_time, chargers, res_parking_time, res_charging_demand