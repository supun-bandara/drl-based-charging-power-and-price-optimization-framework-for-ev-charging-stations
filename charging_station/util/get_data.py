# read data to train the drl model in each timestep

import pandas as pd

def get_data(self):
    #print('Data')
    #print()
    ID = self.timestep
    price = self.df_price.iloc[ID:ID + 13]['price'].values
    ev_forecast = self.df_ev_forecast.iloc[ID:ID + 10:4]['Total_Energy'].values

    new_evs = []

    if ID in self.df_station['Start_Time_Index'].values:
        rows = self.df_station.loc[self.df_station['Start_Time_Index'] == ID]
        for index, row in rows.iterrows():
            duration = row['Duration_Count']
            SOC = row['SOC_Level']
            Capacity = row['Battery_Capacity_kWh']
            energy_demand = (100-SOC)*Capacity/100
            new_evs.append([duration, energy_demand])
            
    return price, new_evs, ev_forecast