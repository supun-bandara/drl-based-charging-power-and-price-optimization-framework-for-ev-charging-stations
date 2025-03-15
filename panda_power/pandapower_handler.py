from panda_power.panda import Pandapower

class PandapowerHandler:
    def __init__(self, current_time):
        self.grid_power = Pandapower(current_time)
        self.current_time = current_time

    def calculate_max_grid_demand(self):
        self.grid_power.time_loads_uniform(self.current_time)
        self.grid_power.uniform_load()  
        self.grid_power.run_calculation()
        
        max_grid_demand = float(self.grid_power.maximum_power(78)) * 1000  # Convert to kilowatts
        return max_grid_demand

    def set_charging_station_power(self, total_power):
        self.grid_power.charging_station_power(78, total_power, 2)
        self.grid_power.run_calculation()
        self.grid_power.open_network()
