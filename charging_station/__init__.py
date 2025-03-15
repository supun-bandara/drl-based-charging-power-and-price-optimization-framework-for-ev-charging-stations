# for creating the python package

from gymnasium import register #gym.envs.registration

register(
    id="charging_station/ChargingEnv-v0",
    entry_point="charging_station.environment:env",
)
