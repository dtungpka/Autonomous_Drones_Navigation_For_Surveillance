from gym.envs.registration import register

register(
    id="Drone-v0",
    entry_point="Autonomus_Drones_Navigation_For_Surveillance.envs:DroneEnv",
    max_episode_steps=1000,
    reward_threshold=1000.0,

)
