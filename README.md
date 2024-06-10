

# Autonomus Drone Navigation for Surveillance Environment

### Environments
This repository contains the implementation of Gym environment for the drone navigation in surveillance environment by [dtungpka](https://github.com/dtungpka)


>Drone-v0

### Installation
```bash
python3 -m pip install DroneNavigationGym-RL
```

### Usage
```python

import gymnasium as gym
import Autonomus_Drones_Navigation_For_Surveillance

env = gym.make('Drone-v0',drones=2,render_mode=None,size=20,targets=2,obstacles=2,battery=100)
obs, info = env.reset()
while True:
    action = env.action_space.sample()

    # Processing:
    obs, reward, terminated, _, info = env.step(action)
    total_reward += reward
    
    if terminated:
        break

env.close()

```

