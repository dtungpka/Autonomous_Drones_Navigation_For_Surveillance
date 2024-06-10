import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np


class DroneEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 15}

    def __init__(self, render_mode=None, size=20,drones=1,targets=1,obstacles=0,battery=100,seed=None,options=None):
        self.size = size  # The size of the square grid
        self.window_size = 1024  # The size of the PyGame window
        self.n_drones = drones
        self.n_targets = targets
        self.n_obstacles = obstacles
        self.max_battery = battery
        if seed is not None:
            self.seed(seed)
        self.options = options

        observation_space_dims = 2 + (7**2 + 2 + 1 + 1) * self.n_drones
        #inf
        observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(observation_space_dims,), dtype=int)
        # observation_space = {}
        # observation_space_drones = {}
        # for i in range(self.n_drones):
        #     observation_space_drones["drone_position_"+str(i)] = spaces.Box(0, size - 1, shape=(2,), dtype=int)
        #     observation_space_drones["drone_battery_"+str(i)] = spaces.Box(0, battery, shape=(1,), dtype=int)
        #     #drone elevation
        #     observation_space_drones["drone_elevation_"+str(i)] = spaces.Box(0, 2, shape=(1,), dtype=int) #view 0: 3x3, view 1: 5x5, view 2: 7x7
        #     observation_space_drones["drone_camera_"+str(i)] = spaces.Box(-1, targets, shape=(7,7), dtype=int)
        # observation_space["drones"] = spaces.Dict(observation_space_drones)
        
        # #Agent should not be able to see the target's location
        # observation_space_target = {}
        # for i in range(self.n_targets):
        #     observation_space_target["target_"+str(i)] = spaces.Box(2, size - 1, shape=(2,), dtype=int)
        # #observation_space["n_targets"] = spaces.Dict(observation_space_target)

        # #TBA
        # #for i in range(self.n_obstacles):
        #     #observation_space["obstacle_"+str(i)] = spaces.Box(1, size - 1, shape=(2,), dtype=int)
        
        # #Base station at 0,0
        # observation_space["base_station"] = spaces.Box(0, size - 1, shape=(2,), dtype=int)


        # # Observations are dictionaries with the agent's and the target's location.
        # # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        # self.observation_space = spaces.Dict(observation_space)
        

        # Actions are discrete values in {0,1,2,3}, where 0 corresponds to "right", 1 to "up" etc. 5,6 are elevation up and down
        #with number of drones
        self.action_space = spaces.MultiDiscrete([7]*self.n_drones)
        self.observation_space = observation_space
        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
            4: np.array([1, 1]), # up elevation
            5: np.array([-1, -1]), # down elevation
            6: np.array([0, 0]) #stay
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """

        self.window = None
        self.clock = None

    def _get_obs(self):
        #convert to a single array
        return self.to_array()
    def to_array(self):
        observation_json = {
            "base_station": self.base_station,
            "drones": self.drones,
            #"n_targets": self.n_targets,
            #"obstacles": self.obstacles, TBA
            
        }
        observation = np.array([])
        for key in observation_json:
            if key == "drones":
                for drone_key in observation_json[key]:
                    observation = np.concatenate((observation, observation_json[key][drone_key]),axis=None)
            else:
                observation = np.concatenate((observation, observation_json[key]),axis=None)

        return observation

        


    def _get_info(self):
        return {
            "base_station": self.base_station,
            "drones": self.drones,
            #"n_targets": self.n_targets,
            #"obstacles": self.obstacles, TBA
            
        }

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        self.drones = {}
        for i in range(self.n_drones):
            #always start at base station
            self.drones["drone_position_"+str(i)] = np.array([0,0])
            self.drones["drone_battery_"+str(i)] = np.array(self.max_battery)
            self.drones["drone_elevation_"+str(i)] = np.array(0)
            self.drones["drone_camera_"+str(i)] = np.zeros((7,7),dtype=int)
        
        self.targets = {}
        for i in range(self.n_targets):
            self.targets["target_"+str(i)] = self.np_random.integers(1, self.size-1, size=2, dtype=int)

        self.obstacles = {}
        for i in range(self.n_obstacles):
            self.obstacles["obstacle_"+str(i)] = self.np_random.integers(1, self.size-1, size=2, dtype=int)

        self.base_station = np.array([0,0])


        self.targets_found = []

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info
    def _actions_to_directions(self, actions):
        #from list of spaces.MultiDiscrete([6]*self.drones) to list of np.array([1, 0])
        return [self._action_to_direction[action] for action in actions]
    def _update_camera(self,drone:int):
        elevation_penalty = [0,0.5,0.75]
        
        position = self.drones["drone_position_"+str(drone)]
        elevation = self.drones["drone_elevation_"+str(drone)]
        camera = np.zeros((7,7),dtype=int)
        found_prob = 1 - elevation_penalty[elevation]
        top_left = position - np.array([elevation + 1, elevation + 1])
        bottom_right = position + np.array([elevation + 1, elevation + 1])

        #for each target, if in view, add to camera as (target_id + 1) in their position
        for i in range(self.n_targets):
            if np.all(top_left <= self.targets["target_"+str(i)]) and np.all(self.targets["target_"+str(i)] <= bottom_right):
                #if probability of finding target is less than 1, check if found
                if self.np_random.random() > found_prob:
                    continue
                else:
                    pos = self.targets["target_"+str(i)] - top_left
                    camera[pos[1],pos[0]] = i + 1
                    if i not in self.targets_found:
                        self.targets_found.append(i)
                    #Remember to reset target found each step
        #for each obstacle, if in view, add to camera as -1 in their position
        for i in range(self.n_obstacles):
            if np.all(top_left <= self.obstacles["obstacle_"+str(i)]) and np.all(self.obstacles["obstacle_"+str(i)] <= bottom_right):
                pos = self.obstacles["obstacle_"+str(i)] - top_left
                camera[pos[1],pos[0]] = -1
        self.drones["drone_camera_"+str(drone)] = camera
        return camera
    def _move_drone(self,drone:int,action:int):
        is_dead = False
        #Reward for moving drone
        reward = -0.01
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        #if action is 4,5, then change elevation
        if action == 4:
            self.drones["drone_elevation_"+str(drone)] += 1
        elif action == 5:
            self.drones["drone_elevation_"+str(drone)] -= 1
        elif action == 6:
            pass
        else:
            direction = self._action_to_direction[action]
            self.drones["drone_position_"+str(drone)] = np.clip(
                self.drones["drone_position_"+str(drone)] + direction, 0, self.size - 1
            )
        if not np.all(self.drones["drone_position_"+str(drone)] == self.base_station):
            self.drones["drone_battery_"+str(drone)] -= 1
        if self.drones["drone_battery_"+str(drone)] <= 0:
            is_dead = True
            #reward for dying
            reward = -1
        self.drones["drone_elevation_"+str(drone)] = np.clip(self.drones["drone_elevation_"+str(drone)],0,2)

        #if drone is at base station, recharge battery by 5
        if np.all(self.drones["drone_position_"+str(drone)] == self.base_station) and self.drones["drone_battery_"+str(drone)] < self.max_battery and not is_dead:
            self.drones["drone_battery_"+str(drone)] = np.clip(self.drones["drone_battery_"+str(drone)] + (self.max_battery//20),0,self.max_battery)
            #reward for recharging
            reward = .1
        return is_dead,reward
    def _is_target_waypoint_valid(self,waypoint:np.array):
        #check if waypoint is within grid and not on obstacle
        if np.all(waypoint >= 2) and np.all(waypoint < self.size):
            for i in range(self.n_obstacles):
                if np.all(waypoint == self.obstacles["obstacle_"+str(i)]):
                    return False
            return True
        return False
    def _move_target(self,target:int):
        # Randomly move the target, but:
        # 1. The target must stay within the grid
        # 2. The target must not move onto the obstacle
        # 3. The target can go in one direction with 0,1,2 steps (speed)

        # Randomly choose a direction to move in
        
        while True:
            new_position = self.targets["target_"+str(target)].copy()
            direction = self.np_random.integers(0,4)
            steps = self.np_random.integers(0,3)
            new_position += self._action_to_direction[direction] * steps
            if self._is_target_waypoint_valid(new_position):
                break
            #print(f"Invalid target waypoint {new_position}")
        self.targets["target_"+str(target)] = new_position



    def step(self, action):
        directions = self._actions_to_directions(action)
        self.targets_found = []
        total_reward = 0
        terminated = False
        for i in range(self.n_drones):
            is_dead,_reward = self._move_drone(i,action[i])
            self._update_camera(i)
            total_reward += _reward
            if is_dead:
                terminated = True
                break
        total_reward += len(self.targets_found)
        for i in range(self.n_targets):
            self._move_target(i)

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, total_reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels
         # add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=1,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=1,
            )
        # Draw base station with green hollow square
        pygame.draw.rect(
            canvas,
            (0, 255, 0),
            pygame.Rect(
                0,
                0,
                pix_square_size,
                pix_square_size,
            ),
            width=3,
        )

        # Draw targets with red circles
        for i in range(self.n_targets):
            #gray if not found, red if found
            clr = (255, 0, 0) if i in self.targets_found else (100, 100, 100)
        
            pygame.draw.circle(
                canvas,
                clr,
                (self.targets["target_"+str(i)] + 0.5) * pix_square_size,
                pix_square_size / 3,
            )
        
        # Draw obstacles with black squares
        for i in range(self.n_obstacles):
            pygame.draw.rect(
                canvas,
                (0, 0, 0),
                pygame.Rect(
                    self.obstacles["obstacle_"+str(i)] * pix_square_size,
                    (pix_square_size, pix_square_size),
                ),
            )
        
        # Draw drones with hollow blue circles with ID number in middle
        for i in range(self.n_drones):
            battery_percentage = self.drones["drone_battery_"+str(i)] / self.max_battery
            #gradually change color from blue to red as battery decreases
            clr = ( 255 * (1 - battery_percentage), 255 * battery_percentage,0)
            pygame.draw.circle(
                canvas,
                (0,100,255),
                (self.drones["drone_position_"+str(i)] + 0.5) * pix_square_size,
                pix_square_size / 3,
                width=3,
            )
            font = pygame.font.Font(None, 36)
            text = font.render(str(i), True, clr)
            text_rect = text.get_rect(center=(self.drones["drone_position_"+str(i)] + 0.5) * pix_square_size)
            canvas.blit(text, text_rect)

            elevation = self.drones["drone_elevation_"+str(i)]
            #ensure the top left and bottom right of the view is within grid
            top_left = self.drones["drone_position_"+str(i)] - np.array([elevation + 1, elevation + 1])
            bottom_right = self.drones["drone_position_"+str(i)] + np.array([elevation + 2, elevation + 2])
            top_left = np.clip(top_left,0,self.size-1)
            bottom_right = np.clip(bottom_right,0,self.size-1)
            
            _size = bottom_right - top_left
            pygame.draw.rect(
                canvas,
                (100, 100, 0),
                pygame.Rect(
                    top_left * pix_square_size,
                    _size * pix_square_size,
                ),
                width=2,
            )
       

                        
       

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
