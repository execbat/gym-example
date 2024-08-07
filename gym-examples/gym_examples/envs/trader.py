import numpy as np
import pygame

import gymnasium as gym
from gymnasium import spaces
from gymnasium.spaces import Dict, Box, Discrete
import random




class Market: # params relatively to USD only
    def __init__(self, num_of_currencies = 3, currency_names = ['USD', 'EUR', 'RUB']):
        self.currency_names = currency_names         
    	
    	#####
    	#Currency params
        self.USD_EUR = 0.92
        self.USD_RUB = 85.74
        
        self.USD_EUR_volatility = 0.1
        self.USD_RUB_volatility = 5
        #####
        self.currency_mtx = np.eye(len(currency_names))
        self.update_course_mtx()  
            
    def update_course_mtx(self):
        # make 1st row
        # self.currency_mtx = np.eye(len(currency_names))
        self.currency_mtx[0 , 1] = self.USD_EUR
        self.currency_mtx[0 , 2] = self.USD_RUB
        # make other rows
        for row in range(self.currency_mtx.shape[0]):    
            for col in range (self.currency_mtx.shape[1]):
                if row == 0 and row != col:
                    self.currency_mtx[col, row] = 1.0 / self.currency_mtx[row, col]
        
                if row != 0 and col != 0 and row != col:
                    self.currency_mtx[row, col] = self.currency_mtx[row, 0] * self.currency_mtx[0, col]
                    
    def update_cources(self):
        #print('old course', self.USD_EUR)
        self.USD_EUR += random.uniform(- self.USD_EUR_volatility,+ self.USD_EUR_volatility)
        self.USD_RUB += random.uniform(- self.USD_RUB_volatility,+ self.USD_RUB_volatility)
        print('new course', self.USD_EUR)
        #self.update_course_mtx()
        
    def get_course_mtx(self):
        return self.currency_mtx
        
        
        
        



class AgentsAccount:
    def __init__(self, num_of_currencies = 3, start_amount = 1000.0, currency_names = ['USD', 'EUR', 'RUB']):
        self.num_of_currencies = num_of_currencies
        self.start_amount = start_amount 
        
        self.currency_names = currency_names
        self.personal_currencies = dict()
        self.reset() # reset all repsonal currencies        
        
    def reset(self):    
        for i in range(self.num_of_currencies):
            if i == 0:
                self.personal_currencies[self.currency_names[i]] = self.start_amount
            else:
                self.personal_currencies[self.currency_names[i]] = 0.0 
                
    def get_total_wealth(self):
        pass
        
    def reserve_curr_for_broker(self, currency_amount): # take some amount of certain currency -> {'USD' : 110}
        # currency_amount = {currency_name : amount} for example {"USD" : 100}
        for currency, amount in currency_amount.items():
            if self.personal_currencies[currency] >= amount: # only if we have it
                self.personal_currencies[currency] -= amount
                return {currency : amount}
            
        return None    
        
    def adopt_curr_from_broker(self, curr_amount):  # curr_amount = {'USD' : 110}
        for currency, amount in curr_amount.items():
            self.personal_currencies[currency] += amount
            
    def get_wallet_state(self):
        return self.personal_currencies
    
                    
            
    





#class GridWorldEnv(gym.Env):
class TraderEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=5, num_of_currencies = 3, buy_amount_max = 999):
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window
        
        self.num_of_currencies = num_of_currencies # total amount of currencies
        self.buy_amount_max = buy_amount_max # how many on target currency to buy
        
        
        self.currency_names = ['USD', 'EUR', 'RUB']
        #CREATING OBJECTS
        self.agents_account = AgentsAccount(num_of_currencies = 3, start_amount = 1000.0, currency_names = self.currency_names) # create personal agent's wallet with several currencies
        self.market = Market(num_of_currencies = 3, currency_names = self.currency_names)
        
        
        
        

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "target": spaces.Box(0, size - 1, shape=(2,), dtype=int),
            }
        )

        # We have 2 actions, 0 - do nothing, 1 - exchange
        #self.action_space = spaces.Discrete(4)
        self.action_space = Dict(
                                    {"Action_type" : Discrete(2),  # {0,1}
                                    "Exchangable_currency" : Dict(
                                                                    {"Source_curr" : Discrete(self.num_of_currencies),  # {0,1,2} one of them
                                                                    "Target_curr" : Discrete(self.num_of_currencies)
                                                                    }
                                                                ),
                                    "Amount_to_buy": Discrete(self.buy_amount_max, start = 1)                                    
                                    
                                    }                
        )
        

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
    	return {"agent": self._agent_location, "target": self._target_location}

    def _get_info(self):
    	return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
                    )
                }

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the agent's location uniformly at random
        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        # We will sample the target's location randomly until it does not coincide with the agent's location
        self._target_location = self._agent_location
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = self.np_random.integers(
                0, self.size, size=2, dtype=int
            )

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
    #Format of action:
    #OrderedDict([('Action_type', 0),
    #         ('Amount_to_buy', 899),
    #         ('Exchangable_currency',
    #          OrderedDict([('Source_curr', 1), ('Target_curr', 1)]))])
    
    
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self._action_to_direction[action]
        # We use `np.clip` to make sure we don't leave the grid
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )
        # An episode is done iff the agent has reached the target
        terminated = np.array_equal(self._agent_location, self._target_location)
        reward = 1 if terminated else 0  # Binary sparse rewards
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # First we draw the target
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * self._target_location,
                (pix_square_size, pix_square_size),
            ),
        )
        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
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
