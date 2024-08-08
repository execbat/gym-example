import numpy as np
import pygame

import gymnasium as gym
from gymnasium import spaces
#from gymnasium.spaces import Dict, Box, Discrete
import random

np.set_printoptions(suppress=True,precision=5)


class Broker:
    def __init__(self, market = None, currency_names = ['USD', 'EUR', 'RUB'] 
                ):
        self.market = market
        self.currency_names = currency_names
        #self.received_to_exchange = {"USD" : 0}
        #self.exchanged_to_give_back = {"USD" : 0}
        
    def exchange(self, to_sell = {0 : 0}, target_curr_idx = 2 ): # --> {2 : 110} as output. that should be adopted by AgentAccount
        actual_course_mtx = self.market.get_course_mtx()
        amount_to_sell = np.fromiter(to_sell.values(), dtype=float)[0]
        source_curr_idx = int(np.fromiter(to_sell.keys(), dtype=float)[0])
        return {target_curr_idx : actual_course_mtx[source_curr_idx, target_curr_idx] * amount_to_sell}
                  
    #  calc_amount_to_be_sold used with Action == 1. When we know how many to buy, but don't know how many to sell.
    #  this function calculates what amount of Source currency you have to sell to buy Known amount of Target currency. 
    def calc_amount_to_be_sold(self, exchange_details = {"Action_type" : 0,  # {-1,0,1}
                                                        "Exchangable_currency" : {"Source_curr" : 0,  # {0,1,2} one of them
                                                                                "Target_curr" : 0
                                                                                },                                                               
                                                        "Amount": 1 
                                                        }                                                             
                                                                
                                ): # --> makes {0 : 123} currency and amount which sholu be taken from AgentsAccount for exchanging. where 0 - is idx of ['USD', 'EUR', 'RUB'] 
        if exchange_details["Action_type"] != 1:
            print('INCORRECT TYPE OF ACTION FOR THIS FUNCTION')
        source_curr_idx = exchange_details["Exchangable_currency"]["Source_curr"]
        target_curr_idx = exchange_details["Exchangable_currency"]["Target_curr"]
        amount_to_buy = exchange_details["Amount"]
        actual_course_mtx = self.market.get_course_mtx()
        
        amount_to_sell = actual_course_mtx[target_curr_idx, source_curr_idx] * amount_to_buy
        
        return {source_curr_idx : amount_to_sell}
        
        
                     


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
        self.currency_mtx = np.eye(len(currency_names), dtype=float)
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
                    
    def update_cources(self): # used in every new STEP
        #print('old course', self.USD_EUR)
        self.USD_EUR += random.uniform(- self.USD_EUR_volatility,+ self.USD_EUR_volatility)
        self.USD_RUB += random.uniform(- self.USD_RUB_volatility,+ self.USD_RUB_volatility)
        #print('new course', self.USD_EUR)
        self.update_course_mtx()
        
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
        
    def reserve_curr_for_broker(self, currency_amount): # take some amount of certain currency -> {0: 110}
        # currency_amount = {currency_name_idx : amount} for example {0 : 100}, where 0 in idx of ['USD', 'EUR', 'RUB']
        amount = np.fromiter(currency_amount.values(), dtype=float)[0]
        curr_idx = int(np.fromiter(currency_amount.keys(), dtype=float)[0])
        curr_name = self.currency_names[curr_idx]
        
        if self.personal_currencies[curr_name] >= amount: # only if we have it
            self.personal_currencies[curr_name] -= amount
            return {curr_idx : amount}
            
        return None    
        
    def adopt_curr_from_broker(self, currency_amount):  # curr_amount = {0 : 110}, where 0 in idx of ['USD', 'EUR', 'RUB']
        amount = np.fromiter(currency_amount.values(), dtype=float)[0]
        curr_idx = int(np.fromiter(currency_amount.keys(), dtype=float)[0])
        curr_name = self.currency_names[curr_idx]           
        self.personal_currencies[curr_name] += amount
            
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
        self.broker = Broker(market = self.market, currency_names = self.currency_names)
        
        
        
        #self.observation_space = spaces.Dict(
        #    {
        #        #"Wallet": spaces.Box(0.0, 9999.0, shape=(1,self.num_of_currencies), dtype=float),
        #        #"Market": spaces.Box(0.0, 9999.0, shape=(self.num_of_currencies,self.num_of_currencies), dtype=float),
        #        
        #    }
        #)
        self.observation_space = spaces.Box(0.0, 9999.0, shape=(self.num_of_currencies + 1,self.num_of_currencies), dtype=float)

        # We have 2 actions, 0 - do nothing, 1 - exchange
        #self.action_space = spaces.Discrete(4)
        self.action_space = spaces.Dict(
                                    {"Action_type" : spaces.Discrete(3, start = -1),  # {-1,0,1}
                                    "Exchangable_currency" : spaces.Dict(
                                                                    {"Source_curr" : spaces.Discrete(self.num_of_currencies),  # {0,1,2} one of them
                                                                    "Target_curr" : spaces.Discrete(self.num_of_currencies)
                                                                    }
                                                                ),
                                    "Amount": spaces.Discrete(self.buy_amount_max, start = 1)                                    
                                    
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
    	# convert wallet values from dict values into np array
        wallet_dict = self.agents_account.get_wallet_state()
        wallet_np_arr = np.fromiter(wallet_dict.values(), dtype=float)
        course_mtx = self.market.get_course_mtx()
        
        observation = np.vstack((wallet_np_arr, course_mtx), dtype=float)
        return observation

    def _get_info(self):
    	return {"Wallet": self.agents_account.get_wallet_state(), "Market": self.market.get_course_mtx()}

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
