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
    def calc_amount_to_be_sold(self, exchange_details = (1,  # {-1,0,1}
                                                        (0,2), # {0,1,2} one of them
                                                        1)
                                                             
                                                                
                                ): # --> makes {0 : 123} currency and amount which sholu be taken from AgentsAccount for exchanging. where 0 - is idx of ['USD', 'EUR', 'RUB'] 
        if exchange_details[0] != 1:
            print('INCORRECT TYPE OF ACTION FOR THIS FUNCTION')
        source_curr_idx = exchange_details[1][0]
        target_curr_idx = exchange_details[1][1]
        amount_to_buy = exchange_details[2]
        actual_course_mtx = self.market.get_course_mtx()
        
        amount_to_sell = actual_course_mtx[target_curr_idx, source_curr_idx] * amount_to_buy
        
        return {source_curr_idx : amount_to_sell}
        
    def get_total_wealth(self, agents_wallet_idx):
        total_wealth = 0
        actual_course_mtx = self.market.get_course_mtx()
        
        USD_idx = self.currency_names.index("USD")
        for i_key, i_val in agents_wallet_idx.items():
            if self.currency_names[i_key] == "USD":
                total_wealth += i_val
            else:
                total_wealth += i_val * actual_course_mtx[i_key, USD_idx]
        
        return total_wealth
        
        
                     


class Market: # params relatively to USD only
    def __init__(self, num_of_currencies = 3, currency_names = ['USD', 'EUR', 'RUB']):
        self.currency_names = currency_names         
    	
    	#####
    	#Currency params
        self.USD_EUR = 0.0
        self.USD_RUB = 0.0
        
        self.USD_EUR_volatility = 0.0
        self.USD_RUB_volatility = 0.0
        #####
        self.USD_EUR_limit_l = 0.0
        self.USD_EUR_limit_h = 0.0
        
        self.USD_RUB_limit_l = 0.0
        self.USD_RUB_limit_h = 0.0
        
        self.currency_mtx = None

    
    def reset(self):
        #####
    	#Currency params
        self.USD_EUR = 0.92
        self.USD_RUB = 85.74
        
        self.USD_EUR_volatility = 0.1
        self.USD_RUB_volatility = 5
        
        self.USD_EUR_limit_l = 0.5
        self.USD_EUR_limit_h = 1.5
        
        self.USD_RUB_limit_l = 50
        self.USD_RUB_limit_h = 150
        
        #####
        self.currency_mtx = np.eye(len(self.currency_names), dtype=float)
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
        USD_EUR_new_val = self.USD_EUR + random.uniform(- self.USD_EUR_volatility,+ self.USD_EUR_volatility)
        if self.USD_EUR_limit_l < USD_EUR_new_val < self.USD_EUR_limit_h:
            self.USD_EUR = USD_EUR_new_val
        
        
        USD_RUB_new_val = self.USD_RUB + random.uniform(- self.USD_RUB_volatility,+ self.USD_RUB_volatility)
        if USD_RUB_new_val > self.USD_RUB_limit_l and USD_RUB_new_val < self.USD_RUB_limit_h:
            self.USD_RUB = USD_RUB_new_val
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
            
    def get_wallet_state(self): # in currency_name mode, like "USD", "RUB"
        return self.personal_currencies
        
    def get_wallet_state_idx(self):
        return dict(zip([self.currency_names.index(i_key) for i_key in self.personal_currencies.keys()], self.personal_currencies.values()))
       
    
                    
            
    





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
        #self.action_space = spaces.Dict(
        #                            {"Action_type" : spaces.Discrete(3, start = -1),  # {-1,0,1}
        #                            "Exchangable_currency" : spaces.Dict(
        #                                                            {"Source_curr" : spaces.Discrete(self.num_of_currencies),  # {0,1,2} one of them
        #                                                            "Target_curr" : spaces.Discrete(self.num_of_currencies)
        #                                                            }
        #                                                        ),
        #                            "Amount": spaces.Discrete(self.buy_amount_max, start = 1)                                    
        #                            
        #                            }                
        #)
        self.action_space = spaces.Tuple((
                                    spaces.Discrete(3, start = -1),                                                                     # {-1,0,1} Action_type
                                    spaces.Tuple((spaces.Discrete(self.num_of_currencies),  spaces.Discrete(self.num_of_currencies) )), # {0,1,2}  one of them
                                    spaces.Discrete(999, start = 1)                                                                     #  Amount 1- 999
                                    ))                                
                                    
                                                    
        
        

        """
        The following dictionary maps abstract actions from `self.action_space` to
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        #self._action_to_direction = {
        #    0: np.array([1, 0]),
        #    1: np.array([0, 1]),
        #    2: np.array([-1, 0]),
        #    3: np.array([0, -1]),
        #}

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
        idx_wallet = self.agents_account.get_wallet_state_idx() # Get current wallet state in idx format of keys
        total_wealth = self.broker.get_total_wealth(idx_wallet) # Calculate total wealth of Agents wallet. All converted to "USD"
        return {"Wallet": self.agents_account.get_wallet_state(), "Market": self.market.get_course_mtx(), "Total_wealth" : total_wealth}

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        
        # RESET MARKET TO INITIAL STATE
        self.market.reset()
        # RESET AGENT'S WALLET
        self.agents_account.reset()

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
    #{"Action_type" : 0,  # {-1,0,1}
    # "Exchangable_currency" : {"Source_curr" : 0,  # {0,1,2} one of them
    #                             "Target_curr" : 0
    #                          },                                                               
    # "Amount": 1 
    #}  
        
        # Parse Action   
        action_type = action[0]
        source_curr_idx = int(action[1][0])
        target_curr_idx = int(action[1][1])
        amount = action[2]
        
        penalty = 0
        deal_completed_reward = 0
        while True:
            
            # CHECKING IF CHANGING THE DIFFERENT CURRENCIES
            if source_curr_idx == target_curr_idx:
                penalty = -100 # penalty because agent trying to exchange the same currencies. change to global par
                break        
        
            # FIGURING OUT HOW MANY SOURCE CURRENCY TO BE PROCESSED
            if action_type == 1: # 1 = buy certain amount. I don't know how many source_currency to sell.        
                curr_amount_to_sell = self.broker.calc_amount_to_be_sold(exchange_details = action) # calculate how many source_currency to sell : {0 : 100}
            elif action_type == -1:
                curr_amount_to_sell = {source_curr_idx : amount}
            else:
                penalty = -10 # penalty because agent prefer to don't do anything. change to global par
                break # no sense to proceed due to Action_type is 0.          
                        
            # CHECK IF I HAVE ENOUGH TO SELL
            reserved_amount_to_sell = self.agents_account.reserve_curr_for_broker(currency_amount = curr_amount_to_sell)
            if reserved_amount_to_sell is None:
                penalty = -100 # penalty because agent trying to sell more than he has in his wallet. change to global par
                break # no sense to proceed due to incorrect amount to sell.
                
            # EXCHANGE SOURCE CURRENCY TO TARGET CURRENCY WITH BROKER
            exchanged_amount_by_broker = self.broker.exchange(to_sell = reserved_amount_to_sell, target_curr_idx = target_curr_idx)
            
            # ADOPT EXCHANGED CURRENCY INTO AGENTS WALLET
            self.agents_account.adopt_curr_from_broker(exchanged_amount_by_broker)
            
            # IF YOU REACHED THIS POINT, THEN EXCHANGE HAS BEEN SUCCESSFULL. AND WE NEED TO GET OUT FROM while loop anyway.
            deal_completed_reward += 100
            break
        
        
        # GET INFO. INFO CONTAINS total_wealth METRICS.
        info = self._get_info() # {"Wallet": self.agents_account.get_wallet_state(), "Market": self.market.get_course_mtx(), "Total_wealth" : total_wealth}
        
        # CALCULATE REWARD
        #idx_wallet = self.agents_account.get_wallet_state_idx() # Get current wallet state in idx format of keys
        #total_wealth = self.broker.get_total_wealth(idx_wallet) # Calculate total wealth of Agents wallet. All converted to "USD"
        total_wealth = info["Total_wealth"]
        reward = total_wealth + penalty + deal_completed_reward # TOTAL STEP REWARD
        
        terminated = False
        truncated = False
                
        # UPDATE MARKET STATE TO GET NEW OBSERVATION
        self.market.update_cources()        
        observation = self._get_obs()
        
        return observation, reward, terminated, truncated, info 
        

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
