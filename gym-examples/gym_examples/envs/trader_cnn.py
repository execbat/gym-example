import numpy as np
import pygame

import gymnasium as gym
from gymnasium import spaces
from gymnasium.spaces.utils import flatten_space
from gym_examples.envs.currecnies import World_currencies_unique_list
from gym_examples.envs.broker_settings import Broker_commission
import random

np.set_printoptions(suppress=True,precision=5)

class Market_activ:
    def __init__(self, market_name = "EUR", init_value = 0.92, limit_l = 0.5, limit_h = 1.5, volatility = 1):
        self.market_name = market_name
        self.init_value = init_value #in respect to USD. 1 USD = some value of this currency
        self.limit_l = limit_l
        self.limit_h = limit_h
        self.volatility = volatility # window of volatility in %

        self.current_value = self.init_value

    def get_current_value(self):
        return self.current_value

    def put_new_current_value(self, value):
        self.current_value = value



class Broker:
    def __init__(self, market = None, agents_account = None):
        self.agents_account = agents_account
        self.market = market
        self.currency_names = self.market.get_actives_names()
        self.commission_count = 0.0

    def exchange(self, exchange_details = (1,0,2,10)): # --> {2 : 110} as output. that should be adopted by AgentAccount afterwards
        actual_course_mtx = self.market.get_course_mtx()
        action_type = int(exchange_details[0])
        source_curr_idx = int(exchange_details[1])
        target_curr_idx = int(exchange_details[2])
        amount = exchange_details[3]
        
        # Check wether source and target actives are share or currency
        source_is_share = not self.currency_names[source_curr_idx] in World_currencies_unique_list
        target_is_share = not self.currency_names[target_curr_idx] in World_currencies_unique_list
        # We need to know if there is a share exchange or currency exchange to calculate commission appropriately 
        if source_is_share == True or target_is_share == True:
            #print('operation with shares')
            operation_with_shares = True
        else:
            #print('operation with currencies')
            operation_with_shares = False      
        
        if action_type == 1: # buy. We know how many to buy, need to calculate how many to be sold.
            amount_to_sell = actual_course_mtx[int(target_curr_idx), int(source_curr_idx)] * amount # this amount doesn't include the commission
            #print('amount_to_sell ', amount_to_sell)
            commission = self.calc_commission(value = amount_to_sell, operation_with_shares = operation_with_shares)    
            self.commission_count += commission * actual_course_mtx[int(source_curr_idx), 0] # count commission in "USD" only
            #print('commission ', commission)        
            taken_amount_from_wallet = self.agents_account.reserve_curr_for_broker({source_curr_idx : amount_to_sell + commission})
            if taken_amount_from_wallet is None:
                return False # CODE OF NOT ENOUGH TO SELL
            self.agents_account.adopt_curr_from_broker({target_curr_idx : amount})
            
        elif action_type == 2: # sell. We know how many to sell, need to calculate how many to be bought
            amount_to_be_bought = actual_course_mtx[source_curr_idx, target_curr_idx] * amount
            #print('amount_to_be_bought ', amount_to_be_bought)
            commission = self.calc_commission(value = amount_to_be_bought, operation_with_shares = operation_with_shares)  
            self.commission_count += commission * actual_course_mtx[int(target_curr_idx), 0] # count commission in "USD" only
            #print('commission ', commission)        
            taken_amount_from_wallet = self.agents_account.reserve_curr_for_broker({source_curr_idx : amount})
            if taken_amount_from_wallet is None:
                return False# CODE OF NOT ENOUGH TO SELL
            self.agents_account.adopt_curr_from_broker({target_curr_idx : amount_to_be_bought - commission})
            
        else:
            print('INCORRECT action_type for this method') 
            return False
            
        return True # in exchange has been completed successfully
        
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
        
    def calc_commission(self, value = 0, operation_with_shares = False): 
        if operation_with_shares:
            commission_value = Broker_commission["Share_exchange"]
        else:
            commission_value = Broker_commission["Currency_exchange"] 
            
        return  value * commission_value 
        
    def get_accrued_commission(self):
        return self.commission_count
        
    def reset(self):
        self.commission_count = 0.0
    
        
        
                     


class Market: # params relatively to USD only
    def __init__(self, list_of_market_actives = ['USD']):
        self.actives_names = None    
        self.list_of_market_actives = list_of_market_actives
        self.create_list_of_actives_names() # build the list of the names of used Market_actives with "USD" on the 1st place
        self.num_of_currencies = len(self.list_of_market_actives)

        self.course_mtx = None    
        self.reset()   
        
    def create_list_of_actives_names(self):
        self.actives_names = [active.market_name for active in self.list_of_market_actives]
        if "USD" in self.actives_names:
            self.actives_names.insert(0, self.actives_names.pop(self.actives_names.index("USD"))) # Move "USD" to 0 place
        else:
            self.actives_names.insert(0, "USD")
        
    
    def reset(self):
        self.course_mtx = np.eye(len(self.actives_names), dtype=float)
        self.update_course_mtx()  
    
            
    def update_course_mtx(self):
        # make 1st row
        for i_num, i_active in enumerate(self.list_of_market_actives):
            self.course_mtx[0, i_num + 1] = i_active.get_current_value()
        # make other rows
        for row in range(self.course_mtx.shape[0]):    
            for col in range (self.course_mtx.shape[1]):
                if row == 0 and row != col:
                    self.course_mtx[col, row] = 1.0 / self.course_mtx[row, col]
        
                if row != 0 and col != 0 and row != col:
                    self.course_mtx[row, col] = self.course_mtx[row, 0] * self.course_mtx[0, col]
                    
    def update_cources(self): # used in every new STEP
        #print('old course', self.USD_EUR)
        for i_active in self.list_of_market_actives:
            current_value = i_active.get_current_value()
            volatility_limit = current_value * (i_active.volatility / 100.0) / 2.0
            new_val = i_active.get_current_value() + random.uniform(- volatility_limit,+ volatility_limit)
            if i_active.limit_l < new_val < i_active.limit_h:
                i_active.put_new_current_value(new_val)
        
        self.update_course_mtx()
        
    def get_course_mtx(self):
        return self.course_mtx

    def get_actives_names(self):
        return self.actives_names


class AgentsAccount:
    def __init__(self, start_amount = 30.0, currency_names = ['USD', 'EUR', 'RUB']):
    
        self.currency_names = currency_names
        self.num_of_currencies = len(self.currency_names)
        self.start_amount = start_amount 
        self.maximal_total_wealth_ever = 0
        
        
        self.personal_currencies = dict()
        self.reset() # reset all repsonal currencies        
        
    def reset(self):    
        for i in range(self.num_of_currencies):
            if i == 0:
                self.personal_currencies[self.currency_names[i]] = self.start_amount
            else:
                self.personal_currencies[self.currency_names[i]] = self.start_amount
                
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
        
    def put_new_maximal_total_wealth_ever(self, value): # updates maximal_total_wealth_ever and returns difference. shows how good was the action
        res = value - self.maximal_total_wealth_ever
        if value > self.maximal_total_wealth_ever:
            self.maximal_total_wealth_ever  = value
            
        return res
            
    def get_new_maximal_total_wealth_ever(self):
        return self.maximal_total_wealth_ever   
    
                    
            
    





#class GridWorldEnv(gym.Env):
class TraderEnvCnn(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=5, buy_amount_max = 100):
        super().__init__()
        #self.current_step = 0
        #self.max_episode_steps = max_episode_steps
        
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window 
        self.buy_amount_max = buy_amount_max # how many on target currency to buy        
        

        # CREATING OBJECTS
        # Actives i.e. shares or currencies
        list_of_market_actives = [
                Market_activ(market_name = "EUR", init_value = 0.91, limit_l = 0.5, limit_h = 1.5, volatility = 1),
                Market_activ(market_name = "RUB", init_value = 88.50, limit_l = 50, limit_h = 150, volatility = 7),
                Market_activ(market_name = "AEROFLOT", init_value = 148.85, limit_l = 100, limit_h = 200, volatility = 5)
                ]
        
        self.market = Market(list_of_market_actives = list_of_market_actives)
        self.currency_names = self.market.get_actives_names()        
        
            
        self.agents_account = AgentsAccount( start_amount = 100.0, currency_names = self.currency_names) # create personal agent's wallet with several currencies        
        self.broker = Broker(market = self.market, agents_account = self.agents_account)
        
        # SPACES
        #self.observation_space = flatten_space(spaces.Box(0.0, 9999.0, shape=(len(self.currency_names) + 1,len(self.currency_names)), dtype=float))
        self.observation_space = spaces.Box(0.0, 9999.0, shape=(len(self.currency_names) + 1,len(self.currency_names)), dtype=float)
        self.action_space = spaces.Tuple((
                                    spaces.Discrete(3, seed=42),                                                                     # {0,1,2} Action_type: 0 - nothing, 1 - buy, 2 -sell
                                    spaces.Discrete(len(self.currency_names), seed=42),  # Source_curr {0,1,2}  one of them
                                    spaces.Discrete(len(self.currency_names), seed=42) , # Target_curr {0,1,2}  one of them
                                    spaces.Discrete(self.buy_amount_max, start = 1, seed=42)                                                                     #  Amount 1- 999
                                    ))                             
                                    
                                                    
 

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
        
        
        #observation = np.vstack((wallet_np_arr, course_mtx), dtype=float).flatten()
        observation = np.vstack((wallet_np_arr, course_mtx), dtype=float)
        return observation

    def _get_info(self):
        idx_wallet = self.agents_account.get_wallet_state_idx() # Get current wallet state in idx format of keys
        total_wealth = self.broker.get_total_wealth(idx_wallet) # Calculate total wealth of Agents wallet. All converted to "USD"
        
        wallet_state = self.agents_account.get_wallet_state()
        market_state = self.market.get_course_mtx()
        brokers_accrued_commission = self.broker.get_accrued_commission()
        maximal_wallet_wealth_ever = self.agents_account.get_new_maximal_total_wealth_ever()
        return {"Wallet": wallet_state, 
        "Market": market_state, 
        "Total_wealth" : total_wealth, 
        "Broker_overall_commission" : brokers_accrued_commission, 
        "Maximum_wallet_wealth" : maximal_wallet_wealth_ever}

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        #self.current_step = 0 #experimental
        
        
        
        # RESET MARKET TO INITIAL STATE
        #self.market.reset()
        # RESET AGENT'S WALLET
        #self.agents_account.reset()        
        # RESET BROKERS ACCRUED COMMISSION
        #self.broker.reset()

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        #self.current_step += 1 #experimental
    

        action_type = int(action[0])
        source_curr_idx = int(action[1])
        target_curr_idx = int(action[2])
        amount = action[3]
        
        penalty = 0
        deal_completed_reward = 0
        forced_to_learn_reward = 0
        total_wealth_increased_reward = 0
        
        # Check wether source and target actives are share or currency
        source_is_share = not self.currency_names[source_curr_idx] in World_currencies_unique_list
        target_is_share = not self.currency_names[target_curr_idx] in World_currencies_unique_list
        
        while True:
            
            # CHECKING IF CHANGING THE DIFFERENT CURRENCIES
            if source_curr_idx == target_curr_idx:
                #print('Error. exchanging the same actives')
                penalty = -100 # penalty because agent trying to exchange the same currencies. change to global par
                break        
        
            # Check if it's legal to sell/buy currencies/shares
            if action_type == 1: # 1 = buy certain amount. I don't know how many source_currency to sell.  
                if source_is_share:
                    #print('Error. Incorrect exchanging types. Source active cant be a Share')
                    penalty = -100 # what is going to be sold Must not to be a Share. Currency only 
                    break    
                
            elif action_type == 2: # 2 = sell certain amount. I don't know how many tarcet_currency to buy.
                if target_is_share:
                    #print('Error. Incorrect exchanging types. Target active cant be a Share')
                    penalty = -100 # what is going to be bought Must not to be a Share. Currency only 
                    break
                
            else:
                penalty = 0 # penalty because agent prefer to don't do anything. change to global par
                break # no sense to proceed due to Action_type is 0.   
                
                        
            # CHECK IF I HAVE ENOUGH TO SELL
            if not self.broker.exchange(action): # try to exchange
                #print('Error. Trying to sell more than have in the wallet')
                penalty = -100 # have not enought amount of source active to sell
                break  
               
            # IF YOU REACHED THIS POINT, THEN EXCHANGE HAS BEEN SUCCESSFULL. AND WE NEED TO GET OUT FROM while loop anyway.
            #print('Good. Exchange has been completed successfully!')
            deal_completed_reward = 0 #10
            break
        
        
        # GET INFO. INFO CONTAINS total_wealth METRICS.
        info = self._get_info() # {"Wallet": self.agents_account.get_wallet_state(), "Market": self.market.get_course_mtx(), "Total_wealth" : total_wealth}
        
        # CALCULATE REWARD
        total_wealth = info["Total_wealth"] # Calculate total wealth of Agents wallet. All converted to "USD"
        
        # CHECK IF TOTAL WEALTH HAS BEEn INCREASED BECAUSE OF AGENT ACTIONS. i.e. action_type == 1 or 2, not 0    
        if ((action_type != 0) and (penalty == 0)):
            difference = self.agents_account.put_new_maximal_total_wealth_ever(total_wealth)
            if difference > 0:
                total_wealth_increased_reward = difference * 100.0
            
            

        
        #reward = total_wealth + total_wealth_increased_reward + penalty + deal_completed_reward + forced_to_learn_reward # TOTAL STEP REWARD
        #reward =  total_wealth_increased_reward + penalty + deal_completed_reward + forced_to_learn_reward # TOTAL STEP REWARD
        reward =  total_wealth * 0.01  + penalty + total_wealth_increased_reward
        
        terminated = False
        truncated = False # self.current_step >= self.max_episode_steps # experimental
                
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
