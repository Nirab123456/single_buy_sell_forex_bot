import argparse
from typing import Optional, Tuple
import gym
from gym import Env
from gym.spaces import Box, MultiBinary, Discrete
import numpy as np
import os
from stable_baselines3 import PPO , SAC
from stable_baselines3.common.callbacks import BaseCallback
import gym
from gym import spaces
from gym.wrappers import FrameStack
from collections import deque
import pandas as pd
import matplotlib.pyplot as plt
import uuid
from torch.utils.tensorboard import SummaryWriter

# Initialize SummaryWriter


# i) can buy (to sell in upper price(buy)) and hold for a time period and then close,
# ii) can buy (to sell in lower price(sell)) and hold for a time period and then close,
# iii) observe the market and do nothing
# 1) buy_open 2) sell_open 3) Close 4) hold 5) Do nothing
class TradingEnv(gym.Env):
    def __init__(self, data):
        super(TradingEnv, self).__init__()
        self.data = data
        #WILL CHANGE max step later
        self.MAXIMUM_AMOUNT_OF_TIME_FOR_HOLDING = 100
        
        log_dir = "./logs"  # Change this to the desired log directory
        self.writer = SummaryWriter(log_dir)

        self.OWN_CURRENCY_AMOUNT = np.random.randint(10, 100)
        self.USED_CURRENCY_AMOUNT = np.random.randint(1000, 100000)
        self.USED_LEVERAGE = self.USED_CURRENCY_AMOUNT / self.OWN_CURRENCY_AMOUNT

        #loss tollaranace is the 3%  of own currency amount
        self.LOSS_TOLLARANCE = self.OWN_CURRENCY_AMOUNT * 0.03
        
        #accumulated loss tollarance is the 10%  of own currency amount
        self.ACCUMULATED_LOSS_TOLLARANCE = self.OWN_CURRENCY_AMOUNT * 0.1
        self.MINIMUM_GAINS = self.OWN_CURRENCY_AMOUNT * 0.05
        self.MAXIMUM_DOING_NOTHING_STEPS_TOLLARANCE = 10
        self.WRONG_STEPS_TOLLARANCE = 10
        self.log_interval = 30  # Log every log_interval episodes   
        self.taking_wrong_action_count = 0
        self.auto_terminated_trades = 0
        self.buy_open_uuids = {}
        self.sell_open_uuids = {}        
        self.previous_trade_details = {}
        self.previous_reward = 0
        self.net_gains = 0
        self.current_step = 0
        self.current_price = data['Close'][self.current_step]

        # Action space:
        # 0: buy
        # 1: sel
        # 2: buy close
        # 3: sell close
        # 4: hold
        # 5: do nothing
        self.action_space = spaces.MultiDiscrete(6)

        
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(12 ,), dtype=np.float32)

    def calculate_buy_profit_loss(self, opening_price, closing_price, position_size, leverage):
        # Calculate profit or loss based on opening and closing prices, position size, and leverage
        profit_loss = ((closing_price - opening_price) / opening_price) * position_size * leverage
        return profit_loss
    
    def calculate_sell_profit_loss(self, opening_price, closing_price, position_size, leverage):
        # Calculate profit or loss based on opening and closing prices, position size, and leverage
        profit_loss = ((opening_price - closing_price) / opening_price) * position_size * leverage
        return profit_loss

    def Terminate_lossing_buy_open_trades(self):
        trades_to_close = []
        for buy_open_uuid, trade_info in self.buy_open_uuids.items():
            # Use calculate_buy_profit_loss function to calculate the profit or loss
            if self.calculate_buy_profit_loss(trade_info['open_price'], self.current_price, self.OWN_CURRENCY_AMOUNT, self.USED_LEVERAGE) < self.LOSS_TOLLARANCE:
                trades_to_close.append(buy_open_uuid)

        for buy_open_uuid in trades_to_close:
            self.close_trade(buy_open_uuid, self.buy_open_uuids, 'buy_open')
            self.auto_terminated_trades += 1


    def Terminate_lossing_sell_open_trades(self):
        trades_to_close = []
        for sell_open_uuid, trade_info in self.sell_open_uuids.items():
            # Use calculate_sell_profit_loss function to calculate the profit or loss
            if self.calculate_sell_profit_loss(trade_info['open_price'], self.current_price, self.OWN_CURRENCY_AMOUNT, self.USED_LEVERAGE) < self.LOSS_TOLLARANCE:
                trades_to_close.append(sell_open_uuid)

        for sell_open_uuid in trades_to_close:
            self.close_trade(sell_open_uuid, self.sell_open_uuids, 'sell_open')
            self.auto_terminated_trades += 1
            
    def Terminate_after_maximum_step(self):
        if self.current_step > self.MAXIMUM_AMOUNT_OF_TIME_FOR_HOLDING:
            # Make a copy of the dictionary keys before iterating

            for buy_open_uuid in list(self.buy_open_uuids.keys()):
                trade_info = self.buy_open_uuids[buy_open_uuid]
                self.close_trade(buy_open_uuid, self.buy_open_uuids, 'buy_open')
                self.auto_terminated_trades += 1
                
            for sell_open_uuid in list(self.sell_open_uuids.keys()):
                trade_info = self.sell_open_uuids[sell_open_uuid]
                self.close_trade(sell_open_uuid, self.sell_open_uuids, 'sell_open')
                self.auto_terminated_trades += 1

    def calculate_current_profit_loss(self):
        # Calculate the current profit or loss
        current_profit_loss = 0
        for buy_open_uuid, trade_info in self.buy_open_uuids.items():
            current_profit_loss += self.calculate_buy_profit_loss(trade_info['open_price'], self.current_price, self.OWN_CURRENCY_AMOUNT, self.USED_LEVERAGE)
        for sell_open_uuid, trade_info in self.sell_open_uuids.items():
            current_profit_loss += self.calculate_sell_profit_loss(trade_info['open_price'], self.current_price, self.OWN_CURRENCY_AMOUNT, self.USED_LEVERAGE)
        return current_profit_loss

            

    def update_available_time_for_each_trade(self):
        for buy_open_uuid, trade_info in self.buy_open_uuids.items():
            trade_info['available_time'] -= 1
            if trade_info['available_time'] == 0:
                self.close_trade(buy_open_uuid, self.buy_open_uuids, 'buy_open')
        for sell_open_uuid, trade_info in self.sell_open_uuids.items():
            trade_info['available_time'] -= 1
            if trade_info['available_time'] == 0:
                self.close_trade(sell_open_uuid, self.sell_open_uuids, 'sell_open')




    def close_trade(self, trade_uuid, trade_dict, trade_type):
        trade_dict[trade_uuid]['close_price'] = self.current_price
        if trade_type == 'buy_open':
            self.net_gains += trade_dict[trade_uuid]['close_price'] - trade_dict[trade_uuid]['open_price']
            self.current_buy_open_trades -= 1
            #add to previous trade details
            self.previous_trade_details[trade_uuid] = trade_dict[trade_uuid]
            #fix currntt amount of money and leverage keep the used currency amount same
            self.OWN_CURRENCY_AMOUNT += self.calculate_buy_profit_loss(trade_dict[trade_uuid]['open_price'], trade_dict[trade_uuid]['close_price'], self.OWN_CURRENCY_AMOUNT, self.USED_LEVERAGE)
            self.USED_LEVERAGE = self.USED_CURRENCY_AMOUNT / self.OWN_CURRENCY_AMOUNT
            self.LOSS_TOLLARANCE = self.OWN_CURRENCY_AMOUNT * 0.03



        elif trade_type == 'sell_open':
            self.net_gains += trade_dict[trade_uuid]['open_price'] - trade_dict[trade_uuid]['close_price']
            self.current_sell_open_trades -= 1
            #add to previous trade details
            self.previous_trade_details[trade_uuid] = trade_dict[trade_uuid]
            #fix currntt amount of money and leverage keep the used currency amount same
            self.OWN_CURRENCY_AMOUNT += self.calculate_sell_profit_loss(trade_dict[trade_uuid]['open_price'], trade_dict[trade_uuid]['close_price'], self.OWN_CURRENCY_AMOUNT, self.USED_LEVERAGE)
            self.USED_LEVERAGE = self.USED_CURRENCY_AMOUNT / self.OWN_CURRENCY_AMOUNT
            self.LOSS_TOLLARANCE = self.OWN_CURRENCY_AMOUNT * 0.03


        del trade_dict[trade_uuid]

    def calculate_total_profit_loss(self):
        #use previous and current trade details to calculate total profit loss
        current_profit_loss = 0
        for buy_open_uuid, trade_info in self.buy_open_uuids.items():
            current_profit_loss += self.calculate_buy_profit_loss(trade_info['open_price'], self.current_price, self.OWN_CURRENCY_AMOUNT, self.USED_LEVERAGE)
        for sell_open_uuid, trade_info in self.sell_open_uuids.items():
            current_profit_loss += self.calculate_sell_profit_loss(trade_info['open_price'], self.current_price, self.OWN_CURRENCY_AMOUNT, self.USED_LEVERAGE)
        previous_profit_loss = 0
        for buy_open_uuid, trade_info in self.previous_trade_details.items():
            previous_profit_loss += self.calculate_buy_profit_loss(trade_info['open_price'], trade_info['close_price'], self.OWN_CURRENCY_AMOUNT, self.USED_LEVERAGE)
        for sell_open_uuid, trade_info in self.previous_trade_details.items():
            previous_profit_loss += self.calculate_sell_profit_loss(trade_info['open_price'], trade_info['close_price'], self.OWN_CURRENCY_AMOUNT, self.USED_LEVERAGE)
        total_profit_loss = current_profit_loss + previous_profit_loss
        return total_profit_loss



    

    def calculate_number_of_profitable_trades(self):
        profitable_trades = 0
        for buy_open_uuid, trade_info in self.previous_trade_details.items():
            if self.calculate_buy_profit_loss(trade_info['open_price'], trade_info['close_price'], self.OWN_CURRENCY_AMOUNT, self.USED_LEVERAGE) > 0:
                profitable_trades += 1
        for sell_open_uuid, trade_info in self.previous_trade_details.items():
            if self.calculate_sell_profit_loss(trade_info['open_price'], trade_info['close_price'], self.OWN_CURRENCY_AMOUNT, self.USED_LEVERAGE) > 0:
                profitable_trades += 1
        return profitable_trades





    def calculate_number_of_total_trades_profotalbe_and_lossing_trades(self):
        total_trades = len(self.previous_trade_details) + self.current_buy_open_trades + self.current_sell_open_trades
        profitable_trades = self.calculate_number_of_profitable_trades()
        lossing_trades = total_trades - profitable_trades
        return total_trades, profitable_trades, lossing_trades


    def reward(self,wrong_action:bool = False,do_nothing:bool = False,done:bool = False):
        # 1) reward is proportional to the profit or loss
        # 2) reward is proportional to the number of trades
        # 4) reward is inversely proportional to number of auto terminated trades
        # 5)panalty for doing nothing
        # 6) reward for closing profitable trades
        # 7) panalty for closing lossing trades
        # 8) extra reward for each 2% profit
        # 9) i step == 404 then means done and panalty
        # 10)  if current reward is more than previous reward then reward is positive else negative
        # 11) panalty for done
        total_profit_loss = self.calculate_total_profit_loss() 
        current_profit_loss = self.calculate_current_profit_loss()

        total_profit_loss_percentage = (total_profit_loss / self.OWN_CURRENCY_AMOUNT) * 100
        current_profit_loss_percentage = (current_profit_loss / self.OWN_CURRENCY_AMOUNT) * 100

        current_number_of_open_trade = self.current_buy_open_trades + self.current_sell_open_trades
        auto_terminated_trades = self.auto_terminated_trades
        total_trades, profitable_trades, lossing_trades = self.calculate_number_of_total_trades_profotalbe_and_lossing_trades()
        profit_loss_percentage = total_profit_loss / self.OWN_CURRENCY_AMOUNT            
        K = 0.1  # You can adjust this value based on the desired magnitude of the rewards/penalties

        reward = (
            K * total_profit_loss_percentage +                  # Reward proportional to profit or loss
            K * current_profit_loss_percentage +                # Reward proportional to profit or loss
            K * total_trades +                       # Reward proportional to the number of trades
            -K * auto_terminated_trades +            # Penalty inversely proportional to the number of auto-terminated trades
            -K * (not current_profit_loss) +  # Penalty for doing nothing
            K * profitable_trades +                  # Reward for closing profitable trades
            -K * lossing_trades +               
                 
                 
                      # Penalty for closing losing trades
            K * (
                
                
                
                   profit_loss_percentage // 0.02)     # Extra reward for each 2% profit
        )
        reward= reward + K * (1 if reward > self.previous_reward else -1)  # Reward for improving or penalizing the reward

        if wrong_action:
            reward = -K * 10
            self.WRONG_STEPS_TOLLARANCE -= 1
        if done:
            reward = -K * 100

        self.previous_reward = reward
        

        

        return reward
    

    def reset(self, new_data=None):
        self.current_step = 0
        self.current_price = self.data['Close'][self.current_step]
        self.current_buy_open_trades = 0
        self.current_sell_open_trades = 0
        self.auto_terminated_trades = 0
        self.buy_open_uuids = {}
        self.sell_open_uuids = {}
        self.previous_trade_details = {}
        self.previous_reward = 0
        self.net_gains = 0
        self.MAXIMUM_DOING_NOTHING_STEPS_TOLLARANCE = 10

        # Reset currency amounts
        self.OWN_CURRENCY_AMOUNT = np.random.randint(10, 100)
        self.USED_CURRENCY_AMOUNT = np.random.randint(1000, 100000)
        self.USED_LEVERAGE = self.USED_CURRENCY_AMOUNT / self.OWN_CURRENCY_AMOUNT
        self.LOSS_TOLLARANCE = self.OWN_CURRENCY_AMOUNT * 0.03
        self.ACCUMULATED_LOSS_TOLLARANCE = self.OWN_CURRENCY_AMOUNT * 0.1

        # Reset maximum steps for holding
        self.MAXIMUM_AMOUNT_OF_TIME_FOR_HOLDING = 100

        # Reset initial step conditions
        self.current_price = self.data['Close'][self.current_step]

        # Reset action and observation spaces
        self.action_space = spaces.Discrete(6)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(12 ,), dtype=np.float32)
        
        # Reset the environment and return the initial observation
        obs = self._next_observation()
        return obs
    
    def _next_observation(self):
        obs = np.array([
            self.current_price,
            self.OWN_CURRENCY_AMOUNT,
            self.USED_CURRENCY_AMOUNT,
            self.USED_LEVERAGE,
            self.LOSS_TOLLARANCE,
            self.ACCUMULATED_LOSS_TOLLARANCE,
            self.MAXIMUM_AMOUNT_OF_TIME_FOR_HOLDING,
            self.current_buy_open_trades,
            self.current_sell_open_trades,
            self.auto_terminated_trades,
            self.MAXIMUM_DOING_NOTHING_STEPS_TOLLARANCE,
            self.current_step

        ])
        
        return obs


    def render(self, mode='human'):
        profit_loss = self.calculate_total_profit_loss()
        print(f'Step: {self.current_step}')
        print(f'Price: {self.current_price}')
        print(f'Profit/Loss: {profit_loss}')
        print(f'Net Gains: {self.net_gains}')
        print(f'Current Buy Open Trades: {self.current_buy_open_trades}')
        print(f'Current Sell Open Trades: {self.current_sell_open_trades}')
        print(f'Auto Terminated Trades: {self.auto_terminated_trades}')
        print(f'Previous Reward: {self.previous_reward}')
        print(f'Action Space: {self.action_space}')
        print(f'Observation Space: {self.observation_space}')
        print(f'Buy Open Trades: {self.buy_open_uuids}')
        print(f'Sell Open Trades: {self.sell_open_uuids}')
        print(f'Previous Trade Details: {self.previous_trade_details}')

    def step(self, action ):
        self.current_step += 1.
        self.current_price = self.data['Close'][self.current_step]
        self.Terminate_after_maximum_step()
        self.Terminate_lossing_buy_open_trades()
        self.Terminate_lossing_sell_open_trades()
        self.current_buy_open_trades = len(self.buy_open_uuids)
        self.current_sell_open_trades = len(self.sell_open_uuids)
        total_open_trades = self.current_buy_open_trades + self.current_sell_open_trades
        wrong_action = False
        do_nothing = False

        if action == 0:
            if total_open_trades == 0:
                self.buy_open_uuids[str(uuid.uuid4())] = {
                    'open_price': self.current_price,
                    'available_time': self.MAXIMUM_AMOUNT_OF_TIME_FOR_HOLDING
                }
            else:
                action = 4
                self.taking_wrong_action_count += 1
                wrong_action = True
        elif action == 1:
            if total_open_trades == 0:
                self.sell_open_uuids[str(uuid.uuid4())] = {
                    'open_price': self.current_price,
                    'available_time': self.MAXIMUM_AMOUNT_OF_TIME_FOR_HOLDING
                }
            else:
                action = 4
                self.taking_wrong_action_count += 1
                wrong_action = True
        elif action == 2:
            if self.current_buy_open_trades > 0:
                buy_open_uuid = list(self.buy_open_uuids.keys())[0]
                self.close_trade(buy_open_uuid, self.buy_open_uuids, 'buy_open')
            elif self.current_sell_open_trades > 0:
                action = 4 # Hold
                self.taking_wrong_action_count += 1
                wrong_action = True
            else:
                action = 5 # Do nothing
                self.taking_wrong_action_count += 1
                wrong_action = True
                self.MAXIMUM_DOING_NOTHING_STEPS_TOLLARANCE -= 1
                do_nothing = True
            
        elif action == 3:
            if self.current_sell_open_trades > 0:
                sell_open_uuid = list(self.sell_open_uuids.keys())[0]
                self.close_trade(sell_open_uuid, self.sell_open_uuids, 'sell_open')
            elif self.current_buy_open_trades > 0:
                action = 4
                self.taking_wrong_action_count += 1
                wrong_action = True
            else:
                action = 5 # Do nothing
                self.taking_wrong_action_count += 1
                wrong_action = True
                self.MAXIMUM_DOING_NOTHING_STEPS_TOLLARANCE -= 1
                do_nothing = True

        elif action == 4:
            if total_open_trades == 0:
                action = 5
                self.taking_wrong_action_count += 1
                wrong_action = True
            else:
                action = 4
        elif action == 5:
            if total_open_trades > 0:
                action = 4
                self.taking_wrong_action_count += 1
                wrong_action = True
            else:
                action = 5
                self.MAXIMUM_DOING_NOTHING_STEPS_TOLLARANCE -= 1
                do_nothing = True
            
        else:
            action = 5
            self.taking_wrong_action_count += 1
            wrong_action = True
            self.MAXIMUM_DOING_NOTHING_STEPS_TOLLARANCE -= 1
            do_nothing = True



        obs = self._next_observation()
        done = self.done()
        reward = self.reward(wrong_action,do_nothing,done)

        info = {'action':action,'current_price': self.current_price, 'current_step': self.current_step, 'current_profit_loss': self.calculate_current_profit_loss(), 'total_profit_loss': self.calculate_total_profit_loss(), 'net_gains': self.net_gains, 'current_buy_open_trades': self.current_buy_open_trades, 'current_sell_open_trades': self.current_sell_open_trades, 'auto_terminated_trades': self.auto_terminated_trades, 'previous_reward': self.previous_reward, 'buy_open_uuids': self.buy_open_uuids, 'sell_open_uuids': self.sell_open_uuids, 'previous_trade_details': self.previous_trade_details}      

        #if done then add to tensorboard
        if done:
            self.log_to_tensorboard(reward)
        
        return obs, reward, done, info



    def done(self):
    #1)if current loss is greater than 3% of own currency amount
    #2)if auto terminated trades are greater than 3
    #3)if all comolative loss is greater than 10% of own currency amount
    #4)maximum doing nothing steps tollarance
    #5) IF 3 loss trades over 10 trades and 2 consecutive loss trades
    #6) if maximum doing nothing steps tollarance is 0
    #7) if total rades are 3 or more but cutent totatal profit is less than 10% of own currency amount
        current_profit_loss = self.calculate_current_profit_loss()
        # print("current loss is ", current_profit_loss)
        # print("loss tollarance is ", self.LOSS_TOLLARANCE)
        # print("accumulated loss tollarance is ", self.ACCUMULATED_LOSS_TOLLARANCE)


        autoterminated_trades = self.auto_terminated_trades
        cumulative_profit_loss = self.calculate_total_profit_loss()
        doing_nothing_steps_tollarance = self.MAXIMUM_DOING_NOTHING_STEPS_TOLLARANCE
        total_trades, profitable_trades, lossing_trades = self.calculate_number_of_total_trades_profotalbe_and_lossing_trades()
        wrong_steps_tollarance = self.WRONG_STEPS_TOLLARANCE

        
        if current_profit_loss < -self.LOSS_TOLLARANCE:
            print("current loss is greater than 3% of own currency amount")
            return True
        elif autoterminated_trades > 2:
            print("auto terminated trades are greater than 3")
            return True
        elif cumulative_profit_loss < -self.ACCUMULATED_LOSS_TOLLARANCE:
            print("all comolative loss is greater than 10% of own currency amount")
            return True
        elif doing_nothing_steps_tollarance == 0:
            print("maximum doing nothing steps tollarance")
            return True
        elif wrong_steps_tollarance == 0:
            print("wrong steps tollarance")
            return True
        elif lossing_trades > 2 and lossing_trades / total_trades > 0.3:
            print("3 loss trades over 10 trades and 2 consecutive loss trades")
            return True
        elif total_trades >= 2 and cumulative_profit_loss < self.MINIMUM_GAINS:
            print("total rades are 2 or more but cutent totatal profit is less than 5% of own currency amount")
            return True
        
        else:
            return False






    def log_to_tensorboard(self, reward):
        # Log relevant metrics to TensorBoard
        self.current_buy_open_trades = len(self.buy_open_uuids)
        self.current_sell_open_trades = len(self.sell_open_uuids)
        total_open_trades = self.current_buy_open_trades + self.current_sell_open_trades
    
        self.writer.add_scalar("Reward", reward, self.current_step)
        self.writer.add_scalar("Profit/Loss", self.calculate_total_profit_loss(), self.current_step)
        self.writer.add_scalar("Net Gains", self.net_gains, self.current_step)
        self.writer.add_scalar("Auto Terminated Trades", self.auto_terminated_trades, self.current_step)
        self.writer.add_scalar("Previous Reward", self.previous_reward, self.current_step)
        self.writer.add_scalar("Current Step", self.current_step, self.current_step)
        self.writer.add_scalar("current profit loss", self.calculate_current_profit_loss(), self.current_step)
        self.writer.add_scalar("total trades", total_open_trades, self.current_step)
        self.writer.add_scalar("number of profitable trades", self.calculate_number_of_profitable_trades(), self.current_step)
        self.writer.add_scalar("number of lossing trades", total_open_trades - self.calculate_number_of_profitable_trades(), self.current_step)
        self.writer.flush()

