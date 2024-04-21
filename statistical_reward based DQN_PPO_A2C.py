

"""# Import"""

from openai import OpenAI
import yfinance as yf
import numpy as np
import pandas as pd
import gym
import json
from stable_baselines3 import DQN, PPO, A2C
from stable_baselines3.common.env_util import make_vec_env
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import ta
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from datasets import load_dataset

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

"""# Preprocessing"""

def add_TI(stock_data):
  stock_data['SMA'] = ta.trend.sma_indicator(stock_data['Close'], window=14) # Trend Indicators
  stock_data['RSI'] = ta.momentum.rsi(stock_data['Close'], window=14) # Momentum Indicators
  stock_data['OBV'] = ta.volume.on_balance_volume(stock_data['Close'], stock_data['Volume']) # Volume Indicators
  stock_data['ATR_14'] = ta.volatility.average_true_range(stock_data['High'], stock_data['Low'], stock_data['Close'], window=14) # Volatility Indicators
  stock_data['CCI_20'] = ta.trend.cci(stock_data['High'], stock_data['Low'], stock_data['Close'], window=20) # Commodity Channel Index : An oscillator used to identify cyclical trends in commodities.
  stock_data['NextDayClose'] = stock_data['Close'].shift(-1)
  stock_data['Target'] = (stock_data['NextDayClose'] > stock_data['Close']).astype(int)
  stock_data.dropna(inplace=True)
  return stock_data

"""### data visualization"""

temp_data = add_TI(yf.download('AMZN', '2020-01-01', '2021-01-01'))
temp_data.head()

nrows, ncols = 1, 4

fig, axs = plt.subplots(nrows, ncols, figsize=(18, 4))  # Adjust figsize as needed

# Flatten the axs array for easy indexing
axs = axs.flatten()

axs[0].plot(temp_data['Close'], label="Close")
axs[0].plot(temp_data['SMA'], label="SMA")
axs[0].plot(temp_data['RSI'], label="RSI")
axs[0].legend()

axs[1].plot(temp_data['ATR_14'], label="ATR-14", color="darkorange")
axs[1].legend()

axs[2].plot(temp_data['CCI_20'], label="CCI-14", color="forestgreen")
axs[2].legend()

bar_width = 0.35
positions = np.arange(len(temp_data.index))

axs[3].bar(positions - bar_width/2, temp_data['Volume'], bar_width, color='skyblue', label="volume")
axs[3].bar(positions + bar_width/2, temp_data['Volume'], bar_width, color='orange', label="OBV")
axs[3].legend()

plt.tight_layout()
plt.show()

# classification of stocks based on https://www.nasdaq.com/solutions/nasdaq-100/companies#utilities

consumer_discretionary = ["AMZN", "CPRT", "SBUX", "PAYX", "MNST"]
consumer_staples = ["PEP", "KHC", "WBA", "CCEP", "MDLZ"]
health_care = ["AMGN", "VRTX", "ISRG", "MRNA", "ILMN"]
industrial = ["CSX", "BKR", "AAPL", "ROP", "HON"]
technology = ["QCOM", "MSFT", "INTC", "MDB", "GOOG"]
telecommunications = ["CMCSA", "WBD", "CSCO", "TMUS", "AEP"]
utilities = ["XEL", "EXC", "PCG", "SRE", "OGE"]

all_stocks = consumer_discretionary + consumer_staples + health_care + industrial + technology + telecommunications + utilities
print(all_stocks)

"""# Reinforce

## Reward functions


"""### Statistical reward"""

# Basic reward function
def calculate_reward(self, action):
        observation = self.stock_data.iloc[self.current_step]

        sma = observation['SMA']
        close_price = observation['Close']
        rsi = observation['RSI']
        obv = observation['OBV']
        atr = observation["ATR_14"]
        cci_20 = observation['CCI_20']
        reward = 0

        price_above_sma = close_price > sma # Determine the trend direction based on SMA
        price_below_sma = close_price < sma

        oversold = rsi < 30 # Determine if the stock is overbought or oversold using RSI
        overbought = rsi > 70

        cci_oversold = cci_20 < -100 # Determine CCI indication
        cci_overbought = cci_20 > 100

        if action == 0: # Buy action
            if price_above_sma:
              reward += 2
            if oversold:
              reward += 2
            if cci_oversold:
              reward += 2
            if price_above_sma and oversold and cci_oversold:
                reward += 5 # Reward for buying under IDEAL conditions

        elif action == 1: # Sell action
            if price_below_sma:
              reward += 2
            if overbought:
              reward += 2
            if cci_overbought:
              reward += 2
            if price_below_sma and overbought and cci_overbought:
                reward += 5 # Reward for selling under IDEAL conditions

        return reward*reward

def basic_reward(portfolio,balance):
  return portfolio - balance


"""## Environment"""

class StockEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self, stock_symbols, window_size, start_date, end_date, render_mode=None):
        super(StockEnv, self).__init__()
        self.start_date = start_date
        self.end_date = end_date
        self.stock_symbols = stock_symbols
        self.current_symbol = np.random.choice(self.stock_symbols)
        self.stock_data = add_TI(yf.download(self.current_symbol, start=self.start_date, end=self.end_date))
        self.window_size = window_size
        self.current_step = window_size

        self.action_space = gym.spaces.Discrete(3) # Actions : Buy (0), Sell (1), Hold (2)
        self.observation_space = gym.spaces.Box(low=0, high=np.inf, shape=(window_size, 10), dtype=np.float64) # Observation space is the stock data of the last 'window_size' days

        # Initial portfolio
        self.initial_balance = 10000
        self.current_balance = self.initial_balance
        self.shares_held = 0
        self.current_portfolio_value = self.current_balance


    def reset(self):
        self.current_symbol = np.random.choice(self.stock_symbols)
        print(f"\n{self.current_symbol}\n")
        self.stock_data = add_TI(yf.download(self.current_symbol, start=self.start_date, end=self.end_date))
        self.current_step = self.window_size
        self.current_balance = self.initial_balance
        self.shares_held = 0
        self.current_portfolio_value = self.current_balance
        return self._next_observation()


    def _next_observation(self):
        frame = self.stock_data.iloc[self.current_step-self.window_size:self.current_step].copy()
        return frame[['Open', 'High', 'Low', 'Close', 'Volume', 'SMA', 'RSI', 'OBV', 'ATR_14', 'CCI_20']].values


    def step(self, action):
        reward = 0
        current_price = self.stock_data.iloc[self.current_step]['Close']
        INVALID = False

        if action == 0:  # Buy
            if self.current_balance >= current_price: # Can buy only if there is enough balance
                self.shares_held += 1
                self.current_balance -= current_price
            else:
              INVALID = True
        elif action == 1:  # Sell
            if self.shares_held > 0: # Can sell only if shares are held
                self.shares_held -= 1
                self.current_balance += current_price
            else:
              INVALID = True


        self.current_portfolio_value = self.current_balance + self.shares_held * current_price

        if INVALID:
          reward = -10
        else:
          reward = calculate_reward(self, action)

        self.current_step += 1
        done = self.current_step >= len(self.stock_data) - 1

        return self._next_observation(), reward, done, {}

    def render(self, mode='human', close=False):
        print(f'Step: {self.current_step}, Action: {action}, Reward: {reward} ,Balance: {self.current_balance}, Shares held: {self.shares_held}, Current Portfolio Value: {self.current_portfolio_value}')
        portfolio_value.append(self.current_portfolio_value)

env = StockEnv(all_stocks, 10, '2020-01-01', '2021-01-01', render_mode='human')

"""## Random action selection"""

state = env.reset()
done = False
portfolio_value = []
while not done:
    action = env.action_space.sample()  # Choose a random action
    state, reward, done, info = env.step(action)
    env.render()

plt.plot(portfolio_value, label="CCEP")
plt.xlabel('step')
plt.ylabel('value')
plt.legend()
plt.show()

"""## DQN"""

vec_env = make_vec_env(lambda: env, n_envs=1)
dqn_model = DQN('MlpPolicy', vec_env, verbose=1, learning_rate=1e-3, learning_starts=1000, batch_size=32, tau=1.0, gamma=0.99, train_freq=4, gradient_steps=1, target_update_interval=500, exploration_fraction=0.1, exploration_initial_eps=1.0, exploration_final_eps=0.6, max_grad_norm=10, tensorboard_log="./dqn_tensorboard/")

dqn_model.learn(total_timesteps=10000)

log_path = './dqn_tensorboard/run1'
dqn_model.learn(total_timesteps=10000, tb_log_name=log_path)

dqn_model.save('dqn_stock_1M')

obs = env.reset()
done = False
episode_rewards = 0
portfolio_value = []
while not done:
    env.render()
    action, _states = dqn_model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    episode_rewards += reward
env.close()

plt.plot(portfolio_value, label="WBD")
plt.xlabel('step')
plt.ylabel('value')
plt.legend()
plt.show()

"""## PPO"""

ppo_model = PPO('MlpPolicy', vec_env, verbose=1, learning_rate=1e-3, n_steps=2048, batch_size=64, n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.0, max_grad_norm=10, tensorboard_log="./ppo_tensorboard/")

log_path = './ppo_tensorboard/run1'
ppo_model.learn(total_timesteps=1_000_000, tb_log_name=log_path)

ppo_model.save('ppo_stock_1M')

obs = env.reset()
done = False
episode_rewards = 0
portfolio_value = []
while not done:
    env.render()
    action, _states = ppo_model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    episode_rewards += reward
env.close()

plt.plot(portfolio_value, label="MSFT")
plt.xlabel('step')
plt.ylabel('value')
plt.legend()
plt.show()

"""## A2C"""

a2c_model = A2C('MlpPolicy', vec_env, verbose=1, learning_rate=1e-3, n_steps=5, gamma=0.99, gae_lambda=1.0, ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5, use_rms_prop=True, use_sde=False, sde_sample_freq=-1, normalize_advantage=False, tensorboard_log="./a2c_tensorboard/")

log_path = './a2c_tensorboard/run1'
a2c_model.learn(total_timesteps=1_000_000, tb_log_name=log_path)

a2c_model.save('a2c_stock_1M')

obs = env.reset()
done = False
episode_rewards = 0
portfolio_value = []
while not done:
    env.render()
    action, _states = a2c_model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    episode_rewards += reward
env.close()

plt.plot(portfolio_value, label="MSFT")
plt.xlabel('step')
plt.ylabel('value')
plt.legend()
plt.show()

"""# Testing"""

class StockEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self, stock_symbols, window_size, start_date, end_date, render_mode=None, specific_symbol=None):
        super(StockEnv, self).__init__()
        self.start_date = start_date
        self.end_date = end_date
        self.specific_symbol = specific_symbol

        if specific_symbol:
            self.current_symbol = specific_symbol
        else:
            self.current_symbol = np.random.choice(self.stock_symbols)
        self.stock_data = add_TI(yf.download(self.current_symbol, start=self.start_date, end=self.end_date))
        self.window_size = window_size
        self.current_step = window_size

        self.action_space = gym.spaces.Discrete(3) # Actions : Buy (0), Sell (1), Hold (2)
        self.observation_space = gym.spaces.Box(low=0, high=np.inf, shape=(window_size, 10), dtype=np.float64) # Observation space is the stock data of the last 'window_size' days

        # Initial portfolio
        self.initial_balance = 10000
        self.current_balance = self.initial_balance
        self.shares_held = 0
        self.current_portfolio_value = self.current_balance

    def reset(self):
        if not self.specific_symbol:
            self.current_symbol = np.random.choice(self.stock_symbols)
        print(f"\n{self.current_symbol}\n")
        self.stock_data = add_TI(yf.download(self.current_symbol, start=self.start_date, end=self.end_date))
        self.current_step = self.window_size
        self.current_balance = self.initial_balance
        self.shares_held = 0
        self.current_portfolio_value = self.current_balance
        return self._next_observation()

    def _next_observation(self):
        frame = self.stock_data.iloc[self.current_step-self.window_size:self.current_step].copy()
        return frame[['Open', 'High', 'Low', 'Close', 'Volume', 'SMA', 'RSI', 'OBV', 'ATR_14', 'CCI_20']].values

    def step(self, action):
        self.current_step += 1
        done = self.current_step >= len(self.stock_data) - 1
        reward = 0
        current_price = self.stock_data.iloc[self.current_step]['Close']

        if action == 0:  # Buy
            if self.current_balance >= current_price: # Can buy only if there is enough balance
                self.shares_held += 1
                self.current_balance -= current_price
        elif action == 1:  # Sell
            if self.shares_held > 0: # Can sell only if shares are held
                self.shares_held -= 1
                self.current_balance += current_price
        self.current_portfolio_value = self.current_balance + self.shares_held * current_price

        reward = self.current_portfolio_value - self.initial_balance
        return self._next_observation(), reward, done, {}

    def render(self, mode='human', close=False):
        print(f'Step: {self.current_step}, Action: {action}, Balance: {self.current_balance}, Shares held: {self.shares_held}, Current Portfolio Value: {self.current_portfolio_value}')
        portfolio_value.append(self.current_portfolio_value)

specific_symbols = ["AAPL","NVAX","AMZN","GE","TSLA", "INTC"]
test_env = StockEnv(all_stocks, 10, '2020-01-01', '2021-01-01', render_mode='human',specific_symbol="INTC")
state = test_env.reset()
done = False
portfolio_value = []
while not done:
    action = test_env.action_space.sample()  # Choose a random action
    state, reward, done, info = test_env.step(action)
    test_env.render()
print(portfolio_value[-1]/10000)

specific_symbols = ["AAPL","NVAX","AMZN","GE","TSLA", "INTC"]
a2c_v = []
dqn_v = []
ppo_v = []
a2c_r = []
dqn_r = []
ppo_r = []


for i in specific_symbols:
  print(i)
  test_env = StockEnv(all_stocks, 10, '2020-01-01', '2021-01-01', render_mode='human',specific_symbol=i)
  obs = test_env.reset()
  done = False
  episode_rewards = 0
  e_reward = [0]
  portfolio_value = []
  while not done:
      test_env.render()
      action, _states = a2c_model.predict(obs, deterministic=True)
      obs, reward, done, info = test_env.step(action)
      episode_rewards += reward
      e_reward.append(episode_rewards)
  test_env.close()
  a2c_v.append(portfolio_value)
  a2c_r.append(e_reward)


  obs = test_env.reset()
  done = False
  episode_rewards = 0
  e_reward = [0]
  portfolio_value = []
  while not done:
      test_env.render()
      action, _states = ppo_model.predict(obs, deterministic=True)
      obs, reward, done, info = test_env.step(action)
      episode_rewards += reward
      e_reward.append(episode_rewards)
  test_env.close()
  ppo_v.append(portfolio_value)
  ppo_r.append(e_reward)

  obs = test_env.reset()
  done = False
  episode_rewards = 0
  e_reward = [0]
  portfolio_value = []
  while not done:
      test_env.render()
      action, _states = dqn_model.predict(obs, deterministic=True)
      obs, reward, done, info = test_env.step(action)
      episode_rewards += reward
      e_reward.append(episode_rewards)
  test_env.close()
  dqn_v.append(portfolio_value)
  dqn_r.append(e_reward)

import matplotlib.pyplot as plt

# Number of rows and columns for subplot grid
nrows, ncols = 2, 3

fig, axs = plt.subplots(nrows, ncols, figsize=(18, 12))  # Adjust figsize as needed

# Flatten the axs array for easy indexing
axs = axs.flatten()

# Loop through the specific_symbols list and plot
for i, symbol in enumerate(specific_symbols):
    axs[i].plot(a2c_v[i], label="A2C")
    axs[i].plot(ppo_v[i], label="PPO")
    axs[i].plot(dqn_v[i], label="DQN")
    axs[i].set_title(symbol)
    axs[i].set_xlabel('Step')
    axs[i].set_ylabel('Portfolio value')
    axs[i].legend()

# If there are any empty subplots (because of an uneven grid), hide them
for j in range(i + 1, nrows * ncols):
    axs[j].axis('off')

plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt

# Number of rows and columns for subplot grid
nrows, ncols = 2, 3

fig, axs = plt.subplots(nrows, ncols, figsize=(18, 12))  # Adjust figsize as needed

# Flatten the axs array for easy indexing
axs = axs.flatten()

# Loop through the specific_symbols list and plot
for i, symbol in enumerate(specific_symbols):
    axs[i].plot(a2c_r[i], label="A2C")
    axs[i].plot(ppo_r[i], label="PPO")
    axs[i].plot(dqn_r[i], label="DQN")
    axs[i].set_title(symbol)
    axs[i].set_xlabel('Step')
    axs[i].set_ylabel('Cumulative Reward')
    axs[i].legend()

# If there are any empty subplots (because of an uneven grid), hide them
for j in range(i + 1, nrows * ncols):
    axs[j].axis('off')

plt.tight_layout()
plt.show()

for i in range(6):
  print(f"{dqn_v[i][-1]/10000} {ppo_v[i][-1]/10000} {a2c_v[i][-1]/10000}")

"""# Policy"""

dqn_model = DQN.load('dqn_stock_1M')
a2c_model = A2C.load('a2c_stock_1M')
ppo_model = PPO.load('ppo_stock_1M')

dqn_policy = dqn_model.policy
a2c_policy = a2c_model.policy
ppo_policy = ppo_model.policy

import torch

# Example: Fetching an observation from your environment
obs = env.reset()  # Get the initial observation
obs_tensor = torch.tensor(obs, dtype=torch.float).unsqueeze(0)  # Reshape and convert to tensor
# Assuming `ppo_model` is your trained PPO model
a2c_policy.eval()  # Set the policy network to evaluation mode


with torch.no_grad():  # Disable gradient computation
    # Perform the forward pass
    logits = a2c_policy(obs_tensor)
    # Depending on your setup, you might directly work with logits here or further process them
logits
