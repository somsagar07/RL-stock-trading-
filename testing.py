

import yfinance as yf
import numpy as np
import pandas as pd
import gym
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.env_util import make_vec_env
import matplotlib.pyplot as plt
import torch
import ta

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

def add_TI(stock_data):
  stock_data['SMA'] = ta.trend.sma_indicator(stock_data['Close'], window=14) # Trend Indicators
  stock_data['RSI'] = ta.momentum.rsi(stock_data['Close'], window=14) # Momentum Indicators
  stock_data['OBV'] = ta.volume.on_balance_volume(stock_data['Close'], stock_data['Volume']) # Volume Indicators
  stock_data['ATR_14'] = ta.volatility.average_true_range(stock_data['High'], stock_data['Low'], stock_data['Close'], window=14) # Volatility Indicators
  stock_data['CCI_20'] = ta.trend.cci(stock_data['High'], stock_data['Low'], stock_data['Close'], window=20) # Commodity Channel Index : An oscillator used to identify cyclical trends in commodities.
  stock_data.dropna(inplace=True)
  return stock_data

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

# vec_env = make_vec_env(lambda: env, n_envs=1)

a2c_model = A2C.load('models/a2c_stock_1M')
dqn_model = DQN.load('models/dqn_stock_1M')
ppo_model = PPO.load('models/ppo_stock_1M')

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

nrows, ncols = 2, 3

fig, axs = plt.subplots(nrows, ncols, figsize=(18, 12))
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

fig, axs = plt.subplots(nrows, ncols, figsize=(18, 12))
axs = axs.flatten()
for i, symbol in enumerate(specific_symbols):
    axs[i].plot(a2c_r[i], label="A2C")
    axs[i].plot(ppo_r[i], label="PPO")
    axs[i].plot(dqn_r[i], label="DQN")
    axs[i].set_title(symbol)
    axs[i].set_xlabel('Step')
    axs[i].set_ylabel('Cumulative Reward')
    axs[i].legend()
for j in range(i + 1, nrows * ncols):
    axs[j].axis('off')

plt.tight_layout()
plt.show()
