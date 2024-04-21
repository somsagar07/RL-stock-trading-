
"""# Import"""

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

### Linear Reward model

#### Train
"""

class RewardPredictor(nn.Module):
    def __init__(self):
        super(RewardPredictor, self).__init__()
        self.fc1 = nn.Linear(6, 64)  # 6 features
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 8)
        self.fc5 = nn.Linear(8, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
        return x

def train_model(model, train_loader, criterion, optimizer, num_epochs=10000):
    model.train()
    losses = []  # Initialize an empty list to store the losses
    for epoch in range(num_epochs):
        epoch_losses = []
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets.unsqueeze(1))  # Ensure targets are the correct shape
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        epoch_loss = sum(epoch_losses) / len(epoch_losses)
        losses.append(epoch_loss)
        if epoch % 25 == 0:  # Print the loss every 100 epochs
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')
    return losses

symbols = all_stocks

start_date = '2020-01-01'
end_date = '2021-01-01'

# Combine data from all symbols
all_data = pd.DataFrame()
for symbol in symbols:
    symbol_data = add_TI((yf.download(symbol, '2020-01-01', '2021-01-01')))
    if symbol_data is not None:
        all_data = pd.concat([all_data, symbol_data])

features = all_data[['Close','SMA', 'RSI', 'OBV', 'ATR_14', 'CCI_20']].values
targets = all_data['NextDayClose'].values


X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

model = RewardPredictor()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
loss_history = train_model(model, train_loader, criterion, optimizer, num_epochs=10000)

torch.save(model.state_dict(), './model.pth')

loss_history

plt.figure(figsize=(10, 6))
plt.plot(loss_history[50:], label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss Over Time')
plt.legend()
plt.show()

"""#### Load NN reward model"""

random_input = torch.rand((1, 6))  # Batch size of 1, 5 features
print(random_input)
model.eval()

def calculate_reward(inputs):
  # Pass the random input through the model
  with torch.no_grad():  # No need to track gradients for inference
      output = model(inputs)

  # # Print the output
  # print(f"Model output for random input: {random_output}")
  return output

"""## Environment"""

jax    metadata = {'render.modes': ['human']}
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

        k = 0
        if action == 0:  # Buy
            k = 1
            if self.current_balance >= current_price: # Can buy only if there is enough balance
                self.shares_held += 1
                self.current_balance -= current_price
            else:
              INVALID = True
        elif action == 1:  # Sell
            k = -1
            if self.shares_held > 0: # Can sell only if shares are held
                self.shares_held -= 1
                self.current_balance += current_price
            else:
              INVALID = True


        self.current_portfolio_value = self.current_balance + self.shares_held * current_price
        if INVALID:
          reward = -10
        else:
          reward = k*(calculate_reward(self.stock_data.iloc[]) - current_price)

        self.current_step += 1
        done = self.current_step >= len(self.stock_data) - 1

        return self._next_observation(), reward, done, {}

    def render(self, mode='human', close=False):
        print(f'Step: {self.current_step}, Action: {action}, Reward: {reward} ,Balance: {self.current_balance}, Shares held: {self.shares_held}, Current Portfolio Value: {self.current_portfolio_value}')
        portfolio_value.append(self.current_portfolio_value)

env = StockEnv(all_stocks, 10, '2020-01-01', '2021-01-01', render_mode='human')

"""## DQN"""

vec_env = make_vec_env(lambda: env, n_envs=1)
dqn_model = DQN('MlpPolicy', vec_env, verbose=1, learning_rate=1e-3, learning_starts=1000, batch_size=32, tau=1.0, gamma=0.99, train_freq=4, gradient_steps=1, target_update_interval=500, exploration_fraction=0.1, exploration_initial_eps=1.0, exploration_final_eps=0.6, max_grad_norm=10, tensorboard_log="./dqn_tensorboard/")

log_path = './dqn_tensorboard/run1'
dqn_model.learn(total_timesteps=1_000_000, tb_log_name=log_path)

dqn_model.save('dqn_stock_1M')

obs = env.reset()
done = False
episode_rewards = 0
portfolio_value = []
while not done:
    action, _states = dqn_model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    episode_rewards += reward
    env.render()
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