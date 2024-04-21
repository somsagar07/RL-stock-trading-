
from google.colab import drive
drive.mount('/content/drive')

"""# Installations"""

! pip install yfinance
! pip install stable-baselines3[extra]
! pip install gym
! pip install ta
! pip install tensorboard
! pip install datasets
! pip install openai
! pip install feedparser

"""# Imports"""

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
import re
import ast
from datetime import datetime, timedelta
import time
import feedparser
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

sentiment_tokenizer = AutoTokenizer.from_pretrained("mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")
sentiment_model = AutoModelForSequenceClassification.from_pretrained("mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")

if torch.cuda.is_available():
    sentiment_model = sentiment_model.cuda()

"""# Preprocessing and Functions"""

def convert_string_to_list(string):
  '''
  param : string
  desc : converts string(in list format) to list
  '''
  numbers = re.findall(r"[\d\.\-e]+", string)
  return [float(num) for num in numbers]

def sum_df(data):
  '''
  SPF
  param : datafram
  desc : finds sum along column
  '''
  data_lists = data.apply(convert_string_to_list)
  data_array = np.array(data_lists.tolist())
  sum_along_columns = np.sum(data_array, axis=0)
  return sum_along_columns

def add_TI(stock_data):
  '''
  param : Dataframe (OHLCV)
  desc : finds additional technical indicators
  '''
  stock_data['SMA'] = ta.trend.sma_indicator(stock_data['Close'], window=14) # Trend Indicators
  stock_data['RSI'] = ta.momentum.rsi(stock_data['Close'], window=14) # Momentum Indicators
  stock_data['OBV'] = ta.volume.on_balance_volume(stock_data['Close'], stock_data['Volume']) # Volume Indicators
  stock_data['ATR_14'] = ta.volatility.average_true_range(stock_data['High'], stock_data['Low'], stock_data['Close'], window=14) # Volatility Indicators
  stock_data['CCI_20'] = ta.trend.cci(stock_data['High'], stock_data['Low'], stock_data['Close'], window=20) # Commodity Channel Index : An oscillator used to identify cyclical trends in commodities.
  stock_data.dropna(inplace=True)
  return stock_data

def past_data(tickerSymbol):
  '''
  param : string (ticker symbols only eg: AAPL)
  desc : get 30 days data of the stock
  '''
  end_date = datetime.today()
  start_date = end_date - timedelta(days=30)  # Increased days to account for weekends
  start_date_str = start_date.strftime('%Y-%m-%d')
  end_date_str = end_date.strftime('%Y-%m-%d')
  df_data = yf.download(tickerSymbol, start=start_date_str, end=end_date_str)
  return df_data

def get_current_obs(tickerSymbol):
  '''
  param : string (ticker symbols only eg: AAPL)
  desc : get current observation - OHLCV + additional technical indicators
  '''
  ticker = yf.Ticker(tickerSymbol)
  info = ticker.info
  current_price = info.get('currentPrice')
  open_price = info.get('open')
  dayLow = info.get('dayLow')
  dayHigh = info.get('dayHigh')
  volume = info.get('volume')

  # Create a DataFrame with the extracted information
  data = {
      'Open': open_price,
      'High': dayHigh,
      'Low': dayLow,
      'Close': current_price,
      'Adj Close': current_price,
      'Volume': volume
  }
  df = pd.DataFrame([data], index=[pd.Timestamp('today').normalize()])
  past_df = past_data(tickerSymbol)

  # Append the new observation
  combined_df = pd.concat([past_df, df])
  return list(add_TI(combined_df).drop(columns = ['Adj Close']).iloc[-1]) + get_current_news_setiment(tickerSymbol)

def get_current_OHLC(tickerSymbol):
  '''
  param : string (ticker symbols only eg: AAPL)
  desc : get OLH of ticker symbol
  '''
  ticker = yf.Ticker(tickerSymbol)
  info = ticker.info
  open_price = info.get('open')
  dayLow = info.get('dayLow')
  dayHigh = info.get('dayHigh')

  return ["Open:"+str(open_price), "High:" + str(dayHigh), "Low:" + str(dayLow)]

def get_sentiment(text):
  '''
  param : String
  desc : get sentiment of string
  '''
  inputs = sentiment_tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
  if torch.cuda.is_available():
      inputs = {key: value.to('cuda') for key, value in inputs.items()}
  with torch.no_grad():
      logits = sentiment_model(**inputs).logits

  # Move logits back to CPU to work with them in numpy/pandas
  logits = logits.cpu()
  predictions = torch.softmax(logits, dim=-1)
  return predictions.numpy()[0]

def get_current_news_setiment(tickerSymbol):
  '''
  param : string (ticker symbols only eg: AAPL)
  desc : get aggregate sentiment of the current news of ticker symbol
  '''
  rss_url = 'https://news.google.com/rss/search?q=' + tickerSymbol + '&hl=en-US&gl=US&ceid=US:en'

  feed = feedparser.parse(rss_url)
  sentiment_sum = np.zeros(3)  # Initialize a sum array of size 3 with zeros
  count = 0
  for entry in feed.entries:
      sentiment = get_sentiment(entry.title)
      sentiment_sum += sentiment  # Sum the sentiment arrays element-wise
      count+=1
  return (sentiment_sum/count).tolist()

def get_live_news(tickerSymbol, topk):
  '''
  param : string (ticker symbols only eg: AAPL)
  desc : get topk news of the day
  '''
  res_url = 'https://news.google.com/rss/search?q=' + tickerSymbol + '&hl=en-US&gl=US&ceid=US:en'
  news = []
  feed = feedparser.parse(res_url);
  for entry in feed.entries:
    news.append(entry.title)

  if topk < len(news):
    return news[0:topk]
  else:
    return news


def getPrice(tickerSymbol):
  '''
  param : string (ticker symbols only eg: AAPL)
  desc : get current price of ticker symbol
  '''
  ticker = yf.Ticker(tickerSymbol)
  info = ticker.info
  current_price = info.get('currentPrice')
  return current_price

"""# Train

## Data
"""

# classification of stocks based on https://www.nasdaq.com/solutions/nasdaq-100/companies#utilities

consumer_discretionary = ["AMZN", "TSLA", "SBUX", "PAYX", "MNST"]
consumer_staples = ["PEP", "KHC", "WBA", "CCEP", "WMT"]
health_care = ["AMGN", "VRTX", "ISRG", "PFE", "ILMN"]
industrial = ["CSX", "BKR", "AAPL", "ROP", "HON"]
technology = ["QCOM", "MSFT", "INTC", "NVDA", "GOOG"]
telecommunications = ["CMCSA", "VZ", "CSCO", "T", "AEP"]
utilities = ["XEL", "EXC", "PCG", "SRE", "NEE"]

all_stocks = consumer_discretionary + consumer_staples + health_care + industrial + technology + telecommunications + utilities
print(all_stocks)

df = pd.read_csv('/content/drive/MyDrive/financial_oped_tweets_sentiment_with_index.csv')
df.tail()

news_df = df.drop('Unnamed: 0', axis=1)
news_df = news_df.drop('Unnamed: 0.1', axis=1)
news_df.sample(n=5)

"""## Reward function"""

client = OpenAI(
    api_key = '' # enter your key here
)

"""data for human feedback"""

hf_data = add_TI(yf.download('AMZN', '2016-01-01', '2021-01-01'))
hf1 = list(hf_data.loc['2018-12-17'][:11]) # buy good
current_news1 = news_df[(news_df['ticker'] == 'AMZN') & (news_df['release_date'] == '2018-12-17')]
sentiment_score1 = sum_df(current_news1['sentiment_content'].reset_index(drop= True))/len(current_news1['sentiment_content']) # bad sentiment
hf1 = hf1 + list(sentiment_score1)

hf2 = list(hf_data.loc['2019-04-22'][:11]) # sell good
current_news2 = news_df[(news_df['ticker'] == 'AMZN') & (news_df['release_date'] == '2019-04-22')]
sentiment_score2 = sum_df(current_news2['sentiment_content'].reset_index(drop= True))/len(current_news2['sentiment_content']) # goof sentiment
hf2 = hf2 + list(sentiment_score2)


hf3 = list(hf_data.loc['2020-02-03'][:11]) # buy bad
current_news3 = news_df[(news_df['ticker'] == 'AMZN') & (news_df['release_date'] == '2020-02-03')]
sentiment_score3 = sum_df(current_news3['sentiment_content'].reset_index(drop= True))/len(current_news3['sentiment_content']) # bad sentiment
hf3 = hf3 + list(sentiment_score3)

def get_LLM_reward(action, obs):
  response = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "system",
          "content": "You are a assistant which gives a single numerical as a output. For the following environment" +
          '''
           class StockEnv(gym.Env):
            def __init__(self, stock_symbols, start_date, end_date, render_mode=None):
                super(StockEnv, self).__init__()
                self.start_date = start_date
                self.current_date = None
                self.end_date = end_date
                self.stock_symbols = stock_symbols
                self.current_symbol = np.random.choice(self.stock_symbols)
                self.stock_data = add_TI(yf.download(self.current_symbol, start=self.start_date, end=self.end_date))
                self.current_step = 0
                self.current_news = None
                self.action_space = gym.spaces.Discrete(3) # Actions : Buy (0), Sell (1), Hold (2)
                self.observation_space = gym.spaces.Box(low=0, high=np.inf, shape=(1, 13), dtype=np.float64)

                # Initial portfolio
                self.initial_balance = 10000
                self.current_balance = self.initial_balance
                self.shares_held = 0
                self.current_portfolio_value = self.current_balance


            def reset(self):
                self.current_symbol = np.random.choice(self.stock_symbols)
                print(f"\n{self.current_symbol}\n")
                self.stock_data = add_TI(yf.download(self.current_symbol, start=self.start_date, end=self.end_date))
                self.current_step = 0
                self.current_balance = self.initial_balance
                self.shares_held = 0
                self.current_portfolio_value = self.current_balance

                return self._next_observation()


            def _next_observation(self,):
                frame = self.stock_data.iloc[self.current_step:self.current_step+1]
                self.current_date = str(frame.index[0])[0:10]

                sentiment_score = [0,0,0]
                if not news_df[(news_df['ticker'] == self.current_symbol) & (news_df['release_date'] == self.current_date)].empty:
                  self.current_news = news_df[(news_df['ticker'] == self.current_symbol) & (news_df['release_date'] == self.current_date)]
                  sentiment_score = sum_df(self.current_news['sentiment_content'].reset_index(drop= True))/len(self.current_news['sentiment_content'])
                return np.concatenate((frame[['Open', 'High', 'Low', 'Close', 'Volume', 'SMA', 'RSI', 'OBV', 'ATR_14', 'CCI_20']].values[0], sentiment_score))


            def step(self, action):
                reward = 0

                current_observation = self._next_observation()
                current_price = current_observation[3]

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
                  reward = calculate_reward(self, action, obs)

                self.current_step += 1
                done = self.current_step >= len(self.stock_data) - 1

                return current_observation, reward, done, {}

          '''

      },
    {"role": "system", "content": "The last 3 numbers in the list of the observation space will be taken as sentiment scores in the format [negetive,positive,neutral] "},
    {"role": "system", "content": "You always give a single numerical as output"},
    {"role": "user", "content": "Act as the calculate_reward which takes in action Buy(0), Sell(1) and Hold(2) and 13 observations ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA', 'RSI', 'OBV', 'ATR_14', 'CCI_20', 'Negative Sentiment', 'Neutral Sentiment', 'Positive Sentiment']"},
    {"role": "assistant", "content": "I will come up a ideal calculate_reward function for this environment and give a single numerical as output. Please provide the action and observation"},
    {"role": "user", "content": "for action 0 and observation " + str(hf1) + " What is the final reward?"},
    {"role": "assistant", "content": "75"},
    {"role": "user", "content": "for action 1 and observation " + str(hf2) + " What is the final reward?"},
    {"role": "assistant", "content": "100"},
    {"role": "user", "content": "for action 0 and observation " + str(hf2) + " What is the final reward?"},
    {"role": "assistant", "content": "0"},
    {"role": "user", "content": "for action" + str(action) + "and observation " + str(obs) + " What is the final reward?"},
    ]
  )
  return response.choices[0].message.content

obs = torch.rand((1, 13)).tolist()[0]
action = 0
get_LLM_reward(action,obs).split(" ")

"""## Reinforce"""

class StockEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self, stock_symbols, start_date, end_date, render_mode=None):
        super(StockEnv, self).__init__()
        self.start_date = start_date
        self.current_date = None
        self.end_date = end_date
        self.stock_symbols = stock_symbols
        self.current_symbol = np.random.choice(self.stock_symbols)
        self.stock_data = add_TI(yf.download(self.current_symbol, start=self.start_date, end=self.end_date))
        self.current_step = 0
        self.current_news = None
        self.action_space = gym.spaces.Discrete(3) # Actions : Buy (0), Sell (1), Hold (2)
        self.observation_space = gym.spaces.Box(low=0, high=np.inf, shape=(1, 13), dtype=np.float64)

        # Initial portfolio
        self.initial_balance = 10000
        self.current_balance = self.initial_balance
        self.shares_held = 0
        self.current_portfolio_value = self.current_balance


    def reset(self):
        self.current_symbol = np.random.choice(self.stock_symbols)
        print(f"\n{self.current_symbol}\n")
        self.stock_data = add_TI(yf.download(self.current_symbol, start=self.start_date, end=self.end_date))
        self.current_step = 0
        self.current_balance = self.initial_balance
        self.shares_held = 0
        self.current_portfolio_value = self.current_balance

        return self._next_observation()


    def _next_observation(self,):
        frame = self.stock_data.iloc[self.current_step:self.current_step+1]
        self.current_date = str(frame.index[0])[0:10]

        sentiment_score = [0,0,0]
        if not news_df[(news_df['ticker'] == self.current_symbol) & (news_df['release_date'] == self.current_date)].empty:
          self.current_news = news_df[(news_df['ticker'] == self.current_symbol) & (news_df['release_date'] == self.current_date)]
          sentiment_score = sum_df(self.current_news['sentiment_content'].reset_index(drop= True))/len(self.current_news['sentiment_content'])
        return np.concatenate((frame[['Open', 'High', 'Low', 'Close', 'Volume', 'SMA', 'RSI', 'OBV', 'ATR_14', 'CCI_20']].values[0], sentiment_score))


    def step(self, action):
        reward = 0

        current_observation = self._next_observation()
        current_price = current_observation[3]

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
          # reward = -1
          llm_output = get_LLM_reward(action, current_observation)
          if len(llm_output.split(" ")) == 1:
            reward = llm_output
          else:
            print(f"LLM OUTPUT OUT OF PATTERN : {llm_output.split(' ')}")
            reward = 0

        self.current_step += 1
        done = self.current_step >= len(self.stock_data) - 1

        self.render(mode='human', action=action, reward=reward)

        return current_observation, reward, done, {}

    def render(self, mode='human', close=False, action=None, reward=None):
      if action is not None and reward is not None:
          print(f'Step: {self.current_step}, Action: {action}, Reward: {reward}, Balance: {self.current_balance}, Shares held: {self.shares_held}, Current Portfolio Value: {self.current_portfolio_value}')
      else:
          print(f'Step: {self.current_step}, Balance: {self.current_balance}, Shares held: {self.shares_held}, Current Portfolio Value: {self.current_portfolio_value}')

env = StockEnv(all_stocks, '2020-01-01', '2021-01-01', render_mode='human')
vec_env = make_vec_env(lambda: env, n_envs=1)
ppo_model = PPO('MlpPolicy', vec_env, verbose=1, learning_rate=1e-3, n_steps=1024, batch_size=64, n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.0, max_grad_norm=10)

ppo_model.learn(total_timesteps=1)

ppo_model.save('PPO_llm')

"""# Test"""

from IPython.display import clear_output
import textwrap

ppo_model = PPO.load('/content/PPO_llm.zip')

tickerSymbol = 'RELIANCE.NS'
ticker = yf.Ticker(tickerSymbol)

prices = []
actions = []  # List to store actions ('buy', 'sell', 'hold')
timestamps = []
size_length = 20

def update_plot():
    current_price = getPrice(tickerSymbol)
    prices.append(current_price)

    action, _states = ppo_model.predict(np.array(get_current_obs(tickerSymbol)).reshape(1, 13))
    actions.append(action)

    now = datetime.now()
    timestamps.append(now)

    if len(actions) > size_length:
        actions.pop(0)
        timestamps.pop(0)

    if len(prices) > size_length:
        prices.pop(0)

    clear_output(wait=True)
    plt.figure(figsize=(12, 6))

    # Formatting timestamps for plotting
    formatted_timestamps = [time.strftime('%H:%M:%S') for time in timestamps]

    plt.plot(formatted_timestamps, prices, label=tickerSymbol, marker='', linestyle='-', color='gray')

    for i, price in enumerate(prices):
        if actions[i] == 0:
            plt.plot(formatted_timestamps[i], price, 'go')  # Green dot for buy
        elif actions[i] == 'sell':
            plt.plot(formatted_timestamps[i], price, 'ro')  # Red dot for sell
        else:
            plt.plot(formatted_timestamps[i], price, 'bo')  # Blue dot for hold

    legend_handles = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='g', markersize=10, label='Buy'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='r', markersize=10, label='Sell'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='b', markersize=10, label='Hold')
    ]

    # Adjust x-axis to better display timestamps
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Time')
    plt.ylabel('Price')
    current_date = now.strftime('%Y-%m-%d')
    plt.title(f'Real-time Price Plot for {tickerSymbol} on {current_date}: {current_price}')
    plt.legend(handles=legend_handles)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')
    plt.tight_layout()  # Adjust layout to make room for the rotated x-axis labels

    news_str = '\n'.join(get_live_news(tickerSymbol,5))
    wrapped_text = textwrap.fill(news_str, width=150)  # Adjust `width` as needed based on your plot size and font size
    plt.figtext(0.5, -0.1, "Live news : " + wrapped_text, ha="center", fontsize=10, color="gray", wrap=True)

    stock_OHL = '\n\n'.join(get_current_OHLC(tickerSymbol))
    plt.figtext(1.05, 0.5, stock_OHL, ha="center", fontsize=13, color="black", wrap=True)

    plt.show()

try:
    while True:
        update_plot()
        time.sleep(1)  # Adjust the sleep time as needed
except KeyboardInterrupt:
    print("Stopped live plotting")
