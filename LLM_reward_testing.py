# %%
from openai import OpenAI
import yfinance as yf
import numpy as np
import pandas as pd
import gym
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
from IPython.display import clear_output
import textwrap
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import warnings
import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objs as go
import numpy as np
import pandas as pd
import json
warnings.filterwarnings("ignore", category=DeprecationWarning)


# %%
sentiment_tokenizer = AutoTokenizer.from_pretrained("mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")
sentiment_model = AutoModelForSequenceClassification.from_pretrained("mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")

if torch.cuda.is_available():
    sentiment_model = sentiment_model.cuda()

# %%
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

  return "Open : "+str(open_price) + ", High : " + str(dayHigh) + ", Low : " + str(dayLow)

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


def append_feedback_to_file(feedback_data, filepath='feedback.json'):
    try:
        # Read the existing data
        with open(filepath, 'r') as file:
            data = json.load(file)
    except FileNotFoundError:
        # If the file does not exist, start with an empty list
        data = []
    except json.JSONDecodeError:
        # If the file is empty or corrupted, start with an empty list
        data = []
    
    # Append the new feedback
    data.append(feedback_data)
    
    # Write the updated data back to the file
    with open(filepath, 'w') as file:
        json.dump(data, file, indent=4)

def get_updated_portfolio(action, num_stocks, current_price, optional_inhand):
   if action == 0 and optional_inhand > current_price:
      num_stocks + 1
      optional_inhand = optional_inhand - current_price
   elif action == 1 and num_stocks !=0:
      num_stocks - 1
      optional_inhand = optional_inhand + current_price

   return (num_stocks, optional_inhand + num_stocks*current_price, optional_inhand)


# %%
ppo_model = PPO.load('PPO_llm.zip')
tickerSymbol = 'AMZN'  
# Initialize the Dash app
app = dash.Dash(__name__)

app.layout = html.Div([
    html.Link(
        rel='stylesheet',
        href='https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.1/css/all.min.css'
    ),
    dcc.Graph(id="stock-graph"),
    html.Div([
        html.Button(html.I(className="fas fa-thumbs-up"), id="good-btn", n_clicks=0,style={"display": "none",}),
        html.Button(html.I(className="fas fa-thumbs-down"), id="bad-btn", n_clicks=0,style={"display": "none",}),
    ], style={"background-color": "white", "text-align": "center", "display": "flex", "justify-content": "center", "gap": "10px"}),  # Use flex display for buttons
    dcc.Interval(id='update-interval', interval=5000, n_intervals=0),
    dcc.Store(id="stock-data", data={"prices": [], "timestamps": [], "actions": []}),
    html.Div(id="news-container", children=[
        html.H4("Live News"),
        html.Ol(id="news-list")
    ], style={"background-color": "white", "padding": "10px"}),
])

action = None
num_stocks = 0
# in_hand = 1000
# portfolio_value = (0,0,0,in_hand)
@app.callback(
    Output("stock-data", "data"),
    Input("update-interval", "n_intervals"),
    State("stock-data", "data")
)
def update_data(n_intervals, data):
    global action, num_stocks, portfolio_value
    price = getPrice(tickerSymbol)
    observation = get_current_obs(tickerSymbol)
    action, _states = ppo_model.predict(np.array(observation).reshape(1, 13))

    if action == 0:
        num_stocks +=1
    elif action == 1 and num_stocks > 0:
        num_stocks -=1
    # Append the new price, timestamp, and action to the respective lists in the data dictionary
    data["prices"].append(price)
    data["timestamps"].append(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    data["actions"].append(action)
    
    # portfolio_value = get_updated_portfolio(action, num_stocks, price, in_hand)
    # in_hand = portfolio_value[3]
    
    # Keep only the latest 20 data points
    data["prices"] = data["prices"][-20:]
    data["timestamps"] = data["timestamps"][-20:]
    data["actions"] = data["actions"][-20:]

    return data

@app.callback(
    Output("stock-graph", "figure"),
    Input("stock-data", "data")
)
def update_graph(data):
    fig = go.Figure()
    if data["prices"]:
        fig.add_trace(go.Scatter(x=data["timestamps"], y=data["prices"], mode="lines+markers",
                                 marker=dict(color=['green' if action == 0 else 'red' if action == 1 else 'blue' for action in data["actions"]], size=12)))
    fig.update_layout(title="TICKER : " + tickerSymbol, xaxis_title="Time", yaxis_title="Price",
                      annotations=[dict(
                          xref='paper', yref='paper', x=0.5, y=1.1,
                          xanchor='center', yanchor='top',
                          text = get_current_OHLC(tickerSymbol) + ", Current : " + str(getPrice(tickerSymbol)) + "      Portfolio value :",
                          font=dict(family='Arial', size=16, color='grey'),
                          showarrow=False)]
                      )
    return fig


@app.callback(
    [Output("good-btn", "style"), Output("bad-btn", "style")],  # Outputs remain the same
    [Input("stock-graph", "clickData"),  # Triggered by clicking a point on the graph
     Input("good-btn", "n_clicks"),  # Triggered by clicking the "Good" button
     Input("bad-btn", "n_clicks")],  # Triggered by clicking the "Bad" button
    [State("good-btn", "n_clicks"), State("bad-btn", "n_clicks")]  # Current state of button clicks
)


def handle_feedback_buttons(clickData, good_clicks, bad_clicks, state_good_clicks, state_bad_clicks):
    
    ctx = dash.callback_context
    if not ctx.triggered:
        # Prevent update if there's no interaction
        raise dash.exceptions.PreventUpdate
    # Define the base styles for the buttons
    good_button_style = {
        "margin": "10px",
        "color": "white",
        "background-color": "green",
        "border": "none",
        "padding": "10px 20px",
        "border-radius": "5px",
        "cursor": "pointer",
        "font-size": "16px",
        "display": "none"  
    }

    bad_button_style = {
        "margin": "10px",
        "color": "white",
        "background-color": "red",
        "border": "none",
        "padding": "10px 20px",
        "border-radius": "5px",
        "cursor": "pointer",
        "font-size": "16px",
        "display": "none"  
    }
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if button_id == "stock-graph":
        # Show buttons if a point on the graph was clicked
        good_button_style["display"] = "block"
        bad_button_style["display"] = "block"
        
    elif button_id in ["good-btn", "bad-btn"]:
        # Logic to append feedback
        feedback_type = "good" if button_id == "good-btn" else "bad"
        #store the timestamp or some other identifier from the clickData
        if clickData:
            clicked_point_time = clickData['points'][0]['x']
            feedback_entry = {
                    "timestamp": clicked_point_time,
                    "observation": get_current_obs(tickerSymbol),
                    "feedback": feedback_type
                }
        append_feedback_to_file(feedback_entry)
        # Hide buttons after clicking either "Good" or "Bad"
    else:
        # Default case to hide buttons
        pass
    return good_button_style, bad_button_style

@app.callback(
    Output("news-list", "children"),
    [Input("stock-graph", "figure")]  #news updates when the graph is rendered/updated
)
def update_news(_): 
    topk = 5  
    news_items = get_live_news(tickerSymbol, topk)
    
    # Create a list of html.Li elements for each news item
    news_list = [html.Li(news) for news in news_items]
    return news_list


if __name__ == '__main__':
    app.run_server(debug=True)



