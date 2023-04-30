import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
def get_data(stock, start, end):
    data = yf.download(stock, start=start, end=end)
    return data

def turtle_trading(data, window_short=20, window_long=55):
    highs = data['High']
    lows = data['Low']
    close = data['Close']
    
    # Entry signals
    data['breakout_high'] = highs.rolling(window=window_short).max()
    data['breakout_low'] = lows.rolling(window=window_short).min()
    data['long_entry'] = np.where(data['High'] > data['breakout_high'].shift(1), 1, 0)
    data['short_entry'] = np.where(data['Low'] < data['breakout_low'].shift(1), -1, 0)
    
    # Exit signals
    data['exit_long'] = close.rolling(window=window_long).min()
    data['exit_short'] = close.rolling(window=window_long).max()
    data['long_exit'] = np.where(data['Low'] < data['exit_long'].shift(1), -1, 0)
    data['short_exit'] = np.where(data['High'] > data['exit_short'].shift(1), 1, 0)
    
    # Positions
    data['positions'] = data['long_entry'] + data['short_entry'] + data['long_exit'] + data['short_exit']
    data['positions'] = data['positions'].cumsum().shift(1)
    
    # Profit and loss calculation
    data['daily_returns'] = data['Close'].pct_change()
    data['strategy_returns'] = data['daily_returns'] * data['positions']
    data['strategy_returns_sum']=(data['strategy_returns']).cumsum()
    data['cumulative_returns'] = (1 + data['strategy_returns']).cumprod()
    
    return data

if __name__ == '__main__':
    stock = '002049.sz'
    start = '2000-01-01'
    end = '2023-04-30'

    data = get_data(stock, start, end)
    turtle_data = turtle_trading(data)
    print(turtle_data[['Close', 'positions', 'daily_returns', 'strategy_returns', 'cumulative_returns']])
    plt.plot(turtle_data['strategy_returns_sum'])
    plt.title(stock)
    plt.show()