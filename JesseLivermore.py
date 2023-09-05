import pandas as pd
import numpy as np
import yfinance as yf

def get_data(stock, start, end):
    data = yf.download(stock, start=start, end=end)
    return data

def livermore_trading(data, ma_period=50):
    close = data['Close']
    ma = close.rolling(window=ma_period).mean()
    
    # Entry signals
    data['long_entry'] = np.where(close > ma, 1, 0)
    data['short_entry'] = np.where(close < ma, -1, 0)
    
    # Exit signals
    data['long_exit'] = np.where(close < ma, -1, 0)
    data['short_exit'] = np.where(close > ma, 1, 0)
    
    # Positions
    data['positions'] = data['long_entry'] + data['short_entry'] + data['long_exit'] + data['short_exit']
    data['positions'] = data['positions'].cumsum().shift(1)
    
    return data

if __name__ == '__main__':
    stock = 'AAPL'
    start = '2018-01-01'
    end = '2021-12-31'

    data = get_data(stock, start, end)
    livermore_data = livermore_trading(data)
    print(livermore_data[['Close', 'positions']])
