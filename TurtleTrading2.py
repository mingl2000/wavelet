import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import mplfinance as mpf
def get_data(stock, start, end):
    data = yf.download(stock, start=start, end=end)
    #data = yf.download(stock, period=1000, interval='h')
    return data

def turtle_trading(data, window_short=20, window_long=55, allowshort=False):
    highs = data['High']
    lows = data['Low']
    close = data['Close']
    
    # Entry signals
    data['breakout_high'] = highs.rolling(window=window_short).max()
    data['breakout_low'] = lows.rolling(window=window_short).min()
    data['long_entry'] = np.where(data['High'] > data['breakout_high'].shift(1), 1, 0)
    if allowshort:
        data['short_entry'] = np.where(data['Low'] < data['breakout_low'].shift(1), -1, 0)
    
    # Exit signals
    data['exit_long'] = close.rolling(window=window_long).min()
    data['exit_short'] = close.rolling(window=window_long).max()
    data['long_exit'] = np.where(data['Low'] < data['exit_long'].shift(1), -1, 0)
    if allowshort:
        data['short_exit'] = np.where(data['High'] > data['exit_short'].shift(1), 1, 0)
    
    # Positions
    if allowshort:
        data['positions'] = data['long_entry'] + data['short_entry'] + data['long_exit'] + data['short_exit']
    else:
        data['positions'] = data['long_entry'] + data['long_exit'] 
    data['positions'] = data['positions'].cumsum().shift(1)
    
    # Profit and loss calculation
    data['daily_returns'] = data['Close'].pct_change()
    data['strategy_returns'] = data['daily_returns'] * data['positions']
    data['strategy_returns_sum']=(data['strategy_returns']).cumsum()
    data['cumulative_returns'] = (1 + data['strategy_returns']).cumprod()
    
    return data

import sys
import os
if __name__ == '__main__':
    if len(sys.argv) >1:
        stocks=sys.argv[1]
        
    stocks = '002049.sz'
    start = '2000-01-01'
    end = '2023-04-30'

    stocks=stocks.split(',')
    for stock in stocks:
        
        data = get_data(stock, start, end)
        allowshort=True
        if stock.lower().endswith(('.sz','.ss')):
            allowshort=False
        print(stock, 'allowshort=',allowshort)
        turtle_data = turtle_trading(data,allowshort=False)
        print(turtle_data[['Close', 'positions', 'daily_returns', 'strategy_returns', 'cumulative_returns']])
        plt.plot(turtle_data['strategy_returns_sum'])
        plt.title(stock)
        apdict = []
        apdict.append(mpf.make_addplot(turtle_data['positions'], panel=0, width=1,secondary_y=False))
        apdict.append(mpf.make_addplot(turtle_data['strategy_returns_sum'], panel=0, width=2,secondary_y=True, color='m'))
        mpf.plot(data,addplot=apdict)
    plt.show()