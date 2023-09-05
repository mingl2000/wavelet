import pandas as pd
import numpy as np
import yfinance as yf

def get_data(stock, start_date, end_date):
    data = yf.download(stock, start=start_date, end=end_date)
    return data

def average_true_range(data, atr_window=14):
    high_low = data['High'] - data['Low']
    high_close = np.abs(data['High'] - data['Close'].shift())
    low_close = np.abs(data['Low'] - data['Close'].shift())
    true_ranges = pd.DataFrame({'HL': high_low, 'HC': high_close, 'LC': low_close})
    tr = true_ranges.max(axis=1)
    atr = tr.rolling(window=atr_window).mean()
    return atr

def turtle_trading(data, entry_window=20, exit_window=10, atr_window=14, risk_per_trade=0.01, account_size=10000):
    data['ATR'] = average_true_range(data, atr_window)

    data['High_Entry'] = data['High'].rolling(window=entry_window).max().shift(1)
    data['Low_Entry'] = data['Low'].rolling(window=entry_window).min().shift(1)
    data['High_Exit'] = data['High'].rolling(window=exit_window).max().shift(1)
    data['Low_Exit'] = data['Low'].rolling(window=exit_window).min().shift(1)

    data['Long_Entry'] = data['High'] > data['High_Entry']
    data['Short_Entry'] = data['Low'] < data['Low_Entry']
    data['Long_Exit'] = data['Low'] < data['Low_Exit']
    data['Short_Exit'] = data['High'] > data['High_Exit']

    data['Units'] = np.nan
    data['Position'] = np.nan
    position = 0
    trade_prices = []

    profit_loss = []

    for i in range(1, len(data)):
        if data['Long_Entry'][i] and position == 0:
            risk = risk_per_trade * account_size
            units = risk // (data['ATR'][i] * 1)
            position = units
            trade_prices.append(data['Close'][i])

        elif data['Short_Entry'][i] and position == 0:
            risk = risk_per_trade * account_size
            units = risk // (data['ATR'][i] * 1)
            position = -units
            trade_prices.append(data['Close'][i])

        elif data['Long_Exit'][i] and position > 0:
            exit_price = data['Close'][i]
            entry_price = trade_prices.pop(0)
            profit = (exit_price - entry_price) * position
            profit_loss.append(profit)
            position = 0

        elif data['Short_Exit'][i] and position < 0:
            exit_price = data['Close'][i]
            entry_price = trade_prices.pop(0)
            profit = (entry_price - exit_price) * abs(position)
            profit_loss.append(profit)
            position = 0

        data.loc[data.index[i], 'Units'] = position
        data.loc[data.index[i], 'Position'] = position * data.loc[data.index[i], 'Close']

    data['Position'].fillna(method='ffill', inplace=True)
    data['Units'].fillna(method='ffill', inplace=True)

    data['Daily_Return'] = data['Close'].pct_change()
   
if __name__ == "__main__":
    stock = 'AAPL'
    start_date = '2021-01-01'
    end_date = '2021-12-31'

    data = get_data(stock, start_date, end_date)
    data = turtle_trading(data, entry_window=20, exit_window=10, atr_window=14, risk_per_trade=0)