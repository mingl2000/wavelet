import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

def get_data(stock, start_date, end_date):
    data = yf.download(stock, start=start_date, end=end_date)
    return data

def moving_average_crossover(data, short_window=50, long_window=200):
    data['Short_MA'] = data['Close'].rolling(window=short_window).mean()
    data['Long_MA'] = data['Close'].rolling(window=long_window).mean()

    data['Signal'] = 0
    data.loc[data['Short_MA'] > data['Long_MA'], 'Signal'] = 1
    data.loc[data['Short_MA'] < data['Long_MA'], 'Signal'] = -1

    return data

def backtest(data, initial_capital=100000):
    positions = pd.DataFrame(index=data.index).fillna(0)
    positions['Stock'] = data['Signal']
    portfolio = positions.multiply(data['Close'], axis=0)
    pos_diff = positions.diff()

    portfolio['Holdings'] = (positions.multiply(data['Close'], axis=0)).sum(axis=1)
    portfolio['Cash'] = initial_capital - (pos_diff.multiply(data['Close'], axis=0)).sum(axis=1).cumsum()
    portfolio['Total'] = portfolio['Cash'] + portfolio['Holdings']

    portfolio['Returns'] = portfolio['Total'].pct_change()

    return portfolio

def plot_equity_curve(portfolio):
    plt.figure(figsize=(12, 6))
    plt.plot(portfolio['Total'], label='Equity Curve', color='blue')
    plt.title('Simple Moving Average Crossover Strategy - Equity Curve')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value')
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    stock = 'AAPL'
    start_date = '2010-01-01'
    end_date = '2021-12-31'

    data = get_data(stock, start_date, end_date)
    data = moving_average_crossover(data, short_window=50, long_window=200)
    portfolio = backtest(data)
    plot_equity_curve(portfolio)

    total_profit_loss = (portfolio['Total'][-1] - portfolio['Total'][0]) / portfolio['Total'][0] * 100
    print(f"Total profit/loss: {total_profit_loss:.2f}%")
