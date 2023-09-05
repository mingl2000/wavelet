import pandas as pd
import numpy as np
import yfinance as yf

# Fetch historical data
def get_data(stock, start_date, end_date):
    data = yf.download(stock, start=start_date, end=end_date)
    return data

# Calculate pivot point
def pivot_point(data):
    data['Pivot'] = (data['High'] + data['Low'] + data['Close']) / 3
    return data

# Generate buy and sell signals
import math
def generate_signals(data):
    position = None
    buy_signals = [math.nan]
    sell_signals = [math.nan]
    entry_price = 0
    profit_loss = [math.nan]
    
    for i in range(1, len(data)):
        # Buy signal
        if data['Close'][i] > data['Pivot'][i-1] and position != 'long':
            buy_signals.append(data['Close'][i])
            sell_signals.append(np.nan)
            entry_price = data['Close'][i]
            position = 'long'
            
        # Sell signal
        elif data['Close'][i] < data['Pivot'][i-1] and position == 'long':
            sell_signals.append(data['Close'][i])
            buy_signals.append(np.nan)
            profit_loss.append(data['Close'][i] - entry_price)
            position = None
        # No signal
        else:
            buy_signals.append(np.nan)
            sell_signals.append(np.nan)
            profit_loss.append(0)

    return buy_signals, sell_signals, profit_loss

# Main function
import matplotlib.pyplot as plt

if __name__ == "__main__":
    stock = 'AAPL'
    start_date = '2021-01-01'
    end_date = '2021-12-31'

    data = get_data(stock, start_date, end_date)
    data = pivot_point(data)

    buy_signals, sell_signals, profit_loss = generate_signals(data)

    data['Buy_Signal'] = buy_signals
    data['Sell_Signal'] = sell_signals
    #data['Profit_Loss'] = profit_loss

    print(data)

    profit_loss_curve=(np.array(profit_loss)).cumsum()
    total_profit_loss = sum(profit_loss)
    print(f'Total Profit/Loss: {total_profit_loss}')
    plt.plot(profit_loss_curve)
    plt.show()
