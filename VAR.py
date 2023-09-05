import pandas as pd
import statsmodels.api as sm
from YahooData import *
import sys
import warnings
import matplotlib.pyplot as plt
if __name__ == '__main__':

    interval='1d'
    symbol='QQQ'
    if len(sys.argv) >=2:
        symbol=sys.argv[1]
    if len(sys.argv) >=3:
        interval=sys.argv[2]

    if symbol=='SPX':
        symbol='^GSPC'

    bars=730
    #symbol='^GSPC'
    df=GetYahooData_v2(symbol, bars=bars, interval=interval)
    # Create a cerebro entity

# Load data
    #data = pd.read_csv('data.csv', index_col='Date', parse_dates=True)
    #data=df["Close"]
    data=df
# Create VAR model
    model = sm.tsa.VAR(data)

# Fit VAR model
    results = model.fit()

# Get impulse response functions
    irf = results.irf()

# Plot impulse responses
    irf.plot()

# Get forecast for next 10 periods
    forecast = results.forecast(data.values, 10)
    
    plt.plot(forecast)
# Plot forecast
    #forecast.plot()
    plt.show()