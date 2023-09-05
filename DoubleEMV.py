import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from YahooData import *
# Load or create your time series data (replace with your data)
df=GetYahooData_v2('QQQ',500,'1d')
'''
data = [123, 119, 135, 150, 169, 178, 193, 210, 225, 238]
data=df['Close']
index = pd.date_range(start='2020-01-01', periods=len(data), freq='M')
time_series = pd.Series(data, index=index)
'''
data=df['Close'].to_numpy()
index = pd.date_range(start='2020-01-01', periods=len(data), freq='D')
time_series = pd.Series(data, index=index)
#time_series=data=df['Close'][-21:]

# Fit the Double Exponential Smoothing model
model = ExponentialSmoothing(time_series, trend='add', seasonal=None)
fit = model.fit(smoothing_level=0.8, smoothing_slope=0.2)

#Fit the Triple Exponential Smoothing model
seasonal_period = 5
model = ExponentialSmoothing(time_series, trend='add', seasonal='add', seasonal_periods=seasonal_period)
fit = model.fit()

# Perform forecasting
forecast = fit.forecast(steps=3)

# Plot the original data, the fitted model, and the forecast
plt.figure(figsize=(12, 6))
plt.plot(time_series, label='Original Data')
plt.plot(fit.fittedvalues, label='Fitted Model')
plt.plot(forecast, label='Forecast')
plt.legend(loc='upper left')
plt.xlabel('Date')
plt.ylabel('Value')
plt.title('Double Exponential Smoothing (Holt\'s Linear Trend)')
plt.show()
