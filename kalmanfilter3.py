import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pykalman import KalmanFilter
import yfinance as yf
import matplotlib.pyplot as plt
import mplfinance as mpf
from YahooData import *
import sys
def dfplot(df,colnames):  
  figsize=(26,13)
  mc = mpf.make_marketcolors(
                            volume='lightgray'
                            )

                            
  s  = mpf.make_mpf_style(marketcolors=mc, gridaxis='both')
  apdict = []
  for colname in colnames:
      apdict.append(mpf.make_addplot(df[colname], secondary_y=False,panel=0))

  fig1,ax1=mpf.plot(df,type='candle',volume=True,volume_panel=1,addplot=apdict, figsize=figsize,tight_layout=True,style=s,returnfig=True,block=False, title=ticker,panel_ratios=(3,1))

# Load stock data
#data = pd.read_csv('./data/aapl_730d_1d.csv')

drawchart=True
historylen=512
interval='1wk'
daysprint=89
usecache=True
daystoplot=512
if len(sys.argv) <1:
  print("arguments are : Symbol historylen interval drawchart daysprint brick_size_in_ATR")
  print("interval can be 1m, 5m, 15m, 30m, 1h, 1d, 1wk, 1mo, 3mo")
  print("python .\mingwave3.py QQQ 256 1d True 20 True 128 0l5")
if len(sys.argv) <2:
  ticker='QQQ'
if len(sys.argv) >=2:
  ticker=sys.argv[1]
if len(sys.argv) >=3:
  historylen=int(sys.argv[2])
if len(sys.argv) >=4:
  interval=sys.argv[3]
if len(sys.argv) >=5:
  drawchart=sys.argv[4].lower()=='true'
if len(sys.argv) >=6:
  daysprint=int(sys.argv[5])
if len(sys.argv) >=7:
  usecache=sys.argv[6].lower()=='true'
if len(sys.argv) >=8:
  daystoplot=int(sys.argv[7])
if len(sys.argv) >=9:
  brick_size=float(sys.argv[8])
  #exit()


#ticker="SPX"
data= GetYahooData_v2(ticker,historylen,interval)
'''
ticker = 'AAPL'
start_date = '2010-01-01'
end_date = '2022-03-15'
data= yf.download(ticker, start=start_date, end=end_date)
'''

# Define Kalman filter parameters
kf = KalmanFilter(
    initial_state_mean=data['Close'][0],
    initial_state_covariance=1,
    observation_covariance=1,
    transition_covariance=0.01
)

# Run Kalman filter on stock data
state_means, _ = kf.filter(data['Close'].values)

# Apply RTS smoothing algorithm
state_means_smooth, _ = kf.smooth(data['Close'].values)

# Plot results
plt.plot(data.index, data['Close'], label='Observed')
plt.plot(data.index, state_means, label='Filtered')
#plt.plot(data['date'], state_means_smooth, label='Smoothed')
plt.plot(data.index, state_means_smooth, label='Smoothed')
plt.legend()
#plt.show()
data['Filtered']=state_means
data['Smoothed']=state_means_smooth
dfplot(data, ['Filtered','Smoothed'])
plt.show()
