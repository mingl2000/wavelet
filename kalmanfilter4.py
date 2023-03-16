import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pykalman import KalmanFilter
import yfinance as yf
import matplotlib.pyplot as plt
import mplfinance as mpf
from YahooData import *
import sys
import statsmodels.api as sm
import copy
def DoMarkovRegression(df, col1, col2):
  #build and train the MSDR model
  msdr_model = sm.tsa.MarkovRegression(endog=df[col1], k_regimes=2,
  trend='c', exog=df[col2], switching_variance=True)
  msdr_model_results = msdr_model.fit(iter=1000)

#print model training summary
  print(msdr_model_results.summary())
  return msdr_model_results

def getMaxMin(df):
    _max=max(df['High'].values)
    _min=min(df['Low'].values)
    return (_max, _min)

def dfplot(ticker, name, df,colnames1,colnames2):  
  figsize=(26,13)
  mc = mpf.make_marketcolors(
                            volume='lightgray'
                            )

                            
  s  = mpf.make_mpf_style(marketcolors=mc, gridaxis='both')
  apdict = []
  for colname in colnames1:
      apdict.append(mpf.make_addplot(df[colname], secondary_y=False,panel=0,width=3))
  for colname in colnames2:
      apdict.append(mpf.make_addplot(df[colname], secondary_y=False,panel=0,width=1))

  df['KFDiff']=df['Filtered']-df['Smoothed']
  df['KFDiff_close']=df['Close']-df['Smoothed']
  df['KFDiff_high']=df['High']-df['Smoothed']
  df['KFDiff_low']=df['Low']-df['Smoothed']
  msdr_model_results=DoMarkovRegression(df, 'KFDiff', 'KFDiff_close')
  (_max,_min)=getMaxMin(df)
  df['MarkovRegression00']=msdr_model_results.filtered_joint_probabilities[0][0]*(_max-_min) +_min
  df['MarkovRegression01']=msdr_model_results.filtered_joint_probabilities[0][1]*(_max-_min) +_min
  df['MarkovRegression10']=msdr_model_results.filtered_joint_probabilities[1][0]*(_max-_min) +_min
  df['MarkovRegression11']=msdr_model_results.filtered_joint_probabilities[1][1]*(_max-_min) +_min

  apdict.append(mpf.make_addplot(df['MarkovRegression00'], secondary_y=False,panel=0,width=3))
  #apdict.append(mpf.make_addplot(df['MarkovRegression01'], secondary_y=False,panel=0,width=2))
  #apdict.append(mpf.make_addplot(df['MarkovRegression10'], secondary_y=False,panel=0,width=3))
  apdict.append(mpf.make_addplot(df['MarkovRegression11'], secondary_y=False,panel=0,width=1))
  
  

  def getTitle(ticker, name):
    title="KalmanFilter&MarkovRegression-"+ticker
    if name is not None:
      title=title+"-" +name
    return title
  apdict_vol=copy.deepcopy(apdict)
  apdict_vol.append(mpf.make_addplot(df['Smoothed_vol'], secondary_y=False,panel=1,width=1))
  fig1,ax1=mpf.plot(df,type='candle',volume=True,volume_panel=1,addplot=apdict_vol, figsize=figsize,tight_layout=True,style=s,returnfig=True,block=False, title=ticker,panel_ratios=(3,1))
  fig1.suptitle(getTitle(ticker,name),fontsize=30)
  apdict.append(mpf.make_addplot(df['KFDiff'], secondary_y=False,panel=1,width=3))
  apdict.append(mpf.make_addplot(df['KFDiff_close'], secondary_y=False,panel=1,width=1))
  apdict.append(mpf.make_addplot(df['KFDiff_high'], secondary_y=False,panel=1,width=1))
  apdict.append(mpf.make_addplot(df['KFDiff_low'], secondary_y=False,panel=1,width=1))
  
  

  fig2,ax2=mpf.plot(df,type='candle',volume=False,volume_panel=1,addplot=apdict, figsize=figsize,tight_layout=True,style=s,returnfig=True,block=False, title=ticker)
  
  fig2.suptitle(getTitle(ticker,name),fontsize=30)

def KalmanFilterPlot(ticker, historylen, interval):
  data= GetYahooData_v2(ticker,historylen,interval)
  # Define Kalman filter parameters
  kf = KalmanFilter(
      initial_state_mean=data['Close'][0],
      initial_state_covariance=1,
      observation_covariance=1,
      transition_covariance=0.01
  )

  # Run Kalman filter on stock data
  state_means, _ = kf.filter(data['Close'].values)
  state_means_high, _ = kf.filter(data['High'].values)
  state_means_low, _ = kf.filter(data['Low'].values)
  state_means_vol, _ = kf.filter(data['Volume'].values)
  # Apply RTS smoothing algorithm
  state_means_smooth, _ = kf.smooth(data['Close'].values)
  state_means_smooth_high, _ = kf.smooth(data['High'].values)
  state_means_smooth_low, _ = kf.smooth(data['Low'].values)
  state_means_smooth_vol, _ = kf.smooth(data['Volume'].values)
  #plt.show()
  data['Filtered']=state_means
  data['Smoothed']=state_means_smooth
  data['Filtered_high']=state_means_high
  data['Smoothed_high']=state_means_smooth_high
  data['Filtered_low']=state_means_low
  data['Smoothed_low']=state_means_smooth_low

  data['Filtered_vol']=state_means_vol
  data['Smoothed_vol']=state_means_smooth_vol

  dfplot(ticker, None, data, ['Smoothed','Smoothed_high','Smoothed_low'],['Filtered','Filtered_high','Filtered_low'])
  plt.show()

# Load stock data
#data = pd.read_csv('./data/aapl_730d_1d.csv')
def main():
  drawchart=True
  historylen=512
  interval='1d'
  daysprint=89
  usecache=True
  daystoplot=512
  if len(sys.argv) <1:
    print("arguments are : Symbol historylen interval drawchart daysprint brick_size_in_ATR")
    print("interval can be 1m, 5m, 15m, 30m, 1h, 1d, 1wk, 1mo, 3mo")
    print("python .\mingwave3.py QQQ 256 1d True 20 True 128 0l5")
  if len(sys.argv) <2:
    ticker='SQQQ'
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
  
  KalmanFilterPlot(ticker, historylen, interval)
'''
ticker = 'AAPL'
start_date = '2010-01-01'
end_date = '2022-03-15'
data= yf.download(ticker, start=start_date, end=end_date)
'''

if __name__ == '__main__':
    main()
