import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pykalman import KalmanFilter
import yfinance as yf
import matplotlib.pyplot as plt
import mplfinance as mpf
from YahooData import *
import sys
def dfplot(ticker, name, df,colnames):  
  figsize=(26,13)
  mc = mpf.make_marketcolors(
                            volume='lightgray'
                            )

                            
  s  = mpf.make_mpf_style(marketcolors=mc, gridaxis='both')
  apdict = []
  for colname in colnames:
      apdict.append(mpf.make_addplot(df[colname], secondary_y=False,panel=0,width=3))
  df['KFDiff_close']=df['Close_Filtered']-df['Close_Smoothed']
  if 'vwap' in df.columns:
    df['KFDiff_vwap']=df['vwap_Filtered']-df['vwap_Smoothed']
  

  def getTitle(ticker, name):
    title="KalmanFilter-"+ticker
    if name is not None:
      title=title+"-" +name
    return title
  fig1,ax1=mpf.plot(df,type='candle',volume=True,volume_panel=1,addplot=apdict, figsize=figsize,tight_layout=True,style=s,returnfig=True,block=False, title=ticker,panel_ratios=(3,1))
  fig1.suptitle(getTitle(ticker,name),fontsize=30)
  apdict.append(mpf.make_addplot(df['KFDiff_close'], secondary_y=False,panel=1,width=3))
  if 'vwap' in df.columns:
    apdict.append(mpf.make_addplot(df['KFDiff_vwap'], secondary_y=False,panel=1,width=3))
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

  # Apply RTS smoothing algorithm
  state_means_smooth, _ = kf.smooth(data['Close'].values)

  #plt.show()
  data['Filtered']=state_means
  data['Smoothed']=state_means_smooth
  dfplot(ticker, None, data, ['Filtered','Smoothed'])
  print (state_means_smooth[-1]-state_means[-1])
  plt.show()

def calculateKalmanFilter(df, colname):
  kf = KalmanFilter(
      initial_state_mean=df[colname][0],
      initial_state_covariance=1,
      observation_covariance=1,
      transition_covariance=0.01
  )

  # Run Kalman filter on stock data
  state_means, _ = kf.filter(df[colname].values)

  # Apply RTS smoothing algorithm
  state_means_smooth, _ = kf.smooth(df[colname].values)

  #plt.show()
  df[colname+'_Filtered']=state_means
  df[colname+'_Smoothed']=state_means_smooth
  return df


def KalmanFilterPlot2(ticker,df):
  #data= GetYahooData_v2(ticker,historylen,interval)
  # Define Kalman filter parameters
  df=calculateKalmanFilter(df,'Close')
  if 'vwap' in df.columns:
    df=calculateKalmanFilter(df,'vwap')
    dfplot(ticker, None, df, ['Close_Filtered','Close_Smoothed','vwap_Filtered','vwap_Smoothed'])
  else:
    dfplot(ticker, None, df, ['Close_Filtered','Close_Smoothed'])
  #print (state_means_smooth[-1]-state_means_close[-1])
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
  #KalmanFilterPlot2(ticker,df)
'''
ticker = 'AAPL'
start_date = '2010-01-01'
end_date = '2022-03-15'
data= yf.download(ticker, start=start_date, end=end_date)
'''

if __name__ == '__main__':
    main()
