"""
==========================
Singular Spectrum Analysis
==========================

Signals such as time series can be seen as a sum of different signals such
as trends and noise. Decomposing time series into several time series can
be useful in order to keep the most important information. One decomposition
algorithm is Singular Spectrum Analysis. This example illustrates the
decomposition of a time series into several subseries using this algorithm and
visualizes the different subseries extracted.
It is implemented as :class:`pyts.decomposition.SingularSpectrumAnalysis`.
"""

# Author: Johann Faouzi <johann.faouzi@gmail.com>
# License: BSD-3-Clause
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
#warnings.simplefilter(action='ignore', category=ValueWarning)

import numpy as np
import matplotlib.pyplot as plt
from pyts.decomposition import SingularSpectrumAnalysis
from numpy import pi
import datetime
from datetime import timedelta
from dateutil import parser
import pandas_datareader as pdr
import pandas as pd
import mplfinance as mpf
import yfinance as yf

from os.path import exists
import yfinance as yf
def GetYahooData(symbol, bars=500, interval='1d'):
  #start=datetime.date.today()-datetime.timedelta(days=days)
  #end=datetime.date.today()
  if symbol=='SPX':
    symbol='^GSPC'
  #df=.gepdrt_data_yahoo(symbols=symbol,  start=start, end=end,interval=interval)
  
  #if interval.endswith('m') or interval.endswith('h'):
  #  period='max'
  
  if interval.endswith('1m'):
    period='7d'
  elif  interval.endswith('m'):
    period='60d'
  elif  interval.endswith('h'):
    period='730d'
  else:
    period='max'
  
  #elif interval.endswith('d'):
    #period=str(days)+'d'
  #  period='max'
  #elif  interval.endswith('w'):
  #  period=str(days)+'wk'
  
  dataFileName="data/"+symbol+'_' +period+'_'+ interval +".csv"
  if interval.endswith(('d','D')) and datetime.datetime.now().hour>=13 and exists(dataFileName):
    print('read yahoo data from cache')
    df=pd.read_csv(dataFileName, header=0, index_col=0, encoding='utf-8', parse_dates=True)
    #df.index=df["Date"]
  else:
    print('read yahoo data from web')
    df = yf.download(tickers=symbol, period=period, interval=interval)
    df.to_csv(dataFileName, index=True, date_format='%Y-%m-%d %H:%M:%S')
  #dataFileName="data/"+symbol+".csv"
  
  #df = pd.read_csv(dataFileName,index_col=0,parse_dates=True)
  #df.shape
  df.dropna(inplace = True)
  df =df [-bars:]
  df.head(3)
  df.tail(3)
  df["id"]=np.arange(len(df))
  #df["date1"]=df.index.astype(str)
  #df["datefmt"]=df.index.strftime('%m/%d/%Y')
  
  return df




#data = quandl.get('WIKI/%s' % instrument, start_date='2017-01-01', end_date='2012-02-10')
def calc_ssa(df,colname,window_size=20):
  X=[]
  X.append(df[colname])
  # We decompose the time series into three subseries

  #groups = [np.arange(i, i + 5) for i in range(0, 11, 5)]

  # Singular Spectrum Analysis
  ssa = SingularSpectrumAnalysis(window_size=window_size, groups=None)
  X_ssa = ssa.fit_transform(X)
  for i in range(window_size):
    ssa_col=colname +'_ssa_'+str(i)
    df[ssa_col]=X_ssa[i]
  print(X_ssa)
  return df

def plot_ssa(symbol,df, window_size, X_ssa,dates,predict):
  '''
  N=20
  t = np.arange(0,N)

  trend = 0.001 * (t - 100)**2

  p1, p2 = 20, 30

  periodic1 = 2 * np.sin(2*pi*t/p1)
  periodic2 = 0.75 * np.sin(2*pi*t/p2)

  np.random.seed(123) # So we generate the same noisy time series every time.
  noise = 2 * (np.random.rand(N) - 0.5)
  F = trend + periodic1 + periodic2 + noise

  # Parameters
  n_samples, n_timestamps = 1000, 10

  # Toy dataset
  rng = np.random.RandomState(41)
  X = rng.randn(n_samples, n_timestamps)
  '''
  # Show the results for the first time series and its subseries
  '''
  plt.figure(figsize=(16, 6))

  ax1 = plt.subplot(211)
  ax1.plot(X_ssa[0],  label='X_ssa[0]')
  ax1.plot(closes.to_numpy(), 'o-', label='Original')
  ax1.legend(loc='best', fontsize=14)

  ax2 = plt.subplot(212)
  for i in range(1, window_size):
      ax2.plot(X_ssa[i], 'o--', label='SSA {0}'.format(i + 1))
  ax2.legend(loc='best', fontsize=14)

  plt.suptitle('Singular Spectrum Analysis', fontsize=20)

  plt.tight_layout()
  plt.subplots_adjust(top=0.88)
  plt.show()
  '''

  import mplfinance as mpf
  figsize=(26,13)
  mc = mpf.make_marketcolors(
                            volume='lightgray'
                            )

                            
  s  = mpf.make_mpf_style(marketcolors=mc, gridaxis='both')
  apdict = []

  apdict.append(mpf.make_addplot((dates,predict), panel=0, width=3,secondary_y=False, color='m'))
  for i in range(0, int(window_size/2)):
      newcol='ssa_' +str(i)
      df[newcol]=X_ssa[i]
      if i==0:
        apdict.append(mpf.make_addplot(df[newcol], panel=0, width=3,secondary_y=False))
      else:
        if i>=4:
          width=1
        else:
          width=5-i
        apdict.append(mpf.make_addplot(df[newcol], panel=1, width=width,secondary_y=False))


  fig1,ax1=mpf.plot(df,type='candle',volume=False,volume_panel=2,addplot=apdict, figsize=figsize,tight_layout=True,style=s,returnfig=True,block=False)
  fig1.suptitle(symbol,fontsize=30)

'''
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller
#from statsmodels.tsa.arima_model import ARIMA
#from statsmodels.tsa.arima.model.ARIMA import ARIMA
from statsmodels.tsa.arima.model import ARIMA
import pmdarima as pm
#from statsmodels.tsa.arima.model import plot_predict
from statsmodels.graphics.tsaplots import plot_predict
'''
from statsmodels.tsa.arima.model import ARIMA
from pandas.tseries.offsets import DateOffset
def arimar_predit(df, colname,numberofssacomps, outcol, days2predict=5):
  print('')
  
  for ssa_comp_id in range(numberofssacomps):
    #incol='ssa_'+str(ssa_comp_id)
    ssa_col=colname +'_ssa_'+str(ssa_comp_id)
    arima_model=ARIMA(df[ssa_col], order=(1,1,1))
    model=arima_model.fit()
    forecast_object = model.get_forecast(days2predict)
    mean=forecast_object.predicted_mean
    
    #print(ssa_comp_id, mean)
    
    
    conf_int = forecast_object.conf_int()
    
    if ssa_comp_id==0:
      meantotal = mean
      #dates=[df.index[-1]+ DateOffset(days=x)for x in range(0,days2predict)]
    else:
      meantotal=mean+meantotal
      
      #print(ssa_comp_id, meantotal)


  print('predictions -------------------')
  def nextworkday(dt):
    if dt.date().weekday()<4:
        days2add=1
    elif dt.date().weekday()==4: # friday
        days2add=3
    elif dt.date().weekday()==5: # sat
        days2add=2
    else:
        days2add=1

    return dt+datetime.timedelta(days=days2add)

  #df=df.append(meantotal)
  #return df
  df['predict']=None
  for i in range(len(meantotal)):
    dt=nextworkday(df.index[-1])
    predictprice=round(meantotal[len(df)],2)
    df=df.append(pd.DataFrame(index=[dt]))
    #df = pd.concat(df,pd.DataFrame(index=[dt])) 
    df.loc[df.index[-1],'predict']=predictprice
    print(dt.date(),predictprice)

  #newdf=pd.DataFrame(columns=['date',''])
  #df.append(pd.DataFrame(index=[dates]))
  
  return df




# Extract the forecast dates

# Calculate the confidence intervals

  #pass

import sys
if len(sys.argv) <2:
  #symbols='QQQ,SPX,000001.ss,399001.sz'
  symbols='QQQ'
if len(sys.argv) >=2:
  symbols=sys.argv[1]

for symbol in symbols.split(','):
  df=GetYahooData(symbol, bars=500, interval='1d')
  #closes = df['Adj Close'].rename('close')
  window_size=20
  colname='Close'
  df=calc_ssa(df,colname,window_size)

  
  df=arimar_predit(df,colname, 3,'predict',5)
  #plot_ssa(symbol,df, window_size, X_ssa, dates,predict)
# The first subseries consists of the trend of the original time series.
# The second and third subseries consist of noise.
plt.show()
