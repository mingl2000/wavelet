#!/usr/bin/env python
# coding: utf-8

# # Stock Index Prediction Based on Wavelet Transformation and ARMA-ML

# - [Wavelet Transformation](#Wavelet-Transformation)
# - [ARMA Model](#ARMA-Model)
# - [ARMA_GBR/SVR](#ARMA_GBR/SVR)
#   - [GBR](#GBR)
#   - [SVR](#SVR)
#   - [GBR+SVR](#GBR+SVR)  
# - [Conclusion](#Conclusion)

# In[1]:

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
import statsmodels.api as sm
import pywt
import copy
import warnings
from statsmodels.tsa.arima_model import ARIMA
#from statsmodels.tsa.arima_model import ARMA 
from statsmodels.tsa.stattools import adfuller
from sklearn.ensemble import GradientBoostingRegressor as GBR
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score
from model import WT, AR_MA, NonlinReg, ModelEvaluation
import datetime
import pandas_datareader as pdr
import pandas as pd
import mplfinance as mpf
import yfinance as yf
from reliability.Other_functions import crosshairs
from matplotlib.widgets import MultiCursor
from os.path import exists
# df = 'ohlc dataframe'

# In[2]:


from IPython.display import Image
#get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:
def GetYahooData(symbol, days=500, interval='1d'):
  start=datetime.date.today()-datetime.timedelta(days=days)
  end=datetime.date.today()
  if symbol=='SPX':
    symbol='^GSPC'
  #df=.gepdrt_data_yahoo(symbols=symbol,  start=start, end=end,interval=interval)
  
  #if interval.endswith('m') or interval.endswith('h'):
  #  period='max'
  if interval.endswith('1m'):
    period='7d'
  elif interval.endswith('d'):
    period=str(days)+'d'
  elif  interval.endswith('w'):
    period=str(days)+'wk'
  elif  interval.endswith('h'):
    period=str(days) +'h'
  elif  interval.endswith('m'):
    period='60d'
  else:
    period='max'
  dataFileName="data/"+symbol+'_' + interval +".csv"
  if exists(dataFileName):
    print('read yahoo data from cache')
    df=pd.read_csv(dataFileName, header=0, index_col=0, encoding='utf-8', parse_dates=True)
  else:
    print('read yahoo data from web')
    df = yf.download(tickers=symbol, period=period, interval=interval)
    df.to_csv(dataFileName)
  #dataFileName="data/"+symbol+".csv"
  
  #df = pd.read_csv(dataFileName,index_col=0,parse_dates=True)
  #df.shape
  df.head(3)
  df.tail(3)
  df["id"]=np.arange(len(df))
  #df["date1"]=df.index.astype(str)
  #df["datefmt"]=df.index.strftime('%m/%d/%Y')
  return df

def multi_plot_wt(df, wavlet_close, wavlet_high, wavlet_low):
    plt.margins(0.1)
    fig, ax =  plt.subplots(len(wavlet_close), 1, figsize=figsize)
    for i in range(len(wf_close)):
        if i == 0:
            ax[i].plot(df["id"],df['Close'], label = 'Close',color='lightgray')
            ax[i].plot(df["id"],wavlet_close[i], label = 'Close cA[%.0f]'%(len(wavlet_close)-i-1),linewidth=2)
            ax[i].plot(df["id"],wavlet_high[i], label = 'High cA[%.0f]'%(len(wavlet_close)-i-1))
            ax[i].plot(df["id"],wavlet_low[i], label = 'Low cA[%.0f]'%(len(wavlet_close)-i-1))
            
            ax[i].legend(loc = 'best')
        else:
            ax[i].plot(df["id"],wavlet_close[i], label = 'close-cD[%.0f]'%(len(wavlet_close)-i))
            ax[i].plot(df["id"],wavlet_high[i], label = 'high-cD[%.0f]'%(len(wavlet_close)-i))
            ax[i].plot(df["id"],wavlet_low[i], label = 'low-cD[%.0f]'%(len(wavlet_close)-i))
            ax[i].legend(loc = 'best')
    #cursor = MultiCursor(None, tuple(ax), color='r',lw=0.5, horizOn=True, vertOn=True)
    
    plt.show(block=False)
    return (fig, ax)
    
def plot_wt(df, colname, wavelet):
    plt.margins(0.1)
    fig, ax =  plt.subplots(len(wavelet), 1, figsize=figsize,sharex=True)
    for i in range(len(wf_close)):
        if i == 0:
            ax[i].plot(df["id"],df[colname], label = colname, color='lightgray')
            ax[i].plot(df["id"],wavelet[i], label = 'Vol cA[%.0f]'%(len(wavelet)-i-1))
            
            ax[i].legend(loc = 'best')
        else:
            ax[i].plot(df["id"],wavelet[i], label = 'Vol cD[%.0f]'%(len(wavelet)-i))
            ax[i].legend(loc = 'best')
    #cursor = MultiCursor(None, tuple(ax), color='r',lw=0.5, horizOn=True, vertOn=True)
    
    plt.show(block=False)
    return (fig, ax)

drawchart=True
historylen=500
interval='1d'
daysprint=89
if len(sys.argv) ==2:
  ticker=sys.argv[1]
elif len(sys.argv) ==3:
  ticker=sys.argv[1]
  historylen=int(sys.argv[2])
elif len(sys.argv) ==4:
  ticker=sys.argv[1]
  historylen=int(sys.argv[2])
  interval=sys.argv[3]
elif len(sys.argv) ==5:
  ticker=sys.argv[1]
  historylen=int(sys.argv[2])
  interval=sys.argv[3]
  drawchart=sys.argv[4].lower()=='true'
elif len(sys.argv) ==6:
  ticker=sys.argv[1]
  historylen=int(sys.argv[2])
  interval=sys.argv[3]
  drawchart=sys.argv[4].lower()=='true'
  daysprint=int(sys.argv[5])
else:
  ticker='QQQ'


  print("arguments are : Symbol historylen interval drawchart daysprint")
  print("interval can be 1m, 5m, 15m, 30m, 1h, 1d, 1wk, 1mo, 3mo")
  #exit()


#ticker="SPX"
df= GetYahooData(ticker,historylen,interval)
# import data
#df = pd.read_csv(filename, header=0, index_col=0, encoding='utf-8')
df.head()

#WT(index_list, wavefunc='db4', lv=4, m=1, n=4, plot=False):
wf_close=WT(df['Close'], wavefunc='db4', lv=2, m=1, n=2,plot=True)
wf_high=WT(df['High'], plot=False)
wf_low=WT(df['Low'], plot=False)
wf_vol=WT(df['Volume'], plot=False)

df["coeff_close"] = wf_close[0]
df["coeff_high"]= wf_high[0]
df["coeff_low"]= wf_low[0]
df["coeff_vol"] =wf_vol[0]

df["coeff_close_01"] = wf_close[0]+wf_close[1]
df["coeff_vol_01"] = wf_vol[0]+wf_vol[1]

print('day                  close         close1       high             high1         low               low1              volume                  volume1')
fmt="{0:18}{1:8.2f} {2:4}{3:8.2f} {4:4} * {5:8.2f} {6:4} {7:8.2f} {8:4} * {9:8.2f} {8:4} {11:8.2f} {12:4} * {13:18,.0f} {14:4} {15:18,.0f} {16:4}"
for i in range(daysprint,-1,-1):  
  if wf_close[0][-i-1]>wf_close[0][-i-2]:
    closedir='UP'
  else:
    closedir='DOWN'
  if wf_close[1][-i-1]>wf_close[1][-i-2]:
    close1dir='UP'
  else:
    close1dir='DOWN'

  if wf_high[0][-i-1]>wf_high[0][-i-2]:
    highdir='UP'
  else:
    highdir='DOWN'
  if wf_high[1][-i-1]>wf_high[1][-i-2]:
    high1dir='UP'
  else:
    high1dir='DOWN'

  if wf_low[0][-i-1]>wf_low[0][-i-2]:
    lowdir='UP'
  else:
    lowdir='DOWN'
  if wf_low[1][-i-1]>wf_low[1][-i-2]:
    low1dir='UP'
  else:
    low1dir='DOWN'
  
  if wf_vol[0][-i-1]>wf_vol[0][-i-2]:
    voldir='UP'
  else:
    voldir='DOWN'
  if wf_vol[1][-i-1]>wf_vol[1][-i-2]:
    vol1dir='UP'
  else:
    vol1dir='DOWN'
  
  print(fmt.format(df.index[-i-1].strftime("%m/%d/%Y %H:%M"), wf_close[0][-i-1],closedir,wf_close[1][-i-1],close1dir,wf_high[0][-i-1],highdir,wf_high[1][-i-1],high1dir,wf_low[0][-i-1],lowdir,wf_low[1][-i-1],low1dir,wf_vol[0][-i-1],voldir,wf_vol[1][-i-1],vol1dir))
if not drawchart:
  exit()
figsize=(26,13)



mc = mpf.make_marketcolors(
                           volume='lightgray'
                           )
s  = mpf.make_mpf_style(marketcolors=mc)

apdict = [mpf.make_addplot(df['coeff_close']),
        mpf.make_addplot(df['coeff_high']),
        mpf.make_addplot(df['coeff_low']),
        mpf.make_addplot(df['coeff_close_01']),
        mpf.make_addplot(wf_low[1],panel=2),
        mpf.make_addplot(wf_close[1],panel=2,ylabel='wf_close[1]',y_on_right=True),
        mpf.make_addplot(wf_high[1],panel=2),
        #mpf.make_addplot(wf_vol[1],panel=2,ylabel='wf_vol[1]',y_on_right=False),
        mpf.make_addplot((df['coeff_vol']),panel=1,color='r'),
        mpf.make_addplot((df['coeff_vol_01']),panel=1,color='g')]

fig1,ax1=mpf.plot(df,type='candle',volume=True,addplot=apdict, figsize=figsize,tight_layout=True,style=s,returnfig=True,block=False)
#cursor = MultiCursor(None, tuple(ax), color='r',lw=0.5, horizOn=True, vertOn=True)


fig2,ax2=mpf.plot(df,type='candle',volume=True,addplot=apdict, figsize=figsize,tight_layout=True, panel_ratios=(1,1),style=s,returnfig=True,block=False)
#cursor = MultiCursor(None, tuple(ax), color='r',lw=0.5, horizOn=True, vertOn=True)

apdict = [mpf.make_addplot(df['coeff_close']),
        mpf.make_addplot(df['coeff_high']),
        mpf.make_addplot(df['coeff_low']),
        mpf.make_addplot(df['coeff_close_01']),
        mpf.make_addplot(wf_low[1],panel=1),
        mpf.make_addplot(wf_close[1],panel=1,ylabel='wf_close[1]',y_on_right=True),
        mpf.make_addplot(wf_high[1],panel=1),
        #mpf.make_addplot(wf_vol[1],panel=2,ylabel='wf_vol[1]',y_on_right=False),
        #mpf.make_addplot((df['coeff_vol']),panel=1,color='g'),
        #mpf.make_addplot((df['coeff_vol_01']),panel=1,color='g')
        ]

fig3,ax3=mpf.plot(df,type='candle',volume=False,addplot=apdict, figsize=figsize,tight_layout=True,returnfig=True,block=False)
#cursor = MultiCursor(None, tuple(ax), color='r',lw=0.5, horizOn=True, vertOn=True)
(fig4, ax4)=multi_plot_wt(df, wf_close, wf_high,wf_low)
(fig5, ax5)=plot_wt(df, 'Volume', wf_vol)
#cursor = MultiCursor(None, tuple(ax1)+tuple(ax2)+tuple(ax3)+tuple(ax4)+tuple(ax5), color='r',lw=0.5, horizOn=True, vertOn=True)
plt.show()
#crosshairs(xlabel='t',ylabel='F')


