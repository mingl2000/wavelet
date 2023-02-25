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

import sys
if len(sys.argv) <2:
  symbol='QQQ'
if len(sys.argv) >=2:
  symbol=sys.argv[1]


bars=500
if len(sys.argv) >=3:
  bars=int(sys.argv[2])
interval='1d'
if len(sys.argv) >=4:
  interval=sys.argv[3]


window_sizes=[5, 10,15,20, 25,30]
if len(sys.argv) >=5:
  window_sizes=[]
  for s in sys.argv[4].split(','):
    window_sizes.append(int(s))

#data = quandl.get('WIKI/%s' % instrument, start_date='2017-01-01', end_date='2012-02-10')


df=GetYahooData(symbol, bars=bars, interval='1d')
closes = df['Adj Close'].rename('close')
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
X=[]
X.append(closes)
# We decompose the time series into three subseries
window_size = 2

#groups = [np.arange(i, i + 5) for i in range(0, 11, 5)]

# Singular Spectrum Analysis
#plt.figure(figsize=(16, 6))
#plt.plot(closes.to_numpy(), 'o-', label='Original')

import mplfinance as mpf
figsize=(26,13)
mc = mpf.make_marketcolors(
                           volume='lightgray'
                           )

                          
s  = mpf.make_mpf_style(marketcolors=mc, gridaxis='both')
apdict = []
for window_size in window_sizes:
  #window_size=i
  ssa = SingularSpectrumAnalysis(window_size=window_size, groups=None)
  X_ssa = ssa.fit_transform(X)
  newcol='ssa_'+str(window_size)
  newcol_diff='ssa_diff_'+str(window_size)
  df[newcol]=X_ssa[0]
  df[newcol_diff]=df['Close']-X_ssa[0]
  apdict.append(mpf.make_addplot(df[newcol], ylabel=newcol))
  
  apdict.append(mpf.make_addplot(df[newcol_diff], panel=2,ylabel=newcol_diff))
  stderr=np.std(df[newcol_diff].to_numpy())
  apdict.append(mpf.make_addplot(df[newcol_diff], panel=2,ylabel=newcol_diff))

  newcol_diff_1_std_ub='ssa_diff_1std_up_'+str(window_size)
  newcol_diff_1_std_lb='ssa_diff_1std_low_'+str(window_size)
  df[newcol_diff_1_std_ub]=stderr
  df[newcol_diff_1_std_lb]=-stderr
  apdict.append(mpf.make_addplot(df[newcol_diff_1_std_ub], panel=2,ylabel=newcol_diff))
  apdict.append(mpf.make_addplot(df[newcol_diff_1_std_lb], panel=2,ylabel=newcol_diff))

  newcol_diff_2_std_ub='ssa_diff_2std_up_'+str(window_size)
  newcol_diff_2_std_lb='ssa_diff_2std_low_'+str(window_size)
  df[newcol_diff_2_std_ub]=2*stderr
  df[newcol_diff_2_std_lb]=-2*stderr
  apdict.append(mpf.make_addplot(df[newcol_diff_2_std_ub], panel=2,ylabel=newcol_diff))
  apdict.append(mpf.make_addplot(df[newcol_diff_2_std_lb], panel=2,ylabel=newcol_diff))

  print(window_size, X_ssa[0][-1], len(X_ssa[0]) )

  #print(X_ssa)
  # Show the results for the first time series and its subseries

  #ax1 = plt.subplot(211)
  #plt.plot(X_ssa[0],  label=('X_ssa[0] window=' +str(window_size)))
  
  #plt.legend(loc='best', fontsize=14)

'''
plt.suptitle('Singular Spectrum Analysis', fontsize=20)

plt.tight_layout()
plt.subplots_adjust(top=0.88)
plt.show()
'''

fig1,ax1=mpf.plot(df,type='candle',volume=True,addplot=apdict, figsize=figsize,tight_layout=True,style=s,returnfig=True,block=False)
plt.show()


# The first subseries consists of the trend of the original time series.
# The second and third subseries consist of noise.
