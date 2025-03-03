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
from YahooData import *
from os.path import exists
import yfinance as yf
from nimbusml.timeseries import SsaForecaster
from nimbusml.timeseries import SsaChangePointDetector

#data = quandl.get('WIKI/%s' % instrument, start_date='2017-01-01', end_date='2012-02-10')

def plot_ssa(symbol, window_size=20):
  df=GetYahooData_v2(symbol, bars=500, interval='1d')
  closes = df['Adj Close'].rename('close')
  training_seasons = 13
  #seasonality_size=5
  #training_size = seasonality_size * training_seasons
  training_size=len(df)
  #window_size=8
  forecaster = SsaForecaster(series_length=len(df),
                              train_size=training_size/2,
                              window_size=window_size,
                              horizon=5) <<   {'fc': 'ts'}
  
  X_train = pd.Series(df['Close'], name="ts")

  forecaster.fit(X_train, verbose=1)
  data = forecaster.transform(X_train)
  pd.set_option('display.float_format', lambda x: '%.2f' % x)
  #print(data)



   #training_size = seasonality_size * training_seasons
  print(data)
  
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
  #X=[]
  #X.append(closes)
  # We decompose the time series into three subseries

  #groups = [np.arange(i, i + 5) for i in range(0, 11, 5)]

  # Singular Spectrum Analysis
  #ssa = SingularSpectrumAnalysis(window_size=window_size, groups=None)
  #X_ssa = ssa.fit_transform(X)

  #print(X_ssa)
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
  #for i in range(0, int(window_size/2)):
  for i in range(0,4):
      newcol='ssa_' +str(i)
      #df[newcol]=X_ssa[i]
      df[newcol]=data['fc.'+str(i)].to_numpy()
      if i==0:
        apdict.append(mpf.make_addplot(df[newcol], panel=0, width=3,secondary_y=False))
      else:
        if i>=4:
          width=1
        else:
          width=5-i
        apdict.append(mpf.make_addplot(df[newcol], panel=0, width=width,secondary_y=False))


  fig1,ax1=mpf.plot(df,type='candle',volume=False,volume_panel=2,addplot=apdict, figsize=figsize,tight_layout=True,style=s,returnfig=True,block=False)
  fig1.suptitle(symbol,fontsize=30)
  
  
  seasonality_size=1
  cpd = SsaChangePointDetector(confidence=60,
                                change_history_length=8 ,
                                training_window_size=training_size,
                                seasonal_window_size=seasonality_size + 1) << {'result': 'ts'}
  cpd.fit(X_train, verbose=1)
  data = cpd.transform(X_train)
  print(data)



import sys
window_size=20
if len(sys.argv) <2:
  symbols='QQQ'
if len(sys.argv) >=2:
  symbols=sys.argv[1]
if len(sys.argv) >=3:
  window_size=int(sys.argv[2])

for symbol in symbols.split(','):
  plot_ssa(symbol, window_size)
plt.show()
# The first subseries consists of the trend of the original time series.
# The second and third subseries consist of noise.
