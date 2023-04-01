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
import yfinance as yf
import talib
import warnings
from YahooData import *
warnings.filterwarnings('ignore')
def getStockNames():
  df=pd.read_excel('CHINA_STOCKs2.xlsx', index_col=0)
  return df

def getStockName(stock_df, symbol):
  try:
    row=stock_df.loc[symbol[0:6]]
    name=row['名称']
    sector=row['细分行业']
    return (name,sector)
  except:
    return (symbol, 'NA')
'''
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
  dataFileName1="data/"+symbol+'_' +period+'_'+ 'max' +".csv"
  if interval.endswith(('d','D')) and datetime.datetime.now().hour>=13 and (exists(dataFileName)or exists(dataFileName1)):
    #print('read yahoo data from cache')
    if exists(dataFileName1):
      dataFileName=dataFileName1
    df=pd.read_csv(dataFileName, header=0, index_col=0, encoding='utf-8', parse_dates=True)
    #df.index=df["Date"]
  else:
    try:
    #print('read yahoo data from web')
      df = yf.download(tickers=symbol, period=period, interval=interval)
      df.to_csv(dataFileName, index=True, date_format='%Y-%m-%d %H:%M:%S')
    except:
      print('yfinance failed to download '+symbol)
      return None
  #dataFileName="data/"+symbol+".csv"
  
  #df = pd.read_csv(dataFileName,index_col=0,parse_dates=True)
  #df.shape
  df.dropna(inplace = True)
  df =df [-bars:]
  df.head(3)
  df.tail(3)
  #df["id"]=np.arange(len(df))
  #df["date1"]=df.index.astype(str)
  #df["datefmt"]=df.index.strftime('%m/%d/%Y')
  
  return df

'''
import sys
if len(sys.argv) <2:
  symbols='QQQ,SPX'
if len(sys.argv) >=2:
  symbols=sys.argv[1]

#data = quandl.get('WIKI/%s' % instrument, start_date='2017-01-01', end_date='2012-02-10')
ssa_columns={'ticker':[], 
             'name':[], 
             'sector':[], 
             
            'X_ssa_0':[], 'X_ssa_0_dir':[], 'X_ssa_0_slope':[],  'X_ssa_0_acceleration':[],
            'X_ssa_1':[], 'X_ssa_1_dir':[], 'X_ssa_1_slope':[], 

            'OBV_ssa_0':[], 'OBV_ssa_0_dir':[], 'OBV_ssa_0_slope':[], 'OBV_ssa_0_acceleration':[], 
            'OBV_ssa_1':[], 'OBV_ssa_1_dir':[], 'OBV_ssa_1_slope':[],


            'V_ssa_0':[], 'V_ssa_0_dir':[], 'V_ssa_0_slope':[], 'V_ssa_0_acceleration':[], 
            'V_ssa_1':[], 'V_ssa_1_dir':[], 'V_ssa_1_slope':[],
            'Close':[], 
            'Low':[],
            'High':[], 
            'OBV':[]
            }
ssa_df=pd.DataFrame(ssa_columns)
ssa_df.set_index('ticker')

stock_df=getStockNames()
for symbol in symbols.split(','):
  df=GetYahooData_v2(symbol, bars=500, interval='1d')
  if df is not None:
    
    df['OBV']=talib.OBV(df['Close'], df['Volume'])
    # We decompose the time series into three subseries
    window_size = 20

    #groups = [np.arange(i, i + 5) for i in range(0, 11, 5)]

    # Singular Spectrum Analysis
    ssa = SingularSpectrumAnalysis(window_size=window_size, groups=None)
    X=[]
    closes = df['Close'].rename('close')
    X.append(df['Close'])
    X_ssa = ssa.fit_transform(X)

    V=[]
    V.append(df['Volume'])
    V_ssa = ssa.fit_transform(V)

    OBV=[]
    OBV.append(df['OBV'])
    OBV_ssa = ssa.fit_transform(OBV)


    def upordown(arr):
      if round(arr[-1],2)==round(arr[-2],2):
        return '=='
      elif round(arr[-1],2)>round(arr[-2],2):      
        return 'UP'
      else:
        return 'DOWN'
    def slope(arr):
      return (arr[-1]-arr[-2])/arr[-2]*100

    def acceration(arr):
      v0=arr[-1]-arr[-2]
      v1=arr[-2]-arr[-3]
      if round(v0,2)==round(v1,2):
        return 'no acc'
      elif round(v0,2)>round(v1,2):
        if v0>0:
          return 'up-acc'
        else:
          return 'down-slowed'
      else:
        if v0 >0:
          return 'up-slowed'
        else:
          return 'down-acc'

    def print_ssa(symbol,X_ssa, V_ssa):    
      #fmt="{0:18}{1:8.2f} * {2:8.2f} {3:4} {4:8.2f} {5:4} * {6:8.2f} {7:4} {8:8.2f} {9:4} * {10:8.2f} {11:4} {12:8.2f} {13:4} * {14:18,.0f} {15:4} {16:18,.0f} {17:4} {18:18,.2f} {19:18,.2f}"
      #print(fmt.format(df.index[-i-1].strftime("%m/%d/%Y %H:%M"), df['Close'][-i-1], wf_close[0][-i-1],closedir,wf_close[1][-i-1],close1dir,wf_high[0][-i-1],highdir,wf_high[1][-i-1],high1dir,wf_low[0][-i-1],lowdir,wf_low[1][-i-1],low1dir,wf_vol[0][-i-1],voldir,wf_vol[1][-i-1],vol1dir, gf3[i], gf5[i]))
      fmt="{0:8} {1:8.2f} {2:4} {3:8.2f} {4:11}* {5:8.2f} {6:4} {7:8.2f} *** {8:18,.0f} {9:4} {10:8.2f} {11:11} * {12:18,.0f} {13:4}  {14:8.2f}"

      print(fmt.format(symbol, X_ssa[0][-1],upordown(X_ssa[0]),slope(X_ssa[0]),acceration(X_ssa[0]), X_ssa[1][-1],upordown(X_ssa[1]), slope(X_ssa[1]), V_ssa[0][-1],upordown(V_ssa[0]),slope(V_ssa[0]), acceration(V_ssa[0]),V_ssa[1][-1],upordown(V_ssa[1]),slope(V_ssa[1]) ))
      pass
    def add_ssa_df(symbol,df,X_ssa, V_ssa,ssa_df, OBV_ssa):
      (name, sector)=getStockName(stock_df,symbol)
      ssa_df.loc[symbol] = [symbol,
                            name,
                            sector,
                            X_ssa[0][-1],upordown(X_ssa[0]),slope(X_ssa[0]),acceration(X_ssa[0]),
                            X_ssa[1][-1],upordown(X_ssa[1]), slope(X_ssa[1]),
                            OBV_ssa[0][-1],upordown(OBV_ssa[0]),slope(OBV_ssa[0]), acceration(OBV_ssa[0]),
                            OBV_ssa[1][-1],upordown(OBV_ssa[1]),slope(OBV_ssa[1]),
                            V_ssa[0][-1],upordown(V_ssa[0]),slope(V_ssa[0]),acceration(V_ssa[0]),
                            V_ssa[1][-1],upordown(V_ssa[1]),slope(V_ssa[1]),
                            
                            df['Close'][-1],
                            df['High'][-1],
                            df['Low'][-1],
                            df['OBV'][-1]
]
      return ssa_df
      #fmt="{0:18}{1:8.2f} * {2:8.2f} {3:4} {4:8.2f} {5:4} * {6:8.2f} {7:4} {8:8.2f} {9:4} * {10:8.2f} {11:4} {12:8.2f} {13:4} * {14:18,.0f} {15:4} {16:18,.0f} {17:4} {18:18,.2f} {19:18,.2f}"
      #print(fmt.format(df.index[-i-1].strftime("%m/%d/%Y %H:%M"), df['Close'][-i-1], wf_close[0][-i-1],closedir,wf_close[1][-i-1],close1dir,wf_high[0][-i-1],highdir,wf_high[1][-i-1],high1dir,wf_low[0][-i-1],lowdir,wf_low[1][-i-1],low1dir,wf_vol[0][-i-1],voldir,wf_vol[1][-i-1],vol1dir, gf3[i], gf5[i]))
      fmt="{0:8} {1:8.2f} {2:4} {3:8.2f} {4:11}* {5:8.2f} {6:4} {7:8.2f} *** {8:18,.0f} {9:4} {10:8.2f} {11:11} * {12:18,.0f} {13:4}  {14:8.2f}"

      #print(fmt.format(symbol, X_ssa[0][-1],upordown(X_ssa[0]),slope(X_ssa[0]),acceration(X_ssa[0]), X_ssa[1][-1],upordown(X_ssa[1]), slope(X_ssa[1]), V_ssa[0][-1],upordown(V_ssa[0]),slope(V_ssa[0]), acceration(V_ssa[0]),V_ssa[1][-1],upordown(V_ssa[1]),slope(V_ssa[1]) ))
    pass
    
    ssa_df=add_ssa_df(symbol,df,X_ssa, V_ssa,ssa_df,OBV_ssa)


  #print_ssa(symbol,X_ssa, V_ssa)
ssa_df.head()
ssa_df.to_csv('ssa_search.csv', index=False)
fmt="ssa_search_{0}.xlsx"
filename=fmt.format(datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
ssa_df.to_excel(filename)

from termcolor import colored
print('Check output file:' +colored(filename,'red'))
import os
os.system("start EXCEL.EXE "+filename)
# Show the results for the first time series and its subseries

