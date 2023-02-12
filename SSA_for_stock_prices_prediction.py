#!/usr/bin/env python
# coding: utf-8

# Here we will see how singular decompositions can be used for stock price prediction. For that we will need `quandl` library to be installed (see [here](https://github.com/quandl/quandl-python) for more details). 

# In[221]:


from ssa_core import ssa, ssa_predict, ssaview, inv_ssa, ssa_cutoff_order
#from mpl_utils import set_mpl_theme

import matplotlib.pylab as plt
import quandl
import pandas as pd
import sys
import numpy as np

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

#get_ipython().run_line_magic('matplotlib', 'inline')

# customize mpl a bit
#set_mpl_theme('light')

# some handy functions
def fig(w=16, h=5, dpi=96, facecolor=None, edgecolor=None):
    return plt.figure(figsize=(w, h), dpi=dpi, facecolor=facecolor, edgecolor=edgecolor)

def mape(f, t):
    return 100*((f - t)/t).abs().sum()/len(t)

def mae(f, t):
    return 100*((f - t)).abs().sum()/len(t)


# Now try to load adjusted close prices for MSFT from Quandl.

# In[222]:


symbol = 'MSFT'
symbol = 'SPX'
#data = quandl.get('WIKI/%s' % instrument, start_date='2017-01-01', end_date='2012-02-10')


df=GetYahooData(symbol, bars=1000, interval='1d')
closes = df['Adj Close'].rename('close')

# We split series into train and test intervals and see how it looks on chart

# In[223]:


test_date = '2023-01-01'

train_d = closes[:test_date]
test_d = closes[test_date:]

fig(16, 3); plt.plot(train_d, label='Train'); plt.plot(test_d, 'r', label='Test')
plt.title('%s adjusted daily close prices' % symbol)
plt.legend()
plt.show()

# We can see how SSA decomposes original series into trend components and noise. There is chart of original series, reconstructed from first 4 components and residuals.

# In[224]:

MAX_LAG_NUMBER = 120 # 4*30 = 1 quarter max
MAX_LAG_NUMBER = 200 # 1 year?

fig()
ssaview(df,train_d.values, MAX_LAG_NUMBER, [0,1,2,3])
plt.show()

# We can plot residuals plot using following code

# In[225]:
pc, _, v = ssa(train_d.values, MAX_LAG_NUMBER)
reconstructed = inv_ssa(pc, v, [0,1,2,3])
noise = train_d.values - reconstructed
plt.hist(noise, 50);


# It's possible to reduce embedding space dimension by finding minimal lag

# In[226]:


n_co = ssa_cutoff_order(train_d.values, dim=MAX_LAG_NUMBER, show_plot=True)


# Using minimal lag we could try to make forecast for price series and plot results

# In[227]:


days_to_predict = 15
forecast = ssa_predict(train_d.values, n_co, list(range(8)), days_to_predict, 1e-5)


# In[228]:


fig(16, 5)

prev_ser = closes[datetime.date.isoformat(parser.parse(test_date) - timedelta(MAX_LAG_NUMBER)):test_date]
plt.plot(prev_ser, label='Train Data')

test_d = closes[test_date:]
f_ser = pd.DataFrame(data=forecast, index=test_d.index[:days_to_predict], columns=['close'])
orig = pd.DataFrame(test_d[:days_to_predict])

plt.plot(orig, label='Test Data')
plt.plot(f_ser, 'r-', marker='.', label='Forecast')
plt.legend()
plt.title('Forecasting %s for %d days, MAPE = %.2f%%' % (symbol, days_to_predict, mape(f_ser, orig)));
plt.show()

# In[ ]:





# In[ ]:





# In[ ]:




