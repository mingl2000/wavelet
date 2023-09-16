'''
https://www.statsmodels.org/dev/examples/notebooks/generated/markov_autoregression.html
'''
from YahooData import *
from datetime import datetime
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import statsmodels.api as sm
import mplfinance as mpf
def CalRateChange(df):
  df['ROC']=0
  #mv250=df['Close'].ewm(span = 250, adjust = False).mean()
  #df['ROC']=df['Close']-mv250
  
  hist_ret = df['Close'] / df['Close'].shift(1) - 1.0     # shift 1 shifts forward one day; today has yesterday's price
# hist_ret = hist_close.pct_change(1)
  hist_ret.dropna(inplace=True)
  hist_ret = hist_ret * 100.0
  X = hist_ret.values.reshape(-1, 1)
  
  for i in range(1,len(df)):
    df['ROC'].iloc[i]=df['Close'].iloc[i]/df['Close'].iloc[i-1]*100.0-100.0
    #mv250=df['Close'].ewm(span = 250, adjust = False).mean()
    #df['ROC'].iloc[i]=df['Close'].iloc[i]-mv250
  
  return df 

def dfplot(ticker, name, df):  
  figsize=(26,13)
  mc = mpf.make_marketcolors(
                            volume='lightgray'
                            )

                            
  s  = mpf.make_mpf_style(marketcolors=mc, gridaxis='both')
  apdict = []
  '''
  apdict.append(mpf.make_addplot(df['fit0'], color='r',secondary_y=False,panel=1))
  apdict.append(mpf.make_addplot(df['fit1'], color='g',secondary_y=False,panel=1))
  apdict.append(mpf.make_addplot(df['fit2'], color='b',secondary_y=False,panel=1))
  '''
  apdict.append(mpf.make_addplot(df['fit0_2'], color='r',secondary_y=False,panel=4))
  apdict.append(mpf.make_addplot(df['fit1_2'], color='g',secondary_y=False,panel=5))
  apdict.append(mpf.make_addplot(df['fit2_2'], color='b',secondary_y=False,panel=2))
  apdict.append(mpf.make_addplot(df['fit3_2'], color='b',secondary_y=False,panel=8))

  apdict.append(mpf.make_addplot(df['fit0'], color='r',secondary_y=False,panel=6))
  apdict.append(mpf.make_addplot(df['fit1'], color='g',secondary_y=False,panel=7))
  apdict.append(mpf.make_addplot(df['fit2'], color='b',secondary_y=False,panel=1))
  apdict.append(mpf.make_addplot(df['fit3'], color='b',secondary_y=False,panel=3))

  
  

  def getTitle(ticker, name):
    title="KalmanFilter&MarkovRegression-"+ticker
    if name is not None:
      title=title+"-" +name
    return title
  fig1,ax1=mpf.plot(df,type='candle',volume=False,volume_panel=1,addplot=apdict, figsize=figsize,tight_layout=True,style=s,returnfig=True,block=False, title=ticker,panel_ratios=(3,1))
  fig1.suptitle(getTitle(ticker,name),fontsize=30)

print(len(sys.argv))
if len(sys.argv) <1:
    print("arguments are : Symbol historylen interval drawchart daysprint brick_size_in_ATR")
    print("interval can be 1m, 5m, 15m, 30m, 1h, 1d, 1wk, 1mo, 3mo")
    print("python .\mingwave3.py QQQ 256 1d True 20 True 128 0l5")
if len(sys.argv) <2:
    ticker='000001.ss'
else:
   ticker=sys.argv[1]
if len(sys.argv) >=3:
    interval=sys.argv[2]
else:
   interval='1d'
print(ticker)
df =GetYahooData_v2(ticker, 512,interval)
df['Close'].plot(title=ticker, figsize=(12, 3))
df=CalRateChange(df)
#plt.show()
#df.index = pd.DatetimeIndex(df.index).to_period('D')
# Fit the model
'''
mod_df = sm.tsa.MarkovAutoregression(
    df['ROC'], k_regimes=3, order=4, switching_ar=False
)
res_df = mod_df.fit()
print(res_df.summary())

mod_df = sm.tsa.MarkovAutoregression(
    df['ROC'], k_regimes=3, order=4, switching_ar=False
)
res_df = mod_df.fit()
print(res_df.summary())
df['fit0']=res_df.filtered_marginal_probabilities[0]
df['fit1']=res_df.filtered_marginal_probabilities[1]
df['fit2']=res_df.filtered_marginal_probabilities[2]
'''

from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

model = MarkovRegression(df['ROC'], k_regimes=3, trend='c', switching_variance=True, switching_trend=True )
res_df_2 = model.fit()
res_df_2.summary()
df['fit0']=res_df_2.smoothed_marginal_probabilities[0]
df['fit1']=res_df_2.smoothed_marginal_probabilities[1]
df['fit2']=res_df_2.smoothed_marginal_probabilities[2]
df['fit3']=0.5
for i in range(0,len(df)):
  if df['fit0'][i]>df['fit1'][i]:
    df['fit3'].iloc[i]=1
  else:
     df['fit3'].iloc[i]=0



model = MarkovRegression(df['ROC'], k_regimes=3, trend='n', switching_variance=True, switching_trend=True )
res_df_2 = model.fit()
res_df_2.summary()
df['fit0_2']=res_df_2.smoothed_marginal_probabilities[0]
df['fit1_2']=res_df_2.smoothed_marginal_probabilities[1]
df['fit2_2']=res_df_2.smoothed_marginal_probabilities[2]
df['fit3_2']=0.5
for i in range(0,len(df)):
  if df['fit0_2'][i]>df['fit1_2'][i]:
    df['fit3_2'].iloc[i]=1
  else:
     df['fit3_2'].iloc[i]=0

dfplot(ticker, 'SH index', df)
plt.show()
'''
res_df.index = pd.DatetimeIndex(res_df.index).to_period('D')
fig, axes = plt.subplots(2, figsize=(7, 7))
ax = axes[0]
ax.plot(res_df.filtered_marginal_probabilities[0])
ax.fill_between(df.index, 0, 1, where=df["Close"].values, color="k", alpha=0.1)
ax.set_xlim(df.index[4], df.index[-1])
ax.set(title="Filtered probability of recession")

ax = axes[1]
ax.plot(res_df.smoothed_marginal_probabilities[0])
ax.fill_between(df.index, 0, 1, where=df["Close"].values, color="k", alpha=0.1)
ax.set_xlim(res_df.index[4], res_df.index[-1])
ax.set(title="Smoothed probability of recession")

fig.tight_layout()
plt.show()
'''