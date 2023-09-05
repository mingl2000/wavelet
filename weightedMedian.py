import pandas as pd
import weightedstats as ws
import mplfinance as mpf
from YahooData import *

ticker='QQQ'
df=GetYahooData_v2(ticker,500,'1d')
df['weight']=1
wedf=ws.weighted_median(df['Close'],df['weight'])

df['wedf']=wedf
figsize=(26,13)
apdict = [
        mpf.make_addplot(df["wedf"], width=3, color='r',linestyle='dashdot'),
        ]

fig3,ax3=mpf.plot(df,type='candle',volume=False,addplot=apdict, figsize=figsize,tight_layout=True,returnfig=True,block=False,title=ticker)
