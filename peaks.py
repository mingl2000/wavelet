import numpy as np
import matplotlib.pyplot as plt
#from scipy.datasets import electrocardiogram
from scipy.signal import find_peaks
from sklearn.cluster import DBSCAN
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from matplotlib import style
import mplfinance as mpf
import datetime 
import sys
from talib import ATR
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.signal import find_peaks_cwt
style.use('ggplot')
def getATR(df, ATR_period):
  return ATR(df["High"], df["Low"], df["Close"], ATR_period)[-1]

def get1mYahooData(ticker):
    data=[]
    

    tick = yf.Ticker(ticker)
    df = tick.history(period='7d', interval='1m')
    
    data.append(df)


    #minstart = datetime.date.today()-datetime.timedelta(29)
    #minstart = datetime.datetime(minstart.year, minstart.month, minstart.day)
    minstart =np.min(df.index)-datetime.timedelta(21)
    end=np.min(df.index)
    start=end-datetime.timedelta(7)
    
    while start>minstart:
        df = tick.history(start=start, end=end,interval='1m')
        data.insert(0, df)
        start=start- datetime.timedelta(7)
        end=end- datetime.timedelta(7)

    start=minstart
    df = tick.history(start=start, end=end,interval='1m')
    data.insert(0, df)    
    df=pd.concat(data)
    df.to_csv((ticker +'_1m.csv'), index=True)
    return df

def getYahooData(ticker, interval='1m'):
    if ticker.lower()=='spx':
        ticker='^GSPC'

    if interval=='1m':
        return get1mYahooData(ticker)
    
    if  interval.endswith('m'):
        period='60d'
    elif  interval.endswith('h'):
        period='730d'
    else:
        period='max'

    df = yf.download(tickers=ticker, period=period, interval=interval)
    return df

df=getYahooData('SPX', '1h')
#df=df.drop(['High', 'Low','Open', 'Volume','Adj Close'], axis=1)
#df["id"]=np.arange(len(df))
#df.reset_index()
x=df["High"]
#peaks, _ = find_peaks(df, height=100)

peakind = find_peaks_cwt(df["High"], np.arange(15,20))

#plt.plot(x)
plt.plot(peakind, x[peakind], "x")
#plt.plot(np.zeros_like(x), "--", color="gray")
plt.show()