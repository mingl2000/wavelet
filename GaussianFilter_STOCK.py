from scipy.ndimage import gaussian_filter1d
import numpy as np
import mplfinance as mpf
import yfinance as yf
from os.path import exists
import numpy as np
import pandas as pd
from talib import ATR
import datetime
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

def GetYahooData(symbol, bars=500, interval='1d'):
  #start=datetime.date.today()-datetime.timedelta(days=days)
  #end=datetime.date.today()
  if symbol=='SPX':
    symbol='^GSPC'
  #df=.gepdrt_data_yahoo(symbols=symbol,  start=start, end=end,interval=interval)
  
  #if interval.endswith('m') or interval.endswith('h'):
  #  period='max'
  if interval=='1m':
    return get1mYahooData(symbol)
  elif  interval.endswith('m'):
    period='60d'
  elif  interval.endswith('h') or interval.endswith('d'):
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


#gaussian_filter1d([1.0, 2.0, 3.0, 4.0, 5.0], 1)
#array([ 1.42704095,  2.06782203,  3.        ,  3.93217797,  4.57295905])
#gaussian_filter1d([1.0, 2.0, 3.0, 4.0, 5.0], 4)
#array([ 2.91948343,  2.95023502,  3.        ,  3.04976498,  3.08051657])
import matplotlib.pyplot as plt
import sys

if len(sys.argv) <2:
  ticker='QQQ'
  interval='1d'
if len(sys.argv) >=2:
  ticker=sys.argv[1]
if len(sys.argv) >=3:
  interval=sys.argv[2]

df=GetYahooData(ticker, bars=720, interval=interval)
x= df["Close"].to_numpy() 
y2 = gaussian_filter1d(x, 2)
y5 = gaussian_filter1d(x, 5)
y8 = gaussian_filter1d(x, 8)
y13 = gaussian_filter1d(x, 13)
y21 = gaussian_filter1d(x, 21)
figsize=(26,13)

apdict = {
        "gaussian_filter1d-3":mpf.make_addplot(y2,color='b', width=3),
        #mpf.make_addplot(y5,color='g'),
        "gaussian_filter1d-8":mpf.make_addplot(y8,color='y', width=3),
        "gaussian_filter1d-21":mpf.make_addplot(y21,color='r', width=3)
        #mpf.make_addplot(wf_vol[1],panel=2,ylabel='wf_vol[1]',y_on_right=False),
        #mpf.make_addplot((df['coeff_vol']),panel=1,color='g'),
        #mpf.make_addplot((df['coeff_vol_01']),panel=1,color='g')
}
fig,ax=mpf.plot(df,type='candle',title="gaussian_filter1d", volume=False,addplot=list(apdict.values()), figsize=figsize,tight_layout=True,returnfig=True,block=False,mav=())

ax[0].legend([None]*(len(apdict)+2))
handles = ax[0].get_legend().legendHandles
ax[0].legend(handles=handles[2:],labels=list(apdict.keys()))


plt.grid()

plt.show()