import yfinance as yf
import numpy as np
from os.path import exists
import datetime
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
  dataFileName1="data/"+symbol+'_' +'max'+'_'+ interval +".csv"
  if interval.endswith(('d','D')) and datetime.datetime.now().hour>=13 and (exists(dataFileName) or exists(dataFileName1)):
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
