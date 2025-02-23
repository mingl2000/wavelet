import struct
import datetime
import numpy as np
import os
import pandas as pd
import sys
from datetime import datetime
from os.path import exists
#import akshare as ak


#stock_csv('D:/Apps/goldsun/vipdoc/sz/lday/sz002594.day', 'sz002594')

paths=['D:/Apps/goldsun/vipdoc/sh/lday/','D:/Apps/goldsun/vipdoc/sz/lday/']


def GetTDXData_v3(symbol, bars, interval='1d'):
  record_dtype = np.dtype([
      ('date', 'u4'),
      ('stock_open', 'u4'),      # 2-byte integer (big-endian)
      ('stock_high', 'u4'),      # 2-byte integer (big-endian)
      ('stock_low', 'u4'),      # 2-byte integer (big-endian)
      ('stock_close', 'u4'),      # 2-byte integer (big-endian)
      ('Amount', 'u4'),      # 2-byte integer (big-endian)
      ('Volume', 'u4'),      # 2-byte integer (big-endian)
      ('stock_reservation', 'u4')      # 2-byte integer (big-endian)
  ])
  for path in paths:
    exchange=path[-8:-6]
    filename=path+exchange+symbol[:6]+'.day'
    if exists(filename):
        records =np.fromfile(filename, dtype=record_dtype)
        #a=np.datetime64(records['date'], '%Y%M%d')
        #stock_date=datetime.strptime(str(records['date']), '%Y%M%d')
        df=pd.DataFrame(records)

        df['Open']= np.divide(records['stock_open'], 100)
        df['High']= np.divide(records['stock_high'], 100)
        df['Low']= np.divide(records['stock_low'], 100)
        df['Close']= np.divide(records['stock_close'], 100)
        date_strings = np.char.mod('%08d', df['date'].values) 
        df.index =  pd.to_datetime(date_strings, format='%Y%m%d')
        #df.index = pd.to_datetime([str(x) for x in df['date']]) # working. but slow
        df.drop(columns=['stock_open','stock_high','stock_low','stock_close','stock_reservation','date'], inplace=True)
        
        #df['Date']==datetime.strptime(str(df['date']), '%Y%M%d')
        return df

  return None



if __name__ == '__main__':
  print(sys.version)
  start=datetime.now()
  df=GetTDXData_v3('002049.sz',500,'1d')
  print(len(df))
  end=datetime.now()
  print(end-start)