import struct
import datetime
import numpy as np
import os
import pandas as pd
from datetime import datetime
from os.path import exists
def stock_csv(filepath):
  data = []
  ssa_columns={'Date':[], 
            'Open':[], 
            'High':[],
            'Low':[], 
            'Close':[],
#             'Adj Close':[],
            'Amount':[], 
            'Volume':[],
            'Reservation':[]
  }
  df=pd.DataFrame(ssa_columns)
  df.set_index('Date')
  with open(filepath, 'rb') as f:
      #file_object_path = 'D:/Projects/PlutusPy/importexport/StockDir/' + name +'.csv'
      #file_object = open(file_object_path, 'w+')
      head="Date,High,Low,Open,Close,Volume,Adj Close"+"\r\n"
      #file_object.writelines(head)
      while True:
          stock_date = f.read(4)
          stock_open = f.read(4)
          stock_high = f.read(4)
          stock_low= f.read(4)
          stock_close = f.read(4)
          stock_amount = f.read(4)
          stock_vol = f.read(4)
          stock_reservation = f.read(4)

          # date,open,high,low,close,amount,vol,reservation
          #day,high,low,open,close,volume,Adj Close
          if not stock_date:
              break
          stock_date = struct.unpack("l", stock_date)     # 4字节 如20091229
          stock_open = struct.unpack("l", stock_open)     #开盘价*100
          stock_high = struct.unpack("l", stock_high)     #最高价*100
          stock_low= struct.unpack("l", stock_low)        #最低价*100
          stock_close = struct.unpack("l", stock_close)   #收盘价*100
          stock_amount = struct.unpack("f", stock_amount) #成交额
          stock_vol = struct.unpack("l", stock_vol)       #成交量
          stock_reservation = struct.unpack("l", stock_reservation) #保留值

          #date_format = datetime.strptime(str(stock_date[0]),'%Y%M%d') #格式化日期
          #list= date_format.strftime('%Y-%M-%d')+","+str(stock_open[0]/100)+","+str(stock_high[0]/100.0)+","+str(stock_low[0]/100.0)+","+str(stock_close[0]/100.0)+","+str(stock_vol[0])+"\r\n"
          #list= date_format.strftime('%Y-%M-%d')+","+str(stock_high[0]/100.0)+","+str(stock_low[0]/100.0)+","+str(stock_open[0]/100)+","+str(stock_close[0]/100.0)+","+str(stock_vol[0]/100.0)+","+str(stock_amount[0])+"\r\n"
          #file_object.writelines(list)
          stock_date=datetime.strptime(str(stock_date[0]), '%Y%M%d')
          df.loc[stock_date] = [stock_date,
                          stock_open[0]/100.,stock_high[0]/100.,stock_low[0]/100.,stock_close[0]/100.,stock_amount[0],stock_vol[0],stock_reservation[0]]
  df['vwap']=df['Amount']/df['Volume']
  df["id"]=np.arange(len(df))
  return df
        #file_object.close()

#stock_csv('D:/Apps/goldsun/vipdoc/sz/lday/sz002594.day', 'sz002594')

paths=['D:/Apps/goldsun/vipdoc/sh/lday/','D:/Apps/goldsun/vipdoc/sz/lday/']
def GetTDXData_v2(symbol, bars=500, interval='1d'):
  for path in paths:
    exchange=path[-8:-6]
    filename=path+exchange+symbol[:6]+'.day'
    if exists(filename):
      return stock_csv(filename)

GetTDXData_v2('002049.sz',500,'1d')