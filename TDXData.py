import struct
import datetime
import numpy as np
import os
import pandas as pd
from datetime import datetime
from os.path import exists
def stock_csv(filepath,bars):
  
  data = []
  with open(filepath, 'rb') as f:
      #file_object_path = 'D:/Projects/PlutusPy/importexport/StockDir/' + name +'.csv'
      #file_object = open(file_object_path, 'w+')
      head="Date,High,Low,Open,Close,Volume,Adj Close"+"\r\n"
      #file_object.writelines(head)
      seekat=os.stat(filepath).st_size - bars*4*8
      if seekat>0:
        f.seek(seekat)
      data=f.read(os.stat(filepath).st_size-seekat)
      i=0
      arr=[]
      while i<len(data):
          stock_date = data[i:i+4]
          i=i+4
          stock_open = data[i:i+4]
          i=i+4
          stock_high = data[i:i+4]
          i=i+4
          stock_low = data[i:i+4]
          i=i+4
          stock_close = data[i:i+4]
          i=i+4
          stock_amount = data[i:i+4]
          i=i+4
          stock_vol = data[i:i+4]
          i=i+4
          stock_reservation = data[i:i+4]
          i=i+4

          '''
          stock_date = f.read(4)
          stock_open = f.read(4)
          stock_high = f.read(4)
          stock_low= f.read(4)
          stock_close = f.read(4)
          stock_amount = f.read(4)
          stock_vol = f.read(4)
          stock_reservation = f.read(4)
          '''
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
          stock_date=datetime.strptime(str(stock_date[0]), '%Y%m%d')
          arr.append([stock_date,
                          stock_open[0]/100.,stock_high[0]/100.,stock_low[0]/100.,stock_close[0]/100.,stock_amount[0],stock_vol[0],stock_reservation[0]])
          #df.loc[stock_date] = [stock_date,
          #                stock_open[0]/100.,stock_high[0]/100.,stock_low[0]/100.,stock_close[0]/100.,stock_amount[0],stock_vol[0],stock_reservation[0]]
  
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

  df=pd.DataFrame(arr, columns=ssa_columns)
  df.set_index('Date',inplace=True)
  df['vwap']=df['Amount']/df['Volume']
  df["id"]=np.arange(len(df))
  
  print('stock_csv', filepath, df.index[-1])
  return df
        #file_object.close()

#stock_csv('D:/Apps/goldsun/vipdoc/sz/lday/sz002594.day', 'sz002594')

paths=['D:/Apps/goldsun/vipdoc/sh/lday/','D:/Apps/goldsun/vipdoc/sz/lday/']
def GetTDXData_v2(symbol, bars=10, interval='1d'):
  for path in paths:
    #exchange=path[-8:-6]
    if symbol[7:].lower()=='ss':
      exchange='sh'
    else:
      exchange='sz'
    filename=path+exchange+symbol[:6]+'.day'
    if exists(filename):
      return stock_csv(filename,bars)
  return None
if __name__ == '__main__':
  start=datetime.now()
  GetTDXData_v2('688599.ss',10,'1d')
  end=datetime.now()
  print(end-start)