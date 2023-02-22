# 参考国泰君安刘富兵研报《基于奇异谱分析的均线择时研究》，交易标的为同花顺，默认参数为20日
#https://blog.csdn.net/weixin_36595077/article/details/123670706
import numpy as np
import pandas as pd
import datetime
from datetime import timedelta
from dateutil import parser
import pandas_datareader as pdr
import mplfinance as mpf
import yfinance as yf
from os.path import exists
import matplotlib.pyplot as plt

# Fiddle with figure settings here:
# Set the default colour cycle (in case someone changes it...)
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

# 初始化账户
def init(context):
    # 设置时间周期
    context.m = 20
    # 设置要交易的股票
    context.stocks = ['300033.SZ']

#设置买卖条件，每个交易频率（日/分钟/tick）调用一次
def handle_bar(context,bar_dict):
    num = context.m
    for stk in context.stocks:
        #获取股票历史收盘价数据
        close = history(stk, ['close'], 2*num, '1d')
        # 给c赋值收盘价数据
        c = close['close'].values
        #计算均线价格
        SSA20 = SSA(close.values, num)
        # 若收盘价与SSA均线金叉
        if c[-1] > SSA20[-1] and  c[-2] < SSA20[-2] and stk not in list(context.portfolio.stock_account.positions.keys()):
            #使用所有现金买入股票
            order_target_percent(stk,1)

        # 若收盘价与SSA均线死叉
        if c[-1] < SSA20[-1] and c[-2] > SSA20[-2] and stk in list(context.portfolio.stock_account.positions.keys()):
            #卖出所有股票
            order_target(stk, 0)
 
# 嵌入
def getWindowMatrix(inputArray, t, m):
    temp = []
    n = t-m+1
    for i in range(n):
        tmp = []
        for j in range(m):
            tmp.append(inputArray[i+j][0])
        temp.append(tmp)
    WindowMatrix = np.array(temp)
    return WindowMatrix

# 奇异谱分析，取第一主成分分量，返回重构矩阵
def SVDreduce(WindowMatrix):
    u, s, v = np.linalg.svd(WindowMatrix) #svd分解
    m1, n1 = u.shape
    m2, n2 = v.shape
    index = s.argmax()
    u1 = u[:, index]
    v1 = v[index]
    u1 = u1.reshape((m1, 1))
    v1 = v1.reshape((1, n2))
    value = s.max()
    newMatrix = value*(np.dot(u1, v1))  #重构矩阵
    return newMatrix

# 对角线平均法重构序列
def recreateArray(newMatrix, t, m):
    ret = []
    n = t-m+1
    for p in range(1, t+1):
        if p < m:
            alpha = p
        elif p > t-m+1:
            alpha = t-p+1
        else:
            alpha = m
        sigma = 0
        for j in range(1, m+1):
            i = p-j+1
            if i > 0 and i < n+1:
                sigma += newMatrix[i-1][j-1]
        ret.append(sigma/alpha)
    return ret

# 按不同的序列、不同的窗口大小计算SSA
def SSA(inputArray, m):
    t = 2*m
    WindowMatrix = getWindowMatrix(inputArray, t, m)
    newMatrix = SVDreduce(WindowMatrix)
    newArray = recreateArray(newMatrix, t, m)
    return newArray

df=GetYahooData('QQQ', bars=500, interval='1d')
ssa20=SSA(df.to_numpy(),20)
ssa25=SSA(df.to_numpy(),25)
ssa30=SSA(df.to_numpy(),30)

plt.plot(ssa20)
plt.plot(ssa25)
plt.plot(ssa30)
plt.show()
