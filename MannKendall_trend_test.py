import pymannkendall as mk
from YahooData import *
import sys
'''
gfg_data = [54, 52, 53, 59, 56, 57, 51, 52, 50, 53]
result=mk.original_test(gfg_data)
print(result)
'''
drawchart=True
historylen=512
interval='1d'
usecache=True
daystoplot=512
if len(sys.argv) <1:
  print("arguments are : Symbol historylen interval drawchart daysprint brick_size_in_ATR")
  print("interval can be 1m, 5m, 15m, 30m, 1h, 1d, 1wk, 1mo, 3mo")
  print("python .\mingwave3.py QQQ 256 1d True 20 True 128 0l5")
if len(sys.argv) <2:
  ticker='QQQ'
if len(sys.argv) >=2:
  ticker=sys.argv[1]
if len(sys.argv) >=3:
  historylen=int(sys.argv[2])
if len(sys.argv) >=4:
  interval=sys.argv[3]
if len(sys.argv) >=5:
  drawchart=sys.argv[4].lower()=='true'
if len(sys.argv) >=6:
  daysprint=int(sys.argv[5])
if len(sys.argv) >=7:
  usecache=sys.argv[6].lower()=='true'
if len(sys.argv) >=8:
  daystoplot=int(sys.argv[7])
if len(sys.argv) >=9:
  brick_size=float(sys.argv[8])
  #exit()


#ticker="SPX"
df= GetYahooData_v2(ticker,historylen,interval)
x= df["Close"].to_numpy() 
trend=None
for i in range(2,historylen):
    data=x[-1-i:]
    
    result=mk.original_test(data)
    if result.h:
        print (i,len(data), result.trend)
        if trend is None:
            trend=result.trend
            print('last trend=',trend)
        if trend is not None and  result.trend!=trend:
            print ('trend change point:', df.index[-i])
            break