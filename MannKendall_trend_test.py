import pymannkendall as mk
from YahooData import *
import sys
'''
gfg_data = [54, 52, 53, 59, 56, 57, 51, 52, 50, 53]
result=mk.original_test(gfg_data)
print(result)
'''
def MannKendallTrendTest_multivariate(arr, testmode):
  if testmode is not None:
    testmode=testmode.lower()
  start=0
  lastmkresult=None
  for i in range(2,len(arr)):
      data=arr[-1-i:]
      if testmode is None or testmode=="original_test":
        mkresult=mk.original_test(data)
      elif testmode=="hamed_rao_modification_test":
        mkresult=mk.hamed_rao_modification_test(data)
      elif testmode=="yue_wang_modification_test":
        mkresult=mk.yue_wang_modification_test(data)
      elif testmode=="pre_whitening_modification_test":
        mkresult=mk.pre_whitening_modification_test(data)
      elif testmode=="trend_free_pre_whitening_modification_test":
        mkresult=mk.trend_free_pre_whitening_modification_test(data)
      elif testmode=="trend_free_pre_whitening_modification_test":
        mkresult=mk.trend_free_pre_whitening_modification_test(data)
      
      if not mkresult.h and lastmkresult is None:
         start=i
      else:
        if mkresult.h  and (lastmkresult is None or mkresult.trend==lastmkresult.trend):
          lastmkresult=mkresult
          end=i
        else:
          return start, end, lastmkresult

def MannKendallTrendTest(arr, testmode=None):
  if testmode is not None:
    testmode=testmode.lower()
  start=0
  lastmkresult=None
  for i in range(2,len(arr)):
      data=arr[-1-i:]
      if testmode is None or testmode=="original_test":
        mkresult=mk.original_test(data)
      elif testmode=="hamed_rao_modification_test":
        mkresult=mk.hamed_rao_modification_test(data)
      elif testmode=="yue_wang_modification_test":
        mkresult=mk.yue_wang_modification_test(data)
      elif testmode=="pre_whitening_modification_test":
        mkresult=mk.pre_whitening_modification_test(data)
      elif testmode=="trend_free_pre_whitening_modification_test":
        mkresult=mk.trend_free_pre_whitening_modification_test(data)
      
      if not mkresult.h and lastmkresult is None:
         start=i
      else:
        if mkresult.h  and (lastmkresult is None or mkresult.trend==lastmkresult.trend):
          lastmkresult=mkresult
          end=i
        else:
          return start, end, lastmkresult


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
for mode in [None,'hamed_rao_modification_test','yue_wang_modification_test','pre_whitening_modification_test','trend_free_pre_whitening_modification_test']:
  begin, end, mkresult=MannKendallTrendTest(df["Close"].to_numpy(), mode)
  print(mode,begin, end, mkresult.trend, mkresult)
  print()

