import pymannkendall as mk
from YahooData import *
import sys
import mplfinance as mpf
import matplotlib.pyplot as plt
def plot(ticker, df, segments):
  figsize=(26,13)
  mc = mpf.make_marketcolors(
                            volume='lightgray'
                            )

                            
  s  = mpf.make_mpf_style(marketcolors=mc, gridaxis='both')
  apdict = []
  vlines=[]
  for (start,end, trend) in segments:
    vlines.append(df.index[start])
    vlines.append(df.index[end])
  #apdict.append(mpf.make_addplot(df[newcol], secondary_y=False))
  #fig1,ax1=mpf.plot(df,type='candle',volume=False,volume_panel=2,addplot=apdict, figsize=figsize,tight_layout=True,style=s,returnfig=True,block=False, title=ticker,panel_ratios=(1,2))
  fig1,ax1=mpf.plot(df,type='candle',volume=False,volume_panel=2,vlines=vlines, figsize=figsize,tight_layout=True,style=s,returnfig=True,block=False, title=ticker)
  #mpf.plot(df,vlines=vlines)
  plt.show()
'''
gfg_data = [54, 52, 53, 59, 56, 57, 51, 52, 50, 53]
result=mk.original_test(gfg_data)
print(result)
'''
def MannKendallTrendTest_multivariate(arr):
  start=0
  lastmkresult=None
  for i in range(2,len(arr)):
      data=arr[-1-i:]
      mkresult=mk.correlated_multivariate_test(arr)
      if not mkresult.h and lastmkresult is None:
         start=i
      else:
        if mkresult.h  and (lastmkresult is None or mkresult.trend==lastmkresult.trend):
          lastmkresult=mkresult
          end=i
        else:
          return start, end, lastmkresult

import numpy as np
from scipy.stats import kendalltau
import numpy as np
from scipy.stats import linregress

def trend_segmentation(time_series, w, m, alpha):
    segments = []
    i = 0
    while i+w <= len(time_series):
        segment = time_series[i:i+w]
        slope, intercept, r_value, p_value, std_err = linregress(range(len(segment)), segment)
        if p_value < alpha and len(segment) >= m:
            segments.append((i, i+w-1, slope))
            i += w-m+1
        else:
            i += 1
    merged_segments = []
    current_segment = segments[0]
    for i in range(1, len(segments)):
        if np.sign(current_segment[2]) == np.sign(segments[i][2]):
            current_segment = (current_segment[0], segments[i][1], current_segment[2]+segments[i][2])
        else:
            merged_segments.append(current_segment)
            current_segment = segments[i]
    merged_segments.append(current_segment)
    return merged_segments



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
'''
for mode in [None,'hamed_rao_modification_test','yue_wang_modification_test','pre_whitening_modification_test','trend_free_pre_whitening_modification_test']:
  begin, end, mkresult=MannKendallTrendTest(df["Close"].to_numpy(), mode)
  print(mode,begin, end, mkresult.trend, mkresult)
  print()
'''
mv_data = np.array([[1,1,1],[2,2,2],[3,3,3],[4,4,4],[5,5,5],[6,6,6],[7,7,7],[8,8,8],[9,9,9]])
def arbitrary_2d_data():
    # Generate arbitrary 80, 2 dimensional data
    arbitrary_2d_data = np.array([[ 490.,  458.], [ 540.,  469.], [ 220., 4630.], [ 390.,  321.], [ 450.,  541.],
       [ 230., 1640.], [ 360., 1060.], [ 460.,  264.], [ 430.,  665.], [ 430.,  680.],
       [ 620.,  650.], [ 460., np.nan], [ 450.,  380.], [ 580.,  325.], [ 350., 1020.],
       [ 440.,  460.], [ 530.,  583.], [ 380.,  777.], [ 440., 1230.], [ 430.,  565.],
       [ 680.,  533.], [ 250., 4930.], [np.nan, 3810.], [ 450.,  469.], [ 500.,  473.],
       [ 510.,  593.], [ 490.,  500.], [ 700.,  266.], [ 420.,  495.], [ 710.,  245.],
       [ 430.,  736.], [ 410.,  508.], [ 700.,  578.], [ 260., 4590.], [ 260., 4670.],
       [ 500.,  503.], [ 450.,  469.], [ 500.,  314.], [ 620.,  432.], [ 670.,  279.],
       [np.nan,  542.], [ 470.,  499.], [ 370.,  741.], [ 410.,  569.], [ 540.,  360.],
       [ 550.,  513.], [ 220., 3910.], [ 460.,  364.], [ 390.,  472.], [ 550.,  245.],
       [ 320., np.nan], [ 570.,  224.], [ 480.,  342.], [ 520.,  732.], [ 620.,  240.],
       [ 520.,  472.], [ 430.,  679.], [ 400., 1080.], [ 430.,  920.], [ 490.,  488.],
       [ 560., np.nan], [ 370.,  595.], [ 460.,  295.], [ 390.,  542.], [ 330., 1500.],
       [ 350., 1080.], [ 480.,  334.], [ 390.,  423.], [ 500.,  216.], [ 410.,  366.],
       [ 470.,  750.], [ 280., 1260.], [ 510.,  223.], [np.nan,  462.], [ 310., 7640.],
       [ 230., 2340.], [ 470.,  239.], [ 330., 1400.], [ 320., 3070.], [ 500.,  244.]])
    return arbitrary_2d_data

#result=MannKendallTrendTest_multivariate(arbitrary_2d_data())
print ('2d data')
mkresult=mk.correlated_multivariate_test(arbitrary_2d_data())
print(mkresult)
print ('\n3d data')
mkresult=mk.correlated_multivariate_test(mv_data)
print(mkresult)
data_with_trend_segment=[1,2,3,4,5,6,7,8,9,8,7,6,3,2,1,2,3,4,5,7,8,9]
#rs_result=trend_segmentation(data_with_trend_segment, 9,0.05)
#print("\ntrend_segmentation:", rs_result)
rs_result=trend_segmentation(data_with_trend_segment, 3,3,0.05)
print("\ntrend_segmentation:", rs_result)

data=df["Close"].to_numpy()
segments=trend_segmentation(data, 3,3,0.05)
plot(ticker,df,segments)
print("\ntrend_segmentation:", segments)

