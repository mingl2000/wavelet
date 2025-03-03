import numpy as np
import matplotlib.pyplot as plt
from YahooData import *
#from scipy.datasets import electrocardiogram
from scipy.signal import find_peaks
from scipy.signal import peak_prominences
import mplfinance as mpf
#x = electrocardiogram()[2000:4000]

def pltdf(df, title, highPeaks, lowPeaks,errorInPercent):
  figsize=(26,13)
  mc = mpf.make_marketcolors(
                            volume='lightgray'
                            )

                            
  s  = mpf.make_mpf_style(marketcolors=mc, gridaxis='both')
  #apdict = []
  #apdict.append(mpf.make_addplot(df[newcol_diff_2_std_ub], panel=1,ylabel=newcol_diff_2_std_ub))
  #apdict.append(mpf.make_addplot(df[newcol_diff_2_std_lb], panel=1,ylabel=newcol_diff_2_std_lb))
  
  
  
  df['highPeaks']=np.nan
  df['lowPeaks']=np.nan
  df['peaks']=np.nan
  for i in highPeaks:
    df.iat[i,6]=df.iloc[i].High
    df.iat[i,8]=df.iloc[i].High
  
  for i in lowPeaks:
    df.iat[i,7]=df.iloc[i].Low
    df.iat[i,8]=df.iloc[i].Low

  peaks,nonePeaks,zhongsus=mergePeaks(df,highPeaks,lowPeaks, errorInPercent)
  
  fig1,ax1=mpf.plot(df,type='candle',volume=False, figsize=figsize,tight_layout=True,style=s,returnfig=True,block=False)
  fig1.suptitle(title, fontsize=28)
  peaksX=[]
  peaksY=[]
  for x, y,z in peaks:
    peaksX.append(x)
    peaksY.append(y)
  
  
  ax1[0].plot(peaksX,peaksY, color='b')
  
  if len(nonePeaks)>0:
    peaksX=[]
    peaksY=[]

    peaksX.append(peaks[-1][0])
    peaksY.append(peaks[-1][1])
    peaksX.append(nonePeaks[0])
    peaksY.append(df['Close'][nonePeaks[0]])
    ax1[0].plot(peaksX,peaksY, color='y')
  
  
  
  for low, high, left, right, expandable in zhongsus:
    peaksX=[]
    peaksY=[]

    peaksX.append(left)
    peaksY.append(low)
    
    peaksX.append(left)
    peaksY.append(high)

    peaksX.append(right)
    peaksY.append(high)

    peaksX.append(right)
    peaksY.append(low)
    peaksX.append(left)
    peaksY.append(low)
    ax1[0].plot(peaksX,peaksY, color='r')
  

  return peaks,highPeaks, lowPeaks, nonePeaks
def between(a, low, high):
  return a>=low and a<=high
#import math
def isclose(a, b, minvalue, maxvalue, errorInPercent):
  return abs(a/b-1)<=abs((maxvalue-minvalue))*errorInPercent/100
  
def isPartOfZhongSus(zu, zhongsus,errorInPercent):
  for zuTemp in zhongsus:
    #if zu[0]==zuTemp[0] and zu[1]==zuTemp[1]:
    #and zu[3]>=zuTemp[3] and  zu[4]<=zuTemp[4]:
    if isclose(zu[0], zuTemp[0], zuTemp[0], zuTemp[1], errorInPercent) and isclose(zu[1], zuTemp[1], zuTemp[0], zuTemp[1], errorInPercent):
      return True
    
  return False

def find_zhongsus(peaks, errorInPercent):
  zhongsus=[]

  for i in range(3, len(peaks)):
    curline=[min(peaks[i-1], peaks[i-0]), max(peaks[i-1], peaks[i-0])]
    if len(zhongsus)>0:
      lastzu=zhongsus[-1]
      if lastzu[4] and between(curline[0][1], lastzu[0],lastzu[1]) or between(curline[1][1], lastzu[0],lastzu[1]):
        lastzu[3]=peaks[i-0][0]
        continue
      else:
        lastzu[4]=False
    line1=[min(peaks[i-3][1], peaks[i-2][1]), max(peaks[i-3][1], peaks[i-2][1])]
    line2=[min(peaks[i-2][1], peaks[i-1][1]), max(peaks[i-2][1], peaks[i-1][1])]
    line3=[min(peaks[i-1][1], peaks[i-0][1]), max(peaks[i-1][1], peaks[i-0][1])]
    dd=max(line1[0], line2[0], line3[0])
    gg=min(line1[1], line2[1], line3[1])
    if dd<gg:      
      zu=[dd, gg, peaks[i-3][0],peaks[i-0][0], True]  # True: open to add expand
      if not isPartOfZhongSus(zu, zhongsus, errorInPercent):
        zhongsus.append(zu)

  return zhongsus

    
    


def mergePeaks(df,highPeaks,lowPeaks, errorInPercent):
  #highPeakParis=zip(highPeaks, [highPeaks, df.iloc[highPeaks].highPeaks.to_numpy(),'high'])
  #lowPeakParis=zip(lowPeaks, [lowPeaks, df.iloc[lowPeaks].highPeaks.to_numpy(),'low'])
  highPeakParis=zip(*[highPeaks, df.iloc[highPeaks].highPeaks.to_numpy(),len(highPeaks)*["high"]])
  lowPeakParis=zip(*[lowPeaks, df.iloc[lowPeaks].lowPeaks.to_numpy(),len(lowPeaks)*["low"]])
  highPeakParis=list(map(list, highPeakParis))
  lowPeakParis=list(map(list, lowPeakParis))
  peakPairs=highPeakParis+lowPeakParis
  soretedPeakPairs=sorted(peakPairs)
  merged=[]
  merged.append(soretedPeakPairs[0])
  last_type=soretedPeakPairs[0][2]
  last_val=soretedPeakPairs[0][1]
  for i in range(1, len(soretedPeakPairs)):
    cur_type=soretedPeakPairs[i][2]
    cur_val=soretedPeakPairs[i][1]
    if cur_type!=last_type:
      merged.append(soretedPeakPairs[i])
      last_type=soretedPeakPairs[i][2]
      last_val=soretedPeakPairs[i][1]
    elif cur_type=='high' and last_type=='high' and cur_val>last_val or cur_type=='low' and last_type=='low' and cur_val<last_val:
      merged[-1]=soretedPeakPairs[i]
      last_type=soretedPeakPairs[i][2]
      last_val=soretedPeakPairs[i][1]
    else:
      print('i=',i,'last_type=',last_type,'last_val',last_val)
      #last_type=soretedPeakPairs[0][2]
      #last_val=soretedPeakPairs[0][1]
      print('skip i=',i,'last_type=',last_type,'last_val',last_val, 'cur_type=',cur_type , 'cur_val=',cur_val)
  
  print('merged=',len(merged))
  ''' 
  add df last point into merged
  '''

  ''' add the last data point'''
  ''' add the last data point'''
  peaks=merged
  nonePeaks=[]

  if peaks[-1][2]=='high' and df['High'][-1]>peaks[-1][1]:
    peaks[-1]=[len(df)-1, df['High'][-1],'high']
  elif peaks[-1][2]=='low' and df['Low'][-1]<peaks[-1][1]:
    peaks[-1]=[len(df)-1, df['Low'][-1],'low']
  elif peaks[-1][2]=='low' and df['High'][-1]>peaks[-1][1]:
    #peaks=np.append(peaks, [len(df)-1, df['High'][-1],'high'])
    peaks.append([len(df)-1, df['High'][-1],'high'])
    #pass
  elif peaks[-1][2]=='high' and df['Low'][-1]<peaks[-1][1]:
    peaks.append([len(df)-1, df['Low'][-1],'low'])
    #peaks=np.append(peaks, [len(df)-1, df['Low'][-1],'low'])
    #pass
  else:
    nonePeaks.append(len(df)-1)


  zhongsus=find_zhongsus(peaks, errorInPercent)
  return (merged, nonePeaks,zhongsus)
      

from statistics import stdev
from statistics import mean
def findpeaksImp(df, title, sigmas=1,errorInPercent=1):
    highdata=df['High'].to_numpy()
    lowdata=df['High'].to_numpy()
    #highPeaks, _highProperties = find_peaks(highdata, height=0,distance=5)
    highPeaks, _highProperties = find_peaks(highdata)
    lowPeaks, _lowProperties= find_peaks(-1*lowdata)
    highProminences=peak_prominences(highdata, highPeaks)
    lowProminences=peak_prominences(-1*lowdata, lowPeaks)
    #lowProminences=_lowProperties["prominences"]
    highProminenceThreashold=mean(highProminences[0])+sigmas*stdev(highProminences[0])
    lowProminenceThreashold=mean(lowProminences[0])+sigmas*stdev(lowProminences[0])

    #plt.plot(highdata)
    #plt.plot(highPeaks, highdata[highPeaks], "x")
    #plt.plot(np.zeros_like(x), "--", color="gray")
    #plt.show()
    highPeaks, _highProperties= find_peaks(highdata, prominence=[highProminenceThreashold,None])
    lowPeaks, _lowProperties= find_peaks(-1*lowdata, prominence=[lowProminenceThreashold,None])
    '''
    nonePeaks=[]
    if highdata[-1]>lowdata[lowPeaks[-1]]:
      highPeaks=np.append(highPeaks, len(highdata)-1)
    elif lowdata[-1]<highdata[highPeaks[-1]]:
      lowPeaks=np.append(lowPeaks, len(lowdata)-1)
    else:

      nonePeaks.append(len(lowdata)-1)
    '''
    peaks,highPeaks,lowPeaks, nonePeaks=pltdf(df,title, highPeaks,lowPeaks,errorInPercent)
    return peaks,highPeaks, lowPeaks, nonePeaks



import sys
def main(ticker):
  
  df=getYahooData_v1(ticker,  '5m' )
  title=ticker +" 5m"
  peaks,highPeaks, lowPeaks, nonePeaks=findpeaksImp(df,title,1)
  title=ticker +" 30m"
  df=getYahooData_v1(ticker,  '30m')
  peaks,highPeaks, lowPeaks, nonePeaks=findpeaksImp(df,title,1,0.1)
  title=ticker +" 5h"
  df=getYahooData_v1(ticker,  '1h')
  title=ticker +" 5m"
  peaks,highPeaks, lowPeaks, nonePeaks=findpeaksImp(df,title,1)
  title=ticker +" 1d"
  df=getYahooData_v1(ticker,  '1d')
  peaks,highPeaks, lowPeaks, nonePeaks=findpeaksImp(df,title,1)
  
  plt.show()

if __name__ =="__main__":
  if len(sys.argv)>1:
    ticker=sys.argv[1]
  else:
    ticker='000001.ss'  
  print(ticker)

  main(ticker)


