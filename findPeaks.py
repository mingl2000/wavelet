import numpy as np
import matplotlib.pyplot as plt
from YahooData import *
#from scipy.datasets import electrocardiogram
from scipy.signal import find_peaks
from scipy.signal import peak_prominences
import mplfinance as mpf
#x = electrocardiogram()[2000:4000]

def pltdf(df, highPeaks, lowPeaks,nonePeaks):
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

  peaks=mergePeaks(df,highPeaks,lowPeaks)

  '''
  apdHigh = mpf.make_addplot(df['highPeaks'],type='scatter',markersize=200,marker='^')
  apdLow = mpf.make_addplot(df['lowPeaks'],type='scatter',markersize=200,marker='v')
  apd = mpf.make_addplot(df['peaks'],type='line',markersize=200,marker='X')
  apdict = []
  #apdict.append(apdHigh)
  #apdict.append(apdLow)
  apdict.append(apd)
  '''
  
  #fig1,ax1=mpf.plot(df,type='candle',addplot=apdict,volume=False, figsize=figsize,tight_layout=True,style=s,returnfig=True,block=False)
  '''
  fig1,ax1=mpf.plot(df,type='candle',volume=False, figsize=figsize,tight_layout=True,style=s,returnfig=True,block=False)
  peaksX=np.sort(np.append(highPeaks, lowPeaks))
  peaksY=df.iloc[peaksX]['peaks']
  '''
  fig1,ax1=mpf.plot(df,type='candle',volume=False, figsize=figsize,tight_layout=True,style=s,returnfig=True,block=False)
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

  return highPeaks, lowPeaks, nonePeaks

def mergePeaks(df,highPeaks,lowPeaks):
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
  return merged
      

from statistics import stdev
from statistics import mean
def findpeaksImp(df, sigmas=1):
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
    nonePeaks=[]
    if highdata[-1]>lowdata[lowPeaks[-1]]:
      highPeaks=np.append(highPeaks, len(highdata)-1)
    elif lowdata[-1]<highdata[highPeaks[-1]]:
      lowPeaks=np.append(lowPeaks, len(lowdata)-1)
    else:

      nonePeaks.append(len(lowdata)-1)

    highPeaks,lowPeaks, nonePeaks=pltdf(df,highPeaks,lowPeaks, nonePeaks)
    return highPeaks, lowPeaks, nonePeaks
import sys
def main(args):
  if len(args)>0:
    ticker=args[1]
  else:
    ticker='000001.ss'  
  print(ticker)

  df=getYahooData_v1(ticker,  '5m' )
  highPeaks, lowPeaks, nonePeaks=findpeaksImp(df,1)
  df=getYahooData_v1(ticker,  '30m')
  highPeaks, lowPeaks, nonePeaks=findpeaksImp(df,1)
  df=getYahooData_v1(ticker,  '1d')
  highPeaks, lowPeaks, nonePeaks=findpeaksImp(df,1)

  plt.show()

if __name__ =="__main__":
  main(sys.argv)


