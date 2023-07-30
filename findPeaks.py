import numpy as np
import matplotlib.pyplot as plt
from YahooData import *
#from scipy.datasets import electrocardiogram
from scipy.signal import find_peaks
from scipy.signal import peak_prominences
import mplfinance as mpf
#x = electrocardiogram()[2000:4000]

def pltdf(df, highPeaks, lowPeaks):
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

  apdHigh = mpf.make_addplot(df['highPeaks'],type='scatter',markersize=200,marker='^')
  apdLow = mpf.make_addplot(df['lowPeaks'],type='scatter',markersize=200,marker='v')
  apd = mpf.make_addplot(df['peaks'],type='line',markersize=200,marker='v')
  apdict = []
  #apdict.append(apdHigh)
  #apdict.append(apdLow)
  apdict.append(apd)
  

  fig1,ax1=mpf.plot(df,type='candle',addplot=apdict,volume=False, figsize=figsize,tight_layout=True,style=s,returnfig=True,block=False)
  peaksX=np.sort(np.append(highPeaks, lowPeaks))
  peaksY=df.iloc[peaksX]['peaks']

  ax1[0].plot(peaksX,peaksY, color='b')

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
    
    pltdf(df,highPeaks,lowPeaks)
interval='5m'
ticker='000001.ss'
df=getYahooData_v1(ticker,  interval )
findpeaksImp(df,1)
plt.show()




