import sys
from termcolor import colored
from pykalman import KalmanFilter
from YahooData import *
import sys
import pandas as pd
import warnings

warnings.filterwarnings('ignore')
#from kalmanfilter3 import *

def DoKalmanFilter(arr):
    kf = KalmanFilter(
      initial_state_mean=arr[0],
      initial_state_covariance=1,
      observation_covariance=1,
      transition_covariance=0.01
    )
    kf_means, _ = kf.filter(arr)
    kf_max=kf_means[-1][0]
    kf_min=kf_means[-1][0]
    #find max
    for i in range(len(kf_means)-2,0,-1):
        if kf_means[-2][0]>kf_means[-1][0]: # going down
            if kf_means[i][0]>kf_means[i+1][0]:
                kf_max=kf_means[i][0]
            else:
               break
        if kf_means[-2][0]<kf_means[-1][0]: # going up
           break
    #find min
    for i in range(len(kf_means)-2,0,-1):
        if kf_means[-2][0]<kf_means[-1][0]: # going up
            if kf_means[i][0]<kf_means[i+1][0]:
                kf_min=kf_means[i][0]
            else:
               break
        if kf_means[-2][0]>kf_means[-1][0]: # going down
           break
        
    return (kf_means, kf_min, kf_max)

def GetDaysUpOrDown(df):
   daysUp=0
   DaysDown=0
   
   for i in range(len(df)-2,-1,-1):
      if df['Close'][i]<=df['Close'][i+1]:
         daysUp=daysUp+1
      else:
         break
      
   for i in range(len(df)-2,-1,-1):
      if df['Close'][i]>=df['Close'][i+1]:
         DaysDown=DaysDown+1
      else:
         break
   return (daysUp, DaysDown)
def doKalmarFilerSearch(symbols):

    kf_columns={'ticker':[], 
            'Close':[],             
            'High':[], 
            'Low':[],
            'Open':[],
            'Volume':[], 
            'kf':[],
            'kfmin':[],
            'kfmax':[],
            'kfslope':[],
            'abovekfmin':[],
            'belowkfmax':[],
            'closemin':[],
            'closemax':[],
            'closeslope':[],
            'aboveclosemin':[],
            'belowclosemax':[],
            'kfVolslope':[],
            'daysUp':[],
            'DaysDown':[],
            }
    kf_df=pd.DataFrame(kf_columns)
    kf_df.set_index('ticker')

    for symbol in symbols.split(','):
        df=GetYahooData_v2(symbol, bars=500, interval='1d')
        if len(df) <3:
           continue
        (daysUp, DaysDown)=GetDaysUpOrDown(df)
        (kf_means, kf_min, kf_max)=DoKalmanFilter(df['Close'].to_numpy())
        (kfv_means, kfv_min, kfv_max)=DoKalmanFilter(df['Volume'].to_numpy())
        kf_df.loc[symbol] = [symbol,
                            df['Close'][-1],
                            df['High'][-1],
                            df['Low'][-1],
                            df['Open'][-1],
                            df['Volume'][-1],
                            kf_means[-1][0],
                            kf_min,
                            kf_max,
                            kf_means[-1][0]/kf_means[-2][0]*100-100,
                            kf_means[-1][0]-kf_min,
                            kf_max-kf_means[-1][0],
                            None,
                            None,
                            df['Close'][-1]/df['Close'][-2]*100-100,
                            None,
                            None,
                            kfv_means[-1][0]/kfv_means[-2][0]*100-100,
                            daysUp,
                            DaysDown
                            ]
    return kf_df

if len(sys.argv) <2:
  filePrefix=''
if len(sys.argv) >=2:
  filePrefix=sys.argv[1]

if len(sys.argv) <3:
  symbols='QQQ,SPX'
if len(sys.argv) >=3:
  symbols=sys.argv[2]

df=doKalmarFilerSearch(symbols)
for i in range(len(df)):
   print(df.index[i],'kf=', df['kf'][i],'kfmin=',df['kfmin'][i], 'dfmax=', df['kfmax'][i],'kfslope=',df['kfslope'][i])
df=df.sort_values(by=['kfslope','belowkfmax','abovekfmin'])

fmt="KarlmanFilter_search_{0}_{1}.xlsx"
filename=fmt.format(filePrefix, datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
df.to_excel(filename)

from termcolor import colored
print('Check output file:' +colored(filename,'red'))

