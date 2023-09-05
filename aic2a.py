import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_predict


import numpy as np
import matplotlib.pyplot as plt
from pyts.decomposition import SingularSpectrumAnalysis
from numpy import pi
import datetime
from datetime import timedelta
from dateutil import parser
import pandas_datareader as pdr
import pandas as pd
import mplfinance as mpf
import yfinance as yf


from os.path import exists
import yfinance as yf
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
  dataFileName1="data/"+symbol+'_' +period+'_'+ 'max' +".csv"
  if interval.endswith(('d','D')) and datetime.datetime.now().hour>=13 and (exists(dataFileName)or exists(dataFileName1)):
    #print('read yahoo data from cache')
    if exists(dataFileName1):
      dataFileName=dataFileName1
    df=pd.read_csv(dataFileName, header=0, index_col=0, encoding='utf-8', parse_dates=True)
    #df.index=df["Date"]
  else:
    #print('read yahoo data from web')
    df = yf.download(tickers=symbol, period=period, interval=interval)
    df.to_csv(dataFileName, index=True, date_format='%Y-%m-%d %H:%M:%S')
  #dataFileName="data/"+symbol+".csv"
  
  #df = pd.read_csv(dataFileName,index_col=0,parse_dates=True)
  #df.shape
  df.dropna(inplace = True)
  df =df [-bars:]
  df.head(3)
  df.tail(3)
  #df["id"]=np.arange(len(df))
  #df["date1"]=df.index.astype(str)
  #df["datefmt"]=df.index.strftime('%m/%d/%Y')
  
  return df

def calc_ssa(df,colname,window_size=20):
  X=[]
  newdf=df[colname]
  newdf=newdf.dropna()
  X.append(newdf)
  # We decompose the time series into three subseries

  #groups = [np.arange(i, i + 5) for i in range(0, 11, 5)]

  # Singular Spectrum Analysis
  ssa = SingularSpectrumAnalysis(window_size=window_size, groups=None)
  X_ssa = ssa.fit_transform(X)
  for i in range(window_size):
    ssa_col=colname +'_ssa_'+str(i)
    df[ssa_col]=X_ssa[i]
  print(X_ssa)
  return df

def aic(n, mse, num_params):
    aic = n * np.log(mse) + 2 * num_params
    aic_c = aic + (2*num_params*(num_params+1))/(n - num_params - 1)
    return (aic, aic_c)

# Load data
url = 'https://raw.githubusercontent.com/nevernervous78/nirpyresearch/master/data/peach_spectra_brix.csv'
data = pd.read_csv(url)
X = data.values[:,1:]
y = data["Brix"].values




# Define PCR estimators
pcr1 = make_pipeline(PCA(n_components=5), LinearRegression())
pcr2 = make_pipeline(PCA(n_components=20), LinearRegression())
 
#Cross-validation
y_cv1 = cross_val_predict(pcr1, X, y, cv=10)
y_cv2 = cross_val_predict(pcr2, X, y, cv=10)
 
# Calculate MSE
mse1 = mean_squared_error(y, y_cv1)
mse2 = mean_squared_error(y, y_cv2)
 

'''
symbol='QQQ'
df=GetYahooData(symbol, bars=500, interval='1d')

# We decompose the time series into three subseries
window_size = 20

#groups = [np.arange(i, i + 5) for i in range(0, 11, 5)]

# Singular Spectrum Analysis

ssa = SingularSpectrumAnalysis(window_size=window_size, groups=None)
X_ssa_input=[]

X_ssa_input.append(df['Close'])
X_ssa = ssa.fit_transform(X_ssa_input)

X = X_ssa[0:]
y = df['Close'].values
'''

 
# Compute AIC
aic1, aicc1 = aic(X.shape[0], mse1, pcr1.steps[0][1].n_components+1)
aic2, aicc2 = aic(X.shape[0], mse2, pcr2.steps[0][1].n_components+1)
 
# Print data
print("AIC, model 1:", aic1)
print("AICc, model 1:", aicc1)
print("AIC, model 2:", aic2)
print("AICc, model 2:", aicc2)


'''
expect to the following output
AIC, model 1: 69.30237521019883
AICc, model 1: 71.25586358229185
AIC, model 2: 100.67687348181386
AICc, model 2: 133.67687348181386
'''
'''
ncomp = np.arange(1,20,1)
AIC = np.zeros_like(ncomp)
AICc = np.zeros_like(ncomp)
for i, nc in enumerate(ncomp):
 
    pcr = make_pipeline(PCA(n_components=nc), LinearRegression())
    y_cv = cross_val_predict(pcr, X, y, cv=10)
 
    mse = mean_squared_error(y, y_cv)
    AIC[i], AICc[i] = aic(X.shape[0], mse, pcr.steps[0][1].n_components+1)
'''

symbol='QQQ'
df=GetYahooData(symbol, bars=500, interval='1d')

# We decompose the time series into three subseries
window_size = 20

#groups = [np.arange(i, i + 5) for i in range(0, 11, 5)]

# Singular Spectrum Analysis
ncomp = np.arange(2,100,1)
AIC = np.zeros_like(ncomp)
AICc = np.zeros_like(ncomp)
import math
for i, nc in enumerate(ncomp):
    ssa = SingularSpectrumAnalysis(window_size=nc, groups=None)
    X_ssa_input=[]

    X_ssa_input.append(df['Close'])
    X_ssa = ssa.fit_transform(X_ssa_input)

    mse = mean_squared_error(df['High'].values, X_ssa[0])
    mse =mse+ mean_squared_error(df['Low'].values, X_ssa[0])
    #mse=math.sqrt(mse/2/len(df))
    AIC[i], AICc[i] = aic(len(df), mse, nc)
    print('i==',i,'AIC=' , AIC[i] , '  AICc[i]=', AICc[i])
    #mse = mean_squared_error(df['Close'].values, X_ssa[0])


plt.figure(figsize=(7,6))
with plt.style.context(('seaborn')):
    plt.plot(ncomp, AIC, 'k', lw=2, label="AIC")
    plt.plot(ncomp,AICc, 'r', lw=2, label="AICc")
plt.xlabel("LV")
plt.ylabel("AIC/AICc value")
plt.tight_layout()
plt.legend()
plt.show()