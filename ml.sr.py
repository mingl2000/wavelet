import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from matplotlib import style
import mplfinance as mpf
import datetime 
import sys
from talib import ATR
import warnings
from termcolor import colored
from YahooData import *
warnings.filterwarnings('ignore')
style.use('ggplot')
def getATR(df, ATR_period):
  return ATR(df["High"], df["Low"], df["Close"], ATR_period)[-1]
'''
def get1mYahooData(ticker):
    data=[]
    

    tick = yf.Ticker(ticker)
    df = tick.history(period='7d', interval='1m')
    
    data.append(df)


    #minstart = datetime.date.today()-datetime.timedelta(29)
    #minstart = datetime.datetime(minstart.year, minstart.month, minstart.day)
    minstart =np.min(df.index)-datetime.timedelta(21)
    end=np.min(df.index)
    start=end-datetime.timedelta(7)
    
    while start>minstart:
        df = tick.history(start=start, end=end,interval='1m')
        data.insert(0, df)
        start=start- datetime.timedelta(7)
        end=end- datetime.timedelta(7)

    start=minstart
    df = tick.history(start=start, end=end,interval='1m')
    data.insert(0, df)    
    df=pd.concat(data)
    df.to_csv((ticker +'_1m.csv'), index=True)
    return df

def getYahooData(ticker, interval='1m'):
    if ticker.lower()=='spx':
        ticker='^GSPC'

    if interval=='1m':
        return get1mYahooData(ticker)
    
    if  interval.endswith('m'):
        period='60d'
    elif  interval.endswith('h'):
        period='730d'
    else:
        period='max'

    df = yf.download(tickers=ticker, period=period, interval=interval)
    return df
'''


def get_upper_lower(cluster_centers, price):
    upper=-1
    lower=-1

    for i in range(len(cluster_centers)):
        if cluster_centers[i][0]>=price:
            if upper==-1:            
                upper=i
            else: 
                if abs(cluster_centers[i][0]-price)< abs(cluster_centers[i][0]-cluster_centers[upper][0]):
                    upper=i

        if cluster_centers[i][0]<=price:
            if lower==-1:            
                lower=i
            else: 
                if abs(cluster_centers[i][0]-price)< abs(cluster_centers[i][0]-cluster_centers[lower][0]):
                    lower=i
    return (upper, lower)
figsize=(20,15)
saturation_point=0.05
clustersize=11
noclusters=3
interval='1m'
brick_size=0.5
if len(sys.argv) <2:
  ticker='QQQ'
if len(sys.argv) >=2:
  ticker=sys.argv[1]

if len(sys.argv) >=3:
  interval=sys.argv[2]

if len(sys.argv) >=4:
  saturation_point=float(sys.argv[3])
if len(sys.argv) >=5:
  clustersize=int(sys.argv[4])

if len(sys.argv) >=6:
  noclusters=int(sys.argv[5])
if len(sys.argv) >=7:
  brick_size=float(sys.argv[6])



data=getYahooData_v1(ticker,interval)
data.tail()
brick_size=brick_size*getATR(data, 14)

lastTS=np.max(data.index)
lastPrice=data.iloc[len(data)-1]["Close"]

low = pd.DataFrame(data=data['Low'], index=data.index)
high = pd.DataFrame(data=data['High'], index=data.index)

# finding the optimum k through elbow method
def get_optimum_clusters(data, saturation_point=0.05, size=11):

    wcss = []
    k_models = []

    for i in range(1, size):
        kmeans = KMeans(n_clusters=i)
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)
        k_models.append(kmeans)
        
    plt.plot(range(1,size), wcss)
            
    return k_models,wcss


# index 3 as 4 is the value of K (elbow point)
low_clusters, wcss_low=get_optimum_clusters(low, saturation_point=saturation_point, size=clustersize)
high_clusters, wcss_high = get_optimum_clusters(high, saturation_point=saturation_point, size=clustersize)
if noclusters>= len(low_clusters):
    noclusters=len(low_clusters)-1
if noclusters>= len(high_clusters):
    noclusters=len(high_clusters)-1

low_clusters = low_clusters[noclusters]
high_clusters = high_clusters[noclusters]

low_centers = low_clusters.cluster_centers_
high_centers = high_clusters.cluster_centers_
#plt.plot(data['Close'])
#fig1,ax1=mpf.plot(df,type='candle',volume=True,addplot=apdict, figsize=figsize,tight_layout=True,style=s,returnfig=True,block=False)

hlines=[]
colors=[]
linewidths=[]
#data['Close'].plot(figsize=(16,8), c='b')
for i in low_centers:
    hlines.append(i[0])
    
    #plt.axhline(i, c='g', ls='--')

(upper,lower)=get_upper_lower(low_centers, lastPrice)    
fmt="top custers: i={0}     low={1:8.2f} {3:1}   wcss={2:8.2f}"

for i in range(len(low_centers)):
    if i==upper or i==lower:
        mark='*'
        color='red'
    else:
        mark=' '
        color=None
    if i<3:
        colors.append('b')
        linewidths.append(len(low_centers)+2-i)
    else:
        colors.append('g')
        linewidths.append(len(low_centers)+2-i)
    
    print(colored(fmt.format(i, low_centers[i][0],wcss_low[i], mark ),color))
print('')
for i in high_centers:
    hlines.append(i[0])
    
    #plt.axhline(i, c='r', ls='--')

(upper,lower)=get_upper_lower(high_centers, lastPrice)
fmt="top custers: i={0}     high={1:8.2f} {3:1}    wcss={2:8.2f}"

for i in range(len(high_centers)):
    if i==upper or i==lower:
        mark='*'
        color='red'
    else:
        mark=' '
        color=None
    if i<3:
        colors.append('violet')
        linewidths.append(len(low_centers)+2-i)
    else:
        colors.append('r')
        linewidths.append(len(low_centers)+2-i)

    linewidths.append(len(high_centers)+2-i)
    print(colored(fmt.format(i, high_centers[i][0],wcss_low[i], mark ),color))
print('')
fmt="{0:10} top {1:2} clusters {2:18}     {3:8.2f}"
title=fmt.format(ticker, noclusters, lastTS.strftime("%m/%d/%Y %H:%M"), lastPrice)
#title=ticker+' top ' + str(noclusters) +'clusters ' + str(lastTS) +' ' + str(lastPrice) 

fig, ax =mpf.plot(data,type='candle', hlines=dict(hlines=hlines,colors=colors, linewidths=linewidths),figsize=figsize, block=False,title=title,returnfig=True)
for i in range(len(hlines)):
    ax[0].text(10+40*i, hlines[i], str(round(hlines[i],2)), fontsize=15,  color='black')


fig, ax =mpf.plot(data,type='renko',volume=False,hlines=dict(hlines=hlines,colors=colors, linewidths=linewidths), figsize=figsize,tight_layout=True,returnfig=True,block=False, renko_params=dict(brick_size=brick_size))
for i in range(len(hlines)):
    ax[0].text(10+40*i, hlines[i], str(round(hlines[i],2)), fontsize=15,  color='black')

#finding the optimum k using the silhouette method
def optimum_Kvalue(data):
    kmax = 11
    sil = {}
    k_model = {}
    for k in range(2, kmax+1):
        kmeans = KMeans(n_clusters = k).fit(data)
        k_model[k] = kmeans
        labels = kmeans.labels_
        sil[k]=(silhouette_score(data, labels))
    optimum_cluster = k_model[max(sil, key=sil.get)]
    #plt.plot(range(2,12), sil.values())
    return optimum_cluster
  
low_cl = optimum_Kvalue(high)
high_cl = optimum_Kvalue(low)

low_ce = low_cl.cluster_centers_
high_ce = high_cl.cluster_centers_

hlines=[]
colors=[]

#plt.plot(data['Close'])
#data['Close'].plot(figsize=(16,8), c='b')
for i in low_ce:
    hlines.append(i[0])
    colors.append('g')
for i in range(len(low_ce)):
    print('optimum_Kvalue custers: i=', i,  ' low=', low_ce[i])
print('')

    #plt.axhline(i, c='g', ls='--')
for i in high_ce:
    hlines.append(i[0])
    colors.append('r')
    #plt.axhline(i, c='r', ls='--')
for i in range(len(high_ce)):
    print('optimum_Kvalue custers: i=', i,  ' high=', high_ce[i])
print('')
fmt="{0:10} optimum_Kvalue  {1:18}     {2:8.2f}"
title=fmt.format(ticker, lastTS.strftime("%m/%d/%Y %H:%M"), lastPrice)
#title=ticker+' optimum_Kvalue ' + str(lastTS) +' ' + str(lastPrice) 
fig, ax =mpf.plot(data,type='candle', hlines=dict(hlines=hlines,colors=colors),figsize=figsize, block=False,title=title, returnfig=True)
for i in range(len(hlines)):
    ax[0].text(10+40*i, hlines[i], str(round(hlines[i],2)), fontsize=15,  color='black')



##plt.text(5, 5, 'bbb')
fig, ax =mpf.plot(data,type='renko',volume=False,hlines=dict(hlines=hlines,colors=colors), figsize=figsize,tight_layout=True,returnfig=True,block=False, renko_params=dict(brick_size=brick_size))
fig.suptitle(title)
for i in range(len(hlines)):
    ax[0].text(10+40*i, hlines[i], str(round(hlines[i],2)), fontsize=15,  color='black')




plt.show()
