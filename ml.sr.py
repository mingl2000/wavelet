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
style.use('ggplot')

def get1mYahooData(ticker):
    end=datetime.date.today()
    start = datetime.date.today()-datetime.timedelta(29)
    end=start+datetime.timedelta(7)
    tick = yf.Ticker(ticker)
    data=[]
    while end<datetime.date.today():
        df = tick.history(start=start, end=end,interval='1m')
        data.append(df)
        start=start+ datetime.timedelta(7)
        end=end+ datetime.timedelta(7)

    data=pd.concat(data)
    return data

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

figsize=(20,15)
saturation_point=0.05
clustersize=11
noclusters=3
interval='1m'
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



data=getYahooData(ticker, interval)
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

#data['Close'].plot(figsize=(16,8), c='b')
for i in low_centers:
    hlines.append(i[0])
    colors.append('g')
    #plt.axhline(i, c='g', ls='--')
fmt="top custers: i={0}     low={1:8.2f}    wcss={2:8.2f}"
for i in range(len(low_centers)):
    print(fmt.format(i, low_centers[i][0],wcss_low[i] ))
print('')
for i in high_centers:
    hlines.append(i[0])
    colors.append('r')
    #plt.axhline(i, c='r', ls='--')
fmt="top custers: i={0}     high={1:8.2f}    wcss={2:8.2f}"
for i in range(len(high_centers)):
    print(fmt.format(i, high_centers[i][0],wcss_low[i] ))
print('')

mpf.plot(data,type='candle', hlines=dict(hlines=hlines,colors=colors),figsize=figsize, block=False,title=(ticker+' top ' + str(noclusters) +'clusters'))

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
    plt.plot(range(2,12), sil.values())
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
for i in range(len(low_ce)):
    print('optimum_Kvalue custers: i=', i,  ' high=', high_ce[i])
print('')
mpf.plot(data,type='candle', hlines=dict(hlines=hlines,colors=colors),figsize=figsize, title=(ticker+' optimum_Kvalue'))
plt.show()
