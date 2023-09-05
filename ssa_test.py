# https://ghammad.github.io/pyActigraphy/pyActigraphy-SSA.html

import pyActigraphy
from pyActigraphy.analysis import SSA
import numpy as np
import plotly.graph_objs as go
import os

from os.path import exists
import yfinance as yf
import datetime
import pandas as pd
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
  if interval.endswith(('d','D')) and datetime.datetime.now().hour>=13 and exists(dataFileName):
    print('read yahoo data from cache')
    df=pd.read_csv(dataFileName, header=0, index_col=0, encoding='utf-8', parse_dates=True)
    #df.index=df["Date"]
  else:
    print('read yahoo data from web')
    df = yf.download(tickers=symbol, period=period, interval=interval)
    df.to_csv(dataFileName, index=True, date_format='%Y-%m-%d %H:%M:%S')
  #dataFileName="data/"+symbol+".csv"
  
  #df = pd.read_csv(dataFileName,index_col=0,parse_dates=True)
  #df.shape
  df.dropna(inplace = True)
  df =df [-bars:]
  df.head(3)
  df.tail(3)
  df["id"]=np.arange(len(df))
  #df["date1"]=df.index.astype(str)
  #df["datefmt"]=df.index.strftime('%m/%d/%Y')
  
  return df
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
  if interval.endswith(('d','D')) and datetime.datetime.now().hour>=13 and exists(dataFileName):
    print('read yahoo data from cache')
    df=pd.read_csv(dataFileName, header=0, index_col=0, encoding='utf-8', parse_dates=True)
    #df.index=df["Date"]
  else:
    print('read yahoo data from web')
    df = yf.download(tickers=symbol, period=period, interval=interval)
    df.to_csv(dataFileName, index=True, date_format='%Y-%m-%d %H:%M:%S')
  #dataFileName="data/"+symbol+".csv"
  
  #df = pd.read_csv(dataFileName,index_col=0,parse_dates=True)
  #df.shape
  df.dropna(inplace = True)
  df =df [-bars:]
  df.head(3)
  df.tail(3)
  df["id"]=np.arange(len(df))
  #df["date1"]=df.index.astype(str)
  #df["datefmt"]=df.index.strftime('%m/%d/%Y')
  
  return df

df=GetYahooData('QQQ', 5000, '1d')

#data=df['Close'].to_numpy()
data=df['Close'].dropna()
dfnew = pd.DataFrame(df['Close'].to_numpy(),columns=['Close'])
dfnew['Date']=None
from datetime import timedelta

for i in range(len(df)):
  dfnew['Date'][i]=df.index[0] + timedelta(days=i)

dfnew.index=dfnew['Date']
data=dfnew['Close']

print('aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa')
mySSA = SSA(data,window_length='1d')
print('bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb')
# Access the trajectory matrix
mySSA.trajectory_matrix().shape

# By definition, the sum of the partial variances should be equal to 1:
mySSA.lambda_s.sum()

layout = go.Layout(
    height=600,
    width=800,
    title="Scree diagram",
    xaxis=dict(title="Singular value index", type='log', showgrid=True, gridwidth=1, gridcolor='LightPink', title_font = {"size": 20}),
    yaxis=dict(title=r'$\lambda_{k} / \lambda_{tot}$', type='log', showgrid=True, gridwidth=1, gridcolor='LightPink', ),
    showlegend=False
)

layout = go.Layout(
    height=600,
    width=800,
    title="Scree diagram",
    xaxis=dict(title="Singular value index", type='log', showgrid=True, gridwidth=1, gridcolor='LightPink', title_font = {"size": 20}),
    yaxis=dict(title=r'$\lambda_{k} / \lambda_{tot}$', type='log', showgrid=True, gridwidth=1, gridcolor='LightPink', ),
    showlegend=False
)
x_elem_0 = mySSA.X_elementary(r=0)
x_elem_0.shape
w_corr_mat = mySSA.w_correlation_matrix(10)
go.Figure(data=[go.Heatmap(z=w_corr_mat)], layout=go.Layout(height=800,width=800))

trend = mySSA.X_tilde(0)
# By definition, the reconstructed components must have the same dimension as the original signal:
trend.shape[0] == len(raw.data.index)

et12 = mySSA.X_tilde([1,2])
et34 = mySSA.X_tilde([3,4])
layout = go.Layout(
    height=600,
    width=800,
    title="",
    xaxis=dict(title='Date Time'),
    yaxis=dict(title='Count'),
    shapes=[],
    showlegend=True
)

go.Figure(data=[
    go.Scatter(x=data.index,y=data, name='Activity'),
    go.Scatter(x=data.index,y=trend, name='Trend'),
    go.Scatter(x=data.index,y=trend+et12, name='Circadian component'),
    go.Scatter(x=data.index,y=trend+et34, name='Ultradian component')
], layout=layout)

rec = mySSA.reconstructed_signal([0,1,2,3,4,5,6])
go.Figure(data=[
    go.Scatter(x=raw.data.index, y=raw.data, name='Activity'),
    go.Scatter(x=raw.data.index, y=rec, name='Reconstructed signal')
], layout=go.Layout(height=600,width=800,showlegend=True))


