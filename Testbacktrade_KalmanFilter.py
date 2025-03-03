from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import datetime  # For datetime objects
import os.path  # To manage paths
import sys  # To find out the script name (in argv[0])

# Import the backtrader platform
import backtrader as bt
from pyts.decomposition import SingularSpectrumAnalysis
from backtrader.indicators import Indicator
from YahooData import *
import statsmodels.api as sm
import copy
from pykalman import KalmanFilter
#from backtrader import Cerebro, PercentSizer
def DoMarkovRegression(df, col1, col2):
  #build and train the MSDR model
  msdr_model = sm.tsa.MarkovRegression(endog=df[col1], k_regimes=2,
  trend='c', exog=df[col2], switching_variance=True)
  msdr_model_results = msdr_model.fit(iter=1000)
  return msdr_model_results

def DoKalmanFilter2(arr):
    kf = KalmanFilter(
      initial_state_mean=arr[0],
      initial_state_covariance=1,
      observation_covariance=1,
      transition_covariance=0.01
    )
    state_means, _ = kf.filter(arr)
    state_means_smooth, _ = kf.smooth(arr)
    return (state_means, state_means_smooth)

def DoKalmanFilter(data):
    kf = KalmanFilter(
      initial_state_mean=df['Close'][0],
      initial_state_covariance=1,
      observation_covariance=1,
      transition_covariance=0.01
    )
    state_means, _ = kf.filter(data.close)
    state_means_smooth, _ = kf.smooth(data.close)
    return (state_means, state_means_smooth)
    '''
      # Run Kalman filter on stock data
    state_means, _ = kf.filter(df['Close'].values)
    state_means_high, _ = kf.filter(df['High'].values)
    state_means_low, _ = kf.filter(df['Low'].values)
    state_means_vol, _ = kf.filter(df['Volume'].values)
    # Apply RTS smoothing algorithm
    state_means_smooth, _ = kf.smooth(df['Close'].values)
    state_means_smooth_high, _ = kf.smooth(df['High'].values)
    state_means_smooth_low, _ = kf.smooth(df['Low'].values)
    state_means_smooth_vol, _ = kf.smooth(df['Volume'].values)
    #plt.show()
    df['Filtered_close']=state_means
    df['Smoothed_close']=state_means_smooth
    df['Filtered_high']=state_means_high
    df['Smoothed_high']=state_means_smooth_high
    df['Filtered_low']=state_means_low
    df['Smoothed_low']=state_means_smooth_low

    df['Filtered_vol']=state_means_vol
    df['Smoothed_vol']=state_means_smooth_vol
    return df
    '''
def getMaxMin(df):
    _max=max(df['High'].values)
    _min=min(df['Low'].values)
    return (_max, _min)


class TestStrategy(bt.Strategy):
    params = (
        ('maperiod', 15),
        ('fastPeriod', 13),
        ('slowPeriod', 26),
    )

    def log(self, txt, dt=None):
        ''' Logging function fot this strategy'''
        dt = dt or self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))

    def __init__(self):
        # Keep a reference to the "close" line in the data[0] dataseries
        self.dataclose = self.datas[0].close

        # To keep track of pending orders and buy price/commission
        self.order = None
        self.buyprice = None
        self.buycomm = None

        # Add a MovingAverageSimple indicator
        self.barIdx=0
        #self.df= GetYahooData_v2('^GSPC',730,'1d')
        #currentdf=self.df[0:self.barIdx]
        #(self.KF, self.KFSmooth)=DoKalmanFilter(self.datas[0])
        self.last_kf_min=10000000
        self.last_kf_max=0
        (self.kf, self.kf_smooth)=DoKalmanFilter2(self.datas[0].close.array[:])
        (self.kfv, self.kfv_smooth)=DoKalmanFilter2(self.datas[0].volume.array[:])


    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # Buy/Sell order submitted/accepted to/by broker - Nothing to do
            return

        # Check if an order has been completed
        # Attention: broker could reject order if not enough cash
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(
                    'BUY EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                    (order.executed.price,
                     order.executed.value,
                     order.executed.comm))

                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            else:  # Sell
                self.log('SELL EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                         (order.executed.price,
                          order.executed.value,
                          order.executed.comm))

            self.bar_executed = len(self)

        #elif order.status in [order.Canceled, order.Margin, order.Rejected]:
        elif order.status in [order.Canceled, order.Rejected]:
            self.log('Order Canceled/Rejected')

        # Write down: no pending order
        self.order = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return

        self.log('OPERATION PROFIT, GROSS %.2f, NET %.2f' %
                 (trade.pnl, trade.pnlcomm))

    def next(self):
        # Simply log the closing price of the series from the reference
        self.log('Close, %.2f' % self.dataclose[0])
        # Check if an order is pending ... if yes, we cannot send a 2nd one
        if self.order:
            return
        #if self.barIdx>=self.params.slowPeriod and self.barIdx< len(self.df):
        #if self.barIdx< len(self.datas[0].close)-1 and self.barIdx>1:
        print ('self.barIdx=',self.barIdx)
        if self.barIdx>1 and self.barIdx < len(self.datas[0].close.array):
            #(kf, kf_smooth)=DoKalmanFilter2(self.datas[0].close.array[:self.barIdx+1])
            #(kfv, kfv_smooth)=DoKalmanFilter2(self.datas[0].volume.array[:self.barIdx+1])

            if self.kf[self.barIdx][0]<self.last_kf_min:
                self.last_kf_min=self.kf[self.barIdx][0]
            if self.kf[self.barIdx][0]>self.last_kf_max:
                self.last_kf_max=self.kf[self.barIdx][0]

            #closeReverse=copy.deepcopy(self.datas[0].close.array[:self.barIdx+1])
            #closeReverse.reverse()
            #(kf2, kf_smooth2)=DoKalmanFilter2(closeReverse)
            
            #kf_smooth=np.flip(kf_smooth2)
            '''
            newclose=[]
            for i in range(len(kf)):
                newclose.append(kf[i][0])
            (kf_smooth, _)=DoKalmanFilter2(newclose)
            '''
        #if 1==1:
            #currentdf=self.df[0:self.barIdx]
            #currentdf=MarkovRegressionOverKalmanFilter(currentdf)
            #currentdf=self.df[0:self.barIdx]
            #df['MarkovRegression00']=msdr_model_results.filtered_joint_probabilities[0][0]
            #df['MarkovRegression01']=msdr_model_results.filtered_joint_probabilities[0][1]
            #df['MarkovRegression10']=msdr_model_results.filtered_joint_probabilities[1][0]
            #df['MarkovRegression11']=msdr_model_results.filtered_joint_probabilities[1][1]

            # Check if we are in the market
            if not self.position:

                # Not yet ... we MIGHT BUY if ...
                #if self.dataclose[0] > self.sma[0]:
                
                #if self.df['MarkovRegression00'][self.barIdx-1]> self.df['MarkovRegression11'][self.barIdx-1] and self.df['Smoothed_close'][self.barIdx-1]>self.df['Smoothed_close'][self.barIdx-2]:
                #if self.df['MarkovRegression00'][self.barIdx-1]> 0.8 and self.df['Smoothed_close'][self.barIdx-1]>self.df['Smoothed_close'][self.barIdx-2]:
                #if self.df['Smoothed_close'][self.barIdx-1]>self.df['Filtered_close'][self.barIdx-1]:
                #if self.KF[self.barIdx-1]<self.KFSmooth[self.barIdx-1]:
                #if self.KF[self.barIdx-1][0]>self.datas[0].close[self.barIdx-1]:
                #if kf[self.barIdx-1][0] >self.datas[0].close.array[self.barIdx-1]:
                #if kf[self.barIdx][0] >kf[self.barIdx-1][0] and kf[self.barIdx-1][0] <=kf[self.barIdx-2][0]:
                #if kf_smooth[-1][0] >self.datas[0].close.array[self.barIdx]:
                #if kf[-1][0] >self.datas[0].close.array[self.barIdx] and kfv[-1][0] >self.datas[0].volume.array[self.barIdx]: 19.89% QQQ
                if self.kf[self.barIdx][0] >self.last_kf_min*1.01  and self.kf[self.barIdx][0]>self.kf[self.barIdx-1][0]: #26.57% QQQ   88.05% 300750.sz
                    # BUY, BUY, BUY!!! (with all possible default parameters)
                    self.log('BUY CREATE, %.2f' % self.dataclose[0])
                    print( 'kf=',self.kf[self.barIdx][0], 'kf_max=',self.last_kf_max, 'kf_min',self.last_kf_min , 'close=',self.datas[0].close.array[self.barIdx])
                    #self.log('BUY CREATE, %.2f' % self.dataclose[0], 'kf=',self.kf[self.barIdx][0], 'kf_max=',self.last_kf_max, 'kf_min',self.last_kf_min, 'close=',self.datas[0].close.array[self.barIdx] )

                    # Keep track of the created order to avoid a 2nd order
                    self.order = self.buy()

            else:

                #if self.dataclose[0] < self.sma[0]:
                #print(self.df['MarkovRegression00'][self.barIdx-1], self.df['MarkovRegression11'][self.barIdx-1], self.df['Smoothed_close'][self.barIdx-1],self.df['Smoothed_close'][self.barIdx-2] )
                #if self.df['MarkovRegression00'][self.barIdx-1]< self.df['MarkovRegression11'][self.barIdx-1] or self.df['Smoothed_close'][self.barIdx-1]<self.df['Smoothed_close'][self.barIdx-2]:
                #if self.df['MarkovRegression00'][self.barIdx-1]< 0.2 :
                #if self.df['Smoothed_close'][self.barIdx-1]<self.df['Filtered_close'][self.barIdx-1]:
                #if self.KF[self.barIdx-1]>self.KFSmooth[self.barIdx-1]:
                #if kf[self.barIdx-1][0] <self.datas[0].close.array[self.barIdx-1]:
                #if kf[self.barIdx][0] <kf[self.barIdx-1][0] and kf[self.barIdx-1][0] >=kf[self.barIdx-2][0]:    
                #if kf_smooth[-1][0] <self.datas[0].close.array[self.barIdx]:
               #if kf[-1][0] <self.datas[0].close.array[self.barIdx]: 19.89% QQQ
                if self.kf[self.barIdx][0] <self.last_kf_max/1.01 and self.kf[self.barIdx][0]<self.kf[self.barIdx-1][0]:   #26.57% QQQ | 88.05% 300750.sz | 9.27% 002030.sz | 32.84% 002049.sz
                    # SELL, SELL, SELL!!! (with all possible default parameters)
                    self.log('SELL CREATE, %.2f' % self.dataclose[0])
                    print( 'kf=',self.kf[self.barIdx][0], 'kf_max=',self.last_kf_max, 'kf_min',self.last_kf_min , 'close=',self.datas[0].close.array[self.barIdx])

                    # Keep track of the created order to avoid a 2nd order
                    self.order = self.sell()
                    self.last_kf_min=self.kf[self.barIdx][0]
        self.barIdx=self.barIdx+1

import sys
import warnings
if __name__ == '__main__':

    interval='1d'
    symbol='QQQ'
    if len(sys.argv) >=2:
        symbol=sys.argv[1]
    if len(sys.argv) >=3:
        interval=sys.argv[2]

    if symbol=='SPX':
        symbol='^GSPC'

    bars=730
    #symbol='^GSPC'
    df=GetYahooData_v2(symbol, bars=bars, interval=interval)
    # Create a cerebro entity
    cerebro = bt.Cerebro()

    # Add a strategy
    cerebro.addstrategy(TestStrategy)

    # Datas are in a subfolder of the samples. Need to find where the script is
    # because it could have been called from anywhere
    modpath = os.path.dirname(os.path.abspath(sys.argv[0]))
    #dataFileName="data/"+symbol+'_' +interval+'_'+ interval +".csv"
    dataFileName1="data/"+symbol+'_' +'max'+'_'+ interval +".csv"
    dataFileName2="data/"+symbol+'_' +str(bars)+'d_'+ interval +".csv"
    if (exists(dataFileName1)):
        datapath=dataFileName1
    else:
        datapath=dataFileName2
    #datapath = os.path.join(modpath, dataFileName1)

    # Create a Data Feed
    data = bt.feeds.YahooFinanceCSVData(
        dataname=datapath,
        # Do not pass values before this date
        fromdate=datetime.datetime(2020, 4, 9),
        # Do not pass values before this date
        todate=datetime.datetime(2023, 3, 4),
        # Do not pass values after this date
        reverse=False)

    # Add the Data Feed to Cerebro
    cerebro.adddata(data)
    startPortfolio=10000
    # Set our desired cash start
    cerebro.broker.setcash(startPortfolio)

    size=int(startPortfolio/df['Close'][-1])
    # Add a FixedSize sizer according to the stake
    cerebro.addsizer(bt.sizers.FixedSize, stake=size)
    #cerebro.addsizer(PercentSizer, percents=50, cash=True)
    #cerebro.addsizer(bt.sizers.percents_sizer, stake=0.5)

    # Set the commission
    cerebro.broker.setcommission(commission=0.0)

    # Print out the starting conditions
    startPortfolio=cerebro.broker.getvalue()
    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())

    # Run over everything
    cerebro.run()

    # Print out the final result
    print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())
    endPortfolio=cerebro.broker.getvalue()
    fmt='profit: {0:.2f}   {1:.2f}%'
    print(fmt.format(endPortfolio-startPortfolio, (endPortfolio/startPortfolio-1)*100))
    # Plot the result
    cerebro.plot()