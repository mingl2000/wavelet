import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import pandas as pd
import TDXData3 as TDX
import mplfinance as mpf
# Fetch stock data with High and Low prices

def get_stock_data(ticker, period, interval):
    if ticker.lower().endswith(('.sz','ss')):
        df= TDX.GetTDXData_v3(ticker, period, interval)
    else:
        df = yf.download(ticker,period=period, interval=interval)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
    return df[-period:-1]

def handle_kline_process_include(df):
    df_new =df
    df_new['direction'] = None
    for i in range(1, len(df)): 
        if df.iloc[i]['High'] > df.iloc[i-1]['High'] and df.iloc[i]['Low'] < df.iloc[i-1]['Low'] or df.iloc[i]['High'] < df.iloc[i-1]['High'] and df.iloc[i]['Low'] > df.iloc[i-1]['Low']:
            if df.iloc[i-1]['direction'] == 'up':
                df_new.at[df.index[i],'High']=max(df_new.iloc[i]['High'],df_new.iloc[i-1]['High'])
                df_new.at[df.index[i],'Low']=max(df_new.iloc[i]['Low'],df_new.iloc[i-1]['Low'])
                df_new.at[df.index[i],'direction'] =  'up'
            elif df.iloc[i-1]['direction'] == 'down':
                df_new.at[df.index[i],'High']=min(df_new.iloc[i]['High'],df_new.iloc[i-1]['High'])
                df_new.at[df.index[i],'Low']=min(df_new.iloc[i]['Low'],df_new.iloc[i-1]['Low'])
                df_new.at[df.index[i],'direction'] =  'down'
            else:
                s=""
                pass
        elif df.iloc[i]['High'] > df.iloc[i-1]['High'] and df.iloc[i]['Low'] > df.iloc[i-1]['Low']:
            df_new.at[df.index[i],'direction'] =  'up'
        elif df.iloc[i]['High'] < df.iloc[i-1]['High'] and df.iloc[i]['Low'] < df.iloc[i-1]['Low']:
            df_new.at[df.index[i],'direction'] =  'down'
        else:
            s=""
            pass
    df_new.at[df.index[0],'direction'] =df.iloc[1]['direction'] 
    for i in range(len(df_new)):
        if df.iloc[i]['direction'] == None:
            print(i, df[i-2:i+1])
            print(' direction not set---------------------------------------------------------------')

    return df_new       
# Wave annotation function using High and Low prices
def annotate_waves(df, high_prices, low_prices):
    # Find peaks in High prices (wave tops)
    peaks, _ = find_peaks(high_prices, distance=9, prominence=0.5)
    # Find troughs in Low prices (wave bottoms)
    troughs, _ = find_peaks(-low_prices, distance=9, prominence=0.5)
    
    # Combine and sort critical points
    critical_points = np.concatenate((peaks, troughs))
    critical_points = np.sort(np.unique(critical_points))
    
    # Determine if each critical point is a peak or trough and get corresponding price
    prices_at_points = []
    new_critical_points=[]
    high_low_labels = []
    last = None
    for point in critical_points:
        if point in peaks and point in troughs:
            continue
        elif point in peaks:
            if last == 'peak':
                if high_prices[point] > prices_at_points[-1]:
                    prices_at_points.pop()
                    high_low_labels.pop()
                    new_critical_points.pop()
                else: 
                    continue            
            prices_at_points.append(high_prices[point])  # Use High price for peaks
            high_low_labels.append('P')
            new_critical_points.append(point)
            last = 'peak'
        elif point in troughs:
            if last == 'trough':
                if low_prices[point] < prices_at_points[-1]:
                    prices_at_points.pop() 
                    high_low_labels.pop()
                    new_critical_points.pop()
                else: 
                    continue            
            prices_at_points.append(low_prices[point])  # Use Low price for troughs
            high_low_labels.append('T')
            new_critical_points.append(point)
            last = 'trough'
    
    # Assign wave labels (simplified: 1-5 for impulse, A-C for correction)
    wave_labels = []
    for i in range(len(new_critical_points)):
        if i % 8 < 5:  # Impulse waves (1, 2, 3, 4, 5)
            wave_labels.append(str((i % 5) + 1))
        else:  # Corrective waves (A, B, C)
            wave_labels.append(chr(65 + (i % 3)))  # A, B, C
    
    df['peak'] = np.nan
    df['trough'] = np.nan
    for i in range(len(new_critical_points)):
        idx = critical_points[i]
        if high_low_labels[i] == 'P':
            df.at[df.index[idx],'peak'] =  True
        elif high_low_labels[i] == 'T':
            df.at[df.index[idx],'trough'] =  True
    return critical_points, prices_at_points, wave_labels, new_critical_points, high_low_labels

# Plot the data with wave annotations
def plot_waves(ticker, df,high_prices, low_prices, critical_points, prices_at_points, high_low_labels,wave_labels, mark_weekly_pts):
    figsize = figsize=(12, 6)
    mc = mpf.make_marketcolors(
                           volume='lightgray'
                           )
    s  = mpf.make_mpf_style(marketcolors=mc)
    line_points =[]
    df['Impulse']=np.nan
    df['Corrective']=np.nan
    for idx in range(len(critical_points)):
        line_points.append((df.index[critical_points[idx]], prices_at_points[idx]))
        if wave_labels[idx] in ['1','2','3','4','5']:
            df.at[df.index[critical_points[idx]],'Impulse'] = prices_at_points[idx]        
        elif wave_labels[idx] in ['A','C']:
            df.at[df.index[critical_points[idx]],'Corrective'] = prices_at_points[idx]
        

    apds = [ 

         #mpf.make_addplot(df['Impulse'],type='scatter',color="r",marker='^',markersize=100, label="make_addplot(type='step', label='...')"),
         #mpf.make_addplot(df['Corrective'],type='scatter',color="r",marker='v',markersize=100,label="make_addplot(type='step', label='...')"),
         #mpf.make_addplot((df['PercentB']),panel=1,color='y',label="make_addplot(type='line',panel=1, label='...')")
       ]

    if mark_weekly_pts:
        apds.append(mpf.make_addplot(df['weekly_trough'],type='scatter',color="r",marker='v',markersize=100, label="weekly_trough"))
        apds.append(mpf.make_addplot(df['weekly_peak'],type='scatter',color="g",marker='^',markersize=100, label="weekly_peak"))
    mpf.plot(df,type='candle',volume=False,addplot=apds, alines=line_points, figsize=figsize,tight_layout=True,style=s,returnfig=True,block=False)


    
    #mpf.plot(df, type='candle', style='charles', volume=True, title=f'{ticker} Price with Elliott Wave Annotations', ylabel='Price')
    #plt.plot(high_prices, label=f'{ticker} High Price', color='green')
    #plt.plot(low_prices, label=f'{ticker} Low Price', color='red')
    #plt.plot(critical_points, prices_at_points, "bo", label="Wave Points")  # Mark critical points
    
    # Annotate waves
    '''
    for i, point in enumerate(critical_points):
        plt.annotate(wave_labels[i], (point, prices_at_points[i]), 
                     textcoords="offset points", xytext=(0, 10), ha='center')
    '''
    plt.title(f'{ticker} Price with Elliott Wave Annotations (High/Low)')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    


def get_custom_week_df(df):
    
    df['Custom_Week'] = -((len(df) - df.reset_index().index - (5-len(df)%5)) // 5-len(df)//5+1)
    #df['Custom_Week'] = df.reset_index().index + (5-len(df)%5)-(len(df)) // 5-len(df)//5+1
    # Aggregate data based on the custom 5-day week definition
    df['Date']=df.index
    weekly_df = df.groupby('Custom_Week').agg({
        'Date': 'first',   # Date of the first day in the 5-day period
        'Open': 'first',   # Open price of the first day in the 5-day period
        'High': 'max',     # Highest price within the 5-day period
        'Low': 'min',      # Lowest price within the 5-day period
        'Close': 'last',    # Close price of the last day in the 5-day period
        'Volume': 'sum'   # Total volume over the 5-day period
    }).sort_index(ascending=True)  # Sort to show latest week first

    
    weekly_df.index = weekly_df['Date']
    print(weekly_df)
    return weekly_df

from datetime import datetime, timedelta
def lable_daily_by_weekly(df, weekly_df):
    df['weekly_peak'] = np.nan
    df['weekly_trough'] = np.nan
    
    week_peak_start=0
    week_peak_end =0 
    next_week_peak_date=weekly_df[weekly_df['peak']==True].index[week_peak_start]
    next_week_trough_date=weekly_df[weekly_df['trough']==True].index[week_peak_end]
    next_week_peak_date_end = next_week_peak_date+ timedelta(days=5)
    next_week_trough_date_end = next_week_trough_date+ timedelta(days=5)


    for idx in df.index:
        if df.loc[idx]['peak']==True and idx>=next_week_peak_date and idx<=next_week_peak_date_end:
            df.at[idx,'weekly_peak']=df.loc[idx]['High']
            week_peak_start= week_peak_start+1
            next_week_peak_date=weekly_df[weekly_df['peak']==True].index[week_peak_start]            
            next_week_peak_date_end = next_week_peak_date+ timedelta(days=5)
            

        elif df.loc[idx]['trough']==True and idx>=next_week_trough_date and idx<=next_week_trough_date_end:
            df.at[idx, 'weekly_trough']=df.loc[idx]['Low']
            week_peak_end= week_peak_end+1
            next_week_trough_date=weekly_df[weekly_df['trough']==True].index[week_peak_end]
            next_week_trough_date_end = next_week_trough_date+ timedelta(days=5)

    return df


def process_waves(ticker):
    data = get_stock_data(ticker, 501, '1d')
    weekly_data = get_custom_week_df(data)
    data = handle_kline_process_include(data)
    high_prices = data['High'].values
    low_prices = data['Low'].values
    critical_points, prices_at_points, wave_labels, new_critical_points, high_low_labels = annotate_waves(data, high_prices, low_prices)
    
    
    week_high_prices = weekly_data['High'].values
    week_low_prices = weekly_data['Low'].values
    week_critical_points, week_prices_at_points, week_wave_labels, week_new_critical_points, week_high_low_labels = annotate_waves(weekly_data, week_high_prices, week_low_prices)
    data = lable_daily_by_weekly(data, weekly_data)
    s=""
    plot_waves(ticker, data,high_prices, low_prices, new_critical_points, prices_at_points, high_low_labels, wave_labels, mark_weekly_pts=True)
    #plot_waves(ticker, weekly_data,week_high_prices, week_low_prices, week_new_critical_points, week_prices_at_points, week_high_low_labels, week_wave_labels , mark_weekly_pts=False)
    


process_waves('002049.sz')
plt.show()