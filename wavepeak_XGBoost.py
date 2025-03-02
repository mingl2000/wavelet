import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import pandas as pd
import TDXData3 as TDX
import mplfinance as mpf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Fetch stock data with High and Low prices
def calculate_macd(df, fast=12, slow=26, signal=9):
    """Calculate MACD indicators"""
    df['EMA_fast'] = df['Close'].ewm(span=fast, adjust=False).mean()
    df['EMA_slow'] = df['Close'].ewm(span=slow, adjust=False).mean()
    df['MACD'] = df['EMA_fast'] - df['EMA_slow']
    df['Signal'] = df['MACD'].ewm(span=signal, adjust=False).mean()
    df['Histogram'] = df['MACD'] - df['Signal']
    return df

def find_divergences(df):
    """Detect bullish and bearish divergences"""
    bullish_div = []
    bearish_div = []
    df['MACD'] =np.nan
    # Look for divergences in last 100 periods
    for i in range(len(df) - 26, len(df)):
        # Bullish divergence: Price makes lower low, MACD makes higher low
        if (i > 2 and 
            df['Close'].iloc[i] < df['Close'].iloc[i-1] and 
            df['MACD'].iloc[i] > df['MACD'].iloc[i-1] and
            df['Close'].iloc[i-1] < df['Close'].iloc[i-2] and
            df['MACD'].iloc[i-1] < df['MACD'].iloc[i-2]):
            bullish_div.append(i)
            
        # Bearish divergence: Price makes higher high, MACD makes lower high
        if (i > 2 and 
            df['Close'].iloc[i] > df['Close'].iloc[i-1] and 
            df['MACD'].iloc[i] < df['MACD'].iloc[i-1] and
            df['Close'].iloc[i-1] > df['Close'].iloc[i-2] and
            df['MACD'].iloc[i-1] > df['MACD'].iloc[i-2]):
            bearish_div.append(i)
    
    return bullish_div, bearish_div

def plot_macd_with_divergence(df, bullish_div, bearish_div, symbol):
    """Plot price, MACD and divergences"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[2, 1])
    
    # Price plot
    ax1.plot(df.index, df['Close'], label='Price')
    ax1.set_title(f'{symbol} Price and MACD with Divergences')
    ax1.set_ylabel('Price')
    ax1.legend()
    ax1.grid(True)
    
    # MACD plot
    ax2.plot(df.index, df['MACD'], label='MACD', color='blue')
    ax2.plot(df.index, df['Signal'], label='Signal', color='orange')
    ax2.bar(df.index, df['Histogram'], label='Histogram', color='gray')
    
    # Plot divergences
    for idx in bullish_div:
        ax1.plot(df.index[idx], df['Close'].iloc[idx], 'g^', markersize=10)
        ax2.plot(df.index[idx], df['MACD'].iloc[idx], 'g^', markersize=10)
        
    for idx in bearish_div:
        ax1.plot(df.index[idx], df['Close'].iloc[idx], 'rv', markersize=10)
        ax2.plot(df.index[idx], df['MACD'].iloc[idx], 'rv', markersize=10)
    
    ax2.set_ylabel('MACD')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()
def get_stock_data(ticker, period, interval):
    if ticker.lower().endswith(('.sz','ss')):
        df= TDX.GetTDXData_v3(ticker, period, interval)
    else:
        df = yf.download(ticker,period=period, interval=interval)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
    return df[-period:]

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
    
    df['peak'] = 0
    df['trough'] = 0
    df['peak_trough'] = 0
    for i in range(len(new_critical_points)):
        idx = critical_points[i]
        if high_low_labels[i] == 'P':
            df.at[df.index[idx],'peak'] =  1
        elif high_low_labels[i] == 'T':
            df.at[df.index[idx],'trough'] =  1
        
        if idx+1<len(df) and df.iloc[idx]['peak']==1:
            df.at[df.index[idx+1],'peak_trough'] =  2
        if idx+1<len(df) and df.iloc[idx]['trough']==1:
            df.at[df.index[idx+1],'peak_trough'] =  1    
        
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
        if len(df[(~np.isnan(df['weekly_trough']))])>0:
            apds.append(mpf.make_addplot(df['weekly_trough'],type='scatter',color="r",marker='^',markersize=300, label="weekly_trough"))
        if len(df[(~np.isnan(df['weekly_peak']))])>0:
            apds.append(mpf.make_addplot(df['weekly_peak'],type='scatter',color="g",marker='v',markersize=300, label="weekly_peak"))
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
    week_trough_start =0 
    next_week_peak_date=weekly_df[weekly_df['peak']==True].index[week_peak_start] 
    next_week_trough_date=weekly_df[weekly_df['trough']==True].index[week_trough_start] 
    next_week_peak_date, next_week_peak_date_end = next_week_peak_date+ timedelta(days=-10), next_week_peak_date+ timedelta(days=10)
    next_week_trough_date, next_week_trough_date_end = next_week_trough_date+ timedelta(days=-10), next_week_trough_date+ timedelta(days=10)


    for idx in df.index:
        if df.loc[idx]['peak']==True and idx>=next_week_peak_date and idx<next_week_peak_date_end:
            df.at[idx,'weekly_peak']=df.loc[idx]['High']
            week_peak_start= week_peak_start+1
            if week_peak_start>=len(weekly_df[weekly_df['peak']==True]):
                break
            next_week_peak_date=weekly_df[weekly_df['peak']==True].iloc[week_peak_start]['Date']
            next_week_peak_date, next_week_peak_date_end = next_week_peak_date+ timedelta(days=-10), next_week_peak_date+ timedelta(days=10)
            

        elif df.loc[idx]['trough']==True and idx>=next_week_trough_date and idx<next_week_trough_date_end:
            df.at[idx, 'weekly_trough']=df.loc[idx]['Low']
            week_trough_start= week_trough_start+1
            if week_trough_start>=len(weekly_df[weekly_df['trough']==True]):
                break
            next_week_trough_date = weekly_df[weekly_df['trough']==True].iloc[week_trough_start]['Date']
            #next_week_trough_date=weekly_df[weekly_df['trough']==True].index[week_peak_end]
            next_week_trough_date, next_week_trough_date_end = next_week_trough_date+ timedelta(days=-10), next_week_trough_date+ timedelta(days=10)
    return df


def process_waves(ticker, toprint=False):
    data = get_stock_data(ticker, 5000, '1d')
    weekly_data = get_custom_week_df(data)
    data = handle_kline_process_include(data)
    weekly_data = handle_kline_process_include(weekly_data)
    high_prices = data['High'].values
    low_prices = data['Low'].values
    critical_points, prices_at_points, wave_labels, new_critical_points, high_low_labels = annotate_waves(data, high_prices, low_prices)
    
    
    
    week_high_prices = weekly_data['High'].values
    week_low_prices = weekly_data['Low'].values
    week_critical_points, week_prices_at_points, week_wave_labels, week_new_critical_points, week_high_low_labels = annotate_waves(weekly_data, week_high_prices, week_low_prices)
    data = lable_daily_by_weekly(data, weekly_data)
    
    if toprint:
        print(data[(~np.isnan(data['weekly_peak']))])
        print(weekly_data[weekly_data['peak'] == True])
        plot_waves(ticker, data,high_prices, low_prices, new_critical_points, prices_at_points, high_low_labels, wave_labels, mark_weekly_pts=True)
        plot_waves(ticker, weekly_data,week_high_prices, week_low_prices, week_new_critical_points, week_prices_at_points, week_high_low_labels, week_wave_labels , mark_weekly_pts=False)
    return data

def calculate_features(ticker):
    df = process_waves('399006.sz')
    df['weekly_peak'] = ~np.isnan(df['weekly_peak'])
    df['weekly_trough'] = ~np.isnan(df['weekly_trough'])
    df['Lag1_Return'] =df['Close'].pct_change().shift(1)
    df['Lag5_Return'] =df['Close'].pct_change().shift(5)
    df['Lag10_Return'] =df['Close'].pct_change().shift(10)
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['SMA_Crossover']= (df['MA10']>df['MA50']).astype(int).shift(1)
    
    df['High_Low'] = df['High'] - df['Low']
    df['High_PrevClose'] = abs(df['High'] - df['Close'].shift(1))
    df['Low_PrevClose'] = abs(df['Low'] - df['Close'].shift(1))
    df['TR'] = df[['High_Low', 'High_PrevClose', 'Low_PrevClose']].max(axis=1)
    df['+DM'] = np.where((df['High'] - df['High'].shift(1)) > (df['Low'].shift(1) - df['Low']),
                         (df['High'] - df['High'].shift(1)).clip(lower=0), 0)
    df['-DM'] = np.where((df['Low'].shift(1) - df['Low']) > (df['High'] - df['High'].shift(1)),
                         (df['Low'].shift(1) - df['Low']).clip(lower=0), 0)
    tr14 = df['TR'].rolling(window=14).sum()
    plus_dm14 = df['+DM'].rolling(window=14).sum()
    minus_dm14 = df['-DM'].rolling(window=14).sum()
    df['+DI'] = 100 * (plus_dm14 / tr14)
    df['-DI'] = 100 * (minus_dm14 / tr14)
    dx = 100 * abs(df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI'])
    df['ADX'] = dx.rolling(window=14).mean()

    df['Direction'] = (df['Close'].shift(-1)>df['Close']).astype(int)
    #bullish_div, bearish_div = find_divergences(df)
    '''
    df =df [[ticker]].copy
    df.columns = ['Price']
    df['Lag1_Return'] =df['Price'].pct_change().shift(1)
    df['MA10'] = df['Price'].rolling(window=10).mean()
    df['MA50'] = df['Price'].rolling(window=50).mean()
    df['SMA_Crossover']= (df['MA10']>df['MA50']).astype(int).shift(1)
    df['Weekly_Return'] = df['Price'].pct_change(7).shift(1)
    df['Direction'] = (df['Price'].shift(-1)>df['Price']).astype(int)
    '''
    return df.dropna()
def xgboost_model(df):
    X =df[['Lag1_Return', 'Lag5_Return', 'Lag10_Return', 'SMA_Crossover','ADX','TR']]
    #X =df[['SMA_Crossover']]
    y = df['peak_trough']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = XGBClassifier(max_depth=8, n_estimators=100, learning_rate=0.1)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    print(f'Model Accuracy: {accuracy:4f}')
def main():
    #process_waves('002049.sz', False)
    df =calculate_features('399006.sz')
    xgboost_model(df)
    print('done')
    
    #plt.show()


if __name__ == "__main__":
    main()