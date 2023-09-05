from renko import Renko
from YahooData import *
import math
import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt
# a between b and c
def between (a,low, high):
    return a>=min(low,high) and a<=max(low,high)

brick_size=2
df=GetYahooData_v2('QQQ', 50,'1h')
rnk = Renko(brick_size, df['Close'])
rnk.create_renko()

brick_open_0=rnk.bricks[0]['open']
brick_close_0=rnk.bricks[0]['close']

mybricks=[]
mybricks.append(brick_close_0)

for i in range(1,len(df)):
    df_close=df['Close'][i]
    if df_close>brick_close_0:
        mybricks.append( math.floor((df_close-brick_close_0)/brick_size)*brick_size+brick_close_0)
    else:
        mybricks.append( brick_close_0-1*math.floor((brick_close_0-df_close)/brick_size)*brick_size)

df['bricks']=mybricks
mc = mpf.make_marketcolors(
                           volume='lightgray'
                           )

figsize=(26,13)                          
s  = mpf.make_mpf_style(marketcolors=mc, gridaxis='both')

apdict = [mpf.make_addplot(df['bricks'], width=3, color='r',type='scatter',panel=0,markersize=200,marker='s')]
fig1,ax1=mpf.plot(df,type='candle',volume=False,addplot=apdict, figsize=figsize,tight_layout=True,style=s,returnfig=True,block=False)
plt.show()
'''
i=1  # df
j=1 # rnk.bricks
while i<len(df) and j<len(rnk.bricks):
    df_close=close=df['Close'][i]
    brick_open=rnk.bricks[j]['open']
    brick_close=rnk.bricks[j]['close']
    brick_type=rnk.bricks[j]['type']  # first, down, up
    brick_low=min(brick_open,brick_close)
    brick_high=max(brick_open,brick_close)
    if between(df_close, brick_low, brick_high):
        # df[i] in brick j
        print (i, ' in brick ',j)
        i= i+1
    elif df_close<brick_low and brick_type=='down':
        j=j+1  # move to next brick down
    elif df_close>=brick_high and brick_type=='up':
        j=j+1  # move to next brick up
    elif df_close<brick_low and brick_type=='up':
        i=i+1  # move to next bar
    elif df_close>=brick_high and brick_type=='down':
        i=i+1  # move to next bar
    else:
        print ('i=' ,i, 'j=', j, ' not covered')
        pass
'''    


