from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt


data = [3, 4, 4, 5, 6, 7, 6, 6, 7, 8, 9, 12, 10]
plt.plot(data)

#perform augmented Dickey-Fuller test
(testStatistic,p,_,_,_,_)=result=adfuller(data)
if p<0.05:
    print('No trend')
else:
    print('There is a trend', 'testStatistic=',testStatistic )
plt.show()
