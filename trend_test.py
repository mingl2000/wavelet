import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
file_path = "./Series_1.xlsx"

df = pd.read_excel(file_path)


X = df["Value"].values
result = adfuller(X)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))

if result[0] < result[4]["5%"]:
    print ("Reject Ho - Time Series is Stationary")
else:
    print ("Failed to Reject Ho - Time Series is Non-Stationary")

df["Diff_Value"] = df["Value"].diff()
diff_values = df["Diff_Value"].values
Y = diff_values[~np.isnan(diff_values)]
ts_values_orig = df["Value"].values
ts_values = ts_values_orig[:-1]
X = sm.add_constant(ts_values)
model = sm.OLS(Y,X)
results = model.fit()
print(results.summary())
results.tvalues[1]    