import pandas as pd
from patsy import dmatrices
from collections import OrderedDict
import itertools
import statsmodels.formula.api as smf
import sys
import matplotlib.pyplot as plt

#Read the data set into a pandas DataFrame
df = pd.read_csv('boston_daily_temps_1978_2019.csv', header=0, infer_datetime_format=True, parse_dates=[0], index_col=[0])

#resample at a month level
df_resampled = df.resample('M').mean()

#Plot the data set
fig = plt.figure()
fig.suptitle('Monthly Average temperatures in Boston, MA from 1978 to 2019')
actual, = plt.plot(df_resampled.index, df_resampled['TAVG'], 'go-', label='Monthly Average Temperature')
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.legend(handles=[actual])
plt.show()

print(df_resampled.head())

#Take a copy
df_lagged = df_resampled.copy()

#Add time lagged columns to the data set
for i in range(1, 13, 1):
	df_lagged['TAVG_LAG_' + str(i)] = df_lagged['TAVG'].shift(i)

print(df_lagged.head(13))

#Drop the NaN rows
for i in range(0, 12, 1):
	df_lagged = df_lagged.drop(df_lagged.index[0])

print(df_lagged.head())

#Carve out the test and the training data sets
split_index = round(len(df_lagged)*0.8)
split_date = df_lagged.index[split_index]
df_train = df_lagged.loc[df_lagged.index <= split_date].copy()
df_test = df_lagged.loc[df_lagged.index > split_date].copy()

#Generate and store away, all possible combinations of the list [1,2,3,4,5,6,7,8,9,10,11,12]
lag_combinations = OrderedDict()
l = list(range(1,13,1))

for i in range(1, 13, 1):
	for combination in itertools.combinations(l, i):
		lag_combinations[combination] = 0.0

print('Number of combinations to be tested: ' + str(len(lag_combinations)))

expr_prefix = 'TAVG ~ '

min_aic = sys.float_info.max
best_expr = ''
best_olsr_model_results = None

#Iterate over each combination
for combination in lag_combinations:
	expr = expr_prefix
	i = 1
	#Setup the model expression using patsy syntax
	for lag_num in combination:
		if i < len(combination):
			expr = expr + 'TAVG_LAG_' + str(lag_num) + ' + '
		else:
			expr = expr + 'TAVG_LAG_' + str(lag_num)

		i += 1

	print('Building model for expr: ' + expr)

	#Carve out the X,y vectors using patsy. We will use X_test, y_test later for testing the model.
	y_test, X_test = dmatrices(expr, df_test, return_type='dataframe')

	#Build and train the OLSR model on the training data set
	olsr_results = smf.ols(expr, df_train).fit()

	#Store it's AIC value
	lag_combinations[combination] = olsr_results.aic

	#Keep track of the best model (the one with the lowest AIC score) seen so far
	if olsr_results.aic < min_aic:
		min_aic = olsr_results.aic
		best_expr = expr
		best_olsr_model_results = olsr_results

	print('AIC='+str(lag_combinations[combination]))

#Print out the model expression, AIC score and model summary of the best model
print('Best expr=' + best_expr)
print('min AIC=' + str(min_aic))
print(best_olsr_model_results.summary())

#Generate predictions for TAVG on the test data set
olsr_predictions = best_olsr_model_results.get_prediction(X_test)

olsr_predictions_summary_frame = olsr_predictions.summary_frame()
print(olsr_predictions_summary_frame.head(10))

actual_temps = y_test['TAVG']
predicted_temps=olsr_predictions_summary_frame['mean']

#Plot the actual versus predicted values of TAVG on the test data set
fig = plt.figure()
fig.suptitle('Predicted versus actual monthly average temperatures')
predicted, = plt.plot(X_test.index, predicted_temps, 'go-', label='Predicted monthly average temp')
actual, = plt.plot(X_test.index, actual_temps, 'ro-', label='Actual monthly average temp')
plt.legend(handles=[predicted, actual])
plt.show()