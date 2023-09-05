import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_predict


def aic(n, mse, num_params):
    aic = n * np.log(mse) + 2 * num_params
    aic_c = aic + (2*num_params*(num_params+1))/(n - num_params - 1)
    return (aic, aic_c)

# Load data
url = 'https://raw.githubusercontent.com/nevernervous78/nirpyresearch/master/data/peach_spectra_brix.csv'
data = pd.read_csv(url)
X = data.values[:,1:]
y = data["Brix"].values




# Define PCR estimators
pcr1 = make_pipeline(PCA(n_components=5), LinearRegression())
pcr2 = make_pipeline(PCA(n_components=20), LinearRegression())
 
#Cross-validation
y_cv1 = cross_val_predict(pcr1, X, y, cv=10)
y_cv2 = cross_val_predict(pcr2, X, y, cv=10)
 
# Calculate MSE
mse1 = mean_squared_error(y, y_cv1)
mse2 = mean_squared_error(y, y_cv2)
 
# Compute AIC
aic1, aicc1 = aic(X.shape[0], mse1, pcr1.steps[0][1].n_components+1)
aic2, aicc2 = aic(X.shape[0], mse2, pcr2.steps[0][1].n_components+1)
 
# Print data
print("AIC, model 1:", aic1)
print("AICc, model 1:", aicc1)
print("AIC, model 2:", aic2)
print("AICc, model 2:", aicc2)


'''
expect to the following output
AIC, model 1: 69.30237521019883
AICc, model 1: 71.25586358229185
AIC, model 2: 100.67687348181386
AICc, model 2: 133.67687348181386
'''

ncomp = np.arange(1,20,1)
AIC = np.zeros_like(ncomp)
AICc = np.zeros_like(ncomp)
for i, nc in enumerate(ncomp):
 
    pcr = make_pipeline(PCA(n_components=nc), LinearRegression())
    y_cv = cross_val_predict(pcr, X, y, cv=10)
 
    mse = mean_squared_error(y, y_cv)
    AIC[i], AICc[i] = aic(X.shape[0], mse, pcr.steps[0][1].n_components+1)
 
plt.figure(figsize=(7,6))
with plt.style.context(('seaborn')):
    plt.plot(ncomp, AIC, 'k', lw=2, label="AIC")
    plt.plot(ncomp,AICc, 'r', lw=2, label="AICc")
plt.xlabel("LV")
plt.ylabel("AIC/AICc value")
plt.tight_layout()
plt.legend()
plt.show()