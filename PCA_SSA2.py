import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from scipy.linalg import svd
from scipy.signal import savgol_filter

# Load Yahoo Finance stock data
tickers = ['AAPL', 'MSFT', 'AMZN', 'GOOG', 'FB']
df = pd.DataFrame()
for ticker in tickers:
    temp_df = pd.read_csv(f'https://query1.finance.yahoo.com/v7/finance/download/{ticker}?period1=1514764800&period2=1614764800&interval=1d&events=history&includeAdjustedClose=true')
    temp_df['Date'] = pd.to_datetime(temp_df['Date'])
    temp_df.set_index('Date', inplace=True)
    temp_df.rename(columns={'Adj Close': ticker}, inplace=True)
    df = pd.concat([df, temp_df[ticker]], axis=1)

# Singular Spectrum Analysis (SSA) function
def ssa(X, L):
    N = len(X)
    K = N - L + 1
    Xmat = np.zeros((L, K))
    for i in range(K):
        Xmat[:, i] = X[i:i+L]
    U, S, Vt = svd(Xmat)
    Ur = U[:, :K]
    Sr = np.diag(S[:K])
    Vtr = Vt[:K, :]
    Xhat = Ur @ Sr @ Vtr
    return Xhat, Ur, Sr, Vtr

# Perform SSA on each stock's time series data
L = 100
Xhat = pd.DataFrame()
for col in df.columns:
    X = df[col].to_numpy()
    Xhat_col, _, _, _ = ssa(X, L)
    Xhat[col] = Xhat_col[:, -1]

# Perform PCA on SSA components
n_components = 2
pca = PCA(n_components=n_components)
pca.fit(Xhat)
principal_components = pca.transform(Xhat)
variance_explained = pca.explained_variance_ratio_

# Plot the results
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 6))
plt.scatter(principal_components[:, 0], principal_components[:, 1])
plt.xlabel(f'Principal Component 1 ({variance_explained[0]:.2f} explained variance)')
plt.ylabel(f'Principal Component 2 ({variance_explained[1]:.2f} explained variance)')
plt.title('PCA on SSA of Yahoo Finance stock data')
plt.show()
