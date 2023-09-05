#from pyts.decomposition import SSA
from pyts.decomposition import SingularSpectrumAnalysis
from sklearn.decomposition import PCA
import numpy as np

# Generate some time series data
X = np.random.randn(100, 10)

# Apply SSA to decompose the time series into components
ss = SingularSpectrumAnalysis(window_size=10)
X_ssa = ss.fit_transform(X)

# Apply PCA to each component
X_pca = []
for i in range(X_ssa.shape[1]):
    pca = PCA(n_components=3)
    X_pca.append(pca.fit_transform(X_ssa[:, i, :]))

# Reconstruct the time series
ts_ssa_pca=np.stack(X_pca, axis=1)
print(ts_ssa_pca.shape)
#X_reconstructed = ss.inverse_transform(np.stack(X_pca, axis=1))

# Check the shape of the reconstructed data
#print(X_reconstructed.shape)
