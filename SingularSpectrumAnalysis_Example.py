import matplotlib.pyplot as plt

from pyts.datasets import load_gunpoint
from pyts.decomposition import SingularSpectrumAnalysis
X, _, _, _ = load_gunpoint(return_X_y=True)
X=X[0:1]
transformer = SingularSpectrumAnalysis(window_size=2)
X_new = transformer.transform(X)
X_new.shape
plt.plot(X_new[0])
plt.show()