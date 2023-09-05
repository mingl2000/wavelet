import numpy as np
from scipy.linalg import svd
from YahooData import *
import matplotlib.pyplot as plt
def ssa(X, L):
    """
    Singular Spectrum Analysis function
    """
    N = len(X)
    K = N - L + 1
    Xtraj = np.zeros((L, K))
    for i in range(K):
        Xtraj[:,i] = X[i:i+L]
    Xtraj = Xtraj - np.mean(Xtraj, axis=1, keepdims=True)
    U, S, V = svd(Xtraj)
    return U, S, V

def select_best_window(X, max_window_size):
    """
    Selects the best window length for Singular Spectrum Analysis
    """
    scores = []
    for L in range(5, max_window_size+1):
        U, S, V = ssa(X, L)
        reconstructed_X = np.dot(U[:,0].reshape(-1,1), V[0,:].reshape(1,-1))
        #score = np.mean((X[L-1:] - reconstructed_X[0]+reconstructed_X[1])**2)
        score = np.corrcoef(X[L-1:], reconstructed_X[0]+reconstructed_X[1])[0, 1]
        scores.append(score)
    best_window = np.argmin(scores) + 5
    return best_window, scores,range(5, max_window_size+1)

# Example usage
X = np.sin(np.arange(100))
df=GetYahooData_v2('399001.sz',500,'1d')
X=df['Close'].to_numpy()
(best_window,scores,X) = select_best_window(X, 120)
print(f"Best window size: {best_window}")
plt.plot(X, scores)
plt.show()


