import pywt
from YahooData import *
from model import *
def find_best_wavelet(data):
  """Finds the best wavelet function for a given time series data.

  Args:
    data: A NumPy array of time series data.

  Returns:
    The name of the best wavelet function.
  """

  wavelets = ["haar", "db1", "db2", "db3", "db4", "db5", "db6", "db7", "db8", "db9"]

  # Compute the wavelet transform for each wavelet function.
  wavelet_transforms = []
  for wavelet in wavelets:
    #pywt.wavedec(index_list,wavefunc,mode='sym',level=lv)
    #wavelet_transform = pywt.cwt(data,np.arange(1,len(data)), wavelet)
    wavelet_transform = WT(data,wavelet, 3)
    wavelet_transforms.append(wavelet_transform)

  # Compute the mean squared error for each wavelet transform.
  mean_squared_errors = []
  for wavelet_transform in wavelet_transforms:
    mean_squared_error = np.mean(np.square(wavelet_transform))
    mean_squared_errors.append(mean_squared_error)

  # Find the wavelet function with the lowest mean squared error.
  best_wavelet = wavelets[np.argmin(mean_squared_errors)]

  return best_wavelet


import numpy as np

# Load the time series data.
df=GetYahooData_v2('QQQ',500,'1d')
data=df['Close']
#data = np.loadtxt("data.csv", delimiter=",")

# Find the best wavelet function.
best_wavelet = find_best_wavelet(data)

# Print the name of the best wavelet function.
print(best_wavelet)

