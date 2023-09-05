import numpy as np
from pyspectrum import SSA
from sklearn.metrics import r2_score

# Generate some example time series data
n_samples = 1000
data = np.random.rand(n_samples)

# Choose a range of possible window sizes
window_sizes = range(10, 101, 10)

# Compute the R-squared for each window size
r2_scores = []
for window_size in window_sizes:
    # Create an SSA object with the "Caterpillar" method
    ssa = SSA(window_size=window_size, method="caterpillar")

    # Fit the SSA object to the data
    ssa.fit(data)

    # Extract the reconstructed components from the SSA object
    components = ssa.components()

    # Compute the R-squared for the first reconstructed component
    r2 = r2_score(data[window_size:], components[0][window_size-1:-1])
    r2_scores.append(r2)

# Find the window size with the highest R-squared
best_window_size = window_sizes[np.argmax(r2_scores)]

print("The optimal window size is:", best_window_size)
