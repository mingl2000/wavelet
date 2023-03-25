import numpy as np
from scipy.optimize import minimize
import yfinance as yf

# Load stock data
stock = yf.Ticker("AAPL")
hist = stock.history(period="max")
returns = hist['Close'].pct_change().dropna()

# Define lognormal distribution
def lognorm_pdf(x, mu, sigma):
    return 1/(x * sigma * np.sqrt(2*np.pi)) * np.exp(-(np.log(x) - mu)**2 / (2*sigma**2))

# Define negative log-likelihood function
def neg_loglik(params, data):
    mu, sigma = params
    pdf = lognorm_pdf(data, mu, sigma)
    loglik = np.log(pdf).sum()
    return -loglik

# Find maximum likelihood estimates
start_params = [returns.mean(), returns.std()]
result = minimize(neg_loglik, start_params, args=(returns,))
mu_mle, sigma_mle = result.x

print("MLE estimates: mu = {:.4f}, sigma = {:.4f}".format(mu_mle, sigma_mle))
