'''
PIN stands for Probability of INformed Trading, 
and it is a measure of the likelihood that a trade was initiated by an informed trader, 
as opposed to a liquidity trader. 
Informed traders are traders who possess non-public information about a security, 
and their trades can have a significant impact on the price of the security. 
Liquidity traders, on the other hand, are traders who are simply looking to buy 
or sell a security in order to meet their liquidity needs. 
PIN is calculated using a variety of factors, including the size and timing of trades, 
bid-ask spreads, and market volatility. 
High PIN values can indicate that there is a greater likelihood of informed trading 
in a particular security, which can be useful for traders 
who are looking to make informed investment decisions.
'''
import pandas as pd
import numpy as np

# Load trade data
data = pd.read_csv('trades.csv')

# Calculate VWAP
#vwap = np.sum(data['Price'] * data['Volume']) / np.sum(data['Volume'])

# Calculate PIN
spread = data['AskPrice'] - data['BidPrice']
spread_mean = np.mean(spread)
spread_std = np.std(spread)
z_score = (data['Price'] - np.mean(data['Price'])) / np.std(data['Price'])
pin = 1 - (np.sum(np.abs(z_score) * spread) / (2 * np.sum(data['Volume']) * spread_mean * spread_std))

# Print results
#print(f"VWAP: {vwap}")
print(f"PIN: {pin}")
