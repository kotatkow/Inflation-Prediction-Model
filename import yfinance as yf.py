import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands

# Download historical data for a stock
ticker = 'LLY'
data = yf.download(ticker, start='2023-01-01', end='2024-05-24')

# Calculate Moving Averages
data['50_SMA'] = data['Close'].rolling(window=50).mean()
data['200_SMA'] = data['Close'].rolling(window=200).mean()

# Calculate RSI
rsi = RSIIndicator(close=data['Close'], window=14)
data['RSI'] = rsi.rsi()

# Calculate MACD
macd = MACD(close=data['Close'], window_slow=26, window_fast=12, window_sign=9)
data['MACD'] = macd.macd()
data['MACD_Signal'] = macd.macd_signal()

# Calculate Bollinger Bands
bb = BollingerBands(close=data['Close'], window=20, window_dev=2)
data['BB_High'] = bb.bollinger_hband()
data['BB_Low'] = bb.bollinger_lband()

# Plotting
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(14, 12), gridspec_kw={'height_ratios': [3, 1, 1, 1]})

# Price and Moving Averages
ax1.plot(data['Close'], label='Close Price')
ax1.plot(data['50_SMA'], label='50-Day SMA')
ax1.plot(data['200_SMA'], label='200-Day SMA')
ax1.fill_between(data.index, data['BB_High'], data['BB_Low'], color='gray', alpha=0.3)
ax1.set_title('LLY Stock Price and Moving Averages')
ax1.legend()

# RSI
ax2.plot(data['RSI'], label='RSI', color='purple')
ax2.axhline(70, linestyle='--', alpha=0.5, color='red')
ax2.axhline(30, linestyle='--', alpha=0.5, color='green')
ax2.set_title('Relative Strength Index (RSI)')
ax2.legend()

# MACD
ax3.plot(data['MACD'], label='MACD', color='blue')
ax3.plot(data['MACD_Signal'], label='Signal Line', color='orange')
ax3.fill_between(data.index, data['MACD'] - data['MACD_Signal'], 0, where=(data['MACD'] - data['MACD_Signal']) > 0, color='green', alpha=0.5, interpolate=True)
ax3.fill_between(data.index, data['MACD'] - data['MACD_Signal'], 0, where=(data['MACD'] - data['MACD_Signal']) < 0, color='red', alpha=0.5, interpolate=True)
ax3.set_title('MACD')
ax3.legend()

# Volume
ax4.bar(data.index, data['Volume'], label='Volume', color='gray')
ax4.set_title('Volume')
ax4.legend()

plt.tight_layout()
plt.show()