import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

cpi_data = pd.read_csv('cpi_data.csv', parse_dates=['date'], index_col='date')
ppi_data = pd.read_csv('ppi_data.csv', parse_dates=['date'], index_col='date')
pce_data = pd.read_csv('pce_data.csv', parse_dates=['date'], index_col='date')

# Display the first few rows of each dataframe
print(cpi_data.head())
print(ppi_data.head())
print(pce_data.head())

