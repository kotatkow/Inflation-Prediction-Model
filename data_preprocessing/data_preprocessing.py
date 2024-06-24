import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import MinMaxScaler

# Load datasets
def load_data(file_path):
    data = pd.read_csv(file_path)
    data['date'] = pd.to_datetime(data['date'])
    data.set_index('date', inplace=True)
    return data

cpi_data = load_data('C:/Users/ghkjs/Inflation-Prediction-Model/economic_data/cpi_data.csv')
pce_percentage_change = load_data('C:/Users/ghkjs/Inflation-Prediction-Model/economic_data/pce_percentage_change.csv')
ppi_data = load_data('C:/Users/ghkjs/Inflation-Prediction-Model/economic_data/ppi_data.csv')
pce_data = load_data('C:/Users/ghkjs/Inflation-Prediction-Model/economic_data/pce_data.csv')

# Handle missing values
def handle_missing_values(data):
    return data.ffill().bfill()

cpi_data = handle_missing_values(cpi_data)
pce_percentage_change = handle_missing_values(pce_percentage_change)
ppi_data = handle_missing_values(ppi_data)
pce_data = handle_missing_values(pce_data)

# Seasonal adjustment
def seasonal_adjustment(data, column, period=12):
    decomposition = seasonal_decompose(data[column], model='additive', period=period)
    return data[column] - decomposition.seasonal

cpi_data['CPI_seasonally_adjusted'] = seasonal_adjustment(cpi_data, 'value')
ppi_data['PPI_seasonally_adjusted'] = seasonal_adjustment(ppi_data, 'value')
pce_data['PCE_seasonally_adjusted'] = seasonal_adjustment(pce_data, 'value')

# Differencing
def apply_differencing(data, column):
    return data[column].diff().dropna()

cpi_data['CPI_diff'] = apply_differencing(cpi_data, 'CPI_seasonally_adjusted')
ppi_data['PPI_diff'] = apply_differencing(ppi_data, 'PPI_seasonally_adjusted')
pce_data['PCE_diff'] = apply_differencing(pce_data, 'PCE_seasonally_adjusted')

# Percentage change
def calculate_percentage_change(data, column):
    return data[column].pct_change() * 100

cpi_data['CPI_pct_change'] = calculate_percentage_change(cpi_data, 'CPI_diff')
ppi_data['PPI_pct_change'] = calculate_percentage_change(ppi_data, 'PPI_diff')
pce_data['PCE_pct_change'] = calculate_percentage_change(pce_data, 'PCE_diff')

#Drop Nan values
cpi_data.dropna(inplace=True)
ppi_data.dropna(inplace=True)
pce_percentage_change.dropna(inplace=True)
pce_data.dropna(inplace=True)

# Normalization
scaler = MinMaxScaler()

cpi_data['CPI_pct_scaled'] = scaler.fit_transform(cpi_data[['CPI_pct_change']])
ppi_data['PPI_pct_scaled'] = scaler.fit_transform(ppi_data[['PPI_pct_change']])
pce_data['PCE_pct_scaled'] = scaler.fit_transform(pce_data[['PCE_pct_change']])

# Feature engineering (lagged features)
def create_lag_features(data, column, lags):
    for lag in range(1, lags + 1):
        data[f'{column}_lag_{lag}'] = data[column].shift(lag)
    return data

cpi_data = create_lag_features(cpi_data, 'CPI_pct_scaled', 3)
ppi_data = create_lag_features(ppi_data, 'PPI_pct_scaled', 3)
pce_data = create_lag_features(pce_data, 'PCE_pct_scaled', 3)

# Drop NaN values created by lagging
cpi_data.dropna(inplace=True)
ppi_data.dropna(inplace=True)
pce_data.dropna(inplace=True)

# Merge datasets
merged_data = cpi_data[['CPI_pct_scaled']].join([
    ppi_data[['PPI_pct_scaled']],
    pce_data[['PCE_pct_scaled']]
], how='inner')

# Add lag features from all datasets
for lag in range(1, 4):
    merged_data[f'CPI_pct_scaled_lag_{lag}'] = cpi_data[f'CPI_pct_scaled_lag_{lag}']
    merged_data[f'PPI_pct_scaled_lag_{lag}'] = ppi_data[f'PPI_pct_scaled_lag_{lag}']
    merged_data[f'PCE_pct_scaled_lag_{lag}'] = pce_data[f'PCE_pct_scaled_lag_{lag}']

merged_data.dropna(inplace=True)

# Display the merged dataset
print(merged_data.head())