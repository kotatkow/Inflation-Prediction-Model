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
agriculture_price_index_data = load_data('C:/Users/ghkjs/Inflation-Prediction-Model/economic_data/agriculture_price_index_data.csv')
core_cpi_data = load_data('C:/Users/ghkjs/Inflation-Prediction-Model/economic_data/core_cpi_data.csv')
core_pce_data = load_data('C:/Users/ghkjs/Inflation-Prediction-Model/economic_data/core_pce_data.csv')
crude_oil_price_index_data = load_data('C:/Users/ghkjs/Inflation-Prediction-Model/economic_data/crude_oil_price_index_data.csv')
gdp_deflator_data = load_data('C:/Users/ghkjs/Inflation-Prediction-Model/economic_data/gdp_deflator_data.csv')
hpi_data = load_data('C:/Users/ghkjs/Inflation-Prediction-Model/economic_data/hpi_data.csv')
industrial_price_index  = load_data('C:/Users/ghkjs/Inflation-Prediction-Model/economic_data/industrial_price_index_data.csv')
interest_rates_data = load_data('C:/Users/ghkjs/Inflation-Prediction-Model/economic_data/interest_rates_data.csv')
m2_data = load_data('C:/Users/ghkjs/Inflation-Prediction-Model/economic_data/m2_data.csv')
nominal_gdp_data = load_data('C:/Users/ghkjs/Inflation-Prediction-Model/economic_data/nominal_gdp_data.csv')
pce_data = load_data('C:/Users/ghkjs/Inflation-Prediction-Model/economic_data/pce_data.csv')
pce_percentage_change = load_data('C:/Users/ghkjs/Inflation-Prediction-Model/economic_data/pce_percentage_change.csv')
ppi_data = load_data('C:/Users/ghkjs/Inflation-Prediction-Model/economic_data/ppi_data.csv')
real_gdp_data = load_data('C:/Users/ghkjs/Inflation-Prediction-Model/economic_data/real_gdp_data.csv')
seasonally_adjusted_ppi_data = load_data('C:/Users/ghkjs/Inflation-Prediction-Model/economic_data/seasonally_adjusted_ppi_data.csv')
velocity_of_money = load_data('C:/Users/ghkjs/Inflation-Prediction-Model/economic_data/velocity_of_money.csv')

economic_data=[cpi_data, agriculture_price_index_data, core_cpi_data, core_pce_data, crude_oil_price_index_data,
                 gdp_deflator_data, hpi_data, industrial_price_index, interest_rates_data, 
                 m2_data, nominal_gdp_data, pce_data, pce_percentage_change, ppi_data, real_gdp_data, 
                 seasonally_adjusted_ppi_data, velocity_of_money]

# Handle missing values
def handle_missing_values(data):
    return data.ffill().bfill()

cpi_data = handle_missing_values(cpi_data)
agriculture_price_index_data = handle_missing_values(agriculture_price_index_data)
core_cpi_data = handle_missing_values(core_cpi_data)
crude_oil_price_index_data = handle_missing_values(crude_oil_price_index_data)
gdp_deflator_data = handle_missing_values(gdp_deflator_data)
hpi_data = handle_missing_values(hpi_data)
industrial_price_index = handle_missing_values(industrial_price_index)
interest_rates_data = handle_missing_values(interest_rates_data)
m2_data = handle_missing_values(m2_data)
nominal_gdp_data = handle_missing_values(nominal_gdp_data)
pce_data = handle_missing_values(pce_data)
pce_percentage_change = handle_missing_values(pce_percentage_change)
ppi_data = handle_missing_values(ppi_data)
real_gdp_data = handle_missing_values(real_gdp_data)
seasonally_adjusted_ppi_data = handle_missing_values(seasonally_adjusted_ppi_data)
velocity_of_money = handle_missing_values(velocity_of_money)

# Seasonal adjustment
def seasonal_adjustment(data, column, period=12):
    decomposition = seasonal_decompose(data[column], model='additive', period=period)
    return data[column] - decomposition.seasonal

cpi_data['seasonally_adjusted'] = seasonal_adjustment(cpi_data, 'value')
agriculture_price_index_data['seasonally_adjusted'] = seasonal_adjustment(agriculture_price_index_data,
                                                                          'value')
core_cpi_data['seasonally_adjusted'] = seasonal_adjustment(core_cpi_data, 'value')
core_pce_data['seasonally_adjusted'] = seasonal_adjustment(core_pce_data, 'value')
crude_oil_price_index_data['seasonally_adjusted'] = seasonal_adjustment(crude_oil_price_index_data,
                                                                        'value')
gdp_deflator_data['seasonally_adjusted'] = seasonal_adjustment(gdp_deflator_data,'value')
hpi_data['seasonally_adjusted'] = seasonal_adjustment(hpi_data, 'value')
industrial_price_index['seasonally_adjusted'] = seasonal_adjustment(industrial_price_index, 'value')
nominal_gdp_data['seasonally_adjusted'] = seasonal_adjustment(nominal_gdp_data, 'value')
m2_data['seasonally_adjusted'] = seasonal_adjustment(m2_data, 'value')
pce_data['seasonally_adjusted'] = seasonal_adjustment(pce_data, 'value')
ppi_data['seasonally_adjusted'] = seasonal_adjustment(ppi_data, 'value')
real_gdp_data['seasonally_adjusted'] = seasonal_adjustment(real_gdp_data, 'value')

# Differencing
def apply_differencing(data, column):
    return data[column].diff().dropna()

economic_data_to_drop_for_differencing = [interest_rates_data, pce_percentage_change,
                                          seasonally_adjusted_ppi_data, velocity_of_money]

cpi_data['diff'] = apply_differencing(cpi_data, 'seasonally_adjusted')
agriculture_price_index_data['diff'] = apply_differencing(agriculture_price_index_data, 'seasonally_adjusted')
core_cpi_data['diff'] = apply_differencing(core_cpi_data, 'seasonally_adjusted')
core_pce_data['diff'] = apply_differencing(core_pce_data, 'seasonally_adjusted')
crude_oil_price_index_data['diff'] = apply_differencing(crude_oil_price_index_data, 'seasonally_adjusted')
gdp_deflator_data['diff'] = apply_differencing(gdp_deflator_data, 'seasonally_adjusted')
hpi_data['diff'] = apply_differencing(hpi_data, 'seasonally_adjusted')
industrial_price_index['diff'] = apply_differencing(industrial_price_index, 'seasonally_adjusted')
nominal_gdp_data['diff'] = apply_differencing(nominal_gdp_data, 'seasonally_adjusted')
m2_data['diff'] = apply_differencing(m2_data, 'seasonally_adjusted')
pce_data['diff'] = apply_differencing(pce_data, 'seasonally_adjusted')
ppi_data['diff'] = apply_differencing(ppi_data, 'seasonally_adjusted')
real_gdp_data['diff'] = apply_differencing(real_gdp_data, 'seasonally_adjusted')
