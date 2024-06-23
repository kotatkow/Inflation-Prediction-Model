import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import MinMaxScaler

#Load datasets
def load_data(C:/Users/ghkjs/Inflation-Prediction-Model/economic_data):
              data = pd.read_excel(C:/Users/ghkjs/Inflation-Prediction-Model/economic_data/ppi_data.csv)
              data['date'] = pd.to_datetime(data['date'])
              data.set_index('date', inplace=True)
              return data