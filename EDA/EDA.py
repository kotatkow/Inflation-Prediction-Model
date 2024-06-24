import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import sys
sys.path.append('C:/Users/ghkjs/Inflation-Prediction-Model/data_preprocessing')
import data_preprocessing

# Load the dataset
merged_data = pd.read_csv(
    'C:/Users/ghkjs/Inflation-Prediction-Model/data_preprocessing/merged_data.csv')

# Summary statistics
print("Summary Statistics")
print(merged_data.describe())
