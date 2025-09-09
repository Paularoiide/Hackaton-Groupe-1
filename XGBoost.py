import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
from xgboost import XGBRegressor

def RMSE(x,y):
  return np.sqrt(mean_squared_error(x,y))

# Put the dataset into a pandas DataFrame
datasetmeteo = pd.read_table('weather_data_combined.csv', sep=',', decimal='.')
datasetsansmeteo = pd.read_table('waiting_times_train.csv', sep=',', decimal='.')

def adapter_dataset(dataset):
    #Remplir les missing values avec infini dans 'TIME_TO_PARADE_1','TIME_TO_PARADE_2','TIME_TO_NIGHT_SHOW'
    dataset['TIME_TO_PARADE_1'] = dataset['TIME_TO_PARADE_1'].fillna(10000)
    dataset['TIME_TO_PARADE_2'] = dataset['TIME_TO_PARADE_2'].fillna(10000)
    dataset['TIME_TO_NIGHT_SHOW'] = dataset['TIME_TO_NIGHT_SHOW'].fillna(10000)
    if 'snow_1h' in dataset.columns:
        dataset['snow_1h'] = dataset['snow_1h'].fillna(0.05)

    # Convert 'DATETIME' to datetime object and extract features
    dataset['DATETIME'] = pd.to_datetime(dataset['DATETIME'])
    dataset['DAY_OF_WEEK'] = dataset['DATETIME'].dt.dayofweek
    dataset['DAY'] = dataset['DATETIME'].dt.day
    dataset['MONTH'] = dataset['DATETIME'].dt.month
    dataset['YEAR'] = dataset['DATETIME'].dt.year
    dataset['HOUR'] = dataset['DATETIME'].dt.hour
    dataset['MINUTE'] = dataset['DATETIME'].dt.minute
    dataset['TIME_TO_PARADE_UNDER_2H'] = np.where((abs(dataset['TIME_TO_PARADE_1']) <= 120) | (abs(dataset['TIME_TO_PARADE_2']) <= 120), 1, 0)


#On va considérer l'impression du client sur l'influence du COVID 

def ajouter_covid(dataset):
    # Ajouter une colonne 'COVID_IMPACT' basée sur les dates
    dataset['COVID_IMPACT'] = np.where(dataset['DATETIME'] >= pd.to_datetime('2020-03-01'), 1, 0)

dataset = datasetsansmeteo
adapter_dataset(dataset)

predictors = ['DAY_OF_WEEK', 'DAY', 'MONTH', 'YEAR', 'HOUR', 'MINUTE', 'ADJUST_CAPACITY','DOWNTIME','CURRENT_WAIT_TIME','TIME_TO_PARADE_1','TIME_TO_PARADE_2','TIME_TO_NIGHT_SHOW', 'TIME_TO_PARADE_UNDER_2H']
X = dataset[predictors]
y = dataset['WAIT_TIME_IN_2H'] # Response variable


my_model = XGBRegressor()
my_model.fit(X, y)