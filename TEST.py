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

print("Numpy:", np.__version__)
print("Matplotlib:", plt.matplotlib.__version__)
print("Pandas:", pd.__version__)
print("Statsmodels:", sm.__version__)

def RMSE(x,y):
  return np.sqrt(mean_squared_error(x,y))

# Put the dataset into a pandas DataFrame
dataset = pd.read_table('waiting_times_train.csv', sep=',', decimal='.')
dataset.info()
dataset.head()

#Remplir les missing values avec infini dans 'TIME_TO_PARADE_1','TIME_TO_PARADE_2','TIME_TO_NIGHT_SHOW'
dataset['TIME_TO_PARADE_1'].fillna(10000, inplace=True)
dataset['TIME_TO_PARADE_2'].fillna(10000, inplace=True)
dataset['TIME_TO_NIGHT_SHOW'].fillna(10000, inplace=True)

#On crée une nouvelle colonne 'TIME_TO_PARADE_UNDER_2H' qui vaut 1 si un des deux parades a lieu dans les 2h, 0 sinon
dataset['TIME_TO_PARADE_UNDER_2H'] = np.where((dataset['TIME_TO_PARADE_1'] <= 120) | (dataset['TIME_TO_PARADE_2'] <= 120), 1, 0)

#On cherche à rendre utilisable DATETIME : on va le separer en trois paramètres --> jour de la semaine, date (dans le type date) et heure de la journée (avec heure et minute)
dataset['DATETIME'] = pd.to_datetime(dataset['DATETIME'])
dataset['DAY_OF_WEEK'] = dataset['DATETIME'].dt.dayofweek
dataset['DAY'] = dataset['DATETIME'].dt.day
dataset['MONTH'] = dataset['DATETIME'].dt.month
dataset['YEAR'] = dataset['DATETIME'].dt.year
dataset['HOUR'] = dataset['DATETIME'].dt.hour
dataset['MINUTE'] = dataset['DATETIME'].dt.minute


predictors = ['DAY_OF_WEEK', 'DAY', 'MONTH', 'YEAR', 'HOUR', 'MINUTE', 'ADJUST_CAPACITY','DOWNTIME','CURRENT_WAIT_TIME','TIME_TO_PARADE_1','TIME_TO_PARADE_2','TIME_TO_NIGHT_SHOW', 'TIME_TO_PARADE_UNDER_2H']
X = dataset[predictors]
y = dataset['WAIT_TIME_IN_2H'] # Response variable

dataset.head()