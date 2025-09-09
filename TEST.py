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
datasetmeteo = pd.read_table('weather_data_combined.csv', sep=',', decimal='.')
datasetsansmeteo = pd.read_table('waiting_times_train.csv', sep=',', decimal='.')

def adapter_dataset(dataset):
    #Remplir les missing values avec infini dans 'TIME_TO_PARADE_1','TIME_TO_PARADE_2','TIME_TO_NIGHT_SHOW'
    dataset['TIME_TO_PARADE_1'] = dataset['TIME_TO_PARADE_1'].fillna(10000)
    dataset['TIME_TO_PARADE_2'] = dataset['TIME_TO_PARADE_2'].fillna(10000)
    dataset['TIME_TO_NIGHT_SHOW'] = dataset['TIME_TO_NIGHT_SHOW'].fillna(10000)

    # Convert 'DATETIME' to datetime object and extract features
    dataset['DATETIME'] = pd.to_datetime(dataset['DATETIME'])
    dataset['DAY_OF_WEEK'] = dataset['DATETIME'].dt.dayofweek
    dataset['DAY'] = dataset['DATETIME'].dt.day
    dataset['MONTH'] = dataset['DATETIME'].dt.month
    dataset['YEAR'] = dataset['DATETIME'].dt.year
    dataset['HOUR'] = dataset['DATETIME'].dt.hour
    dataset['MINUTE'] = dataset['DATETIME'].dt.minute
    dataset['TIME_TO_PARADE_UNDER_2H'] = np.where((abs(dataset['TIME_TO_PARADE_1']) <= 120) | (abs(dataset['TIME_TO_PARADE_2']) <= 120), 1, 0)

def AIC(X, predictors, y):
# AIC and BIC based stepwise forward selection
    selected_predictors = ['const'] # Start with only the intercept
    unselected_predictors = predictors.copy()
    X = sm.add_constant(X)  # Ajoute une constante si elle n'existe pas déjà

    # Compute the Information Criterion (IC) for the model with only the intercept
    current_ic = sm.OLS(y, X[selected_predictors]).fit().aic # AIC

    ic_values = [current_ic]  # Store successive IC values

    # Stepwise forward selection based on minimizing IC
    while len(unselected_predictors)>0:
        best_ic = np.inf  # Initialize with a very high IC
        best_predictor = None

        # Try adding each unselected predictor one by one
        for pred in unselected_predictors:
            test_model = sm.OLS(y, X[selected_predictors + [pred]]).fit()
            test_ic = test_model.aic # AIC

            # Check if this predictor gives the lowest IC so far
            if test_ic < best_ic:
                best_ic = test_ic
                best_predictor = pred

        # If the best IC is lower than the current IC, accept the predictor
        if best_ic < current_ic:
            selected_predictors.append(best_predictor)
            unselected_predictors.remove(best_predictor)
            ic_values.append(best_ic)
            current_ic = best_ic  # Update current IC
        else:
            break  # Stop if no improvement

    # Print results
    print("Selected predictors:", selected_predictors)
    print("IC values over iterations:", ic_values)
    return selected_predictors, ic_values


#On effectue une analyse des composantes en ne considérant pas les variables météorologiques
dataset = datasetmeteo
adapter_dataset(dataset)

predictors = ['DAY_OF_WEEK', 'DAY', 'MONTH', 'YEAR', 'HOUR', 'MINUTE', 'ADJUST_CAPACITY','DOWNTIME','CURRENT_WAIT_TIME','TIME_TO_PARADE_1','TIME_TO_PARADE_2','TIME_TO_NIGHT_SHOW', 'TIME_TO_PARADE_UNDER_2H']
X = dataset[predictors]
y = dataset['WAIT_TIME_IN_2H'] # Response variable

AIC(X, predictors, y)

#On effectue une analyse des composantes en considérant les variables météorologiques
dataset = datasetmeteo
adapter_dataset(dataset)
predictors = ['DAY_OF_WEEK', 'DAY', 'MONTH', 'YEAR', 'HOUR', 'MINUTE', 'ADJUST_CAPACITY','DOWNTIME','CURRENT_WAIT_TIME','TIME_TO_PARADE_1','TIME_TO_PARADE_2','TIME_TO_NIGHT_SHOW', 'TIME_TO_PARADE_UNDER_2H', 'temp', 'humidity', 'wind_speed', 'pressure', 'rain_1h', 'clouds_all']
X = dataset[predictors]
y = dataset['WAIT_TIME_IN_2H'] # Response variable

AIC(X, predictors, y)