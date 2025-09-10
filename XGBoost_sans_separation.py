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
    # Remplir valeurs manquantes
    dataset['TIME_TO_PARADE_1'] = dataset['TIME_TO_PARADE_1'].fillna(10000)
    dataset['TIME_TO_PARADE_2'] = dataset['TIME_TO_PARADE_2'].fillna(10000)
    dataset['TIME_TO_NIGHT_SHOW'] = dataset['TIME_TO_NIGHT_SHOW'].fillna(10000)
    dataset['snow_1h'] = dataset['snow_1h'].fillna(0)

    # Conversion datetime
    dataset['DATETIME'] = pd.to_datetime(dataset['DATETIME'])
    dataset['DAY_OF_WEEK'] = dataset['DATETIME'].dt.dayofweek
    dataset['DAY'] = dataset['DATETIME'].dt.day
    dataset['MONTH'] = dataset['DATETIME'].dt.month
    dataset['YEAR'] = dataset['DATETIME'].dt.year
    dataset['HOUR'] = dataset['DATETIME'].dt.hour
    dataset['MINUTE'] = dataset['DATETIME'].dt.minute

    # Parade proche (< 2h) et temps avant prochaine
    dataset['TIME_TO_PARADE_UNDER_2H'] = np.where(
        (abs(dataset['TIME_TO_PARADE_1']) <= 500) | (abs(dataset['TIME_TO_PARADE_2']) <= 500),
        1, 0
    )

    # Encodage cyclique de l'heure
    dataset['HOUR_SIN'] = np.sin(2 * np.pi * dataset['HOUR'] / 24)
    dataset['HOUR_COS'] = np.cos(2 * np.pi * dataset['HOUR'] / 24)

    # Binarisation des attractions

    dataset['IS_ATTRACTION_Water_Ride'] = np.where(dataset['ENTITY_DESCRIPTION_SHORT'] == "Water Ride", 1, 0)
    dataset['IS_ATTRACTION_Pirate_Ship'] = np.where(dataset['ENTITY_DESCRIPTION_SHORT'] == "Pirate Ship", 1, 0)
    dataset['IS_ATTRACTION__Flying_Coaster'] = np.where(dataset['ENTITY_DESCRIPTION_SHORT'] == "Flying Coaster", 1, 0)

    # Avant ou après COVID
    dataset['IS_PRE_COVID'] = np.where(dataset['DATETIME'] < "2020-03-15", 1, 0)


#On cherche à avoir le meilleur RMSE possible en faisant varier n_estimators 
def tune_hyperparameters(X_train, y_train, X_val, y_val):
    best_rmse = float('inf')
    best_n_estimators = None

    for n_estimators in range(50, 500, 50):
        model = XGBRegressor(n_estimators=n_estimators)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        rmse = RMSE(y_val, y_pred)

        if rmse < best_rmse:
            best_rmse = rmse
            best_n_estimators = n_estimators

    print(f"Best n_estimators: {best_n_estimators} with RMSE: {best_rmse}")
    
    return best_n_estimators, best_rmse

# -----------------------------------------------------
# 3. Entraînement des deux modèles
# -----------------------------------------------------

def train(df, target="WAIT_TIME_IN_2H"):

    features = [c for c in df if c not in [target, "DATETIME", "ENTITY_DESCRIPTION_SHORT"]]

    X, y = df[features], df[target]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

   # Tune hyperparameters
    best_n_estimators, best_rmse = tune_hyperparameters(X_train, y_train, X_val, y_val)

    # Train final model with best hyperparameters
    model = XGBRegressor(n_estimators=best_n_estimators)
    model.fit(X, y)

    return model


#On adapte le dataset
adapter_dataset(datasetmeteo)

#On entraîne le modèle
rf = train(datasetmeteo)

features = [c for c in datasetmeteo if c not in ["WAIT_TIME_IN_2H", "DATETIME", "ENTITY_DESCRIPTION_SHORT"]]

# Test sur validation externe
val = pd.read_csv("valmeteo.csv")
adapter_dataset(val)

X_val = val[features]
Y_val = rf.predict(X_val)
val['y_pred'] = Y_val

columns_valcsv = ['DATETIME','ENTITY_DESCRIPTION_SHORT','y_pred']
valcsv = val[columns_valcsv]
valcsv["KEY"] = "Validation"
valcsv.to_csv("XGBoost_sans_separation.csv", index=False)