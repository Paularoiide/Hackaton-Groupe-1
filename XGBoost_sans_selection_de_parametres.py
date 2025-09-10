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


    
# -----------------------------------------------------
# 2. Séparation pré- / post-COVID
# -----------------------------------------------------
def split_pre_post(df, covid_date="2020-03-15"):
    df_pre = df[df['DATETIME'] < covid_date].copy()
    df_post = df[df['DATETIME'] >= covid_date].copy()
    return df_pre, df_post


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

def train_two_models(df_pre, df_post, target="WAIT_TIME_IN_2H"):

    features = [c for c in df_pre.columns if c not in [target, "DATETIME", "ENTITY_DESCRIPTION_SHORT"]]

    X_pre, y_pre = df_pre[features], df_pre[target]
    X_pre_train, X_pre_val, y_pre_train, y_pre_val = train_test_split(X_pre, y_pre, test_size=0.2, random_state=42)

    X_post, y_post = df_post[features], df_post[target]
    X_post_train, X_post_val, y_post_train, y_post_val = train_test_split(X_post, y_post, test_size=0.2, random_state=42)

    # Optionnel : tuner les hyperparamètres
    best_n_pre, best_rmse_pre = tune_hyperparameters(X_pre_train, y_pre_train, X_pre_val, y_pre_val)
    best_n_post, best_rmse_post = tune_hyperparameters(X_post_train, y_post_train, X_post_val, y_post_val)

    rf_pre = XGBRegressor(n_estimators=best_n_pre)

    rf_post = XGBRegressor(n_estimators=best_n_post)

    rf_pre.fit(X_pre, y_pre)
    rf_post.fit(X_post, y_post)

    return rf_pre, rf_post


# -----------------------------------------------------
# 4. Prédictions avec les deux modèles
# -----------------------------------------------------

def predict_two_models(rf_pre, rf_post, df, covid_date="2020-03-15"):
    target = "WAIT_TIME_IN_2H"
    features = [c for c in df.columns if c not in [target, "DATETIME", "ENTITY_DESCRIPTION_SHORT"]]
    df = df.copy()
    df['DATETIME'] = pd.to_datetime(df['DATETIME'])

    mask_pre = df['DATETIME'] < covid_date
    mask_post = df['DATETIME'] >= covid_date

    preds = np.zeros(len(df))

    if mask_pre.any():
        preds[mask_pre] = rf_pre.predict(df.loc[mask_pre, features])
    if mask_post.any():
        preds[mask_post] = rf_post.predict(df.loc[mask_post, features])

    return preds


#On adapte le dataset
adapter_dataset(datasetmeteo)
dt_pre, dt_post = split_pre_post(datasetmeteo)

#On entraîne les deux modèles
rf_pre, rf_post = train_two_models(dt_pre, dt_post)

# Test sur validation externe
val = pd.read_csv("valmeteo.csv")
adapter_dataset(val)

y_val_pred = predict_two_models(rf_pre, rf_post, val)

# Ajouter dans val + exporter
val['y_pred'] = y_val_pred
val[['DATETIME','ENTITY_DESCRIPTION_SHORT','y_pred']].assign(KEY="Validation").to_csv("val_predictions_xgboost_hyperpar.csv", index=False)
