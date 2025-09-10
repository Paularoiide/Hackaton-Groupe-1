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


# -----------------------------------------------------
# 3. Entraînement des deux modèles
# -----------------------------------------------------
def train_two_models(df_pre, df_post, features_pre, features_post, target="WAIT_TIME_IN_2H"):

    X_pre, y_pre = df_pre[features_pre], df_pre[target]
    X_post, y_post = df_post[features_post], df_post[target]

    rf_pre = XGBRegressor()

    rf_post = XGBRegressor()

    rf_pre.fit(X_pre, y_pre)
    rf_post.fit(X_post, y_post)

    return rf_pre, rf_post

# -----------------------------------------------------
# 4. Prédictions avec les deux modèles
# -----------------------------------------------------

def predict_two_models(rf_pre, rf_post, features_pre, features_post, df, covid_date="2020-03-15"):
    df = df.copy()
    df['DATETIME'] = pd.to_datetime(df['DATETIME'])

    mask_pre = df['DATETIME'] < covid_date
    mask_post = df['DATETIME'] >= covid_date

    preds = np.zeros(len(df))

    if mask_pre.any():
        preds[mask_pre] = rf_pre.predict(df.loc[mask_pre, features_pre])
    if mask_post.any():
        preds[mask_post] = rf_post.predict(df.loc[mask_post, features_post])

    return preds

# -----------------------------------------------------
# On selectionne les meilleures features : qui donnent le meuilleur RSME pour un modèle XGBoost
# -----------------------------------------------------
def meilleur_modele_XGBoost(df, target = 'WAIT_TIME_IN_2H'):

    #On sépare les features de la target:
    features = [col for col in df.columns if col not in [target, 'DATETIME', 'ENTITY_DESCRIPTION_SHORT']]
    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = XGBRegressor()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse = RMSE(y_test, y_pred)
    print(f"RMSE: {rmse}")

    importance = model.feature_importances_
    feature_importance = pd.Series(importance, index=features).sort_values(ascending=False)
    print("Feature Importance:")
    print(feature_importance)
    #On retourne la liste des features avec la plus grande importance : 10 features
    return feature_importance.head(10).index.tolist()


#On adapte le dataset
adapter_dataset(datasetmeteo)
dt_pre, dt_post = split_pre_post(datasetmeteo)

#Affichage des features importante et rmse
features_pre = meilleur_modele_XGBoost(dt_pre, title="Pré-COVID")
features_post = meilleur_modele_XGBoost(dt_post, title="Post-COVID")

#On entraîne les deux modèles
rf_pre, rf_post = train_two_models(dt_pre, dt_post, features_pre, features_post)

# Test sur validation externe
val = pd.read_csv("valmeteo.csv")
adapter_dataset(val)

y_val_pred = predict_two_models(rf_pre, rf_post, features_pre, features_post, val)

# Ajouter dans val + exporter
val['y_pred'] = y_val_pred
val[['DATETIME','ENTITY_DESCRIPTION_SHORT','y_pred']].assign(KEY="Validation").to_csv("val_predictions_xgboost_2.csv", index=False)
