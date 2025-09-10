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

     # Jour weekend
    dataset['IS_WEEKEND'] = dataset['DAY_OF_WEEK'].isin([5, 6]).astype(int)

    # Saison (0=hiver,1=printemps,2=été,3=automne)
    dataset['SEASON'] = (dataset['MONTH'] % 12) // 3

    # Périodes de la journée (catégoriel → peut être one-hot ensuite)
    def get_part_of_day(h):
        if 6 <= h < 12: return 0
        elif 12 <= h < 18: return 1
        elif 18 <= h < 23: return 2
        else: return 3
    dataset['PART_OF_DAY'] = dataset['HOUR'].apply(get_part_of_day)

    # === 3. Proximité événements spéciaux ===
    dataset['IS_PARADE_SOON'] = ((dataset['TIME_TO_PARADE_1'].between(-120, 120)) |
                                 (dataset['TIME_TO_PARADE_2'].between(-120, 120))).astype(int)
    dataset['IS_NIGHT_SHOW_SOON'] = (dataset['TIME_TO_NIGHT_SHOW'].between(-120, 120)).astype(int)

    # === 4. Attractions (one-hot encoding direct) ===
    attractions = dataset['ENTITY_DESCRIPTION_SHORT'].unique()
    for att in attractions:
        dataset[f"IS_ATTRACTION_{att.replace(' ', '_')}"] = (dataset['ENTITY_DESCRIPTION_SHORT'] == att).astype(int)

    # === 5. Capacités et pannes ===
    dataset['CAPACITY_RATIO'] = dataset['CURRENT_WAIT_TIME'] / (dataset['ADJUST_CAPACITY'] + 1e-6)
    dataset['IS_DOWNTIME'] = (dataset['DOWNTIME'] > 0).astype(int)

    # === 6. Météo enrichie ===
    dataset['IS_RAINING'] = (dataset['rain_1h'] > 0.2).astype(int)
    dataset['IS_SNOWING'] = (dataset['snow_1h'] > 0.05).astype(int)
    dataset['IS_HOT'] = (dataset['temp'] > 25).astype(int)
    dataset['IS_COLD'] = (dataset['temp'] < 5).astype(int)
    dataset['IS_BAD_WEATHER'] = ((dataset['rain_1h'] > 2) |
                                 (dataset['snow_1h'] > 0.5) |
                                 (dataset['wind_speed'] > 30)).astype(int)

    # Interaction température-humidité (ressenti de lourdeur)
    dataset['TEMP_HUMIDITY_INDEX'] = dataset['temp'] * dataset['humidity']
    dataset.drop(columns=["temp",'humidity','pressure'], inplace=True)

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

def train(df, features, target="WAIT_TIME_IN_2H"):

    X, y = df[features], df[target]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

   # Tune hyperparameters
    best_n_estimators, best_rmse = tune_hyperparameters(X_train, y_train, X_val, y_val)

    # Train final model with best hyperparameters
    model = XGBRegressor(n_estimators=best_n_estimators)
    model.fit(X, y)

    return model

#On selectionne les features
def meilleur_modele_XGBoost(df, target = 'WAIT_TIME_IN_2H'):

    #On sépare les features de la target:
    features = [col for col in df.columns if col not in [target, 'DATETIME', 'ENTITY_DESCRIPTION_SHORT']]

    print("Features utilisées pour le modèle XGBoost :")
    print(features)

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
    return feature_importance.head(15).index.tolist()


#On adapte le dataset
adapter_dataset(datasetmeteo)

features = [c for c in datasetmeteo if c not in ["WAIT_TIME_IN_2H", "DATETIME", "ENTITY_DESCRIPTION_SHORT"]]
print("Features initiales :", features)

features_importantes = meilleur_modele_XGBoost(datasetmeteo)

#On entraîne le modèle
rf = train(datasetmeteo, features_importantes)


# Test sur validation externe
val = pd.read_csv("valmeteo.csv")
adapter_dataset(val)

X_val = val[features_importantes]
Y_val = rf.predict(X_val)
val['y_pred'] = Y_val

columns_valcsv = ['DATETIME','ENTITY_DESCRIPTION_SHORT','y_pred']
valcsv = val[columns_valcsv]
valcsv["KEY"] = "Validation"

valcsv.to_csv("XGBoost_sans_separation_avec_selection_15_plus_de par.csv", index=False)