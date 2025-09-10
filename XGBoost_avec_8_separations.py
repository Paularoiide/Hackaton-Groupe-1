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
from datetime import datetime

def RMSE(x,y):
  return np.sqrt(mean_squared_error(x,y))

def adapter_dataset_8_groupes(dataset):
    # Faire une copie pour éviter les modifications sur l'original
    dataset = dataset.copy()
    
    # Conversion datetime
    dataset['DATETIME'] = pd.to_datetime(dataset['DATETIME'])
    
    # Fonction pour détecter les vacances scolaires par zone
    def detecter_vacances_par_zone(date):
        # Dates exactes des vacances scolaires françaises 2019-2022 par zone
        vacances_zones = {
            'ZONE_A': [
                # 2018-2019
                (datetime(2018, 10, 20), datetime(2018, 11, 4)),    # Toussaint
                (datetime(2018, 12, 22), datetime(2019, 1, 6)),     # Noël
                (datetime(2019, 2, 16), datetime(2019, 3, 3)),      # Hiver
                (datetime(2019, 4, 13), datetime(2019, 4, 28)),     # Printemps
                (datetime(2019, 7, 6), datetime(2019, 9, 1)),       # Été
                
                # 2019-2020
                (datetime(2019, 10, 19), datetime(2019, 11, 3)),    # Toussaint
                (datetime(2019, 12, 21), datetime(2020, 1, 5)),     # Noël
                (datetime(2020, 2, 8), datetime(2020, 2, 23)),      # Hiver
                (datetime(2020, 4, 4), datetime(2020, 4, 19)),      # Printemps
                (datetime(2020, 7, 4), datetime(2020, 9, 1)),       # Été
                
                # 2020-2021
                (datetime(2020, 10, 17), datetime(2020, 11, 1)),    # Toussaint
                (datetime(2020, 12, 19), datetime(2021, 1, 3)),     # Noël
                (datetime(2021, 2, 6), datetime(2021, 2, 21)),      # Hiver
                (datetime(2021, 4, 10), datetime(2021, 4, 25)),     # Printemps
                (datetime(2021, 7, 6), datetime(2021, 9, 1)),       # Été
                
                # 2021-2022
                (datetime(2021, 10, 23), datetime(2021, 11, 7)),    # Toussaint
                (datetime(2021, 12, 18), datetime(2022, 1, 2)),     # Noël
                (datetime(2022, 2, 12), datetime(2022, 2, 27)),     # Hiver
                (datetime(2022, 4, 16), datetime(2022, 5, 1)),      # Printemps
                (datetime(2022, 7, 7), datetime(2022, 9, 1)),       # Été
            ],
            'ZONE_B': [
                # 2018-2019
                (datetime(2018, 10, 20), datetime(2018, 11, 4)),    # Toussaint
                (datetime(2018, 12, 22), datetime(2019, 1, 6)),     # Noël
                (datetime(2019, 2, 9), datetime(2019, 2, 24)),      # Hiver
                (datetime(2019, 4, 6), datetime(2019, 4, 21)),      # Printemps
                (datetime(2019, 7, 6), datetime(2019, 9, 1)),       # Été
                
                # 2019-2020
                (datetime(2019, 10, 19), datetime(2019, 11, 3)),    # Toussaint
                (datetime(2019, 12, 21), datetime(2020, 1, 5)),     # Noël
                (datetime(2020, 2, 22), datetime(2020, 3, 8)),      # Hiver
                (datetime(2020, 4, 4), datetime(2020, 4, 19)),      # Printemps
                (datetime(2020, 7, 4), datetime(2020, 9, 1)),       # Été
                
                # 2020-2021
                (datetime(2020, 10, 17), datetime(2020, 11, 1)),    # Toussaint
                (datetime(2020, 12, 19), datetime(2021, 1, 3)),     # Noël
                (datetime(2021, 2, 20), datetime(2021, 3, 7)),      # Hiver
                (datetime(2021, 4, 10), datetime(2021, 4, 25)),     # Printemps
                (datetime(2021, 7, 6), datetime(2021, 9, 1)),       # Été
                
                # 2021-2022
                (datetime(2021, 10, 23), datetime(2021, 11, 7)),    # Toussaint
                (datetime(2021, 12, 18), datetime(2022, 1, 2)),     # Noël
                (datetime(2022, 2, 26), datetime(2022, 3, 13)),     # Hiver
                (datetime(2022, 4, 16), datetime(2022, 5, 1)),      # Printemps
                (datetime(2022, 7, 7), datetime(2022, 9, 1)),       # Été
            ],
            'ZONE_C': [
                # 2018-2019
                (datetime(2018, 10, 20), datetime(2018, 11, 4)),    # Toussaint
                (datetime(2018, 12, 22), datetime(2019, 1, 6)),     # Noël
                (datetime(2019, 2, 23), datetime(2019, 3, 10)),     # Hiver
                (datetime(2019, 4, 20), datetime(2019, 5, 5)),      # Printemps
                (datetime(2019, 7, 6), datetime(2019, 9, 1)),       # Été
                
                # 2019-2020
                (datetime(2019, 10, 19), datetime(2019, 11, 3)),    # Toussaint
                (datetime(2019, 12, 21), datetime(2020, 1, 5)),     # Noël
                (datetime(2020, 2, 15), datetime(2020, 3, 1)),      # Hiver
                (datetime(2020, 4, 18), datetime(2020, 5, 3)),      # Printemps
                (datetime(2020, 7, 4), datetime(2020, 9, 1)),       # Été
                
                # 2020-2021
                (datetime(2020, 10, 17), datetime(2020, 11, 1)),    # Toussaint
                (datetime(2020, 12, 19), datetime(2021, 1, 3)),     # Noël
                (datetime(2021, 2, 13), datetime(2021, 2, 28)),     # Hiver
                (datetime(2021, 4, 24), datetime(2021, 5, 9)),      # Printemps
                (datetime(2021, 7, 6), datetime(2021, 9, 1)),       # Été
                
                # 2021-2022
                (datetime(2021, 10, 23), datetime(2021, 11, 7)),    # Toussaint
                (datetime(2021, 12, 18), datetime(2022, 1, 2)),     # Noël
                (datetime(2022, 2, 12), datetime(2022, 2, 27)),     # Hiver
                (datetime(2022, 4, 23), datetime(2022, 5, 8)),      # Printemps
                (datetime(2022, 7, 7), datetime(2022, 9, 1)),       # Été
            ]
        }
        
        result = {'VACANCES_ZONE_A': 0, 'VACANCES_ZONE_B': 0, 'VACANCES_ZONE_C': 0}
        
        for zone, periodes in vacances_zones.items():
            for debut, fin in periodes:
                if debut <= date <= fin:
                    if zone == 'ZONE_A':
                        result['VACANCES_ZONE_A'] = 1
                    elif zone == 'ZONE_B':
                        result['VACANCES_ZONE_B'] = 1
                    elif zone == 'ZONE_C':
                        result['VACANCES_ZONE_C'] = 1
        
        return result
    
    # Appliquer la fonction à toutes les dates et créer les colonnes
    vacances_data = dataset['DATETIME'].apply(detecter_vacances_par_zone)
    dataset['VACANCES_ZONE_A'] = vacances_data.apply(lambda x: x['VACANCES_ZONE_A'])
    dataset['VACANCES_ZONE_B'] = vacances_data.apply(lambda x: x['VACANCES_ZONE_B'])
    dataset['VACANCES_ZONE_C'] = vacances_data.apply(lambda x: x['VACANCES_ZONE_C'])

    #ajouter avant ou après covid 
    dataset['IS_COVID_PRE'] = np.where(dataset['DATETIME'] < pd.Timestamp('2020-03-01'), 1, 0)
    dataset['IS_COVID_POST'] = np.where(dataset['DATETIME'] >= pd.Timestamp('2020-03-01'), 1, 0)
    
    # Créer les masques pour les 8 groupes
    # 1. TIME_TO_PARADE_1 présent
    mask_parade1 = ~dataset['TIME_TO_PARADE_1'].isna()
    # 2. TIME_TO_PARADE_2 présent
    mask_parade2 = ~dataset['TIME_TO_PARADE_2'].isna()
    # 3. TIME_TO_NIGHT_SHOW présent
    mask_night_show = ~dataset['TIME_TO_NIGHT_SHOW'].isna()
    
    # Créer les 8 groupes
    groupes = {}
    
    # Groupe 1: Parade1 présent, Parade2 présent, NightShow présent
    mask1 = mask_parade1 & mask_parade2 & mask_night_show
    groupes['groupe_1'] = dataset[mask1].copy()
    
    # Groupe 2: Parade1 présent, Parade2 présent, NightShow absent
    mask2 = mask_parade1 & mask_parade2 & ~mask_night_show
    groupes['groupe_2'] = dataset[mask2].copy()
    
    # Groupe 3: Parade1 présent, Parade2 absent, NightShow présent
    mask3 = mask_parade1 & ~mask_parade2 & mask_night_show
    groupes['groupe_3'] = dataset[mask3].copy()
    
    # Groupe 4: Parade1 présent, Parade2 absent, NightShow absent
    mask4 = mask_parade1 & ~mask_parade2 & ~mask_night_show
    groupes['groupe_4'] = dataset[mask4].copy()
    
    # Groupe 5: Parade1 absent, Parade2 présent, NightShow présent
    mask5 = ~mask_parade1 & mask_parade2 & mask_night_show
    groupes['groupe_5'] = dataset[mask5].copy()
    
    # Groupe 6: Parade1 absent, Parade2 présent, NightShow absent
    mask6 = ~mask_parade1 & mask_parade2 & ~mask_night_show
    groupes['groupe_6'] = dataset[mask6].copy()
    
    # Groupe 7: Parade1 absent, Parade2 absent, NightShow présent
    mask7 = ~mask_parade1 & ~mask_parade2 & mask_night_show
    groupes['groupe_7'] = dataset[mask7].copy()
    
    # Groupe 8: Parade1 absent, Parade2 absent, NightShow absent
    mask8 = ~mask_parade1 & ~mask_parade2 & ~mask_night_show
    groupes['groupe_8'] = dataset[mask8].copy()
    
    # Pour chaque groupe, ajouter les features supplémentaires
    for groupe_name, groupe_data in groupes.items():
        if not groupe_data.empty:
            # Remplir snow_1h avec 0 si manquant
            groupe_data['snow_1h'] = groupe_data['snow_1h'].fillna(0)
            
            # Extraire les features datetime
            groupe_data['DAY_OF_WEEK'] = groupe_data['DATETIME'].dt.dayofweek
            groupe_data['DAY'] = groupe_data['DATETIME'].dt.day
            groupe_data['MONTH'] = groupe_data['DATETIME'].dt.month
            groupe_data['YEAR'] = groupe_data['DATETIME'].dt.year
            groupe_data['HOUR'] = groupe_data['DATETIME'].dt.hour
            groupe_data['MINUTE'] = groupe_data['DATETIME'].dt.minute
            
            # Encodage cyclique de l'heure
            groupe_data['HOUR_SIN'] = np.sin(2 * np.pi * groupe_data['HOUR'] / 24)
            groupe_data['HOUR_COS'] = np.cos(2 * np.pi * groupe_data['HOUR'] / 24)
            
            # Binarisation des attractions
            groupe_data['IS_ATTRACTION_Water_Ride'] = np.where(groupe_data['ENTITY_DESCRIPTION_SHORT'] == "Water Ride", 1, 0)
            groupe_data['IS_ATTRACTION_Pirate_Ship'] = np.where(groupe_data['ENTITY_DESCRIPTION_SHORT'] == "Pirate Ship", 1, 0)
            groupe_data['IS_ATTRACTION__Flying_Coaster'] = np.where(groupe_data['ENTITY_DESCRIPTION_SHORT'] == "Flying Coaster", 1, 0)
            
            # Parade proche (< 500) - seulement si les données de parade existent
            
            # Initialiser à 0
            groupe_data['TIME_TO_PARADE_UNDER_2H'] = 0
            if 'TIME_TO_PARADE_1' in groupe_data.columns:
                mask_parade1_close = groupe_data['TIME_TO_PARADE_1'].notna() & (abs(groupe_data['TIME_TO_PARADE_1']) <= 500)
                groupe_data.loc[mask_parade1_close, 'TIME_TO_PARADE_UNDER_2H'] = 1
            if 'TIME_TO_PARADE_2' in groupe_data.columns:
                mask_parade2_close = groupe_data['TIME_TO_PARADE_2'].notna() & (abs(groupe_data['TIME_TO_PARADE_2']) <= 500)
                groupe_data.loc[mask_parade2_close, 'TIME_TO_PARADE_UNDER_2H'] = 1

    return groupes

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

# Put the dataset into a pandas DataFrame
dataset = pd.read_table('weather_data_combined.csv', sep=',', decimal='.')

#On adapte le dataset
groupes = adapter_dataset_8_groupes(dataset)
models = {}
for groupe_name, groupe_data in groupes.items():
    if not groupe_data.empty:
        print(f"Training model for {groupe_name} with {len(groupe_data)} samples")
        
        # Define features and target
        features = [col for col in groupe_data.columns if col not in ['WAIT_TIME_IN_2H', 'DATETIME', 'ENTITY_DESCRIPTION_SHORT', 'TIME_TO_PARADE_1', 'TIME_TO_PARADE_2', 'TIME_TO_NIGHT_SHOW']]
        target = 'WAIT_TIME_IN_2H'
        
        # Train model
        model = train(groupe_data, features, target)
        models[groupe_name] = model

# Test sur validation externe
val = pd.read_csv("valmeteo.csv")
adapter_dataset_8_groupes(val)

# Prédire avec les 8 modèles
def predict_eight_models(models, val):
    features = [c for c in val if c not in ["WAIT_TIME_IN_2H", "DATETIME", "ENTITY_DESCRIPTION_SHORT"]]
    # Créer une colonne pour les prédictions
    val['y_pred'] = np.nan
    
    # Créer les masques pour les 8 groupes
    mask_parade1 = ~val['TIME_TO_PARADE_1'].isna()
    mask_parade2 = ~val['TIME_TO_PARADE_2'].isna()
    mask_night_show = ~val['TIME_TO_NIGHT_SHOW'].isna()
    
    # Groupe 1: Parade1 présent, Parade2 présent, NightShow présent
    mask1 = mask_parade1 & mask_parade2 & mask_night_show
    if 'groupe_1' in models:
        X_val_1 = val[mask1][features]
        val.loc[mask1, 'y_pred'] = models['groupe_1'].predict(X_val_1)
    
    # Groupe 2: Parade1 présent, Parade2 présent, NightShow absent
    mask2 = mask_parade1 & mask_parade2 & ~mask_night_show
    if 'groupe_2' in models:
        X_val_2 = val[mask2][features]
        val.loc[mask2, 'y_pred'] = models['groupe_2'].predict(X_val_2)
    
    # Groupe 3: Parade1 présent, Parade2 absent, NightShow présent
    mask3 = mask_parade1 & ~mask_parade2 & mask_night_show
    if 'groupe_3' in models:
        X_val_3 = val[mask3][features]
        val.loc[mask3, 'y_pred'] = models['groupe_3'].predict(X_val_3)
    
    # Groupe 4: Parade1 présent, Parade2 absent, NightShow absent
    mask4 = mask_parade1 & ~mask_parade2 & ~mask_night_show
    if 'groupe_4' in models:
        X_val_4 = val[mask4][features]
        val.loc[mask4, 'y_pred'] = models['groupe_4'].predict(X_val_4)
    
    # Groupe 5: Parade1 absent, Parade2 présent, NightShow présent
    mask5 = ~mask_parade1 & mask_parade2 & mask_night_show
    if 'groupe_5' in models:
        X_val_5 = val[mask5][features]
        val.loc[mask5, 'y_pred'] = models['groupe_5'].predict(X_val_5)
    
    # Groupe 6: Parade1 absent, Parade2 présent, NightShow absent
    mask6 = ~mask_parade1 & mask_parade2 & ~mask_night_show
    if 'groupe_6' in models:
        X_val_6 = val[mask6][features]
        val.loc[mask6, 'y_pred'] = models['groupe_6'].predict(X_val_6)
    # Groupe 7: Parade1 absent, Parade2 absent, NightShow présent
    mask7 = ~mask_parade1 & ~mask_parade2 & mask_night_show
    if 'groupe_7' in models:
        X_val_7 = val[mask7][features]
        val.loc[mask7, 'y_pred'] = models['groupe_7'].predict(X_val_7)
    # Groupe 8: Parade1 absent, Parade2 absent, NightShow absent
    mask8 = ~mask_parade1 & ~mask_parade2 & ~mask_night_show
    if 'groupe_8' in models:
        X_val_8 = val[mask8][features]
        val.loc[mask8, 'y_pred'] = models['groupe_8'].predict(X_val_8)


    return val['y_pred']

features = [c for c in dataset if c not in ["WAIT_TIME_IN_2H", "DATETIME", "ENTITY_DESCRIPTION_SHORT"]]

X_val = val[features]
Y_val = predict_eight_models(models, val)
val['y_pred'] = Y_val

columns_valcsv = ['DATETIME','ENTITY_DESCRIPTION_SHORT','y_pred']
valcsv = val[columns_valcsv]
valcsv["KEY"] = "Validation"

valcsv.to_csv("XGBoost_:_8_groupes.csv", index=False)

