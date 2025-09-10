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
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error  # Ajouter mean_absolute_error

def adapter_dataset_12_groupes_par_date(dataset):
    # Faire une copie pour éviter les modifications sur l'original
    dataset = dataset.copy()
    
    # Conversion datetime
    dataset['DATETIME'] = pd.to_datetime(dataset['DATETIME'])
    
    # Trier par date pour assurer un ordre chronologique
    dataset = dataset.sort_values('DATETIME').reset_index(drop=True)
    
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
    
    # Remplacer les valeurs manquantes par 10^6 pour les colonnes spécifiques
    parade_cols = ['TIME_TO_PARADE_1', 'TIME_TO_PARADE_2', 'TIME_TO_NIGHT_SHOW']
    for col in parade_cols:
        if col in dataset.columns:
            dataset[col] = dataset[col].fillna(1000000)  # 10^6
    
    # Créer 12 groupes uniformément répartis par date
    n = len(dataset)
    groupes = {}
    
    for i in range(12):
        start_idx = i * n // 12
        end_idx = (i + 1) * n // 12
        
        if i == 11:  # Dernier groupe inclut tout le reste
            end_idx = n
        
        groupe_name = f'groupe_{i+1}'
        groupes[groupe_name] = dataset.iloc[start_idx:end_idx].copy()
        
        # Pour chaque groupe, ajouter les features supplémentaires
        if not groupes[groupe_name].empty:
            groupe_data = groupes[groupe_name]
            
            # Remplir snow_1h avec 0 si manquant
            if 'snow_1h' in groupe_data.columns:
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
            if 'ENTITY_DESCRIPTION_SHORT' in groupe_data.columns:
                groupe_data['IS_ATTRACTION_Water_Ride'] = np.where(groupe_data['ENTITY_DESCRIPTION_SHORT'] == "Water Ride", 1, 0)
                groupe_data['IS_ATTRACTION_Pirate_Ship'] = np.where(groupe_data['ENTITY_DESCRIPTION_SHORT'] == "Pirate Ship", 1, 0)
                groupe_data['IS_ATTRACTION_Flying_Coaster'] = np.where(groupe_data['ENTITY_DESCRIPTION_SHORT'] == "Flying Coaster", 1, 0)
            
            # Parade proche (< 500) - maintenant toutes les colonnes ont des valeurs (soit réelles, soit 10^6)
            if 'TIME_TO_PARADE_1' in groupe_data.columns and 'TIME_TO_PARADE_2' in groupe_data.columns:
                mask_parade1_close = abs(groupe_data['TIME_TO_PARADE_1']) <= 500
                mask_parade2_close = abs(groupe_data['TIME_TO_PARADE_2']) <= 500
                groupe_data['TIME_TO_PARADE_UNDER_2H'] = np.where(mask_parade1_close | mask_parade2_close, 1, 0)
            else:
                groupe_data['TIME_TO_PARADE_UNDER_2H'] = 0
    
    return groupes