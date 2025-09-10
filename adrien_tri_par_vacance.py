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


def adapter_dataset_8_groupes_par_vacances(dataset):
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
    
    # Créer 8 groupes basés sur les combinaisons de vacances
    groupes = {}
    
    # Toutes les combinaisons possibles des 3 zones de vacances
    combinaisons = [
        (0, 0, 0),  # Aucune zone en vacances
        (1, 0, 0),  # Seulement zone A en vacances
        (0, 1, 0),  # Seulement zone B en vacances
        (0, 0, 1),  # Seulement zone C en vacances
        (1, 1, 0),  # Zones A et B en vacances
        (1, 0, 1),  # Zones A et C en vacances
        (0, 1, 1),  # Zones B et C en vacances
        (1, 1, 1),  # Toutes les zones en vacances
    ]
    
    noms_groupes = [
        'groupe_aucune_vacances',
        'groupe_vacances_A_seule',
        'groupe_vacances_B_seule',
        'groupe_vacances_C_seule',
        'groupe_vacances_A_B',
        'groupe_vacances_A_C',
        'groupe_vacances_B_C',
        'groupe_toutes_vacances'
    ]
    
    for i, (vac_a, vac_b, vac_c) in enumerate(combinaisons):
        mask = (dataset['VACANCES_ZONE_A'] == vac_a) & \
               (dataset['VACANCES_ZONE_B'] == vac_b) & \
               (dataset['VACANCES_ZONE_C'] == vac_c)
        
        groupe_data = dataset[mask].copy()
        groupe_name = noms_groupes[i]
        groupes[groupe_name] = groupe_data
        
        # Pour chaque groupe, ajouter les features supplémentaires
        if not groupe_data.empty:
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
            
            # Parade proche (< 500)
            if 'TIME_TO_PARADE_1' in groupe_data.columns and 'TIME_TO_PARADE_2' in groupe_data.columns:
                mask_parade1_close = abs(groupe_data['TIME_TO_PARADE_1']) <= 500
                mask_parade2_close = abs(groupe_data['TIME_TO_PARADE_2']) <= 500
                groupe_data['TIME_TO_PARADE_UNDER_2H'] = np.where(mask_parade1_close | mask_parade2_close, 1, 0)
            else:
                groupe_data['TIME_TO_PARADE_UNDER_2H'] = 0
    
    return groupes


# Fonction d'entraînement 8 modèles (par groupe de vacances)
# ------------------------
def train_by_attr_and_vacation_groups(df, target="WAIT_TIME_IN_2H"):
    models = {}
    features = [c for c in df.columns if c not in [target, "DATETIME", "ENTITY_DESCRIPTION_SHORT", 
                                                  "TIME_TO_PARADE_1", "TIME_TO_PARADE_2", "TIME_TO_NIGHT_SHOW",
                                                  "VACANCES_ZONE_A", "VACANCES_ZONE_B", "VACANCES_ZONE_C"]]
    
    # Définir les 8 groupes de vacances
    vacation_groups = {
        'groupe_1': {'zone_a': True, 'zone_b': True, 'zone_c': True},
        'groupe_2': {'zone_a': True, 'zone_b': True, 'zone_c': False},
        'groupe_3': {'zone_a': True, 'zone_b': False, 'zone_c': True},
        'groupe_4': {'zone_a': True, 'zone_b': False, 'zone_c': False},
        'groupe_5': {'zone_a': False, 'zone_b': True, 'zone_c': True},
        'groupe_6': {'zone_a': False, 'zone_b': True, 'zone_c': False},
        'groupe_7': {'zone_a': False, 'zone_b': False, 'zone_c': True},
        'groupe_8': {'zone_a': False, 'zone_b': False, 'zone_c': False}
    }

    for attraction in df["ENTITY_DESCRIPTION_SHORT"].unique():
        df_attr = df[df["ENTITY_DESCRIPTION_SHORT"] == attraction]
        
        for group_name, group_conditions in vacation_groups.items():
            # Filtrer selon les conditions du groupe de vacances
            mask = pd.Series(True, index=df_attr.index)
            
            if group_conditions['zone_a']:
                mask = mask & (df_attr['VACANCES_ZONE_A'] == 1)
            else:
                mask = mask & (df_attr['VACANCES_ZONE_A'] == 0)
                
            if group_conditions['zone_b']:
                mask = mask & (df_attr['VACANCES_ZONE_B'] == 1)
            else:
                mask = mask & (df_attr['VACANCES_ZONE_B'] == 0)
                
            if group_conditions['zone_c']:
                mask = mask & (df_attr['VACANCES_ZONE_C'] == 1)
            else:
                mask = mask & (df_attr['VACANCES_ZONE_C'] == 0)
            
            df_group = df_attr[mask]
            
            if len(df_group) < 30:  # sécurité si trop peu de données
                print(f"⚠️ Pas assez de données pour {attraction} ({group_name}): {len(df_group)} lignes")
                continue
            
            print(f"\n--- Entraînement modèle {attraction} ({group_name}) ---")
            print(f"Taille du dataset: {len(df_group)} lignes")
            print(f"Vacances - Zone A: {group_conditions['zone_a']}, Zone B: {group_conditions['zone_b']}, Zone C: {group_conditions['zone_c']}")
            
            X, y = df_group[features], df_group[target]
            
            # Si très peu de données, on utilise tout pour l'entraînement
            if len(df_group) < 100:
                X_train, y_train = X, y
                X_test, y_test = X, y  # Pour l'évaluation
            else:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            rf = RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_leaf=3,
                max_features="sqrt",
                random_state=42,
                n_jobs=-1
            )
            rf.fit(X_train, y_train)
            
            y_pred = rf.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            print(f"RMSE {attraction} ({group_name}): {rmse:.2f}")
            print(f"MAE {attraction} ({group_name}): {mae:.2f}")
            
            # Sauvegarde du modèle
            models[(attraction, group_name)] = rf
            
            # Importance des features
            if len(df_group) > 50:
                feat_importances = pd.DataFrame({
                    "Feature": features,
                    "Importance": rf.feature_importances_
                }).sort_values("Importance", ascending=False)
                
                print(f"\nTop 5 features {attraction} ({group_name}):\n", feat_importances.head(5))

    return models, features


# ------------------------
# Prédiction avec les 8 modèles (par groupe de vacances)
# ------------------------
def predict_by_attr_and_vacation_groups(models, features, df):
    preds = np.zeros(len(df))
    df = df.copy()
    
    for attraction in df["ENTITY_DESCRIPTION_SHORT"].unique():
        mask_attr = df["ENTITY_DESCRIPTION_SHORT"] == attraction
        
        for group_name in ['groupe_1', 'groupe_2', 'groupe_3', 'groupe_4', 
                          'groupe_5', 'groupe_6', 'groupe_7', 'groupe_8']:
            
            if (attraction, group_name) not in models:
                continue
                
            # Déterminer le masque pour ce groupe de vacances
            group_conditions = {
                'groupe_1': {'zone_a': True, 'zone_b': True, 'zone_c': True},
                'groupe_2': {'zone_a': True, 'zone_b': True, 'zone_c': False},
                'groupe_3': {'zone_a': True, 'zone_b': False, 'zone_c': True},
                'groupe_4': {'zone_a': True, 'zone_b': False, 'zone_c': False},
                'groupe_5': {'zone_a': False, 'zone_b': True, 'zone_c': True},
                'groupe_6': {'zone_a': False, 'zone_b': True, 'zone_c': False},
                'groupe_7': {'zone_a': False, 'zone_b': False, 'zone_c': True},
                'groupe_8': {'zone_a': False, 'zone_b': False, 'zone_c': False}
            }[group_name]
            
            mask_group = mask_attr.copy()
            
            if group_conditions['zone_a']:
                mask_group = mask_group & (df['VACANCES_ZONE_A'] == 1)
            else:
                mask_group = mask_group & (df['VACANCES_ZONE_A'] == 0)
                
            if group_conditions['zone_b']:
                mask_group = mask_group & (df['VACANCES_ZONE_B'] == 1)
            else:
                mask_group = mask_group & (df['VACANCES_ZONE_B'] == 0)
                
            if group_conditions['zone_c']:
                mask_group = mask_group & (df['VACANCES_ZONE_C'] == 1)
            else:
                mask_group = mask_group & (df['VACANCES_ZONE_C'] == 0)
            
            if mask_group.any():
                preds[mask_group] = models[(attraction, group_name)].predict(df.loc[mask_group, features])
    
    return preds


# ------------------------
# Fonction pour adapter le dataset aux 8 groupes de vacances
# ------------------------
def adapter_dataset_8_groupes_vacances(dataset):
    """
    Adapte le dataset pour les 8 groupes de vacances en utilisant votre fonction existante
    """
    # Utiliser votre fonction existante pour obtenir les données avec les colonnes de vacances
    groupes = adapter_dataset_12_groupes_par_date(dataset)
    
    # Combiner tous les groupes (on ignore les groupes temporels pour se concentrer sur les vacances)
    df_combined = pd.concat(groupes.values(), ignore_index=True)
    
    return df_combined


# ------------------------
# Utilisation complète
# ------------------------
# Préparation des données avec les 8 groupes de vacances
df = pd.read_csv("weather_data_combined.csv")
df_processed = adapter_dataset_8_groupes_vacances(df)

# Entraînement des modèles
print("Début de l'entraînement des 8 modèles de vacances...")
models_8, features = train_by_attr_and_vacation_groups(df_processed)

# Prédiction sur les données de validation
val = pd.read_csv("valmeteo.csv")
print("\nDébut des prédictions sur les données de validation...")

val_processed = adapter_dataset_8_groupes_vacances(val)
y_val_pred = predict_by_attr_and_vacation_groups(models_8, features, val_processed)

# Export CSV
val_processed['y_pred'] = y_val_pred
val_processed[['DATETIME','ENTITY_DESCRIPTION_SHORT','y_pred']].assign(KEY="Validation").to_csv("val_predictions_8_groupes_vacances.csv", index=False)

print("Prédictions terminées et sauvegardées dans val_predictions_8_groupes_vacances.csv")


# ------------------------
# Fonction pour analyser la distribution des groupes de vacances
# ------------------------
def analyser_distribution_vacances(df):
    """
    Analyse la distribution des données dans les 8 groupes de vacances
    """
    print("Distribution des données par groupe de vacances:")
    print("=" * 60)
    
    vacation_groups = {
        'groupe_1': {'zone_a': True, 'zone_b': True, 'zone_c': True},
        'groupe_2': {'zone_a': True, 'zone_b': True, 'zone_c': False},
        'groupe_3': {'zone_a': True, 'zone_b': False, 'zone_c': True},
        'groupe_4': {'zone_a': True, 'zone_b': False, 'zone_c': False},
        'groupe_5': {'zone_a': False, 'zone_b': True, 'zone_c': True},
        'groupe_6': {'zone_a': False, 'zone_b': True, 'zone_c': False},
        'groupe_7': {'zone_a': False, 'zone_b': False, 'zone_c': True},
        'groupe_8': {'zone_a': False, 'zone_b': False, 'zone_c': False}
    }
    
    total_rows = len(df)
    
    for group_name, conditions in vacation_groups.items():
        mask = pd.Series(True, index=df.index)
        
        if conditions['zone_a']:
            mask = mask & (df['VACANCES_ZONE_A'] == 1)
        else:
            mask = mask & (df['VACANCES_ZONE_A'] == 0)
            
        if conditions['zone_b']:
            mask = mask & (df['VACANCES_ZONE_B'] == 1)
        else:
            mask = mask & (df['VACANCES_ZONE_B'] == 0)
            
        if conditions['zone_c']:
            mask = mask & (df['VACANCES_ZONE_C'] == 1)
        else:
            mask = mask & (df['VACANCES_ZONE_C'] == 0)
        
        n_rows = len(df[mask])
        print(f"{group_name}: {n_rows} lignes ({n_rows/total_rows*100:.1f}%) - "
              f"Zone A: {conditions['zone_a']}, Zone B: {conditions['zone_b']}, Zone C: {conditions['zone_c']}")
    
    print("=" * 60)
    print(f"Total: {total_rows} lignes")
    
    # Distribution générale des vacances
    print("\nDistribution générale des vacances:")
    print(f"Zone A en vacances: {len(df[df['VACANCES_ZONE_A'] == 1])} lignes")
    print(f"Zone B en vacances: {len(df[df['VACANCES_ZONE_B'] == 1])} lignes")
    print(f"Zone C en vacances: {len(df[df['VACANCES_ZONE_C'] == 1])} lignes")
    print(f"Aucune zone en vacances: {len(df[(df['VACANCES_ZONE_A'] == 0) & (df['VACANCES_ZONE_B'] == 0) & (df['VACANCES_ZONE_C'] == 0)])} lignes")

# Analyser la distribution
analyser_distribution_vacances(df_processed)


# ------------------------
# Fonction pour expliquer les groupes de vacances
# ------------------------
def expliquer_groupes_vacances():
    """
    Explique ce que représente chaque groupe de vacances
    """
    print("\nExplication des 8 groupes de vacances:")
    print("=" * 50)
    
    groupes = {
        'groupe_1': "Toutes les zones en vacances (A+B+C)",
        'groupe_2': "Zones A et B en vacances, Zone C non",
        'groupe_3': "Zones A et C en vacances, Zone B non",
        'groupe_4': "Seule la Zone A en vacances",
        'groupe_5': "Zones B et C en vacances, Zone A non",
        'groupe_6': "Seule la Zone B en vacances",
        'groupe_7': "Seule la Zone C en vacances",
        'groupe_8': "Aucune zone en vacances"
    }
    
    for groupe, description in groupes.items():
        print(f"{groupe}: {description}")

expliquer_groupes_vacances()