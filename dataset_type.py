import pandas as pd
import numpy as np
from datetime import datetime

def adapter_dataset_PREDPOSTcovid_VACANCE(dataset):
    # Faire une copie pour éviter les modifications sur l'original
    dataset = dataset.copy()
    
    # 1. Séparer les données sans aucune info de parade
    mask_no_parade = dataset['TIME_TO_PARADE_1'].isna() & dataset['TIME_TO_PARADE_2'].isna()
    df_no_parade = dataset[mask_no_parade].copy()
    df_with_parade = dataset[~mask_no_parade].copy()
    
    # 2. Pour les données avec parade, remplir UNIQUEMENT les valeurs manquantes par 300
    # Ne pas toucher aux valeurs existantes
    df_with_parade['TIME_TO_PARADE_1'] = df_with_parade['TIME_TO_PARADE_1'].fillna(300)
    df_with_parade['TIME_TO_PARADE_2'] = df_with_parade['TIME_TO_PARADE_2'].fillna(300)
    
    # 3. Remplir les autres valeurs manquantes
    df_with_parade['TIME_TO_NIGHT_SHOW'] = df_with_parade['TIME_TO_NIGHT_SHOW'].fillna(10000)
    df_with_parade['snow_1h'] = df_with_parade['snow_1h'].fillna(0)
    
    # 4. Conversion datetime
    df_with_parade['DATETIME'] = pd.to_datetime(df_with_parade['DATETIME'])
    df_with_parade['DAY_OF_WEEK'] = df_with_parade['DATETIME'].dt.dayofweek
    df_with_parade['DAY'] = df_with_parade['DATETIME'].dt.day
    df_with_parade['MONTH'] = df_with_parade['DATETIME'].dt.month
    df_with_parade['YEAR'] = df_with_parade['DATETIME'].dt.year
    df_with_parade['HOUR'] = df_with_parade['DATETIME'].dt.hour
    df_with_parade['MINUTE'] = df_with_parade['DATETIME'].dt.minute
    
    # 5. Parade proche (< 500) et temps avant prochaine - CORRIGÉ
    # On utilise les valeurs originales (sauf les NaN remplacés par 300)
    df_with_parade['TIME_TO_PARADE_UNDER_2H'] = np.where(
        (abs(df_with_parade['TIME_TO_PARADE_1']) <= 500) | (abs(df_with_parade['TIME_TO_PARADE_2']) <= 500),
        1, 0
    )
    
    # 6. Encodage cyclique de l'heure
    df_with_parade['HOUR_SIN'] = np.sin(2 * np.pi * df_with_parade['HOUR'] / 24)
    df_with_parade['HOUR_COS'] = np.cos(2 * np.pi * df_with_parade['HOUR'] / 24)
    
    # 7. Binarisation des attractions
    df_with_parade['IS_ATTRACTION_Water_Ride'] = np.where(df_with_parade['ENTITY_DESCRIPTION_SHORT'] == "Water Ride", 1, 0)
    df_with_parade['IS_ATTRACTION_Pirate_Ship'] = np.where(df_with_parade['ENTITY_DESCRIPTION_SHORT'] == "Pirate Ship", 1, 0)
    df_with_parade['IS_ATTRACTION__Flying_Coaster'] = np.where(df_with_parade['ENTITY_DESCRIPTION_SHORT'] == "Flying Coaster", 1, 0)
    
    # 8. Fonction pour détecter les vacances scolaires par zone
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
    
    # 9. Appliquer la fonction à toutes les dates et créer les colonnes
    vacances_data = df_with_parade['DATETIME'].apply(detecter_vacances_par_zone)
    df_with_parade['VACANCES_ZONE_A'] = vacances_data.apply(lambda x: x['VACANCES_ZONE_A'])
    df_with_parade['VACANCES_ZONE_B'] = vacances_data.apply(lambda x: x['VACANCES_ZONE_B'])
    df_with_parade['VACANCES_ZONE_C'] = vacances_data.apply(lambda x: x['VACANCES_ZONE_C'])
    
    # 10. Séparation pré/post COVID
    covid_date = "2020-03-15"
    df_pre = df_with_parade[df_with_parade['DATETIME'] < covid_date].copy()
    df_post = df_with_parade[df_with_parade['DATETIME'] >= covid_date].copy()
    
    # 11. Ajouter aussi les colonnes vacances aux données sans parade
    df_no_parade['DATETIME'] = pd.to_datetime(df_no_parade['DATETIME'])
    vacances_data_no_parade = df_no_parade['DATETIME'].apply(detecter_vacances_par_zone)
    df_no_parade['VACANCES_ZONE_A'] = vacances_data_no_parade.apply(lambda x: x['VACANCES_ZONE_A'])
    df_no_parade['VACANCES_ZONE_B'] = vacances_data_no_parade.apply(lambda x: x['VACANCES_ZONE_B'])
    df_no_parade['VACANCES_ZONE_C'] = vacances_data_no_parade.apply(lambda x: x['VACANCES_ZONE_C'])
    
    # 12. Ajouter les colonnes vacances aux datasets pré et post COVID
    df_pre['VACANCES_ZONE_A'] = df_pre['DATETIME'].apply(lambda x: detecter_vacances_par_zone(x)['VACANCES_ZONE_A'])
    df_pre['VACANCES_ZONE_B'] = df_pre['DATETIME'].apply(lambda x: detecter_vacances_par_zone(x)['VACANCES_ZONE_B'])
    df_pre['VACANCES_ZONE_C'] = df_pre['DATETIME'].apply(lambda x: detecter_vacances_par_zone(x)['VACANCES_ZONE_C'])
    
    df_post['VACANCES_ZONE_A'] = df_post['DATETIME'].apply(lambda x: detecter_vacances_par_zone(x)['VACANCES_ZONE_A'])
    df_post['VACANCES_ZONE_B'] = df_post['DATETIME'].apply(lambda x: detecter_vacances_par_zone(x)['VACANCES_ZONE_B'])
    df_post['VACANCES_ZONE_C'] = df_post['DATETIME'].apply(lambda x: detecter_vacances_par_zone(x)['VACANCES_ZONE_C'])
    
    # Retourner les 3 datasets : sans parade, pré-COVID, post-COVID
    return df_no_parade, df_pre, df_post



def adapter_dataset_PREDPOSTcovid_VACANCE_PAOLO(dataset):
    # Faire une copie pour éviter les modifications sur l'original
    dataset = dataset.copy()
    
    # 1. Remplir les valeurs manquantes de TIME_TO_PARADE par 300
    dataset['TIME_TO_PARADE_1'] = dataset['TIME_TO_PARADE_1'].fillna(300)
    dataset['TIME_TO_PARADE_2'] = dataset['TIME_TO_PARADE_2'].fillna(300)
    
    # 2. Remplir les autres valeurs manquantes
    dataset['TIME_TO_NIGHT_SHOW'] = dataset['TIME_TO_NIGHT_SHOW'].fillna(10000)
    dataset['snow_1h'] = dataset['snow_1h'].fillna(0)
    
    # 3. Conversion datetime
    dataset['DATETIME'] = pd.to_datetime(dataset['DATETIME'])
    dataset['DAY_OF_WEEK'] = dataset['DATETIME'].dt.dayofweek
    dataset['DAY'] = dataset['DATETIME'].dt.day
    dataset['MONTH'] = dataset['DATETIME'].dt.month
    dataset['YEAR'] = dataset['DATETIME'].dt.year
    dataset['HOUR'] = dataset['DATETIME'].dt.hour
    dataset['MINUTE'] = dataset['DATETIME'].dt.minute
    
    # 4. Parade proche (< 500) et temps avant prochaine
    dataset['TIME_TO_PARADE_UNDER_2H'] = np.where(
        (abs(dataset['TIME_TO_PARADE_1']) <= 500) | (abs(dataset['TIME_TO_PARADE_2']) <= 500),
        1, 0
    )
    
    # 5. Encodage cyclique de l'heure
    dataset['HOUR_SIN'] = np.sin(2 * np.pi * dataset['HOUR'] / 24)
    dataset['HOUR_COS'] = np.cos(2 * np.pi * dataset['HOUR'] / 24)
    
    # 6. Binarisation des attractions
    dataset['IS_ATTRACTION_Water_Ride'] = np.where(dataset['ENTITY_DESCRIPTION_SHORT'] == "Water Ride", 1, 0)
    dataset['IS_ATTRACTION_Pirate_Ship'] = np.where(dataset['ENTITY_DESCRIPTION_SHORT'] == "Pirate Ship", 1, 0)
    dataset['IS_ATTRACTION__Flying_Coaster'] = np.where(dataset['ENTITY_DESCRIPTION_SHORT'] == "Flying Coaster", 1, 0)
    
    # 7. Fonction pour détecter les vacances scolaires par zone
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
                (datetime(2021, 7, 极), datetime(2021, 9, 1)),       # Été
                
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
    
    # Appliquer la détection des vacances
    vacances_data = dataset['DATETIME'].apply(detecter_vacances_par_zone)
    vacances_df = pd.DataFrame(vacances_data.tolist(), index=dataset.index)
    dataset = pd.concat([dataset, vacances_df], axis=1)
    
    return dataset