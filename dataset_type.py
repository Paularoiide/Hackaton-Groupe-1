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



#On va tout binariser les variables qualitatives
def adapter_dataset_2_groupes(dataset):
        # === 1. Gestion des valeurs manquantes ===
    dataset['TIME_TO_PARADE_1'] = dataset['TIME_TO_PARADE_1'].fillna(1e6)
    dataset['TIME_TO_PARADE_2'] = dataset['TIME_TO_PARADE_2'].fillna(1e6)
    dataset['TIME_TO_NIGHT_SHOW'] = dataset['TIME_TO_NIGHT_SHOW'].fillna(1e6)
    dataset['snow_1h'] = dataset['snow_1h'].fillna(0)

    # === 2. Conversion datetime et features temporelles ===
    dataset['DATETIME'] = pd.to_datetime(dataset['DATETIME'])
    dataset['DAY_OF_WEEK'] = dataset['DATETIME'].dt.dayofweek   # 0 = lundi
    dataset['DAY'] = dataset['DATETIME'].dt.day
    dataset['MONTH'] = dataset['DATETIME'].dt.month
    dataset['YEAR'] = dataset['DATETIME'].dt.year
    dataset['HOUR'] = dataset['DATETIME'].dt.hour
    dataset['MINUTE'] = dataset['DATETIME'].dt.minute

    # On binarise les jours, les mois et les années
    # Binarisation des mois 
    dataset['IS_MONTH_January'] = np.where(dataset['MONTH'] == 1, 1, 0)
    dataset['IS_MONTH_February'] = np.where(dataset['MONTH'] == 2, 1, 0)
    dataset['IS_MONTH_March'] = np.where(dataset['MONTH'] == 3, 1, 0)
    dataset['IS_MONTH_April'] = np.where(dataset['MONTH'] == 4, 1, 0)
    dataset['IS_MONTH_May'] = np.where(dataset['MONTH'] == 5, 1, 0)
    dataset['IS_MONTH_June'] = np.where(dataset['MONTH'] == 6, 1, 0)
    dataset['IS_MONTH_July'] = np.where(dataset['MONTH'] == 7, 1, 0)
    dataset['IS_MONTH_August'] = np.where(dataset['MONTH'] == 8, 1, 0)
    dataset['IS_MONTH_September'] = np.where(dataset['MONTH'] == 9, 1, 0)       
    dataset['IS_MONTH_October'] = np.where(dataset['MONTH'] == 10, 1, 0)
    dataset['IS_MONTH_November'] = np.where(dataset['MONTH'] == 11, 1, 0)
    dataset['IS_MONTH_December'] = np.where(dataset['MONTH'] == 12, 1, 0)

    # Binarisation des jours de la semaine
    dataset['IS_DAY_Monday'] = np.where(dataset['DAY_OF_WEEK'] == 0, 1, 0)
    dataset['IS_DAY_Tuesday'] = np.where(dataset['DAY_OF_WEEK'] == 1, 1, 0)
    dataset['IS_DAY_Wednesday'] = np.where(dataset['DAY_OF_WEEK'] == 2, 1, 0)
    dataset['IS_DAY_Thursday'] = np.where(dataset['DAY_OF_WEEK'] == 3, 1, 0)
    dataset['IS_DAY_Friday'] = np.where(dataset['DAY_OF_WEEK'] == 4, 1, 0)
    dataset['IS_DAY_Saturday'] = np.where(dataset['DAY_OF_WEEK'] == 5, 1, 0)
    dataset['IS_DAY_Sunday'] = np.where(dataset['DAY_OF_WEEK'] == 6, 1, 0)

    # Binarisation des années
    dataset['IS_YEAR_2019'] = np.where(dataset['YEAR'] == 2019, 1, 0)
    dataset['IS_YEAR_2020'] = np.where(dataset['YEAR'] == 2020, 1, 0)
    dataset['IS_YEAR_2021'] = np.where(dataset['YEAR'] == 2021, 1, 0)
    dataset['IS_YEAR_2022'] = np.where(dataset['YEAR'] == 2022, 1, 0)

    #On droppe les anciennes colonnes
    dataset.drop(columns=['DAY_OF_WEEK','DAY','MONTH','YEAR'], inplace=True)

    # Cyclic encoding des heures
    dataset['HOUR_SIN'] = np.sin(2 * np.pi * dataset['HOUR'] / 24)
    dataset['HOUR_COS'] = np.cos(2 * np.pi * dataset['HOUR'] / 24)

    # Saison (0=hiver,1=printemps,2=été,3=automne)
    dataset['SEASON'] = (dataset['MONTH'] % 12) // 3
    
    #On binarise les saisons
    dataset['IS_SEASON_Winter'] = np.where(dataset['SEASON'] == 0, 1, 0)
    dataset['IS_SEASON_Spring'] = np.where(dataset['SEASON'] == 1, 1, 0)
    dataset['IS_SEASON_Summer'] = np.where(dataset['SEASON'] == 2, 1, 0)
    dataset['IS_SEASON_Autumn'] = np.where(dataset['SEASON'] == 3, 1, 0)
    dataset.drop(columns=['SEASON'], inplace=True)  

    # Périodes de la journée (catégoriel → peut être one-hot ensuite)
    def get_part_of_day(h):
        if 6 <= h < 12: return 0
        elif 12 <= h < 18: return 1
        elif 18 <= h < 23: return 2
        else: return 3
    dataset['PART_OF_DAY'] = dataset['HOUR'].apply(get_part_of_day)

    #On binarise les parties de la journée
    dataset['IS_PART_OF_DAY_Morning'] = np.where(dataset['PART_OF_DAY'] == 0, 1, 0)
    dataset['IS_PART_OF_DAY_Afternoon'] = np.where(dataset['PART_OF_DAY'] == 1, 1, 0)
    dataset['IS_PART_OF_DAY_Evening'] = np.where(dataset['PART_OF_DAY'] == 2, 1, 0)
    dataset.drop(columns=['PART_OF_DAY'], inplace=True) 

    # === 3. Proximité événements spéciaux ===
    dataset['IS_PARADE_SOON'] = ((dataset['TIME_TO_PARADE_1'].between(-120, 120)) |
                                 (dataset['TIME_TO_PARADE_2'].between(-120, 120))).astype(int)
    dataset['IS_NIGHT_SHOW_SOON'] = (dataset['TIME_TO_NIGHT_SHOW'].between(-120, 120)).astype(int)

    # === 4. Attractions (one-hot encoding direct) ===
    dataset['IS_ATTRACTION_Water_Ride'] = np.where(dataset['ENTITY_DESCRIPTION_SHORT'] == "Water Ride", 1, 0)
    dataset['IS_ATTRACTION_Pirate_Ship'] = np.where(dataset['ENTITY_DESCRIPTION_SHORT'] == "Pirate Ship", 1, 0)

    # === 5. Capacités et pannes ===
    dataset['CAPACITY_RATIO'] = dataset['CURRENT_WAIT_TIME'] / (dataset['ADJUST_CAPACITY'] + 1e-6)
    dataset['CURRENT_WAIT_TIME'] = dataset['CURRENT_WAIT_TIME'] + dataset['DOWNTIME']

    # === 6. Météo enrichie ===
    dataset['IS_RAINING'] = (dataset['rain_1h'] > 0.2).astype(int)
    dataset['IS_SNOWING'] = (dataset['snow_1h'] > 0.05).astype(int)
    dataset['IS_HOT'] = (dataset['feels_like'] > 25).astype(int)
    dataset['IS_COLD'] = (dataset['feels_like'] < 5).astype(int)
    dataset['IS_BAD_WEATHER'] = ((dataset['rain_1h'] > 2) |
                                 (dataset['snow_1h'] > 0.5) |
                                 (dataset['wind_speed'] > 30)).astype(int)

    # Interaction température-humidité (ressenti de lourdeur)
    dataset['TEMP_HUMIDITY_INDEX'] = dataset['feels_like'] * dataset['humidity']
    dataset.drop(columns=["temp",'humidity','pressure','rain_1h','snow_1h','TIME_TO_NIGHT_SHOW','HOUR','dew_point'], inplace=True)

    dataset['DATETIME'] = pd.to_datetime(dataset['DATETIME'])


# IDEE DE CLARA LGINE 337



import pandas as pd
import numpy as np
from datetime import datetime

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