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


# Fonction d'entraînement 12 modèles (par groupe temporel)
# ------------------------
def train_by_attr_and_time_groups(df, target="WAIT_TIME_IN_2H"):
    models = {}
    features = [c for c in df.columns if c not in [target, "DATETIME", "ENTITY_DESCRIPTION_SHORT", 
                                                  "TIME_TO_PARADE_1", "TIME_TO_PARADE_2", "TIME_TO_NIGHT_SHOW"]]
    
    # Préparer les groupes avec votre fonction existante
    groupes = adapter_dataset_12_groupes_par_date(df)
    
    # Combiner tous les groupes pour avoir toutes les données avec l'identifiant de groupe
    df_combined = pd.concat([groupe.assign(TIME_GROUP=group_name) 
                           for group_name, groupe in groupes.items()], ignore_index=True)

    for attraction in df_combined["ENTITY_DESCRIPTION_SHORT"].unique():
        df_attr = df_combined[df_combined["ENTITY_DESCRIPTION_SHORT"] == attraction]
        
        for group_name in [f'groupe_{i}' for i in range(1, 13)]:
            
            # Filtrer selon le groupe temporel
            df_group = df_attr[df_attr['TIME_GROUP'] == group_name]
            
            if len(df_group) < 30:  # sécurité si trop peu de données
                print(f"⚠️ Pas assez de données pour {attraction} ({group_name}): {len(df_group)} lignes")
                continue
            
            print(f"\n--- Entraînement modèle {attraction} ({group_name}) ---")
            print(f"Taille du dataset: {len(df_group)} lignes")
            print(f"Plage temporelle: {group_name}")
            
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
# Prédiction avec les 12 modèles (par groupe temporel)
# ------------------------
def predict_by_attr_and_time_groups(models, features, df):
    preds = np.zeros(len(df))
    df = df.copy()
    
    # Préparer les groupes avec votre fonction existante
    groupes = adapter_dataset_12_groupes_par_date(df)
    
    # Combiner tous les groupes pour avoir toutes les données avec l'identifiant de groupe
    df_combined = pd.concat([groupe.assign(TIME_GROUP=group_name) 
                           for group_name, groupe in groupes.items()], ignore_index=True)

    for attraction in df_combined["ENTITY_DESCRIPTION_SHORT"].unique():
        mask_attr = df_combined["ENTITY_DESCRIPTION_SHORT"] == attraction
        
        for group_name in [f'groupe_{i}' for i in range(1, 13)]:
            
            if (attraction, group_name) not in models:
                continue
                
            # Filtrer selon le groupe temporel
            mask_group = mask_attr & (df_combined['TIME_GROUP'] == group_name)
            
            if mask_group.any():
                # Obtenir les indices originaux pour la prédiction
                original_indices = df_combined.loc[mask_group].index
                preds[original_indices] = models[(attraction, group_name)].predict(df_combined.loc[mask_group, features])
    
    return preds


# ------------------------
# Fonction pour adapter le dataset aux prédictions (simplifiée)
# ------------------------
def prepare_dataset_for_prediction(df):
    """
    Prépare le dataset pour la prédiction en utilisant votre fonction existante
    """
    groupes = adapter_dataset_12_groupes_par_date(df)
    df_combined = pd.concat([groupe.assign(TIME_GROUP=group_name) 
                           for group_name, groupe in groupes.items()], ignore_index=True)
    return df_combined


# ------------------------
# Utilisation complète
# ------------------------
# Chargement des données
df = pd.read_csv("weather_data_combined.csv")

# Entraînement des modèles
print("Début de l'entraînement des 12 modèles...")
models_12, features = train_by_attr_and_time_groups(df)

# Prédiction sur les données de validation
val = pd.read_csv("valmeteo.csv")
print("\nDébut des prédictions sur les données de validation...")

val_processed = prepare_dataset_for_prediction(val)
y_val_pred = predict_by_attr_and_time_groups(models_12, features, val)

# Export CSV
val_processed['y_pred'] = y_val_pred
val_processed[['DATETIME','ENTITY_DESCRIPTION_SHORT','y_pred']].assign(KEY="Validation").to_csv("val_predictions_12_groupes_temporels.csv", index=False)

print("Prédictions terminées et sauvegardées dans val_predictions_12_groupes_temporels.csv")


# ------------------------
# Fonction pour analyser la distribution des groupes
# ------------------------
def analyser_distribution_groupes(df):
    """
    Analyse la distribution des données dans les 12 groupes
    """
    groupes = adapter_dataset_12_groupes_par_date(df)
    
    print("Distribution des données par groupe temporel:")
    print("=" * 50)
    
    total_rows = 0
    for group_name, groupe_data in groupes.items():
        n_rows = len(groupe_data)
        total_rows += n_rows
        print(f"{group_name}: {n_rows} lignes ({n_rows/len(df)*100:.1f}%)")
    
    print("=" * 50)
    print(f"Total: {total_rows} lignes")
    
    # Distribution par attraction dans chaque groupe
    print("\nDistribution par attraction dans chaque groupe:")
    for group_name, groupe_data in groupes.items():
        print(f"\n{group_name}:")
        attraction_counts = groupe_data['ENTITY_DESCRIPTION_SHORT'].value_counts()
        for attraction, count in attraction_counts.items():
            print(f"  {attraction}: {count} lignes")

# Analyser la distribution
analyser_distribution_groupes(df)