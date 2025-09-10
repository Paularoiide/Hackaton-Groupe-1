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

def adapter_dataset_16_groupes_avec_covid(dataset):
    """
    Adapte le dataset pour créer 16 groupes basés sur les combinaisons de vacances scolaires
    et la période avant/après COVID, avec ajout de features supplémentaires.
    
    Parameters:
    dataset (DataFrame): Dataset contenant les colonnes DATETIME et autres features
    
    Returns:
    dict: Dictionnaire contenant les 16 groupes de données
    """
    
    # Faire une copie pour éviter les modifications sur l'original
    dataset = dataset.copy()
    
    # Conversion datetime
    dataset['DATETIME'] = pd.to_datetime(dataset['DATETIME'])
    
    # Trier par date pour assurer un ordre chronologique
    dataset = dataset.sort_values('DATETIME').reset_index(drop=True)
    
    # Ajouter la colonne période COVID
    def categoriser_periode_covid(date):
        if date < datetime(2020, 3, 1):  # Avant mars 2020
            return "Avant COVID"
        else:  # Après mars 2020
            return "Après COVID"
    
    dataset['PERIODE_COVID'] = dataset['DATETIME'].apply(categoriser_periode_covid)
    
    # Fonction pour détecter les vacances scolaires par zone
    def detecter_vacances_par_zone(date):
        # Dates exactes des vacances scolaires françaises 2018-2022 par zone
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
    
    # Créer 16 groupes basés sur les combinaisons de vacances ET période COVID
    groupes = {}
    
    # Toutes les combinaisons possibles des 3 zones de vacances
    combinaisons_vacances = [
        (0, 0, 0),  # Aucune zone en vacances
        (1, 0, 0),  # Seulement zone A en vacances
        (0, 1, 0),  # Seulement zone B en vacances
        (0, 0, 1),  # Seulement zone C en vacances
        (1, 1, 0),  # Zones A et B en vacances
        (1, 0, 1),  # Zones A et C en vacances
        (0, 1, 1),  # Zones B et C en vacances
        (1, 1, 1),  # Toutes les zones en vacances
    ]
    
    # Périodes COVID
    periodes_covid = ["Avant COVID", "Après COVID"]
    
    noms_groupes = []
    
    for periode in periodes_covid:
        for i, (vac_a, vac_b, vac_c) in enumerate(combinaisons_vacances):
            mask = (dataset['VACANCES_ZONE_A'] == vac_a) & \
                   (dataset['VACANCES_ZONE_B'] == vac_b) & \
                   (dataset['VACANCES_ZONE_C'] == vac_c) & \
                   (dataset['PERIODE_COVID'] == periode)
            
            groupe_data = dataset[mask].copy()
            
            # Nom du groupe avec période COVID
            suffixe_vacances = [
                'aucune_vacances',
                'vacances_A_seule',
                'vacances_B_seule',
                'vacances_C_seule',
                'vacances_A_B',
                'vacances_A_C',
                'vacances_B_C',
                'toutes_vacances'
            ][i]
            
            groupe_name = f'groupe_{suffixe_vacances}_{periode.lower().replace(" ", "_")}'
            noms_groupes.append(groupe_name)
            
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
            
            groupes[groupe_name] = groupe_data
    
    return groupes

# Exemple d'utilisation
if __name__ == "__main__":
    # Charger ton dataset
    # df = pd.read_csv('ton_dataset.csv')
    
    # Exemple avec des données fictives pour tester
    dates_example = pd.date_range('2019-01-01', '2022-12-31', freq='D')
    df_example = pd.DataFrame({
        'DATETIME': dates_example,
        'ENTITY_DESCRIPTION_SHORT': np.random.choice(["Water Ride", "Pirate Ship", "Flying Coaster", "Other"], len(dates_example)),
        'TIME_TO_PARADE_1': np.random.randint(-1000, 1000, len(dates_example)),
        'TIME_TO_PARADE_2': np.random.randint(-1000, 1000, len(dates_example)),
        'TIME_TO_NIGHT_SHOW': np.random.randint(-1000, 1000, len(dates_example)),
        'snow_1h': np.random.choice([0, 1, np.nan], len(dates_example), p=[0.8, 0.1, 0.1])
    })
    
    # Appliquer la fonction
    groupes_result = adapter_dataset_16_groupes_avec_covid(df_example)
    
    # Afficher les informations sur les groupes
    print("Nombre de groupes créés:", len(groupes_result))
    print("\nTaille de chaque groupe:")
    for groupe_name, groupe_data in groupes_result.items():
        print(f"{groupe_name}: {len(groupe_data)} observations")
    
    # Afficher un exemple de groupe
    print(f"\nExemple du premier groupe:")
    premier_groupe = list(groupes_result.keys())[0]
    print(f"Nom: {premier_groupe}")
    print(f"Données: {groupes_result[premier_groupe].head()}")


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from datetime import datetime

# ------------------------
# Fonction d'entraînement pour 16 groupes
# ------------------------
def train_16_groupes_models(groupes_dict, target="WAIT_TIME_IN_2H"):
    """
    Entraîne un modèle Random Forest pour chacun des 16 groupes
    
    Parameters:
    groupes_dict (dict): Dictionnaire contenant les 16 groupes de données
    target (str): Nom de la colonne cible
    
    Returns:
    dict: Modèles entraînés pour chaque groupe
    list: Liste des features utilisées
    """
    
    models = {}
    features_info = {}
    
    for groupe_name, groupe_data in groupes_dict.items():
        if len(groupe_data) < 50:  # Minimum d'observations requis
            print(f"⚠️ Pas assez de données pour {groupe_name}: {len(groupe_data)} observations")
            continue
        
        print(f"\n--- Entraînement modèle {groupe_name} ({len(groupe_data)} observations) ---")
        
        # Définir les features (exclure les colonnes non pertinentes)
        exclude_cols = [target, "DATETIME", "ENTITY_DESCRIPTION_SHORT", "PERIODE_COVID", 
                       "VACANCES_ZONE_A", "VACANCES_ZONE_B", "VACANCES_ZONE_C"]
        
        features = [col for col in groupe_data.columns if col not in exclude_cols and col != target]
        
        X = groupe_data[features]
        y = groupe_data[target]
        
        # Vérifier qu'il y a des données
        if len(X) == 0 or len(y) == 0:
            print(f"❌ Aucune donnée pour les features ou la target dans {groupe_name}")
            continue
        
        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Entraînement du modèle
        rf = RandomForestRegressor(
            n_estimators=500,
            max_depth=20,
            min_samples_leaf=2,
            max_features="log2",
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X_train, y_train)
        
        # Évaluation
        y_pred = rf.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        
        print(f"RMSE {groupe_name}: {rmse:.2f}")
        print(f"MAE {groupe_name}: {mae:.2f}")
        
        # Sauvegarde du modèle
        models[groupe_name] = rf
        features_info[groupe_name] = features
        
        # Importance des features
        feat_importances = pd.DataFrame({
            "Feature": features,
            "Importance": rf.feature_importances_
        }).sort_values("Importance", ascending=False)
        
        print(f"\nTop 10 features {groupe_name}:\n", feat_importances.head(10))
        
        # Visualisation des features importantes
        plt.figure(figsize=(10, 8))
        plt.barh(feat_importances["Feature"][:15][::-1], feat_importances["Importance"][:15][::-1])
        plt.title(f"Top 15 Features - {groupe_name}")
        plt.xlabel("Importance")
        plt.tight_layout()
        plt.show()
    
    return models, features_info

# ------------------------
# Prédiction avec les 16 modèles
# ------------------------
def predict_16_groupes(models_dict, features_info, df):
    """
    Prédit les valeurs pour un dataset en utilisant les modèles appropriés
    
    Parameters:
    models_dict (dict): Dictionnaire des modèles par groupe
    features_info (dict): Dictionnaire des features par groupe
    df (DataFrame): Données à prédire
    
    Returns:
    array: Prédictions
    """
    
    # Préparer le dataset pour la prédiction (appliquer les mêmes transformations)
    df_pred = adapter_dataset_16_groupes_avec_covid(df)
    
    preds = np.zeros(len(df))
    df_original_index = df.index  # Conserver l'index original
    
    for groupe_name, groupe_data in df_pred.items():
        if groupe_name not in models_dict:
            continue
            
        if len(groupe_data) == 0:
            continue
        
        # Récupérer le modèle et les features pour ce groupe
        model = models_dict[groupe_name]
        features = features_info[groupe_name]
        
        # Vérifier que toutes les features sont présentes
        missing_features = set(features) - set(groupe_data.columns)
        if missing_features:
            print(f"⚠️ Features manquantes dans {groupe_name}: {missing_features}")
            # Ajouter les features manquantes avec des valeurs par défaut
            for feat in missing_features:
                groupe_data[feat] = 0
        
        # Faire la prédiction
        try:
            groupe_preds = model.predict(groupe_data[features])
            
            # Assigner les prédictions aux indices correspondants
            original_indices = groupe_data.index
            preds[original_indices] = groupe_preds
            
        except Exception as e:
            print(f"❌ Erreur lors de la prédiction pour {groupe_name}: {e}")
    
    return preds

# ------------------------
# Fonction utilitaire pour appliquer les transformations
# ------------------------
def prepare_dataset_for_prediction(df):
    """
    Applique les mêmes transformations que pendant l'entraînement
    """
    # Faire une copie
    df_prepared = df.copy()
    
    # Conversion datetime
    df_prepared['DATETIME'] = pd.to_datetime(df_prepared['DATETIME'])
    
    # Trier par date pour assurer un ordre chronologique
    df_prepared = df_prepared.sort_values('DATETIME').reset_index(drop=True)
    
    # Appliquer la fonction de groupe
    groupes = adapter_dataset_16_groupes_avec_covid(df_prepared)
    
    return groupes

# ------------------------
# Exemple d'utilisation complet
# ------------------------
if __name__ == "__main__":
    # Charger vos données
    df_train = pd.read_csv("weather_data_combined.csv")
    
    # Préparer les groupes d'entraînement
    print("Préparation des groupes d'entraînement...")
    groupes_train = adapter_dataset_16_groupes_avec_covid(df_train)
    
    # Entraîner les modèles
    print("Entraînement des modèles...")
    models_16, features_info = train_16_groupes_models(groupes_train)
    
    # Charger les données de validation
    df_val = pd.read_csv("valmeteo.csv")
    
    # Prédire sur les données de validation
    print("Prédiction sur les données de validation...")
    y_val_pred = predict_16_groupes(models_16, features_info, df_val)
    
    # Ajouter les prédictions au dataset de validation
    df_val['y_pred'] = y_val_pred
    
    # Calculer les métriques de performance
    if 'WAIT_TIME_IN_2H' in df_val.columns:
        rmse_val = np.sqrt(mean_squared_error(df_val['WAIT_TIME_IN_2H'], df_val['y_pred']))
        mae_val = mean_absolute_error(df_val['WAIT_TIME_IN_2H'], df_val['y_pred'])
        print(f"\n📊 Performance sur validation:")
        print(f"RMSE: {rmse_val:.2f}")
        print(f"MAE: {mae_val:.2f}")
    
    # Exporter les résultats
    output_cols = ['DATETIME', 'ENTITY_DESCRIPTION_SHORT', 'y_pred']
    if all(col in df_val.columns for col in output_cols):
        df_val[output_cols].assign(KEY="Validation").to_csv("val_predictions_16_groupes.csv", index=False)
        print("✅ Prédictions exportées vers val_predictions_16_groupes.csv")
    
    # Afficher quelques statistiques
    print(f"\n📈 Statistiques des prédictions:")
    print(f"Nombre total de prédictions: {len(y_val_pred)}")
    print(f"Moyenne des prédictions: {y_val_pred.mean():.2f}")
    print(f"Écart-type des prédictions: {y_val_pred.std():.2f}")
    print(f"Min/Max des prédictions: {y_val_pred.min():.2f} / {y_val_pred.max():.2f}")