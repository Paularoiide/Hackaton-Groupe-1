import pandas as pd
from datetime import datetime
# en gros tu mets en 1 le fichier que tu veux modifier, en 2 le fichier reference

def ajuster_proportion_simple(csv1_path, csv2_path):
    """
    Version simplifiée pour ajuster les proportions
    """
    # Charger les données
    df1 = pd.read_csv(csv1_path)
    df2 = pd.read_csv(csv2_path)
    
    # Date de référence
    ref_date = datetime(2020, 3, 1)
    
    # Calculer les proportions
    prop1 = (df1['DATETIME'] >= ref_date).mean()
    prop2 = (df2['DATETIME'] >= ref_date).mean()
    
    print(f"Proportion CSV1: {prop1:.3f}")
    print(f"Proportion CSV2: {prop2:.3f}")
    
    # Créer une copie pour ne pas modifier l'original
    df1_modified = df1.copy()
    
    # Si CSV1 a plus de données récentes que CSV2, on en supprime
    if prop1 > prop2:
        recent_data = df1_modified[df1_modified['DATETIME'] >= ref_date]
        to_remove = int(len(recent_data) * (prop1 - prop2) / prop1)
        
        # Sélection aléatoire des lignes à supprimer
        indices_to_remove = recent_data.sample(to_remove, random_state=42).index
        df1_modified = df1_modified.drop(indices_to_remove)
    
    # Calculer la nouvelle proportion
    new_prop = (df1_modified['DATETIME'] >= ref_date).mean()
    print(f"Nouvelle proportion CSV1: {new_prop:.3f}")
    
    return df1_modified





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
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns
import xgboost as xgb
from xgboost import XGBClassifier
from datetime import datetime
from datetime import date
import shap
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
import lightgbm as lgb
import catboost as cb
import xgboost as xgb
from copy import deepcopy
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error

valsetsansmeteo = pd.read_table('waiting_times_X_test_val.csv', sep=',', decimal='.')
valsetmeteo = pd.read_table('valmeteo.csv', sep=',', decimal='.')
datasetmeteo = pd.read_table('weather_data_combined.csv', sep=',', decimal='.')
datasetsansmeteo = pd.read_table('waiting_times_train.csv', sep=',', decimal='.')


def compute_time_weights(train, val, freq="M"):
    """
    Calcule des poids pour aligner la distribution temporelle du train
    sur celle du val/test. 
    freq : "M" pour mensuel, "Q" pour trimestriel
    """
    # S'assurer que DATETIME est bien en datetime
    train = train.copy()
    val = val.copy()
    train["DATETIME"] = pd.to_datetime(train["DATETIME"])
    val["DATETIME"] = pd.to_datetime(val["DATETIME"])
    
    # Comptages normalisés par période
    train_counts = train.resample(freq, on="DATETIME").size()
    val_counts = val.resample(freq, on="DATETIME").size()
    
    train_dist = train_counts / train_counts.sum()
    val_dist = val_counts / val_counts.sum()
    
    # Ratio val/train = poids relatifs
    ratios = (val_dist / train_dist).replace([np.inf, np.nan], 0)
    
    # Appliquer ratio à chaque ligne
    weights = train["DATETIME"].dt.to_period(freq).map(ratios)
    
    return weights.fillna(0.01).values


def assign_era(dt):
    if dt < pd.to_datetime("2020-03-01"):
        return 0  # Pré-COVID
    elif dt < pd.to_datetime("2021-01-01"):
        return 1  # COVID strict
    else:
        return 2  # Post-COVID (réouverture)


def RMSE(x,y):
  return np.sqrt(mean_squared_error(x,y))

def adapt_data_paul_GX(dataset):
    # Faire une copie pour éviter les modifications sur l'original
    dataset = dataset.copy()

    dataset['IS_RAINING'] = (dataset['rain_1h'] > 0.2).astype(int) #No changes
    dataset['IS_SNOWING'] = (dataset['snow_1h'] > 0.05).astype(int) #No changes
    dataset['IS_HOT'] = (dataset['feels_like'] > 25).astype(int) #No changes
    dataset['IS_COLD'] = (dataset['feels_like'] < 0).astype(int) #Could be to remove
    #dataset['IS_BAD_WEATHER'] = ((dataset['rain_1h'] > 2) |     #Could be to remove
                                 #(dataset['snow_1h'] > 0.5) |
                                 #(dataset['wind_speed'] > 30)).astype(int)
    dataset['TEMP_HUMIDITY_INDEX'] = dataset['feels_like'] * dataset['humidity']  #No changes
    

    dataset['CAPACITY_RATIO'] = dataset['CURRENT_WAIT_TIME'] / (dataset['ADJUST_CAPACITY'] + 1e-6) #No changes with wait time removed
    # Remplir les autres valeurs manquantes
    dataset['snow_1h'] = dataset['snow_1h'].fillna(0)
    
    # Conversion datetime
    dataset['DATETIME'] = pd.to_datetime(dataset['DATETIME'])
    dataset['DAY_OF_WEEK'] = dataset['DATETIME'].dt.dayofweek
    dataset['DAY'] = dataset['DATETIME'].dt.day
    dataset['MONTH'] = dataset['DATETIME'].dt.month
    dataset['YEAR'] = dataset['DATETIME'].dt.year
    dataset['HOUR'] = dataset['DATETIME'].dt.hour
    dataset['MINUTE'] = dataset['DATETIME'].dt.minute


    # Colonne weekend (1 si weekend, 0 si jour de semaine)
    dataset['WEEKEND'] = np.where(dataset['DAY_OF_WEEK'] >= 5, 1, 0)

    # Colonne POST_COVID (1 si après Mars 2020, 0 si avant)
    #dataset['POST_COVID'] = np.where(dataset['DATETIME'] >= '2020-03-01', 1, 0)
    dataset["ERA"] = dataset["DATETIME"].apply(assign_era)
    # Binarisation des attractions
    dataset['IS_ATTRACTION_Water_Ride'] = np.where(dataset['ENTITY_DESCRIPTION_SHORT'] == "Water Ride", 1, 0)
    dataset['IS_ATTRACTION_Pirate_Ship'] = np.where(dataset['ENTITY_DESCRIPTION_SHORT'] == "Pirate Ship", 1, 0)
    dataset['IS_ATTRACTION__Flying_Coaster'] = np.where(dataset['ENTITY_DESCRIPTION_SHORT'] == "Flying Coaster", 1, 0)
    
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

    # Appliquer la détection des vacances
    vacances_data = dataset['DATETIME'].apply(detecter_vacances_par_zone)
    vacances_df = pd.DataFrame(list(vacances_data))
    
    # Fusionner avec le dataset principal
    dataset = pd.concat([dataset, vacances_df], axis=1)



    dataset.drop(columns=['CURRENT_WAIT_TIME','dew_point'], inplace=True) #feels_like, humidity,dew_point et parade_2, (day peut etre) peuvent être enlévés
    
    return dataset


# ---------- helpers ----------
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def round_to_5_floor(a):
    return (np.floor(a / 5) * 5).astype(int)

def build_xgb_models(xgb_sets, seeds=(42,133)):
    """
    Construit une liste [(name, model), ...] à partir d'une liste de dicts de params et de seeds.
    """
    models = []
    for i, params in enumerate(xgb_sets):
        for s in seeds:
            name = f"xgb_{i}_seed{s}"
            mdl = xgb.XGBRegressor(
                random_state=s, n_jobs=-1, tree_method="hist",
                objective="reg:squarederror", eval_metric="rmse",
                **params
            )
            models.append((name, mdl))
    return models

def get_oof_matrix(models, X, y, n_splits=5, random_state=42):
    """
    Calcule la matrice OOF des prédictions des modèles.
    Retourne: (oof_preds: [n_samples, n_models], per_model_rmse: [n_models], fitted_full_models: [(name, fitted_model), ...])
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    n_samples, n_models = X.shape[0], len(models)
    oof = np.zeros((n_samples, n_models), dtype=float)

    # Fit/preds OOF
    for m_idx, (name, mdl_proto) in enumerate(models):
        preds = np.zeros(n_samples, dtype=float)
        for tr_idx, va_idx in kf.split(X, y):
            mdl = deepcopy(mdl_proto)
            mdl.fit(X[tr_idx], y[tr_idx])
            preds[va_idx] = mdl.predict(X[va_idx])
        oof[:, m_idx] = preds

    per_model_rmse = np.array([rmse(y, oof[:, j]) for j in range(n_models)])

    # Refit sur tout le train (pour servir en prédiction finale)
    fitted_full = []
    for name, mdl_proto in models:
        mdl = deepcopy(mdl_proto)
        mdl.fit(X, y)
        fitted_full.append((name, mdl))

    return oof, per_model_rmse, fitted_full

def nnls_like_weights(P, y):
    """
    Approx NNLS simple: moindres carrés puis projection dans le simplex (>=0, somme=1).
    P: [n_samples, n_models], y: [n_samples]
    """
    # moindres carrés
    w, *_ = np.linalg.lstsq(P, y, rcond=None)
    # >= 0
    w = np.maximum(w, 0)
    # éviter tout zéro
    if w.sum() == 0:
        w = np.ones_like(w) / len(w)
    else:
        w = w / w.sum()
    return w

def greedy_subset_selection(oof, y, max_models=6, verbose=True):
    """
    Sélection gloutonne d'un sous-ensemble de modèles qui minimise le RMSE OOF de la combinaison (poids appris par NNLS-like).
    Retourne: idx_list (indices retenus), weights (poids sur ce sous-ensemble), rmse_best
    """
    n_models = oof.shape[1]
    remaining = set(range(n_models))
    chosen = []
    best_rmse = np.inf
    best_weights = None

    while len(chosen) < max_models and remaining:
        improved = False
        candidate_best = None
        candidate_weights = None
        candidate_rmse = None

        for j in list(remaining):
            idx_try = chosen + [j]
            P = oof[:, idx_try]
            w = nnls_like_weights(P, y)
            y_hat = P.dot(w)
            r = rmse(y, y_hat)
            if r < best_rmse - 1e-6:
                improved = True
                candidate_best = j
                candidate_rmse = r
                candidate_weights = w

        if improved:
            chosen.append(candidate_best)
            remaining.remove(candidate_best)
            best_rmse = candidate_rmse
            best_weights = candidate_weights
            if verbose:
                print(f"➕ Ajout modèle {candidate_best} | OOF RMSE={best_rmse:.4f} | k={len(chosen)}")
        else:
            break

    return chosen, best_weights, best_rmse

def predict_ensemble(fitted_models, selected_idx, weights, X):
    """
    Combine les modèles sélectionnés avec les poids appris.
    """
    preds_mat = []
    for k, (name, mdl) in enumerate(fitted_models):
        if k in selected_idx:
            preds_mat.append(mdl.predict(X))
    P = np.column_stack(preds_mat)  # [n_samples, k]
    return P.dot(weights)

# 1) Prépare les données
df = pd.read_csv("weather_data_combined.csv")
df = adapt_data_paul_GX(df)  # ta fonction d’ingénierie
features = [c for c in df.columns if c not in ["WAIT_TIME_IN_2H", "DATETIME", "ENTITY_DESCRIPTION_SHORT"]]
X = df[features].values
y = df["WAIT_TIME_IN_2H"].values

# 2) Construis les modèles à partir de tes configs
#    (utilise EXACTEMENT ton format xgb_sets existant)
xgb_sets = [
    # --- Autour de B : depth 6, réglages fins ---
    {"n_estimators": 1000, "learning_rate": 0.09, "max_depth": 6, "min_child_weight": 3, "subsample": 0.90, "colsample_bytree": 0.90, "gamma": 0.0, "reg_alpha": 0.0,   "reg_lambda": 1.0},
    {"n_estimators": 1100, "learning_rate": 0.08, "max_depth": 6, "min_child_weight": 3, "subsample": 0.90, "colsample_bytree": 0.90, "gamma": 0.5, "reg_alpha": 0.0,   "reg_lambda": 2.0},
    {"n_estimators": 1200, "learning_rate": 0.07, "max_depth": 6, "min_child_weight": 4, "subsample": 0.90, "colsample_bytree": 0.90, "gamma": 0.0, "reg_alpha": 0.0,   "reg_lambda": 2.0},
    {"n_estimators": 1300, "learning_rate": 0.06, "max_depth": 6, "min_child_weight": 5, "subsample": 0.90, "colsample_bytree": 0.90, "gamma": 1.0, "reg_alpha": 0.0,   "reg_lambda": 2.0},
    {"n_estimators": 900,  "learning_rate": 0.10, "max_depth": 6, "min_child_weight": 4, "subsample": 0.85, "colsample_bytree": 0.90, "gamma": 0.0, "reg_alpha": 0.0,   "reg_lambda": 1.0},
    {"n_estimators": 1000, "learning_rate": 0.10, "max_depth": 6, "min_child_weight": 5, "subsample": 0.85, "colsample_bytree": 0.85, "gamma": 0.5, "reg_alpha": 0.0,   "reg_lambda": 2.0},
    {"n_estimators": 1200, "learning_rate": 0.08, "max_depth": 6, "min_child_weight": 6, "subsample": 0.95, "colsample_bytree": 0.90, "gamma": 0.5, "reg_alpha": 0.0,   "reg_lambda": 3.0},
    {"n_estimators": 1400, "learning_rate": 0.07, "max_depth": 6, "min_child_weight": 4, "subsample": 0.85, "colsample_bytree": 0.95, "gamma": 0.0, "reg_alpha": 0.0,   "reg_lambda": 2.0},
    {"n_estimators": 1600, "learning_rate": 0.05, "max_depth": 6, "min_child_weight": 3, "subsample": 0.90, "colsample_bytree": 0.90, "gamma": 0.0, "reg_alpha": 0.0,   "reg_lambda": 2.0},
    {"n_estimators": 1500, "learning_rate": 0.06, "max_depth": 6, "min_child_weight": 5, "subsample": 0.95, "colsample_bytree": 0.85, "gamma": 0.5, "reg_alpha": 0.0,   "reg_lambda": 3.0},
]
models = build_xgb_models(xgb_sets, seeds=(42,133))  # 2 seeds → robuste

# 3) OOF + sélection gloutonne + poids
oof, per_rmse, fitted_full = get_oof_matrix(models, X, y, n_splits=5, random_state=42)
print("RMSE OOF par modèle :", per_rmse.round(4))

# sélection d'un sous-ensemble de taille max (ex: 6 modèles)
idx_sel, w_sel, oof_rmse = greedy_subset_selection(oof, y, max_models=6, verbose=True)
print("Indices retenus :", idx_sel)
print("Poids retenus   :", np.round(w_sel, 4))
print("OOF RMSE blend  :", round(oof_rmse, 4))

# 4) Validation externe
val = pd.read_csv("valmeteo.csv")
val = adapt_data_paul_GX(val)
X_val = val[features].values

# Prédiction ensemble sélectionné
y_val_pred = predict_ensemble(fitted_full, idx_sel, w_sel, X_val)

# Post-traitement (clips + arrondi multiples de 5 vers le bas)
y_val_pred = np.clip(y_val_pred, 0, 180)
y_val_pred = round_to_5_floor(y_val_pred)

val["y_pred"] = y_val_pred
val[["DATETIME","ENTITY_DESCRIPTION_SHORT","y_pred"]].assign(KEY="Validation")\
   .to_csv("val_predictions_xgb_subset_blend.csv", index=False)

print("✅ Écrit : val_predictions_xgb_subset_blend.csv")

