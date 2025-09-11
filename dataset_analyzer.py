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

#------------------------------------------------------------------

# ---------- 1) Liste de modèles de base (diversifiés) ----------
def get_base_models():
    base = []

    # XGB – variations (seeds/params)
    xgb_sets = [
        dict(n_estimators=800, max_depth=6, learning_rate=0.1, subsample=1.0, gamma=0, colsample_bytree=0.8),
        dict(n_estimators=600, max_depth=8, learning_rate=0.1, subsample=0.7, gamma=1, colsample_bytree=0.9),
        dict(n_estimators=800, max_depth=5, learning_rate=0.05, subsample=0.7, gamma=0, colsample_bytree=0.7),
        dict(n_estimators=200, max_depth=10, learning_rate=0.1, subsample=0.9, gamma=0, colsample_bytree=0.8),
        dict(n_estimators=600, max_depth=7, learning_rate=0.05, subsample=0.8, gamma=1, colsample_bytree=1.0),
        dict(n_estimators=1000, max_depth=9, learning_rate=0.05, subsample=1.0, gamma=0, colsample_bytree=0.9),
        dict(n_estimators=400, max_depth=4, learning_rate=0.2, subsample=0.85, gamma=1, colsample_bytree=0.7),
        dict(n_estimators=700, max_depth=7, learning_rate=0.1, subsample=0.75, gamma=0, colsample_bytree=0.85),
        dict(n_estimators=500, max_depth=6, learning_rate=0.15, subsample=0.75, gamma=2, colsample_bytree=0.9),
        dict(n_estimators=900, max_depth=8, learning_rate=0.03, subsample=0.9, gamma=0, colsample_bytree=1.0),
    ]
    for i, p in enumerate(xgb_sets):
        base.append((
            f"xgb_{i}",
            xgb.XGBRegressor(
                random_state=42 + i, n_jobs=-1,
                tree_method="hist",
                **p
            )
        ))

    # Forêts (robustes, patterns différents)
    base.append(("rf_0", RandomForestRegressor(
        n_estimators=600, max_depth=12, min_samples_leaf=2,
        n_jobs=-1, random_state=7
    )))
    base.append(("et_0", ExtraTreesRegressor(
        n_estimators=700, max_depth=None, min_samples_leaf=1,
        n_jobs=-1, random_state=11
    )))

    return base

# ---------- 2) Stacking OOF ----------
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from packaging import version
import xgboost as xgb
import numpy as np

from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge


def compute_oof_weights(oof_preds, y, power=1.0):
    # RMSE par modèle
    rmses = np.array([np.sqrt(mean_squared_error(y, oof_preds[:, i]))
                      for i in range(oof_preds.shape[1])])
    # poids = 1 / rmse^power (power=1 par défaut)
    w = 1.0 / (rmses ** power + 1e-9)
    w = w / w.sum()
    return w, rmses

def predict_with_blend(fitted_base, features, df, weights):
    X = df[features].values
    base_mat = np.column_stack([mdl.predict(X) for mdl in fitted_base])
    return base_mat.dot(weights)

def train_stacking_oof(df, val_ref=None, target="WAIT_TIME_IN_2H", n_splits=5, weight_mode="none"):
    df["DATETIME"] = pd.to_datetime(df["DATETIME"])

    features = [c for c in df.columns if c not in [target, "DATETIME", "ENTITY_DESCRIPTION_SHORT"]]
    X, y = df[features].values, df[target].values

    # -------------------------
    # Pondération
    # -------------------------
    if weight_mode == "none":
        weights = np.ones(len(df))
    elif weight_mode == "linear":
        days = (df["DATETIME"] - df["DATETIME"].min()).dt.days
        weights = days / days.max()
    elif weight_mode == "sqrt":
        days = (df["DATETIME"] - df["DATETIME"].min()).dt.days
        weights = np.sqrt(days / days.max())
    elif weight_mode == "log":
        days = (df["DATETIME"] - df["DATETIME"].min()).dt.days
        weights = np.log1p(days) / np.log1p(days.max())
    elif weight_mode == "distribution_match":
        if val_ref is None:
            raise ValueError("⚠️ Pour weight_mode='distribution_match', il faut fournir val_ref (jeu de validation/test).")
        weights = compute_time_weights(df, val_ref, freq="M")
    else:
        raise ValueError("Mode de pondération inconnu")

    weights = weights + 0.01  # éviter poids nuls


    # Modèles de base
    base_models = get_base_models()
    oof_preds = np.zeros((len(X), len(base_models)))

    # Split CV
    folds = KFold(n_splits=n_splits, shuffle=True, random_state=42).split(X)

    for i, (tr_idx, va_idx) in enumerate(folds):
        Xtr, Xva = X[tr_idx], X[va_idx]
        ytr, yva = y[tr_idx], y[va_idx]
        wtr, wva = weights[tr_idx], weights[va_idx]

        for m, (name, mdl) in enumerate(base_models):
            print(f"🔄 Fold {i+1}, Model {name}")
            try:
                mdl.fit(Xtr, ytr, sample_weight=wtr)
            except TypeError:
                # Certains modèles n'acceptent pas sample_weight → fallback
                mdl.fit(Xtr, ytr)
            oof_preds[va_idx, m] = mdl.predict(Xva)

    # Calcul des poids de blend
    blend_weights, base_rmses = compute_oof_weights(oof_preds, y, power=1.0)
    print("Poids blend (1/rmse):", blend_weights)
    print("RMSE bases:", base_rmses)

    # Métamodèle
    meta_model = Ridge(alpha=1.0)
    meta_model.fit(oof_preds, y)

    # Réentraîner tous les modèles de base sur 100% du dataset
    fitted_base = []
    for name, mdl in base_models:
        try:
            mdl.fit(X, y, sample_weight=weights)
        except TypeError:
            mdl.fit(X, y)
        fitted_base.append(mdl)

    return fitted_base, meta_model, features, blend_weights




# ---------- 3) Prédiction ----------
def predict_stacking(fitted_base, meta_model, features, df):
    X = df[features].values
    # Prédictions des modèles de base
    base_mat = np.column_stack([mdl.predict(X) for mdl in fitted_base])
    # Métamodèle combine les prédictions
    return meta_model.predict(base_mat)


# ---------- 4) Post-traitement ----------
def round_floor_to_5(arr):
    return (np.floor(arr / 5) * 5).astype(int)

if __name__ == "__main__":
    # Entraînement
    df = pd.read_csv("weather_data_combined.csv")
    df = adapt_data_paul_GX(df)

    # Jeu de validation comme référence pour la distribution
    val = pd.read_csv("valmeteo.csv")
    val = adapt_data_paul_GX(val)

    # Entraînement stacking avec distribution matching
    fitted_base, meta, features, blend_weights = train_stacking_oof(
        df, val_ref=val, weight_mode="distribution_match"
    )

    # Validation externe
    y_pred_stack = predict_stacking(fitted_base, meta, features, val)
    y_pred_blend = predict_with_blend(fitted_base, features, val, blend_weights)
    y_pred = 0.5 * y_pred_stack + 0.5 * y_pred_blend

    val['y_pred'] = y_pred
    val[['DATETIME','ENTITY_DESCRIPTION_SHORT','y_pred']].assign(KEY="Validation").to_csv("val_predictions_stacking.csv", index=False)
    print("✅ Prédictions stacking exportées (mode distribution_match)")


