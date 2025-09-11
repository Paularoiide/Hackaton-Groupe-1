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

# Put the dataset into a pandas DataFrame

valsetsansmeteo = pd.read_table('waiting_times_X_test_val.csv', sep=',', decimal='.')
valsetmeteo = pd.read_table('valmeteo.csv', sep=',', decimal='.')
datasetmeteo = pd.read_table('weather_data_combined.csv', sep=',', decimal='.')
datasetsansmeteo = pd.read_table('waiting_times_train.csv', sep=',', decimal='.')

def AIC(X, predictors, y):
# AIC and BIC based stepwise forward selection
    selected_predictors = ['const'] # Start with only the intercept
    unselected_predictors = predictors.copy()
    X = sm.add_constant(X)  # Ajoute une constante si elle n'existe pas déjà

    # Compute the Information Criterion (IC) for the model with only the intercept
    current_ic = sm.OLS(y, X[selected_predictors]).fit().aic # AIC

    ic_values = [current_ic]  # Store successive IC values

    # Stepwise forward selection based on minimizing IC
    while len(unselected_predictors)>0:
        best_ic = np.inf  # Initialize with a very high IC
        best_predictor = None

        # Try adding each unselected predictor one by one
        for pred in unselected_predictors:
            test_model = sm.OLS(y, X[selected_predictors + [pred]]).fit()
            test_ic = test_model.aic # AIC

            # Check if this predictor gives the lowest IC so far
            if test_ic < best_ic:
                best_ic = test_ic
                best_predictor = pred

        # If the best IC is lower than the current IC, accept the predictor
        if best_ic < current_ic:
            selected_predictors.append(best_predictor)
            unselected_predictors.remove(best_predictor)
            ic_values.append(best_ic)
            current_ic = best_ic  # Update current IC
        else:
            break  # Stop if no improvement

    # Print results
    print("Selected predictors:", selected_predictors)
    print("IC values over iterations:", ic_values)
    return selected_predictors, ic_values

def RMSE(x,y):
  return np.sqrt(mean_squared_error(x,y))


#________________________________________________________________________________________

from datetime import datetime

def ajuster_proportion_exact(df1, df2):
    ref_date = datetime(2020, 3, 1)

    # Stats
    n1 = len(df1)
    post1 = (df1['DATETIME'] >= ref_date).sum()
    pre1 = n1 - post1
    prop1 = post1 / n1
    prop2 = (df2['DATETIME'] >= ref_date).mean()

    # Proportion cible = moyenne
    target_prop = (prop1 + prop2) / 2

    print(f"Proportion CSV1: {prop1:.3f}")
    print(f"Proportion CSV2: {prop2:.3f}")
    print(f"Proportion cible: {target_prop:.3f}")

    df1_mod = df1.copy()

    if prop1 > target_prop:
        # Trop de post-COVID → on en enlève
        x = int((post1 - target_prop * n1) / (1 - target_prop))
        x = min(x, post1)
        idx_remove = df1_mod[df1_mod['DATETIME'] >= ref_date].sample(x, random_state=42).index
        df1_mod = df1_mod.drop(idx_remove)

    elif prop1 < target_prop:
        # Trop de pre-COVID → on en enlève
        x = int(n1 - (post1 / target_prop))
        x = min(x, pre1)
        idx_remove = df1_mod[df1_mod['DATETIME'] < ref_date].sample(x, random_state=42).index
        df1_mod = df1_mod.drop(idx_remove)

    new_prop = (df1_mod['DATETIME'] >= ref_date).mean()
    print(f"Nouvelle proportion CSV1: {new_prop:.3f}")

    return df1_mod.reset_index(drop=True)



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
    dataset['POST_COVID'] = np.where(dataset['DATETIME'] >= '2020-03-01', 1, 0)
    
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

#________________________________________________________________________________________
# -----------------------------------------------------
# 2. Split pré/post Covid
# -----------------------------------------------------
def split_pre_post(df, covid_date="2020-03-15"):
    df_pre = df[df['DATETIME'] < covid_date].copy()
    df_post = df[df['DATETIME'] >= covid_date].copy()
    return df_pre, df_post

def train_and_feature_importance(df, target="WAIT_TIME_IN_2H", title=""):
    features = [c for c in df.columns if c not in [target, "DATETIME", "ENTITY_DESCRIPTION_SHORT"]]
    X, y = df[features], df[target]

    # split train/test interne
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"RMSE {title} :", rmse)

    # importance
    importances = model.feature_importances_
    feat_importances = pd.DataFrame({
        "Feature": features,
        "Importance": importances
    }).sort_values("Importance", ascending=False)

    print(f"\nTop 15 features {title}:\n", feat_importances.head(15))

    plt.figure(figsize=(8, 6))
    plt.barh(feat_importances["Feature"][:15][::-1], feat_importances["Importance"][:15][::-1])
    plt.title(f"Top 15 Features - {title}")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.show()

    return model, features, rmse, feat_importances, X_train, X_test, y_train, y_test

def train_two_models(df_pre, df_post, target="WAIT_TIME_IN_2H"):
    features = [col for col in df_pre.columns if col not in [target, 'DATETIME', 'ENTITY_DESCRIPTION_SHORT']]

    X_pre, y_pre = df_pre[features], df_pre[target]
    X_post, y_post = df_post[features], df_post[target]

    model_pre = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    model_post = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )

    model_pre.fit(X_pre, y_pre)
    model_post.fit(X_post, y_post)

    return model_pre, model_post, features

def predict_two_models(model_pre, model_post, features, df, covid_date="2020-03-15"):
    df = df.copy()
    df['DATETIME'] = pd.to_datetime(df['DATETIME'])

    mask_pre = df['DATETIME'] < covid_date
    mask_post = df['DATETIME'] >= covid_date

    preds = np.zeros(len(df))

    if mask_pre.any():
        preds[mask_pre] = model_pre.predict(df.loc[mask_pre, features])
    if mask_post.any():
        preds[mask_post] = model_post.predict(df.loc[mask_post, features])

    return preds

def get_sample(df, frac=0.3, random_state=42):
    """
    Retourne un sous-échantillon fixe du dataset.
    Utile pour tester rapidement différentes combinaisons de features.

    Args:
        df (pd.DataFrame): ton dataset complet
        frac (float): proportion du dataset à garder (0 < frac <= 1)
        random_state (int): graine pour la reproductibilité

    Returns:
        pd.DataFrame: sous-échantillon du dataset
    """
    return df.sample(frac=frac, random_state=random_state).reset_index(drop=True)

def eval_on_sample(df, target="WAIT_TIME_IN_2H", frac=0.3, random_state=42):
    """
    Prend un sous-échantillon fixe du dataset et entraîne un XGBRegressor.
    Retourne le RMSE pour évaluer rapidement les performances.
    """
    # Sous-échantillonnage
    df_sample = df.sample(frac=frac, random_state=random_state).reset_index(drop=True)

    # Séparation features / target
    features = [c for c in df_sample.columns if c not in [target, "DATETIME", "ENTITY_DESCRIPTION_SHORT"]]
    X, y = df_sample[features], df_sample[target]

    # Split train/test
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    # Modèle XGBoost simple
    model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # Prédictions
    y_pred = model.predict(X_val)

    # Calcul du RMSE
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    print(f"✅ RMSE sur {int(frac*100)}% du dataset :", rmse)

    return model, features, rmse, df_sample

def train_promising_models_classifier(df, target="WAIT_TIME_IN_2H", n_seeds=3):
    # Transformer la cible en classes (par tranches de 5 minutes)
    df = df.copy()
    df["WAIT_CLASS"] = (df[target] / 5).round().astype(int)

    # Encodage pour avoir des classes continues 0..N
    le = LabelEncoder()
    y = le.fit_transform(df["WAIT_CLASS"])

    features = [c for c in df.columns if c not in [target, "WAIT_CLASS", "DATETIME", "ENTITY_DESCRIPTION_SHORT"]]
    X = df[features]

    # Paramètres prometteurs
    param_list = [
        {'subsample': 1.0, 'n_estimators': 800, 'max_depth': 6, 'learning_rate': 0.1, 'colsample_bytree': 0.8},
        {'subsample': 0.7, 'n_estimators': 600, 'max_depth': 8, 'learning_rate': 0.1, 'colsample_bytree': 0.9},
        {'subsample': 0.7, 'n_estimators': 800, 'max_depth': 5, 'learning_rate': 0.05, 'colsample_bytree': 0.7},
    ]

    models = []
    for i, params in enumerate(param_list):
        for seed in range(n_seeds):
            model = XGBClassifier(
                objective="multi:softmax",
                num_class=len(le.classes_),  # nb exact de classes après encodage
                random_state=42 + seed,
                n_jobs=-1,
                **params
            )
            model.fit(X, y)
            models.append((model, le))  # on stocke aussi le LabelEncoder
            print(f"✅ Classifieur {len(models)} entraîné avec params {params} (seed={42+seed})")

    return models, features

from sklearn.ensemble import ExtraTreesRegressor

def train_promising_models(df, target="WAIT_TIME_IN_2H", n_seeds=2):
    features = [c for c in df.columns if c not in [target, "DATETIME", "ENTITY_DESCRIPTION_SHORT"]]
    X, y = df[features], df[target]

    # On adapte les paramètres pour ExtraTrees
    param_list = [
        {'n_estimators': 800, 'max_depth': 6, 'max_features': 0.8},
        {'n_estimators': 600, 'max_depth': 8, 'max_features': 0.9},
        {'n_estimators': 800, 'max_depth': 5, 'max_features': 0.7},
        {'n_estimators': 200, 'max_depth': 10, 'max_features': 0.8},
        {'n_estimators': 600, 'max_depth': 7, 'max_features': 1.0},
        {'n_estimators': 1000, 'max_depth': 9, 'max_features': 0.9},
        {'n_estimators': 400, 'max_depth': 4, 'max_features': 0.7},
        {'n_estimators': 700, 'max_depth': 7, 'max_features': 0.85},
        {'n_estimators': 500, 'max_depth': 6, 'max_features': 0.9},
        {'n_estimators': 900, 'max_depth': 8, 'max_features': 1.0}
    ]

    models = []
    for i, params in enumerate(param_list):
        for seed in range(n_seeds):
            model = ExtraTreesRegressor(
                random_state=42 + seed,
                n_jobs=-1,
                **params
            )
            model.fit(X, y)
            models.append(model)
            print(f"✅ Modèle {len(models)} entraîné avec params {params} (seed={42+seed})")

    return models, features


def predict_multiple_models_classifier(models, features, df):
    """
    Prédit avec plusieurs classifieurs, moyenne les classes encodées,
    puis reconvertit en minutes (classe originale * 5).
    """
    preds = np.zeros((len(df), len(models)))
    
    for i, (model, le) in enumerate(models):
        preds[:, i] = model.predict(df[features])

    # Moyenne des classes encodées
    class_pred_encoded = np.rint(preds.mean(axis=1)).astype(int)

    # Décodage vers WAIT_CLASS original
    class_pred = models[0][1].inverse_transform(class_pred_encoded)

    # Conversion en minutes
    return class_pred * 5, preds


def predict_hybrid(models_reg, models_clf, features, df, alpha=0.7):
    """
    Combine régression et classification avec une moyenne pondérée.
    alpha : poids du regressor (0.0 = seulement classifier, 1.0 = seulement regressor)
    """
    # --- Régression ---
    y_pred_reg, _ = predict_multiple_models(models_reg, features, df)

    # --- Classification ---
    y_pred_clf, _ = predict_multiple_models_classifier(models_clf, features, df)

    # --- Hybrid ---
    y_pred_hybrid = alpha * y_pred_reg + (1 - alpha) * y_pred_clf

    # --- Arrondi au multiple de 5 ---
    y_pred_hybrid = (np.round(y_pred_hybrid / 5) * 5).astype(int)

    return y_pred_hybrid, y_pred_reg, y_pred_clf


def predict_multiple_models(models, features, df):
    """
    Prend la moyenne des prédictions de plusieurs modèles
    Retourne : (prédictions moyennes, prédictions individuelles)
    """
    preds = np.zeros((len(df), len(models)))
    for i, model in enumerate(models):
        preds[:, i] = model.predict(df[features])
    return preds.mean(axis=1), preds


def tune_xgb_random(X, y, n_iter=30):
    param_dist = {
        "max_depth": [3, 4, 5, 6, 7, 8],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "n_estimators": [200, 400, 600, 800],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "gamma": [0, 1, 5]
    }

    model = xgb.XGBRegressor(random_state=42, n_jobs=-1)

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=n_iter,   # 👈 paramètre passé ici
        scoring="neg_root_mean_squared_error",
        cv=3,
        verbose=2,
        n_jobs=-1,
        random_state=42
    )

    search.fit(X, y)
    print("✅ Best parameters:", search.best_params_)
    print("✅ Best score (RMSE):", -search.best_score_)

    return search.best_estimator_

def tune_xgb_random_topn(X, y, n_iter=50, top_n=5):
    """
    Lance RandomizedSearchCV, récupère les top_n meilleurs params,
    entraîne un modèle pour chacun et retourne modèles + poids.
    """
    param_dist = {
        "max_depth": [3, 4, 5, 6, 7, 8, 10],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "n_estimators": [200, 400, 600, 800, 1000],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "gamma": [0, 1, 5]
    }

    base_model = xgb.XGBRegressor(random_state=42, n_jobs=-1)

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="neg_root_mean_squared_error",
        cv=3,
        verbose=1,
        n_jobs=-1,
        random_state=42,
        return_train_score=True
    )

    search.fit(X, y)

    results = pd.DataFrame(search.cv_results_)
    results["rmse"] = -results["mean_test_score"]
    results = results.sort_values("rmse")

    print("✅ Meilleurs paramètres trouvés :")
    print(results[["params", "rmse"]].head(top_n))

    models = []
    weights = []

    for i in range(top_n):
        params = results.iloc[i]["params"]
        rmse = results.iloc[i]["rmse"]

        model = xgb.XGBRegressor(random_state=42+i, n_jobs=-1, **params)
        model.fit(X, y)

        models.append(model)
        weights.append(1 / rmse)  # poids inverse du RMSE

    return models, weights


# -----------------------------
# 2. Prédictions moyennées
# -----------------------------
def ensemble_predict(models, weights, X):
    preds = np.zeros((len(X), len(models)))
    for i, model in enumerate(models):
        preds[:, i] = model.predict(X)
    weights = np.array(weights) / np.sum(weights)
    return np.dot(preds, weights)

# -----------------------------------------------------
# 5. Main pipeline
# -----------------------------------------------------


if __name__ == "__main__":

  # Charger dataset complet
    df = pd.read_csv("weather_data_combined.csv")
    df = adapt_data_paul_GX(df)

    val = pd.read_csv("valmeteo.csv") 
    val = adapt_data_paul_GX(val)


    # Compare distributions train vs val
    cols = ["WAIT_TIME_IN_2H", "feels_like", "ADJUST_CAPACITY", "rain_1h"]
    for c in cols:
        plt.figure()
        sns.kdeplot(df[c], label="train")
        sns.kdeplot(val[c], label="val")
        plt.title(c)
        plt.legend()
        plt.show()

    # Relation WAIT_TIME vs HOUR
    sns.lineplot(data=df, x="HOUR", y="WAIT_TIME_IN_2H", label="train")
    sns.lineplot(data=val, x="HOUR", y="WAIT_TIME_IN_2H", label="val")
    plt.show()

