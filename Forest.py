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


def adapter_dataset(dataset):
    # Remplir valeurs manquantes
    dataset['TIME_TO_PARADE_1'] = dataset['TIME_TO_PARADE_1'].fillna(10000)
    dataset['TIME_TO_PARADE_2'] = dataset['TIME_TO_PARADE_2'].fillna(10000)
    dataset['TIME_TO_NIGHT_SHOW'] = dataset['TIME_TO_NIGHT_SHOW'].fillna(10000)
    dataset['snow_1h'] = dataset['snow_1h'].fillna(0)

    # Conversion datetime
    dataset['DATETIME'] = pd.to_datetime(dataset['DATETIME'])
    dataset['DAY_OF_WEEK'] = dataset['DATETIME'].dt.dayofweek
    dataset['DAY'] = dataset['DATETIME'].dt.day
    dataset['MONTH'] = dataset['DATETIME'].dt.month
    dataset['YEAR'] = dataset['DATETIME'].dt.year
    dataset['HOUR'] = dataset['DATETIME'].dt.hour
    dataset['MINUTE'] = dataset['DATETIME'].dt.minute

    # Parade proche (< 2h) et temps avant prochaine
    dataset['TIME_TO_PARADE_UNDER_2H'] = np.where(
        (abs(dataset['TIME_TO_PARADE_1']) <= 500) | (abs(dataset['TIME_TO_PARADE_2']) <= 500),
        1, 0
    )

    # Encodage cyclique de l'heure
    dataset['HOUR_SIN'] = np.sin(2 * np.pi * dataset['HOUR'] / 24)
    dataset['HOUR_COS'] = np.cos(2 * np.pi * dataset['HOUR'] / 24)

    # Binarisation des attractions
    
    dataset['IS_ATTRACTION_Water_Ride'] = np.where(dataset['ENTITY_DESCRIPTION_SHORT'] == "Water Ride", 1, 0)
    dataset['IS_ATTRACTION_Pirate_Ship'] = np.where(dataset['ENTITY_DESCRIPTION_SHORT'] == "Pirate Ship", 1, 0)
    dataset['IS_ATTRACTION__Flying_Coaster'] = np.where(dataset['ENTITY_DESCRIPTION_SHORT'] == "Flying Coaster", 1, 0)
    

    
# -----------------------------------------------------
# 2. Séparation pré- / post-COVID
# -----------------------------------------------------
def split_pre_post(df, covid_date="2020-08-15"):
    df_pre = df[df['DATETIME'] < covid_date].copy()
    df_post = df[df['DATETIME'] >= covid_date].copy()
    return df_pre, df_post

# -----------------------------------------------------
# 3. Entraînement des deux modèles
# -----------------------------------------------------
def train_two_models(df_pre, df_post, target="WAIT_TIME_IN_2H"):
    features = [col for col in df_pre.columns if col not in [target, 'DATETIME', 'ENTITY_DESCRIPTION_SHORT']]

    X_pre, y_pre = df_pre[features], df_pre[target]
    X_post, y_post = df_post[features], df_post[target]

    rf_pre = RandomForestRegressor(
        n_estimators=1000,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='log2',
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )
    rf_post = RandomForestRegressor(
        n_estimators=1000,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='log2',
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )

    rf_pre.fit(X_pre, y_pre)
    rf_post.fit(X_post, y_post)

    return rf_pre, rf_post, features

# ------------------------
# Entraînement par attraction
# ------------------------
def train_by_attraction(df, target="WAIT_TIME_IN_2H"):
    models = {}
    features = [c for c in df.columns if c not in [target, "DATETIME","ENTITY_DESCRIPTION_SHORT"]]

    for attraction in df["ENTITY_DESCRIPTION_SHORT"].unique():
        print(f"\n--- Entraînement modèle pour {attraction} ---")
        df_attr = df[df["ENTITY_DESCRIPTION_SHORT"] == attraction]

        X, y = df_attr[features], df_attr[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        rf = RandomForestRegressor(
            n_estimators=500,
            max_depth=20,
            min_samples_leaf=2,
            max_features="log2",
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X_train, y_train)

        y_pred = rf.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        print(f"RMSE {attraction} :", rmse)

        models[attraction] = rf

        # Importance des features
        feat_importances = pd.DataFrame({
            "Feature": features,
            "Importance": rf.feature_importances_
        }).sort_values("Importance", ascending=False)

        print(f"\nTop 10 features pour {attraction}:\n", feat_importances.head(10))

        plt.figure(figsize=(8, 6))
        plt.barh(feat_importances["Feature"][:10][::-1], feat_importances["Importance"][:10][::-1])
        plt.title(f"Top 10 Features - {attraction}")
        plt.xlabel("Importance")
        plt.tight_layout()
        plt.show()

    return models, features

# ------------------------
# Fonction d’entraînement 6 modèles
# ------------------------
def train_by_attr_and_covid(df, target="WAIT_TIME_IN_2H", covid_date="2020-03-15"):
    models = {}
    features = [c for c in df.columns if c not in [target, "DATETIME", "ENTITY_DESCRIPTION_SHORT"]]

    for attraction in df["ENTITY_DESCRIPTION_SHORT"].unique():
        df_attr = df[df["ENTITY_DESCRIPTION_SHORT"] == attraction]

        # Split pré/post COVID
        df_pre = df_attr[df_attr["DATETIME"] < covid_date]
        df_post = df_attr[df_attr["DATETIME"] >= covid_date]

        for subset_name, subset_df in [("pre", df_pre), ("post", df_post)]:
            if len(subset_df) < 50:  # sécurité si trop peu de données
                print(f"⚠️ Pas assez de données pour {attraction} ({subset_name}-COVID)")
                continue

            print(f"\n--- Entraînement modèle {attraction} ({subset_name}-COVID) ---")
            X, y = subset_df[features], subset_df[target]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            rf = RandomForestRegressor(
                n_estimators=500,
                max_depth=20,
                min_samples_leaf=2,
                max_features="log2",
                random_state=42,
                n_jobs=-1
            )
            rf.fit(X_train, y_train)

            y_pred = rf.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            print(f"RMSE {attraction} ({subset_name}-COVID):", rmse)

            # Sauvegarde du modèle
            models[(attraction, subset_name)] = rf

            # Importance des features
            feat_importances = pd.DataFrame({
                "Feature": features,
                "Importance": rf.feature_importances_
            }).sort_values("Importance", ascending=False)

            print(f"\nTop 10 features {attraction} ({subset_name}-COVID):\n", feat_importances.head(10))

            plt.figure(figsize=(8, 6))
            plt.barh(feat_importances["Feature"][:10][::-1], feat_importances["Importance"][:10][::-1])
            plt.title(f"Top 10 Features - {attraction} ({subset_name}-COVID)")
            plt.xlabel("Importance")
            plt.tight_layout()
            plt.show()

    return models, features

# -----------------------------------------------------
# 4. Prédiction automatique (choisit le bon modèle)
# -----------------------------------------------------
def predict_two_models(rf_pre, rf_post, features, df, covid_date="2020-03-15"):
    df = df.copy()
    df['DATETIME'] = pd.to_datetime(df['DATETIME'])

    mask_pre = df['DATETIME'] < covid_date
    mask_post = df['DATETIME'] >= covid_date

    preds = np.zeros(len(df))

    if mask_pre.any():
        preds[mask_pre] = rf_pre.predict(df.loc[mask_pre, features])
    if mask_post.any():
        preds[mask_post] = rf_post.predict(df.loc[mask_post, features])

    return preds

# ------------------------
# Prédiction avec modèle par attraction
# ------------------------
def predict_by_attraction(models, features, df):
    preds = np.zeros(len(df))

    for attraction in df["ENTITY_DESCRIPTION_SHORT"].unique():
        mask = df["ENTITY_DESCRIPTION_SHORT"] == attraction
        if mask.any():
            preds[mask] = models[attraction].predict(df.loc[mask, features])
    return preds

# ------------------------
# Prédiction avec les 6 modèles
# ------------------------
def predict_by_attr_and_covid(models, features, df, covid_date="2020-03-15"):
    preds = np.zeros(len(df))
    df = df.copy()
    df['DATETIME'] = pd.to_datetime(df['DATETIME'])

    for attraction in df["ENTITY_DESCRIPTION_SHORT"].unique():
        mask_attr = df["ENTITY_DESCRIPTION_SHORT"] == attraction

        # Split pré/post
        mask_pre = (df["DATETIME"] < covid_date) & mask_attr
        mask_post = (df["DATETIME"] >= covid_date) & mask_attr

        if mask_pre.any() and (attraction, "pre") in models:
            preds[mask_pre] = models[(attraction, "pre")].predict(df.loc[mask_pre, features])
        if mask_post.any() and (attraction, "post") in models:
            preds[mask_post] = models[(attraction, "post")].predict(df.loc[mask_post, features])

    return preds

# -----------------------------------------------------
# 5. Affichage des features
# -----------------------------------------------------

# --- entraînement + importance des features
def train_and_feature_importance(df, target="WAIT_TIME_IN_2H", title=""):
    features = [c for c in df.columns if c not in [target, "DATETIME", "ENTITY_DESCRIPTION_SHORT"]]
    X, y = df[features], df[target]

    # split train/test interne
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf = RandomForestRegressor(
        n_estimators=1000,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='log2',
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"RMSE {title} :", rmse)

    # importance
    importances = rf.feature_importances_
    feat_importances = pd.DataFrame({
        "Feature": features,
        "Importance": importances
    }).sort_values("Importance", ascending=False)

    print(f"\nTop 15 features {title}:\n", feat_importances.head(15))

    # graphique
    plt.figure(figsize=(8, 6))
    plt.barh(feat_importances["Feature"][:15][::-1], feat_importances["Importance"][:15][::-1])
    plt.title(f"Top 15 Features - {title}")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.show()

    return rf, features, rmse, feat_importances


# ------------------------
# Application
# ------------------------
# Train

"""
# Train
df = pd.read_csv("weather_data_combined.csv")
adapter_dataset(df)

models_6, features = train_by_attr_and_covid(df)

# Validation externe
val = pd.read_csv("valmeteo.csv")
adapter_dataset(val)

y_val_pred = predict_by_attr_and_covid(models_6, features, val)

# Export CSV
val['y_pred'] = y_val_pred
val[['DATETIME','ENTITY_DESCRIPTION_SHORT','y_pred']].assign(KEY="Validation").to_csv("val_predictions.csv", index=False)
"""
"""
df = pd.read_csv("weather_data_combined.csv")
adapter_dataset(df)

models_by_attr, features = train_by_attraction(df)

# Validation externe
val = pd.read_csv("valmeteo.csv")
adapter_dataset(val)

y_val_pred = predict_by_attraction(models_by_attr, features, val)

# Export CSV
val['y_pred'] = y_val_pred
val[['DATETIME','ENTITY_DESCRIPTION_SHORT','y_pred']].assign(KEY="Validation").to_csv("val_predictions.csv", index=False)
"""

# Charger et préparer
df = pd.read_csv("weather_data_combined.csv")
adapter_dataset(df)


# Split pre/post covid
df_pre, df_post = split_pre_post(df)


#Affichage des features importante et rmse
rf_pre, features_pre, rmse_pre, feat_pre = train_and_feature_importance(df_pre, title="Pré-COVID")
rf_post, features_post, rmse_post, feat_post = train_and_feature_importance(df_post, title="Post-COVID")
# Entraînement
rf_pre, rf_post, features = train_two_models(df_pre, df_post)

# Test sur validation externe
val = pd.read_csv("valmeteo.csv")
adapter_dataset(val)

y_val_pred = predict_two_models(rf_pre, rf_post, features, val)

# Ajouter dans val + exporter
val['y_pred'] = y_val_pred
val[['DATETIME','ENTITY_DESCRIPTION_SHORT','y_pred']].assign(KEY="Validation").to_csv("val_predictions.csv", index=False)



























































































































































"""
# Création du modèle
rf = RandomForestRegressor(
    n_estimators=1000,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='log2',
    max_depth=None,
    random_state=42,
    n_jobs=-1
)

# Séparation des sets

df = datasetmeteo
adapter_dataset(df)

# X = features, y = cible (ex: Temps_attente_dans_2h)
X = df.drop(columns=['WAIT_TIME_IN_2H', 'DATETIME', 'ENTITY_DESCRIPTION_SHORT']+['HOUR', 'DATETIME', 'ENTITY_DESCRIPTION_SHORT','TIME_TO_PARADE_UNDER_2H']) # Predicteurs à enlever])
y = df['WAIT_TIME_IN_2H']

# Split en train (70%) et test (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

rf.fit(X_train, y_train)

           
# Modèle de base
rf = RandomForestRegressor(random_state=42, n_jobs=-1)

# Grille de paramètres à explorer
param_distributions = {
    'n_estimators': [200, 400],   # évite 1000 pour le tuning
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt']
}


# RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_distributions,
    n_iter=10,   # nombre de combinaisons testées (au hasard)
    cv=3,        # cross-validation en 3 folds
    scoring='neg_root_mean_squared_error',
    random_state=42,
    n_jobs=-1,
)

# Entraînement
random_search.fit(X_train, y_train)


print("Meilleurs paramètres :", random_search.best_params_)
print("Meilleur score (CV RMSE):", -random_search.best_score_)


# Test du Modèle

# Prédictions
best_rf= random_search.best_estimator_

# RMSE

y_pred = rf.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("RMSE Random Forest:", rmse)

import matplotlib.pyplot as plt
import pandas as pd

# Supposons que rf soit ton RandomForestRegressor déjà entraîné
importances = rf.feature_importances_
features = X_train.columns

# Mettre dans un DataFrame pour plus de clarté
feat_importances = pd.DataFrame({
    'Feature': features,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print(feat_importances)

# Visualisation
plt.figure(figsize=(10,6))
plt.barh(feat_importances['Feature'], feat_importances['Importance'])
plt.gca().invert_yaxis()  # feature la plus importante en haut
plt.title("Importance des variables (Random Forest)")
plt.show()

vf = valsetmeteo
adapter_dataset(vf)
X_val = vf.drop(columns=['DATETIME', 'ENTITY_DESCRIPTION_SHORT'] +['HOUR', 'DATETIME', 'ENTITY_DESCRIPTION_SHORT','TIME_TO_PARADE_UNDER_2H'])

Y_val = rf.predict(X_val)

# Copier pour éviter de modifier vf directement
val_results = valsetmeteo.copy()
val_results['y_pred'] = Y_val

# Génération du CSV
columns_valcsv = ['DATETIME', 'ENTITY_DESCRIPTION_SHORT', 'y_pred']
valcsv = val_results[columns_valcsv]
valcsv["KEY"] = "Validation"
valcsv.to_csv("mon_nouveau_dataset.csv", index=False)
"""