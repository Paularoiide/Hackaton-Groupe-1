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

"""
# Put the dataset into a pandas DataFrame
valsetsansmeteo = pd.read_table('waiting_times_X_test_val.csv', sep=',', decimal='.')
valsetmeteo = pd.read_table('valmeteo.csv', sep=',', decimal='.')
datasetmeteo = pd.read_table('weather_data_combined.csv', sep=',', decimal='.')
datasetsansmeteo = pd.read_table('waiting_times_train.csv', sep=',', decimal='.')
"""

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

    
# -----------------------------------------------------
# 2. Séparation pré- / post-COVID
# -----------------------------------------------------
def split_pre_post(df, covid_date="2020-03-15"):
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
        n_estimators=500, max_depth=20, min_samples_leaf=2,
        max_features="log2", random_state=42, n_jobs=-1
    )
    rf_post = RandomForestRegressor(
        n_estimators=500, max_depth=20, min_samples_leaf=2,
        max_features="log2", random_state=42, n_jobs=-1
    )

    rf_pre.fit(X_pre, y_pre)
    rf_post.fit(X_post, y_post)

    return rf_pre, rf_post, features

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

# Charger et préparer
df = pd.read_csv("weather_data_combined.csv")
adapter_dataset(df)

# Split pre/post covid
df_pre, df_post = split_pre_post(df)

# Entraînement
rf_pre, rf_post, features = train_two_models(df_pre, df_post)

# Test sur validation externe
val = pd.read_csv("valmeteo.csv")
adapter_dataset(val)

y_val_pred = predict_two_models(rf_pre, rf_post, features, val)

# Ajouter dans val + exporter
val['y_pred'] = y_val_pred
val[['DATETIME','ENTITY_DESCRIPTION_SHORT','y_pred']].assign(KEY="Validation").to_csv("val_predictions.csv", index=False)

rmse = np.sqrt(mean_squared_error(y_test, y_val_pred))
print("RMSE Random Forest:", rmse)

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