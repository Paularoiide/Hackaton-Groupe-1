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


# Put the dataset into a pandas DataFrame
valsetsansmeteo = pd.read_table('waiting_times_X_test_val.csv', sep=',', decimal='.')
valsetmeteo = pd.read_table('valmeteo.csv', sep=',', decimal='.')
datasetmeteo = pd.read_table('weather_data_combined.csv', sep=',', decimal='.')
datasetsansmeteo = pd.read_table('waiting_times_train.csv', sep=',', decimal='.')

def adapter_dataset(dataset):
    #Remplir les missing values avec infini dans 'TIME_TO_PARADE_1','TIME_TO_PARADE_2','TIME_TO_NIGHT_SHOW'
    dataset['TIME_TO_PARADE_1'] = dataset['TIME_TO_PARADE_1'].fillna(10000)
    dataset['TIME_TO_PARADE_2'] = dataset['TIME_TO_PARADE_2'].fillna(10000)
    dataset['TIME_TO_NIGHT_SHOW'] = dataset['TIME_TO_NIGHT_SHOW'].fillna(10000)

    # Convert 'DATETIME' to datetime object and extract features
    dataset['DATETIME'] = pd.to_datetime(dataset['DATETIME'])
    dataset['DAY_OF_WEEK'] = dataset['DATETIME'].dt.dayofweek
    dataset['DAY'] = dataset['DATETIME'].dt.day
    dataset['MONTH'] = dataset['DATETIME'].dt.month
    dataset['YEAR'] = dataset['DATETIME'].dt.year
    dataset['HOUR'] = dataset['DATETIME'].dt.hour
    dataset['MINUTE'] = dataset['DATETIME'].dt.minute
    dataset['TIME_TO_PARADE_UNDER_2H'] = np.where((abs(dataset['TIME_TO_PARADE_1']) <= 120) | (abs(dataset['TIME_TO_PARADE_2']) <= 120), 1, 0)
    dataset['snow_1h'] = dataset['snow_1h'].fillna(0.05)


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

# Création du modèle
rf = RandomForestRegressor(
    n_estimators=200,   # nombre d’arbres
    max_depth=None,     # profondeur max (None = jusqu’aux feuilles)
    random_state=42,    # pour la reproductibilité
    n_jobs=-1           # utilise tous les cœurs CPU
)

# Séparation des sets

df = datasetmeteo
adapter_dataset(df)

# X = features, y = cible (ex: Temps_attente_dans_2h)
X = df.drop(columns=['WAIT_TIME_IN_2H', 'DATETIME', 'ENTITY_DESCRIPTION_SHORT'])
y = df['WAIT_TIME_IN_2H']

# Split en train (80%) et test (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

rf.fit(X_train, y_train)


# Modèle de base
rf = RandomForestRegressor(random_state=42, n_jobs=-1)

# Grille de paramètres à explorer
param_distributions = {
    'n_estimators': [100, 200, 500, 1000],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
}

# RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_distributions,
    n_iter=20,   # nombre de combinaisons testées (au hasard)
    cv=3,        # cross-validation en 3 folds
    scoring='neg_root_mean_squared_error',
    random_state=42,
    n_jobs=-1
)

# Entraînement
random_search.fit(X_train, y_train)

print("Meilleurs paramètres :", random_search.best_params_)
print("Meilleur score (CV RMSE):", -random_search.best_score_)

# Test du Modèle

# Prédictions
y_pred = rf.predict(X_test)

# RMSE
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
X_val = vf.drop(columns=['DATETIME', 'ENTITY_DESCRIPTION_SHORT'])

Y_val = rf.predict(X_val)
valsetmeteo['y_pred'] = Y_val

columns_valcsv = ['DATETIME','ENTITY_DESCRIPTION_SHORT','y_pred']
valcsv = valsetmeteo[columns_valcsv]
valcsv["KEY"] = "Validation"
valcsv.to_csv("mon_nouveau_dataset.csv", index=False)