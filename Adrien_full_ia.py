
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import time

def RMSE(x,y):
  return np.sqrt(mean_squared_error(x,y))

# Put the dataset into a pandas DataFrame
# Assurez-vous que ces fichiers CSV sont dans le même répertoire que ce script
datasetmeteo = pd.read_csv('weather_data_combined.csv', sep=',', decimal='.')
datasetsansmeteo = pd.read_csv('waiting_times_train.csv', sep=',', decimal='.')

def adapter_dataset(dataset):
    #Remplir les missing values avec infini dans 'TIME_TO_PARADE_1','TIME_TO_PARADE_2','TIME_TO_NIGHT_SHOW'
    dataset['TIME_TO_PARADE_1'] = dataset['TIME_TO_PARADE_1'].fillna(10000)
    dataset['TIME_TO_PARADE_2'] = dataset['TIME_TO_PARADE_2'].fillna(10000)
    dataset['TIME_TO_NIGHT_SHOW'] = dataset['TIME_TO_NIGHT_SHOW'].fillna(10000)
    if 'snow_1h' in dataset.columns:
        dataset['snow_1h'] = dataset['snow_1h'].fillna(0.05)

    # Convert 'DATETIME' to datetime object and extract features
    dataset['DATETIME'] = pd.to_datetime(dataset['DATETIME'])
    dataset['DAY_OF_WEEK'] = dataset['DATETIME'].dt.dayofweek
    dataset['DAY'] = dataset['DATETIME'].dt.day
    dataset['MONTH'] = dataset['DATETIME'].dt.month
    dataset['YEAR'] = dataset['DATETIME'].dt.year
    dataset['HOUR'] = dataset['DATETIME'].dt.hour
    dataset['MINUTE'] = dataset['DATETIME'].dt.minute
    dataset['TIME_TO_PARADE_UNDER_2H'] = np.where((abs(dataset['TIME_TO_PARADE_1']) <= 120) | (abs(dataset['TIME_TO_PARADE_2']) <= 120), 1, 0)
    return dataset

start_time = time.time()

# Appliquer la fonction adapter_dataset aux deux datasets
datasetmeteo = adapter_dataset(datasetmeteo)
datasetsansmeteo = adapter_dataset(datasetsansmeteo)

# Concaténer les datasets pour l'entraînement si nécessaire, ou choisir le dataset principal
# Pour cet exemple, nous allons utiliser datasetmeteo comme dataset principal pour l'entraînement
# Si datasetsansmeteo contient des données d'entraînement complémentaires, il faudrait les fusionner
# ou les concaténer après avoir appliqué adapter_dataset.
# Par simplicité, nous allons nous concentrer sur datasetmeteo pour la prédiction de WAIT_TIME_IN_2H.

# La colonne cible est 'WAIT_TIME_IN_2H'
# Les colonnes à exclure de l'entraînement sont 'DATETIME' et 'ENTITY_DESCRIPTION_SHORT'
# car elles sont soit transformées en features, soit des identifiants.

X = datasetmeteo.drop(columns=['DATETIME', 'ENTITY_DESCRIPTION_SHORT', 'WAIT_TIME_IN_2H'])
y = datasetmeteo['WAIT_TIME_IN_2H']

# Identification des colonnes numériques et catégorielles
# 'ENTITY_DESCRIPTION_SHORT' est catégorielle et sera encodée.
# Les colonnes de temps extraites ('DAY_OF_WEEK', 'DAY', 'MONTH', 'YEAR', 'HOUR', 'MINUTE') sont numériques.
# 'TIME_TO_PARADE_UNDER_2H' est également numérique.

numerical_cols = X.select_dtypes(include=np.number).columns.tolist()
categorical_cols = X.select_dtypes(include='object').columns.tolist()

# Création du préprocesseur
# Utilisation de OneHotEncoder pour les variables catégorielles
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

# Création du pipeline avec le préprocesseur et le modèle RandomForestRegressor
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))
])

# Définition de la grille de paramètres pour RandomizedSearchCV
# Ces paramètres peuvent être ajustés pour de meilleures performances
param_distributions = {
    'regressor__n_estimators': [100, 200, 300],
    'regressor__max_features': ['sqrt', 'log2'], # 'auto' est déprécié pour RandomForestRegressor
    'regressor__max_depth': [10, 20, None],
    'regressor__min_samples_split': [2, 5],
    'regressor__min_samples_leaf': [1, 2]
}

# Division des données en ensembles d'entraînement et de test
# Un test_size de 0.2 (20%) est un bon point de départ
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialisation de RandomizedSearchCV
# n_iter: nombre d'itérations pour la recherche aléatoire. Plus il est élevé, plus la recherche est approfondie.
# cv: nombre de plis pour la validation croisée.
random_search = RandomizedSearchCV(pipeline, param_distributions, n_iter=5, cv=3, verbose=1, random_state=42, n_jobs=-1)

# Entraînement du modèle
print("Début de l'entraînement du modèle...")
random_search.fit(X_train, y_train)
print("Entraînement terminé.")

# Utiliser le meilleur modèle trouvé par RandomizedSearchCV
best_rf = random_search.best_estimator_

# Évaluation du modèle sur l'ensemble de test
y_pred = best_rf.predict(X_test)
rmse = RMSE(y_test, y_pred)
print(f"RMSE Random Forest sur l'ensemble de test: {rmse:.4f}")

# Prédiction sur un ensemble de validation ou de nouvelles données
# Pour cet exemple, nous allons simuler un ensemble de validation en prenant une partie du dataset original.
# En production, 'valsetmeteo' serait un nouveau fichier CSV à prédire.
valsetmeteo = datasetmeteo.sample(frac=0.1, random_state=42).copy() # Utiliser .copy() pour éviter SettingWithCopyWarning

# Préparer les données de validation de la même manière que les données d'entraînement
X_val = valsetmeteo.drop(columns=['DATETIME', 'ENTITY_DESCRIPTION_SHORT', 'WAIT_TIME_IN_2H'])

# Effectuer les prédictions
valsetmeteo['y_pred'] = best_rf.predict(X_val)

# Sauvegarde du fichier CSV de sortie avec la structure spécifiée
# DATETIME, ENTITY_DESCRIPTION_SHORT, y_pred, KEY
valcsv = valsetmeteo[['DATETIME', 'ENTITY_DESCRIPTION_SHORT', 'y_pred']]
valcsv["KEY"] = "Validation" # La clé "Validation" est ajoutée comme demandé

# Convertir 'DATETIME' en format string pour la sauvegarde si nécessaire, ou laisser tel quel
valcsv['DATETIME'] = valcsv['DATETIME'].dt.strftime('%Y-%m-%d %H:%M:%S')

valcsv.to_csv("mon_nouveau_dataset.csv", index=False)

print("Fichier 'mon_nouveau_dataset.csv' généré avec succès.")
print(f"Temps d'exécution total: {time.time() - start_time:.2f} secondes")


