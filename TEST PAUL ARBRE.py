import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error
import time

# Chronométrer le temps d'exécution
start_time = time.time()

# Chargement des données
def load_and_preprocess_data():
    # Charger tous les datasets en une fois
    datasetmeteo = pd.read_csv('weather_data_combined.csv')
    valsetmeteo = pd.read_csv('valmeteo.csv')
    
    # Préprocessing optimisé
    def preprocess_data(df):
        df = df.copy()
        # Remplir les NaN une seule fois
        time_cols = ['TIME_TO_PARADE_1', 'TIME_TO_PARADE_2', 'TIME_TO_NIGHT_SHOW']
        df[time_cols] = df[time_cols].fillna(10000)
        df['snow_1h'] = df['snow_1h'].fillna(0.05)
        
        # Conversion datetime avec extraction directe
        df['DATETIME'] = pd.to_datetime(df['DATETIME'])
        df['DAY_OF_WEEK'] = df['DATETIME'].dt.dayofweek
        df['DAY'] = df['DATETIME'].dt.day
        df['MONTH'] = df['DATETIME'].dt.month
        df['YEAR'] = df['DATETIME'].dt.year
        df['HOUR'] = df['DATETIME'].dt.hour
        df['MINUTE'] = df['DATETIME'].dt.minute
        
        df['TIME_TO_PARADE_UNDER_2H'] = ((df['TIME_TO_PARADE_1'].abs() <= 120) | 
                                        (df['TIME_TO_PARADE_2'].abs() <= 120)).astype(int)
        return df
    
    datasetmeteo = preprocess_data(datasetmeteo)
    valsetmeteo = preprocess_data(valsetmeteo)
    
    return datasetmeteo, valsetmeteo

# Chargement et préprocessing
datasetmeteo, valsetmeteo = load_and_preprocess_data()

# Préparation des features
X = datasetmeteo.drop(columns=['WAIT_TIME_IN_2H', 'DATETIME', 'ENTITY_DESCRIPTION_SHORT'])
y = datasetmeteo['WAIT_TIME_IN_2H']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Recherche d'hyperparamètres + entraînement en une étape
param_distributions = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt', 'log2']
}

rf = RandomForestRegressor(random_state=42, n_jobs=-1)
random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_distributions,
    n_iter=10,  # Réduit pour gagner du temps
    cv=3,
    scoring='neg_root_mean_squared_error',
    random_state=42,
    n_jobs=-1
)

# Un seul entraînement
random_search.fit(X_train, y_train)

# Utiliser le meilleur modèle
best_rf = random_search.best_estimator_

# Évaluation
y_pred = best_rf.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"RMSE Random Forest: {rmse:.4f}")

# Prédiction sur validation set
X_val = valsetmeteo.drop(columns=['DATETIME', 'ENTITY_DESCRIPTION_SHORT'])
valsetmeteo['y_pred'] = best_rf.predict(X_val)

# Sauvegarde
valcsv = valsetmeteo[['DATETIME', 'ENTITY_DESCRIPTION_SHORT', 'y_pred']]
valcsv["KEY"] = "Validation"
valcsv.to_csv("mon_nouveau_dataset.csv", index=False)

print(f"Temps d'exécution total: {time.time() - start_time:.2f} secondes")