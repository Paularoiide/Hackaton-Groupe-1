import pandas as pd
from datetime import datetime
# en gros tu mets en 1 le fichier que tu veux modifier, en 2 le fichier reference
# il faut trouver la date column de chaque dot 

def ajuster_proportion_simple(csv1_path, csv2_path, date_column):
    """
    Version simplifiée pour ajuster les proportions
    """
    # Charger les données
    df1 = pd.read_csv(csv1_path)
    df2 = pd.read_csv(csv2_path)
    
    # Convertir les dates
    df1[date_column] = pd.to_datetime(df1[date_column])
    df2[date_column] = pd.to_datetime(df2[date_column])
    
    # Date de référence
    ref_date = datetime(2020, 3, 1)
    
    # Calculer les proportions
    prop1 = (df1[date_column] >= ref_date).mean()
    prop2 = (df2[date_column] >= ref_date).mean()
    
    print(f"Proportion CSV1: {prop1:.3f}")
    print(f"Proportion CSV2: {prop2:.3f}")
    
    # Créer une copie pour ne pas modifier l'original
    df1_modified = df1.copy()
    
    # Si CSV1 a plus de données récentes que CSV2, on en supprime
    if prop1 > prop2:
        recent_data = df1_modified[df1_modified[date_column] >= ref_date]
        to_remove = int(len(recent_data) * (prop1 - prop2) / prop1)
        
        # Sélection aléatoire des lignes à supprimer
        indices_to_remove = recent_data.sample(to_remove, random_state=42).index
        df1_modified = df1_modified.drop(indices_to_remove)
    
    # Calculer la nouvelle proportion
    new_prop = (df1_modified[date_column] >= ref_date).mean()
    print(f"Nouvelle proportion CSV1: {new_prop:.3f}")
    
    return df1_modified