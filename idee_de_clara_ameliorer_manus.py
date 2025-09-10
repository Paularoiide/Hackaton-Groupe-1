import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from datetime import datetime

# Définition globale des groupes de parade
parade_groups = {
    "groupe_1": {"parade1": True, "parade2": True, "night_show": True},
    "groupe_2": {"parade1": True, "parade2": True, "night_show": False},
    "groupe_3": {"parade1": True, "parade2": False, "night_show": True},
    "groupe_4": {"parade1": True, "parade2": False, "night_show": False},
    "groupe_5": {"parade1": False, "parade2": True, "night_show": True},
    "groupe_6": {"parade1": False, "parade2": True, "night_show": False},
    "groupe_7": {"parade1": False, "parade2": False, "night_show": True},
    "groupe_8": {"parade1": False, "parade2": False, "night_show": False}
}

def adapter_dataset_8_groupes(dataset):
    dataset = dataset.copy()
    dataset["DATETIME"] = pd.to_datetime(dataset["DATETIME"])

    def detecter_vacances_par_zone(date):
        vacances_zones = {
            "ZONE_A": [
                (datetime(2018, 10, 20), datetime(2018, 11, 4)), (datetime(2018, 12, 22), datetime(2019, 1, 6)),
                (datetime(2019, 2, 16), datetime(2019, 3, 3)), (datetime(2019, 4, 13), datetime(2019, 4, 28)),
                (datetime(2019, 7, 6), datetime(2019, 9, 1)),
                (datetime(2019, 10, 19), datetime(2019, 11, 3)), (datetime(2019, 12, 21), datetime(2020, 1, 5)),
                (datetime(2020, 2, 8), datetime(2020, 2, 23)), (datetime(2020, 4, 4), datetime(2020, 4, 19)),
                (datetime(2020, 7, 4), datetime(2020, 9, 1)),
                (datetime(2020, 10, 17), datetime(2020, 11, 1)), (datetime(2020, 12, 19), datetime(2021, 1, 3)),
                (datetime(2021, 2, 6), datetime(2021, 2, 21)), (datetime(2021, 4, 10), datetime(2021, 4, 25)),
                (datetime(2021, 7, 6), datetime(2021, 9, 1)),
                (datetime(2021, 10, 23), datetime(2021, 11, 7)), (datetime(2021, 12, 18), datetime(2022, 1, 2)),
                (datetime(2022, 2, 12), datetime(2022, 2, 27)), (datetime(2022, 4, 16), datetime(2022, 5, 1)),
                (datetime(2022, 7, 7), datetime(2022, 9, 1)),
            ],
            "ZONE_B": [
                (datetime(2018, 10, 20), datetime(2018, 11, 4)), (datetime(2018, 12, 22), datetime(2019, 1, 6)),
                (datetime(2019, 2, 9), datetime(2019, 2, 24)), (datetime(2019, 4, 6), datetime(2019, 4, 21)),
                (datetime(2019, 7, 6), datetime(2019, 9, 1)),
                (datetime(2019, 10, 19), datetime(2019, 11, 3)), (datetime(2019, 12, 21), datetime(2020, 1, 5)),
                (datetime(2020, 2, 22), datetime(2020, 3, 8)), (datetime(2020, 4, 4), datetime(2020, 4, 19)),
                (datetime(2020, 7, 4), datetime(2020, 9, 1)),
                (datetime(2020, 10, 17), datetime(2020, 11, 1)), (datetime(2020, 12, 19), datetime(2021, 1, 3)),
                (datetime(2021, 2, 20), datetime(2021, 3, 7)), (datetime(2021, 4, 10), datetime(2021, 4, 25)),
                (datetime(2021, 7, 6), datetime(2021, 9, 1)),
                (datetime(2021, 10, 23), datetime(2021, 11, 7)), (datetime(2021, 12, 18), datetime(2022, 1, 2)),
                (datetime(2022, 2, 26), datetime(2022, 3, 13)), (datetime(2022, 4, 16), datetime(2022, 5, 1)),
                (datetime(2022, 7, 7), datetime(2022, 9, 1)),
            ],
            "ZONE_C": [
                (datetime(2018, 10, 20), datetime(2018, 11, 4)), (datetime(2018, 12, 22), datetime(2019, 1, 6)),
                (datetime(2019, 2, 23), datetime(2019, 3, 10)), (datetime(2019, 4, 20), datetime(2019, 5, 5)),
                (datetime(2019, 7, 6), datetime(2019, 9, 1)),
                (datetime(2019, 10, 19), datetime(2019, 11, 3)), (datetime(2019, 12, 21), datetime(2020, 1, 5)),
                (datetime(2020, 2, 15), datetime(2020, 3, 1)), (datetime(2020, 4, 18), datetime(2020, 5, 3)),
                (datetime(2020, 7, 4), datetime(2020, 9, 1)),
                (datetime(2020, 10, 17), datetime(2020, 11, 1)), (datetime(2020, 12, 19), datetime(2021, 1, 3)),
                (datetime(2021, 2, 13), datetime(2021, 2, 28)), (datetime(2021, 4, 24), datetime(2021, 5, 9)),
                (datetime(2021, 7, 6), datetime(2021, 9, 1)),
                (datetime(2021, 10, 23), datetime(2021, 11, 7)), (datetime(2021, 12, 18), datetime(2022, 1, 2)),
                (datetime(2022, 2, 12), datetime(2022, 2, 27)), (datetime(2022, 4, 23), datetime(2022, 5, 8)),
                (datetime(2022, 7, 7), datetime(2022, 9, 1)),
            ]
        }

        result = {"VACANCES_ZONE_A": 0, "VACANCES_ZONE_B": 0, "VACANCES_ZONE_C": 0}
        for zone, periodes in vacances_zones.items():
            for debut, fin in periodes:
                if debut <= date <= fin:
                    if zone == "ZONE_A":
                        result["VACANCES_ZONE_A"] = 1
                    elif zone == "ZONE_B":
                        result["VACANCES_ZONE_B"] = 1
                    elif zone == "ZONE_C":
                        result["VACANCES_ZONE_C"] = 1
        return result

    vacances_data = dataset["DATETIME"].apply(detecter_vacances_par_zone)
    dataset["VACANCES_ZONE_A"] = vacances_data.apply(lambda x: x["VACANCES_ZONE_A"])
    dataset["VACANCES_ZONE_B"] = vacances_data.apply(lambda x: x["VACANCES_ZONE_B"])
    dataset["VACANCES_ZONE_C"] = vacances_data.apply(lambda x: x["VACANCES_ZONE_C"])

    mask_parade1 = ~dataset["TIME_TO_PARADE_1"].isna()
    mask_parade2 = ~dataset["TIME_TO_PARADE_2"].isna()
    mask_night_show = ~dataset["TIME_TO_NIGHT_SHOW"].isna()

    groupes = {}
    groupes["groupe_1"] = dataset[mask_parade1 & mask_parade2 & mask_night_show].copy()
    groupes["groupe_2"] = dataset[mask_parade1 & mask_parade2 & ~mask_night_show].copy()
    groupes["groupe_3"] = dataset[mask_parade1 & ~mask_parade2 & mask_night_show].copy()
    groupes["groupe_4"] = dataset[mask_parade1 & ~mask_parade2 & ~mask_night_show].copy()
    groupes["groupe_5"] = dataset[~mask_parade1 & mask_parade2 & mask_night_show].copy()
    groupes["groupe_6"] = dataset[~mask_parade1 & mask_parade2 & ~mask_night_show].copy()
    groupes["groupe_7"] = dataset[~mask_parade1 & ~mask_parade2 & mask_night_show].copy()
    groupes["groupe_8"] = dataset[~mask_parade1 & ~mask_parade2 & ~mask_night_show].copy()

    for groupe_name, groupe_data in groupes.items():
        if not groupe_data.empty:
            # Assurez-vous que la colonne cible est présente dans le DataFrame du groupe
            if "WAIT_TIME_IN_2H" not in groupe_data.columns and "WAIT_TIME_IN_2H" in dataset.columns:
                groupe_data["WAIT_TIME_IN_2H"] = dataset.loc[groupe_data.index, "WAIT_TIME_IN_2H"]

            groupe_data["snow_1h"] = groupe_data["snow_1h"].fillna(0)
            groupe_data["DAY_OF_WEEK"] = groupe_data["DATETIME"].dt.dayofweek
            groupe_data["DAY"] = groupe_data["DATETIME"].dt.day
            groupe_data["MONTH"] = groupe_data["DATETIME"].dt.month
            groupe_data["YEAR"] = groupe_data["DATETIME"].dt.year
            groupe_data["HOUR"] = groupe_data["DATETIME"].dt.hour
            groupe_data["MINUTE"] = groupe_data["DATETIME"].dt.minute
            groupe_data["HOUR_SIN"] = np.sin(2 * np.pi * groupe_data["HOUR"] / 24)
            groupe_data["HOUR_COS"] = np.cos(2 * np.pi * groupe_data["HOUR"] / 24)
            groupe_data["IS_ATTRACTION_Water_Ride"] = np.where(groupe_data["ENTITY_DESCRIPTION_SHORT"] == "Water Ride", 1, 0)
            groupe_data["IS_ATTRACTION_Pirate_Ship"] = np.where(groupe_data["ENTITY_DESCRIPTION_SHORT"] == "Pirate Ship", 1, 0)
            groupe_data["IS_ATTRACTION__Flying_Coaster"] = np.where(groupe_data["ENTITY_DESCRIPTION_SHORT"] == "Flying Coaster", 1, 0)

            groupe_data["TIME_TO_PARADE_UNDER_2H"] = 0
            if "TIME_TO_PARADE_1" in groupe_data.columns:
                mask_parade1_close = groupe_data["TIME_TO_PARADE_1"].notna() & (abs(groupe_data["TIME_TO_PARADE_1"]) <= 500)
                groupe_data.loc[mask_parade1_close, "TIME_TO_PARADE_UNDER_2H"] = 1
            if "TIME_TO_PARADE_2" in groupe_data.columns:
                mask_parade2_close = groupe_data["TIME_TO_PARADE_2"].notna() & (abs(groupe_data["TIME_TO_PARADE_2"]) <= 500)
                groupe_data.loc[mask_parade2_close, "TIME_TO_PARADE_UNDER_2H"] = 1

            # Nouvelle caractéristique: interaction entre l"heure et la capacité
            groupe_data["HOUR_ADJUST_CAPACITY"] = groupe_data["HOUR"] * groupe_data["ADJUST_CAPACITY"]
            # Nouvelle caractéristique: interaction entre le temps d"attente actuel et la température
            groupe_data["CURRENT_WAIT_TEMP"] = groupe_data["CURRENT_WAIT_TIME"] * groupe_data["temp"]
            # Nouvelle caractéristique: Jours de la semaine en vacances
            groupe_data["WEEKDAY_IN_VACATION"] = ((groupe_data["DAY_OF_WEEK"] >= 0) & (groupe_data["DAY_OF_WEEK"] <= 4)) & \
                                                ((groupe_data["VACANCES_ZONE_A"] == 1) | \
                                                 (groupe_data["VACANCES_ZONE_B"] == 1) | \
                                                 (groupe_data["VACANCES_ZONE_C"] == 1))
            groupe_data["WEEKDAY_IN_VACATION"] = groupe_data["WEEKDAY_IN_VACATION"].astype(int)

    return groupes


def train_by_attr_and_parade_groups(df, target="WAIT_TIME_IN_2H"):
    models = {}
    features = [c for c in df.columns if c not in [target, "DATETIME", "ENTITY_DESCRIPTION_SHORT",
                                                  "TIME_TO_PARADE_1", "TIME_TO_PARADE_2", "TIME_TO_NIGHT_SHOW"]]

    for attraction in df["ENTITY_DESCRIPTION_SHORT"].unique():
        df_attr = df[df["ENTITY_DESCRIPTION_SHORT"] == attraction]

        for group_name, conditions in parade_groups.items():
            mask = pd.Series(True, index=df_attr.index)

            if conditions["parade1"]:
                mask = mask & df_attr["TIME_TO_PARADE_1"].notna()
            else:
                mask = mask & df_attr["TIME_TO_PARADE_1"].isna()

            if conditions["parade2"]:
                mask = mask & df_attr["TIME_TO_PARADE_2"].notna()
            else:
                mask = mask & df_attr["TIME_TO_PARADE_2"].isna()

            if conditions["night_show"]:
                mask = mask & df_attr["TIME_TO_NIGHT_SHOW"].notna()
            else:
                mask = mask & df_attr["TIME_TO_NIGHT_SHOW"].isna()

            df_group = df_attr[mask]

            if len(df_group) < 30:
                print(f"⚠️ Pas assez de données pour {attraction} ({group_name}): {len(df_group)} lignes")
                continue

            print(f"\n--- Entraînement modèle {attraction} ({group_name}) ---")
            print(f"Taille du dataset: {len(df_group)} lignes")

            X, y = df_group[features], df_group[target]

            if len(df_group) < 100:
                X_train, y_train = X, y
                X_test, y_test = X, y
            else:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            rf = RandomForestRegressor(
                n_estimators=300,
                max_depth=20,
                min_samples_leaf=2,
                max_features="sqrt",
                random_state=42,
                n_jobs=-1
            )
            rf.fit(X_train, y_train)

            y_pred = rf.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            print(f"RMSE {attraction} ({group_name}): {rmse:.2f}")
            print(f"MAE {attraction} ({group_name}): {mae:.2f}")

            models[(attraction, group_name)] = rf

            if len(df_group) > 50:
                feat_importances = pd.DataFrame({
                    "Feature": features,
                    "Importance": rf.feature_importances_
                }).sort_values("Importance", ascending=False)

                print(f"\nTop 5 features {attraction} ({group_name}):\n", feat_importances.head(5))

    return models, features


def predict_by_attr_and_parade_groups(models, features, df):
    preds = np.zeros(len(df))
    df = df.copy()

    for attraction in df["ENTITY_DESCRIPTION_SHORT"].unique():
        mask_attr = df["ENTITY_DESCRIPTION_SHORT"] == attraction

        for group_name, conditions in parade_groups.items():

            if (attraction, group_name) not in models:
                continue

            mask_group = pd.Series(True, index=df.index)
            mask_group = mask_group & mask_attr

            if conditions["parade1"]:
                mask_group = mask_group & df["TIME_TO_PARADE_1"].notna()
            else:
                mask_group = mask_group & df["TIME_TO_PARADE_1"].isna()

            if conditions["parade2"]:
                mask_group = mask_group & df["TIME_TO_PARADE_2"].notna()
            else:
                mask_group = mask_group & df["TIME_TO_PARADE_2"].isna()

            if conditions["night_show"]:
                mask_group = mask_group & df["TIME_TO_NIGHT_SHOW"].notna()
            else:
                mask_group = mask_group & df["TIME_TO_NIGHT_SHOW"].isna()

            df_group_pred = df[mask_group]

            if df_group_pred.empty:
                continue

            X_pred = df_group_pred[features]
            preds[mask_group] = models[(attraction, group_name)].predict(X_pred)

    return preds


if __name__ == "__main__":
    df_train = pd.read_csv("/home/ubuntu/upload/weather_data_combined.csv")
    df_val = pd.read_csv("/home/ubuntu/upload/valmeteo.csv")

    print("\n--- Adaptation du dataset d\'entraînement ---")
    adapted_train_groups = adapter_dataset_8_groupes(df_train)
    df_train_adapted = pd.concat(adapted_train_groups.values())

    print("\n--- Adaptation du dataset de validation ---")
    adapted_val_groups = adapter_dataset_8_groupes(df_val)
    df_val_adapted = pd.concat(adapted_val_groups.values())

    print("\n--- Entraînement des modèles ---")
    models, features = train_by_attr_and_parade_groups(df_train_adapted)

    print("\n--- Prédiction sur le dataset de validation ---")
    df_val_adapted["PREDICTED_WAIT_TIME_IN_2H"] = predict_by_attr_and_parade_groups(models, features, df_val_adapted)

    # Vérifier si la colonne cible existe dans le DataFrame de validation pour l'évaluation
    target = "WAIT_TIME_IN_2H"
    if target in df_val_adapted.columns:
        all_y_true = []
        all_y_pred = []

        for attraction in df_val_adapted["ENTITY_DESCRIPTION_SHORT"].unique():
            mask_attr = df_val_adapted["ENTITY_DESCRIPTION_SHORT"] == attraction
            for group_name, conditions in parade_groups.items():
                if (attraction, group_name) not in models:
                    continue

                mask = pd.Series(True, index=df_val_adapted.index)
                mask = mask & mask_attr

                if conditions["parade1"]:
                    mask = mask & df_val_adapted["TIME_TO_PARADE_1"].notna()
                else:
                    mask = mask & df_val_adapted["TIME_TO_PARADE_1"].isna()

                if conditions["parade2"]:
                    mask = mask & df_val_adapted["TIME_TO_PARADE_2"].notna()
                else:
                    mask = mask & df_val_adapted["TIME_TO_PARADE_2"].isna()

                if conditions["night_show"]:
                    mask = mask & df_val_adapted["TIME_TO_NIGHT_SHOW"].notna()
                else:
                    mask = mask & df_val_adapted["TIME_TO_NIGHT_SHOW"].isna()

                if df_val_adapted[mask].empty:
                    continue

                X_pred = df_val_adapted[mask][features]
                y_true_group = df_val_adapted[mask][target]
                y_pred_group = models[(attraction, group_name)].predict(X_pred)

                all_y_true.extend(y_true_group)
                all_y_pred.extend(y_pred_group)

        if len(all_y_true) > 0:
            final_rmse = np.sqrt(mean_squared_error(all_y_true, all_y_pred))
            final_mae = mean_absolute_error(all_y_true, all_y_pred)
            print(f"\n--- Évaluation globale sur le dataset de validation ---")
            print(f"RMSE global: {final_rmse:.2f}")
            print(f"MAE global: {final_mae:.2f}")
        else:
            print("Aucune prédiction n\'a pu être faite sur le dataset de validation avec les modèles entraînés pour l\'évaluation.")
    else:
        print(f"La colonne cible \'{target}\' est manquante dans le DataFrame de validation. Impossible d\'évaluer le modèle.")

    df_val_adapted.to_csv("valmeteo_predictions.csv", index=False)
    print("Prédictions sauvegardées dans valmeteo_predictions.csv")