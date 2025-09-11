import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesRegressor
import xgboost as xgb

# =========================================================
# Utils
# =========================================================
def RMSE(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def compute_oof_weights(oof_preds, y, power=1.0):
    rmses = np.array([np.sqrt(mean_squared_error(y, oof_preds[:, i]))
                      for i in range(oof_preds.shape[1])])
    w = 1.0 / (rmses ** power + 1e-9)
    w = w / w.sum()
    return w, rmses

def predict_with_blend(fitted_base, features, df, weights, selected_idx=None):
    X = df[features].values
    models = fitted_base if selected_idx is None else [fitted_base[i] for i in selected_idx]
    base_mat = np.column_stack([mdl.predict(X) for mdl in models])
    if selected_idx is not None:
        weights = weights[selected_idx]
        weights = weights / weights.sum()
    return base_mat.dot(weights)

def evaluate_if_possible(df, y_true_col, y_pred, tag=""):
    if y_true_col in df.columns:
        score = RMSE(df[y_true_col].values, y_pred)
        print(f"[{tag}] RMSE = {score:.4f}")
        return score
    else:
        print(f"[{tag}] Pas de colonne '{y_true_col}' : pas d'évaluation.")
        return None

# =========================================================
# Tes features/ingénierie (reprend et corrige mineures)
# =========================================================
def adapt_data_paul_GX(dataset):
    dataset = dataset.copy()

    dataset['IS_RAINING'] = (dataset['rain_1h'] > 0.2).astype(int)
    dataset['IS_SNOWING'] = (dataset['snow_1h'] > 0.05).astype(int)
    dataset['IS_HOT']     = (dataset['feels_like'] > 25).astype(int)
    dataset['IS_COLD']    = (dataset['feels_like'] < 0).astype(int)
    dataset['TEMP_HUMIDITY_INDEX'] = dataset['feels_like'] * dataset['humidity']
    dataset['CAPACITY_RATIO'] = dataset['CURRENT_WAIT_TIME'] / (dataset['ADJUST_CAPACITY'] + 1e-6)

    dataset['snow_1h'] = dataset['snow_1h'].fillna(0)

    dataset['DATETIME'] = pd.to_datetime(dataset['DATETIME'])
    dataset['DAY_OF_WEEK'] = dataset['DATETIME'].dt.dayofweek
    dataset['DAY'] = dataset['DATETIME'].dt.day
    dataset['MONTH'] = dataset['DATETIME'].dt.month
    dataset['YEAR'] = dataset['DATETIME'].dt.year
    dataset['HOUR'] = dataset['DATETIME'].dt.hour
    dataset['MINUTE'] = dataset['DATETIME'].dt.minute
    dataset['WEEKEND'] = (dataset['DAY_OF_WEEK'] >= 5).astype(int)
    dataset['POST_COVID'] = (dataset['DATETIME'] >= '2020-03-01').astype(int)

    dataset['IS_ATTRACTION_Water_Ride'] = (dataset['ENTITY_DESCRIPTION_SHORT'] == "Water Ride").astype(int)
    dataset['IS_ATTRACTION_Pirate_Ship'] = (dataset['ENTITY_DESCRIPTION_SHORT'] == "Pirate Ship").astype(int)
    dataset['IS_ATTRACTION__Flying_Coaster'] = (dataset['ENTITY_DESCRIPTION_SHORT'] == "Flying Coaster").astype(int)

    def detecter_vacances_par_zone(ts):
        # ts: pandas.Timestamp
        d = ts.to_pydatetime()
        vacances_zones = {
            'ZONE_A': [
                (datetime(2018,10,20), datetime(2018,11,4)),
                (datetime(2018,12,22), datetime(2019,1,6)),
                (datetime(2019,2,16),  datetime(2019,3,3)),
                (datetime(2019,4,13),  datetime(2019,4,28)),
                (datetime(2019,7,6),   datetime(2019,9,1)),
                (datetime(2019,10,19), datetime(2019,11,3)),
                (datetime(2019,12,21), datetime(2020,1,5)),
                (datetime(2020,2,8),   datetime(2020,2,23)),
                (datetime(2020,4,4),   datetime(2020,4,19)),
                (datetime(2020,7,4),   datetime(2020,9,1)),
                (datetime(2020,10,17), datetime(2020,11,1)),
                (datetime(2020,12,19), datetime(2021,1,3)),
                (datetime(2021,2,6),   datetime(2021,2,21)),
                (datetime(2021,4,10),  datetime(2021,4,25)),
                (datetime(2021,7,6),   datetime(2021,9,1)),
                (datetime(2021,10,23), datetime(2021,11,7)),
                (datetime(2021,12,18), datetime(2022,1,2)),
                (datetime(2022,2,12),  datetime(2022,2,27)),
                (datetime(2022,4,16),  datetime(2022,5,1)),
                (datetime(2022,7,7),   datetime(2022,9,1)),
            ],
            'ZONE_B': [
                (datetime(2018,10,20), datetime(2018,11,4)),
                (datetime(2018,12,22), datetime(2019,1,6)),
                (datetime(2019,2,9),   datetime(2019,2,24)),
                (datetime(2019,4,6),   datetime(2019,4,21)),
                (datetime(2019,7,6),   datetime(2019,9,1)),
                (datetime(2019,10,19), datetime(2019,11,3)),
                (datetime(2019,12,21), datetime(2020,1,5)),
                (datetime(2020,2,22),  datetime(2020,3,8)),
                (datetime(2020,4,4),   datetime(2020,4,19)),
                (datetime(2020,7,4),   datetime(2020,9,1)),
                (datetime(2020,10,17), datetime(2020,11,1)),
                (datetime(2020,12,19), datetime(2021,1,3)),
                (datetime(2021,2,20),  datetime(2021,3,7)),
                (datetime(2021,4,10),  datetime(2021,4,25)),
                (datetime(2021,7,6),   datetime(2021,9,1)),
                (datetime(2021,10,23), datetime(2021,11,7)),
                (datetime(2021,12,18), datetime(2022,1,2)),
                (datetime(2022,2,26),  datetime(2022,3,13)),
                (datetime(2022,4,16),  datetime(2022,5,1)),
                (datetime(2022,7,7),   datetime(2022,9,1)),
            ],
            'ZONE_C': [
                (datetime(2018,10,20), datetime(2018,11,4)),
                (datetime(2018,12,22), datetime(2019,1,6)),
                (datetime(2019,2,23),  datetime(2019,3,10)),
                (datetime(2019,4,20),  datetime(2019,5,5)),
                (datetime(2019,7,6),   datetime(2019,9,1)),
                (datetime(2019,10,19), datetime(2019,11,3)),
                (datetime(2019,12,21), datetime(2020,1,5)),
                (datetime(2020,2,15),  datetime(2020,3,1)),
                (datetime(2020,4,18),  datetime(2020,5,3)),
                (datetime(2020,7,4),   datetime(2020,9,1)),
                (datetime(2020,10,17), datetime(2020,11,1)),
                (datetime(2020,12,19), datetime(2021,1,3)),
                (datetime(2021,2,13),  datetime(2021,2,28)),
                (datetime(2021,4,24),  datetime(2021,5,9)),
                (datetime(2021,7,6),   datetime(2021,9,1)),
                (datetime(2021,10,23), datetime(2021,11,7)),
                (datetime(2021,12,18), datetime(2022,1,2)),
                (datetime(2022,2,12),  datetime(2022,2,27)),
                (datetime(2022,4,23),  datetime(2022,5,8)),
                (datetime(2022,7,7),   datetime(2022,9,1)),
            ]
        }
        res = {'VACANCES_ZONE_A': 0, 'VACANCES_ZONE_B': 0, 'VACANCES_ZONE_C': 0}
        for zone, periods in vacances_zones.items():
            for a, b in periods:
                if a <= d <= b:
                    if zone == 'ZONE_A': res['VACANCES_ZONE_A'] = 1
                    if zone == 'ZONE_B': res['VACANCES_ZONE_B'] = 1
                    if zone == 'ZONE_C': res['VACANCES_ZONE_C'] = 1
        return res

    vac_df = pd.DataFrame(list(dataset['DATETIME'].apply(detecter_vacances_par_zone)))
    dataset = pd.concat([dataset, vac_df], axis=1)

    # Nettoyage colonnes inutiles pour l'entraînement
    drop_cols = [c for c in ['CURRENT_WAIT_TIME', 'dew_point'] if c in dataset.columns]
    dataset.drop(columns=drop_cols, inplace=True, errors='ignore')
    return dataset

# =========================================================
# Modèles de base
# =========================================================
def get_base_models():
    base = []
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

    base.append(("et_0", ExtraTreesRegressor(
        n_estimators=700, max_depth=None, min_samples_leaf=1,
        n_jobs=-1, random_state=11
    )))
    base.append(("et_1", ExtraTreesRegressor(
        n_estimators=1000, max_depth=None, min_samples_leaf=1,
        n_jobs=-1, random_state=12
    )))
    return base

# =========================================================
# Stacking OOF + sélection des meilleurs modèles
# =========================================================
def train_stacking_oof(df, target="WAIT_TIME_IN_2H", n_splits=5, random_state=42):
    features = [c for c in df.columns if c not in [target, "DATETIME", "ENTITY_DESCRIPTION_SHORT"]]
    X, y = df[features].values, df[target].values
    base_models = get_base_models()

    oof_preds = np.zeros((len(X), len(base_models)))
    folds = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for i, (tr_idx, va_idx) in enumerate(folds.split(X)):
        Xtr, Xva = X[tr_idx], X[va_idx]
        ytr = y[tr_idx]
        for m, (name, mdl) in enumerate(base_models):
            print(f"🔄 Fold {i+1}, Model {name}")
            mdl.fit(Xtr, ytr)
            oof_preds[va_idx, m] = mdl.predict(Xva)

    # Poids + RMSE par base
    blend_weights, base_rmses = compute_oof_weights(oof_preds, y, power=1.0)
    print("Poids blend (1/rmse):", blend_weights)
    print("RMSE bases:", base_rmses)

    # Ré-entraînement full data
    fitted_base = []
    for name, mdl in base_models:
        mdl.fit(X, y)
        fitted_base.append(mdl)

    return fitted_base, features, oof_preds, y, blend_weights, base_rmses, [n for n, _ in base_models]

def select_best_models(base_rmses, names, threshold=6.5, top_k_min=5):
    idx = [i for i, r in enumerate(base_rmses) if r <= threshold]
    if len(idx) < top_k_min:
        # fallback: prendre les meilleurs top_k_min
        idx = list(np.argsort(base_rmses)[:top_k_min])
    print("📌 Modèles retenus:", [names[i] for i in idx])
    print("   Leurs RMSE:", [float(base_rmses[i]) for i in idx])
    return idx

# =========================================================
# Méta-modèles
# =========================================================
def fit_meta_ridge(oof_preds_sel, y):
    meta = Ridge(alpha=1.0)
    meta.fit(oof_preds_sel, y)
    return meta

def fit_meta_xgb(oof_preds_sel, y, random_state=1337):
    meta = xgb.XGBRegressor(
        n_estimators=400, max_depth=3, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
        random_state=random_state, tree_method="hist", n_jobs=-1
    )
    meta.fit(oof_preds_sel, y)
    return meta

def predict_with_meta(meta_model, fitted_base, selected_idx, features, df):
    X = df[features].values
    base_mat_sel = np.column_stack([fitted_base[i].predict(X) for i in selected_idx])
    return meta_model.predict(base_mat_sel)

# =========================================================
# Hiérarchique (ET → XGB méta)
# =========================================================
def hierarchical_stack(fitted_base, names, features, df, y_train_oof, oof_preds, target_in_val=False):
    et_idx = [i for i, n in enumerate(names) if n.startswith("et_")]
    if not et_idx:
        return None, None

    # OOF de la couche 1 = moyenne ET
    oof_et = oof_preds[:, et_idx].mean(axis=1, keepdims=True)

    # Méta XGB entraîné sur oof_et
    meta = fit_meta_xgb(oof_et, y_train_oof)

    # Prédictions couche 1 sur val = moyenne ET
    X_val_preds_et = np.column_stack([fitted_base[i].predict(df[features].values) for i in et_idx]).mean(axis=1)
    y_pred = meta.predict(X_val_preds_et.reshape(-1, 1))
    return y_pred, et_idx

# =========================================================
# Main
# =========================================================
if __name__ == "__main__":
    TARGET = "WAIT_TIME_IN_2H"
    RMSE_THRESHOLD = 6.5
    TOP_K_MIN = 5

    # --- Entraînement
    train = pd.read_csv("weather_data_combined.csv")
    train = adapt_data_paul_GX(train)

    fitted_base, features, oof_preds, y_oof, blend_weights, base_rmses, model_names = train_stacking_oof(
        train, target=TARGET, n_splits=5, random_state=42
    )

    # Sélection des meilleurs modèles
    best_idx = select_best_models(base_rmses, model_names, threshold=RMSE_THRESHOLD, top_k_min=TOP_K_MIN)

    # Recalcule des poids restreints aux meilleurs
    w_all, _ = compute_oof_weights(oof_preds, y_oof, power=1.0)
    w_sel = w_all[best_idx] / w_all[best_idx].sum()

    # Méta-modèles sur OOF sélectionné
    oof_sel = oof_preds[:, best_idx]
    meta_ridge = fit_meta_ridge(oof_sel, y_oof)
    meta_xgb = fit_meta_xgb(oof_sel, y_oof)

    # --- Validation
    val = pd.read_csv("valmeteo.csv")
    val = adapt_data_paul_GX(val)

    # 1) Weighted average (meilleurs modèles)
    y_pred_weighted = predict_with_blend(fitted_base, features, val, weights=w_all, selected_idx=best_idx)
    evaluate_if_possible(val, TARGET, y_pred_weighted, tag="WeightedAvg (best only)")

    # 2) Méta Ridge (best only)
    y_pred_meta_ridge = predict_with_meta(meta_ridge, fitted_base, best_idx, features, val)
    evaluate_if_possible(val, TARGET, y_pred_meta_ridge, tag="Meta Ridge (best only)")

    # 3) Méta XGB (best only)
    y_pred_meta_xgb = predict_with_meta(meta_xgb, fitted_base, best_idx, features, val)
    evaluate_if_possible(val, TARGET, y_pred_meta_xgb, tag="Meta XGB (best only)")

    # 4) Hiérarchique ET→XGB
    y_pred_hier, et_idx = hierarchical_stack(fitted_base, model_names, features, val, y_oof, oof_preds)
    if y_pred_hier is not None:
        evaluate_if_possible(val, TARGET, y_pred_hier, tag="Hierarchical ET→XGB")

    # --- Choix final (prend le meilleur si y_true dispo, sinon on exporte tout)
    preds = {
        "weighted_best": y_pred_weighted,
        "meta_ridge_best": y_pred_meta_ridge,
        "meta_xgb_best": y_pred_meta_xgb
    }
    if y_pred_hier is not None:
        preds["hier_et_xgb"] = y_pred_hier

    # Si la colonne cible est dispo dans val, on choisit la meilleure
    best_key = None
    if TARGET in val.columns:
        scores = {k: RMSE(val[TARGET].values, v) for k, v in preds.items()}
        best_key = min(scores, key=scores.get)
        print("🏆 Meilleure stratégie sur validation:", best_key, "RMSE =", scores[best_key])
        val[f"y_pred_{best_key}"] = preds[best_key]
        out_cols = ["DATETIME", "ENTITY_DESCRIPTION_SHORT", f"y_pred_{best_key}"]
    else:
        # Pas de cible : on dump toutes les colonnes de prédiction
        for k, v in preds.items():
            val[f"y_pred_{k}"] = v
        out_cols = ["DATETIME", "ENTITY_DESCRIPTION_SHORT"] + [f"y_pred_{k}" for k in preds.keys()]

    val.assign(KEY="Validation")[out_cols + ["KEY"]].to_csv("val_predictions_stacking.csv", index=False)
    print("✅ Export: val_predictions_stacking.csv")
