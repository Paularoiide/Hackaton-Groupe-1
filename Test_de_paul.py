# repartition_sets.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import ks_2samp, chi2_contingency

# --------- Config fichiers ---------
PATH_TRAIN = "waiting_times_train.csv"
PATH_VAL   = "waiting_times_X_test_val.csv"
PATH_TEST  = "waiting_times_X_test_final.csv"

OUT_DIR = Path("repartition_outputs")
OUT_DIR.mkdir(exist_ok=True)

# --------- Utils ---------
def load_df(path, name):
    df = pd.read_csv(path)
    df["__SET__"] = name
    # DATETIME -> datetime si dispo
    if "DATETIME" in df.columns:
        df["DATETIME"] = pd.to_datetime(df["DATETIME"], errors="coerce")
        df["HOUR"] = df["DATETIME"].dt.hour
        df["DAY_OF_WEEK"] = df["DATETIME"].dt.dayofweek
    return df

def add_weather_buckets(df):
    """Crée des catégories météo robustes si colonnes dispo."""
    df = df.copy()
    if "rain_1h" in df.columns:
        df["IS_RAINING"] = (df["rain_1h"] > 0.2).map({True:"Rain", False:"No rain"})
    if "snow_1h" in df.columns:
        df["IS_SNOWING"] = (df["snow_1h"] > 0.05).map({True:"Snow", False:"No snow"})
    if "feels_like" in df.columns:
        # bandes simples
        bins = [-1e9, 0, 10, 20, 25, 30, 1e9]
        labels = ["<0°C","0–10","10–20","20–25","25–30",">30"]
        df["TEMP_BAND"] = pd.cut(df["feels_like"], bins=bins, labels=labels, include_lowest=True)
    if "wind_speed" in df.columns:
        bins = [-1e9, 5, 10, 20, 30, 1e9]
        labels = ["≤5","5–10","10–20","20–30",">30"]
        df["WIND_BAND"] = pd.cut(df["wind_speed"], bins=bins, labels=labels, include_lowest=True)
    if "humidity" in df.columns:
        bins = [-1e9, 40, 60, 80, 1e9]
        labels = ["≤40%","40–60%","60–80%",">80%"]
        df["HUMID_BAND"] = pd.cut(df["humidity"], bins=bins, labels=labels, include_lowest=True)
    return df

def norm_share_table(df, col):
    """Table % par catégorie et par set (colonnes: Train/Val/Test)"""
    if col not in df.columns:
        return None
    tab = (df
           .pivot_table(index=col, columns="__SET__", values=df.columns[0],  # n'importe quelle col pour size
                        aggfunc="size", fill_value=0)
           .apply(lambda s: s / s.sum(), axis=0))
    return tab.sort_index()

def plot_norm_bar(tab, title, fname):
    """Barres normalisées côte à côte."""
    if tab is None or tab.empty:
        return
    plt.figure(figsize=(10,5))
    # Chaque set en lignes -> on transpose pour tracer par set
    tab_t = tab.copy()
    tab_t.plot(kind="bar", figsize=(12,5))
    plt.title(title)
    plt.ylabel("Proportion")
    plt.xlabel("Catégorie")
    plt.legend(title="Set")
    plt.tight_layout()
    plt.savefig(OUT_DIR / fname, dpi=160)
    plt.close()

def top_k_by_set(df, col, k=20):
    """Top-k catégories par set avec % interne au set."""
    if col not in df.columns:
        return None
    out = []
    for s, g in df.groupby("__SET__"):
        cnt = g[col].value_counts(normalize=True).head(k).rename(s)
        out.append(cnt)
    return pd.concat(out, axis=1).fillna(0)

def ks_numeric(train, val, cols):
    rows = []
    for c in cols:
        x = train[c].dropna()
        y = val[c].dropna()
        if len(x)>0 and len(y)>0:
            stat, p = ks_2samp(x, y)
            rows.append((c, stat, p))
    return pd.DataFrame(rows, columns=["feature","ks_stat","p_value"]).sort_values("p_value")

def chi2_categorical(train, val, col):
    # Contingence: catégories x set
    sub = pd.concat([train.assign(__SET__="Train")[col],
                     val.assign(__SET__="Val")[col]], axis=0)
    ct = pd.crosstab(sub[col], sub["__SET__"])
    if ct.shape[0] < 2 or ct.shape[1] < 2:  # besoin min 2 catégories et 2 colonnes
        return None
    chi2, p, dof, _ = chi2_contingency(ct)
    return {"feature": col, "chi2": chi2, "p_value": p, "dof": dof}

# --------- Charge et prépare ---------
train = load_df(PATH_TRAIN, "Train")
val   = load_df(PATH_VAL,   "Val")
test  = load_df(PATH_TEST,  "Test")

all_df = pd.concat([train, val, test], ignore_index=True)
all_df = add_weather_buckets(all_df)

# --------- Listes de colonnes utiles ---------
cat_cols = [
    "ENTITY_DESCRIPTION_SHORT", "IS_RAINING", "IS_SNOWING",
    "TEMP_BAND", "WIND_BAND", "HUMID_BAND", "DAY_OF_WEEK", "HOUR"
]
num_cols = [c for c in ["feels_like","humidity","rain_1h","snow_1h","wind_speed","ADJUST_CAPACITY","CURRENT_WAIT_TIME"]
            if c in all_df.columns]

# --------- Répartition catégorielles (tables + plots) ---------
for col in cat_cols:
    if col not in all_df.columns: 
        continue
    tab = norm_share_table(all_df, col)
    if tab is None: 
        continue
    tab.to_csv(OUT_DIR / f"share_{col}.csv")
    plot_norm_bar(tab, f"Répartition normalisée par set — {col}", f"share_{col}.png")

# Top attractions (Top20 par set)
top_attractions = top_k_by_set(all_df, "ENTITY_DESCRIPTION_SHORT", k=20)
if top_attractions is not None:
    top_attractions.to_csv(OUT_DIR / "top20_attractions_by_set.csv")

# --------- Tests stats Train vs Val ---------
train_only = all_df[all_df["__SET__"]=="Train"].copy()
val_only   = all_df[all_df["__SET__"]=="Val"].copy()

# KS pour numériques
if num_cols:
    ks_tbl = ks_numeric(train_only, val_only, num_cols)
    ks_tbl.to_csv(OUT_DIR / "ks_train_vs_val_numeric.csv", index=False)

# Chi² pour catégorielles
chi_rows = []
for col in cat_cols:
    if col in train_only.columns and col in val_only.columns:
        res = chi2_categorical(train_only, val_only, col)
        if res:
            chi_rows.append(res)
if chi_rows:
    pd.DataFrame(chi_rows).sort_values("p_value").to_csv(OUT_DIR / "chi2_train_vs_val_categorical.csv", index=False)

print("✅ Fini. Dossiers/exports:", OUT_DIR.resolve())
print("- Tables CSV: share_*.csv, top20_attractions_by_set.csv, ks_*.csv, chi2_*.csv")
print("- Graphs PNG: share_*.png")
