# %% [markdown]
# # AirSentinel — Jour 3 : Analyse Exploratoire des Données (EDA)
# Ce script valide statistiquement la qualité et la stationnarité des fonctionnalités (features) engineered.

# %%
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
# Analyse statistique de séries temporelles
from statsmodels.tsa.stattools import adfuller, kpss

# ── CONFIGURATION DES CHEMINS ──────────────────────────────────────────────
# Permet de remonter d'un dossier pour que Python trouve le dossier 'src'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    # Ajuste l'import selon le nom exact de ton fichier dans src/
    from src.features.engineering import build_feature_matrix

    print("✅ Module de Feature Engineering importé avec succès !")
except ImportError:
    print("❌ Erreur d'importation. Assure-toi que src/features/engineering.py existe.")

# Configuration graphique
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_context("notebook", font_scale=1.1)
plt.rcParams["figure.figsize"] = (12, 6)

# %% [markdown]
# ## 1. Simulation / Chargement des Données de Test
# Si tu as déjà tes fichiers réels, remplace cette section par :
# `raw_df = pd.read_csv("../data/raw_sensor_data.csv")`

# %%
print("⏳ Génération de données synthétiques pour valider le pipeline...")
rng = np.random.default_rng(42)
N_SENSORS = 4
N_HOURS = 168  # 1 semaine de données horaires

sensor_ids = [f"SENSOR_{i:02d}" for i in range(N_SENSORS)]
sensor_locations = pd.DataFrame(
    {
        "location_id": sensor_ids,
        "latitude": 35.57 + rng.uniform(-0.02, 0.02, N_SENSORS),
        "longitude": -5.37 + rng.uniform(-0.02, 0.02, N_SENSORS),
    }
)

records = []
base_time = pd.Timestamp("2026-01-01 00:00:00", tz="UTC")

for sensor_id in sensor_ids:
    baseline = rng.uniform(15, 30)
    for h in range(N_HOURS):
        ts = base_time + pd.Timedelta(hours=h)
        diurnal = 12 * np.sin(2 * np.pi * h / 24)  # Cycle quotidien actif
        noise = rng.normal(0, 2)
        value = max(0, baseline + diurnal + noise)

        # Injection de pics de pollution locaux sur SENSOR_00
        if sensor_id == "SENSOR_00" and h in [48, 100, 140]:
            value = rng.uniform(120, 180)

        records.append(
            {
                "timestamp": ts,
                "location_id": sensor_id,
                "pollutant": "pm25",
                "value": round(value, 2),
            }
        )

raw_df = pd.DataFrame(records)

# Application de la matrice de features corrigée
df_features = build_feature_matrix(raw_df, sensor_locations, pollutant="pm25")

# %% [markdown]
# ## 2. Analyse de la Distribution et des Anomalies (Spikes)

# %%
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Histogramme global avec seuil d'alerte EPA
sns.histplot(
    data=df_features, x="value", kde=True, ax=axes[0], color="#1f77b4", bins=40
)
axes[0].axvline(
    55.5, color="#d62728", linestyle="--", linewidth=2, label="Seuil Alerte EPA (PM2.5)"
)
axes[0].set_title("Distribution des concentrations mesurées")
axes[0].set_xlabel("Valeur brute (µg/m³)")
axes[0].legend()

# CODE CORRIGÉ POUR LE BOXPLOT (Lignes 93-95)
# En spécifiant hue et en fixant les ticks explicitement, les avertissements disparaissent !
sns.boxplot(
    data=df_features,
    x="is_anomaly",
    y="value",
    ax=axes[1],
    hue="is_anomaly",
    palette=["#aec7e8", "#ff9896"],
    legend=False,
)
axes[1].set_title("Plage de concentrations : Normal vs Anomaly Spike")
axes[1].set_xticks([0, 1])
axes[1].set_xticklabels(["Normal (0)", "Anomalie (1)"])

# %% [markdown]
# ## 3. Matrice de Corrélation des Variables Clés

# %%
features_to_correlate = [
    "value",
    "spatial_neighbor_mean",
    "spatial_deviation",
    "stl_trend",
    "stl_seasonal",
    "stl_residual",
    "is_anomaly",
]

existing_cols = [c for c in features_to_correlate if c in df_features.columns]
corr_matrix = df_features[existing_cols].corr()

plt.figure(figsize=(10, 8))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

sns.heatmap(
    corr_matrix,
    mask=mask,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    vmin=-1,
    vmax=1,
    square=True,
    linewidths=0.5,
)
plt.title("Matrice de Corrélation de la Feature Matrix")
plt.show()

# %% [markdown]
# ## 4. Analyse Temporelle Avancée (ACF & PACF sur le résidu STL)

# %%
# Sélection d'un capteur témoin pour l'analyse temporelle isolée
sample_sensor = "SENSOR_00"
sensor_series = df_features[df_features["location_id"] == sample_sensor].sort_values(
    "timestamp"
)

# Idéalement sur le résidu pour voir s'il reste des structures temporelles cachées
series_to_plot = (
    sensor_series["stl_residual"].dropna()
    if "stl_residual" in sensor_series.columns
    else sensor_series["value"]
)

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
plot_acf(series_to_plot, lags=48, ax=axes[0], color="#2ca02c")
axes[0].set_title("Autocorrélation (ACF) sur le Résidu STL")

plot_pacf(series_to_plot, lags=48, ax=axes[1], color="#9467bd", method="ywm")
axes[1].set_title("Autocorrélation Partielle (PACF) sur le Résidu STL")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 5. Tests Rigoureux de Stationnarité (ADF / KPSS)

# %%
print("🔬 Lancement des tests statistiques de stationnarité...\n" + "=" * 60)

# Test ADF
adf_res = adfuller(series_to_plot, autolag="AIC")
print("🔹 Test Augmented Dickey-Fuller (ADF) :")
print(f"   • Statistique : {adf_res[0]:.4f} | p-value : {adf_res[1]:.4f}")
print(
    "   ✅ Conclusion : STATIONNAIRE"
    if adf_res[1] < 0.05
    else "   ❌ Conclusion : NON-STATIONNAIRE"
)

print("-" * 60)

# Test KPSS
kpss_res = kpss(series_to_plot, regression="c", nlags="auto")
print("🔹 Test KPSS :")
print(f"   • Statistique : {kpss_res[0]:.4f} | p-value : {kpss_res[1]:.4f}")
print(
    "   ✅ Conclusion : STATIONNAIRE"
    if kpss_res[1] > 0.05
    else "   ❌ Conclusion : NON-STATIONNAIRE"
)
print("=" * 60)
