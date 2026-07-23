"""
src/models/anomaly_detector.py
================================
Isolation Forest training for AirSentinel.

Contains two paths:
  1. run_anomaly_pipeline() — the original synthetic-data demo. Unchanged,
     kept intact because it's already verified working end-to-end (MLflow +
     MinIO logging confirmed).
  2. train_baseline_on_features() / score_batch() — new functions that train
     on a REAL engineered feature matrix (the output of
     src/features/engineering.py), so the batch CLI pipeline
     (ingest -> validate -> features -> train) trains on live data instead
     of synthetic spikes.
"""

import logging
import os
from datetime import datetime
from typing import List, Tuple

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import seaborn as sns
from dotenv import load_dotenv
from loguru import logger as loguru_logger
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import KFold

# Configurer le logging structuré (original demo path uses stdlib logging)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("air_sentinel.analytics")

# Charger les variables d'environnement depuis .env
load_dotenv()

# Configuration des connexions de stockage MLflow & MinIO (S3 compatible)
os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.getenv(
    "MINIO_ENDPOINT", "http://localhost:9000"
)
os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("MINIO_USER", "minioadmin")
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("MINIO_PASSWORD", "minioadmin")

# Fixer explicitement l'URI de tracking
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))

# Initialiser l'expérience MLflow
mlflow.set_experiment("AirSentinel_Anomaly_Detection")


def generate_labeled_spike_data(
    n_samples: int = 500, n_sensors: int = 10
) -> pd.DataFrame:
    """Génère une matrice de features historique fictive mais réaliste avec des
    labels de pics ('spike events') pour simuler le processus de cross-validation.
    """
    np.random.seed(42)
    timestamps = pd.date_range(end=datetime.utcnow(), periods=n_samples, freq="15min")
    sensor_ids = [f"LONDON_{i}" for i in range(1, n_sensors + 1)]

    data_list = []
    for s_id in sensor_ids:
        pm25 = np.random.normal(15, 4, n_samples)
        pm10 = pm25 * 1.4 + np.random.normal(5, 2, n_samples)
        no2 = np.random.normal(30, 8, n_samples)

        df_sensor = pd.DataFrame(
            {
                "timestamp": timestamps,
                "sensor_id": s_id,
                "pm25": pm25,
                "pm10": pm10,
                "no2": no2,
                "is_spike": 0,
            }
        )

        spike_indices = np.random.choice(
            n_samples, size=int(n_samples * 0.05), replace=False
        )
        df_sensor.loc[spike_indices, "pm25"] += np.random.uniform(
            30, 60, size=len(spike_indices)
        )
        df_sensor.loc[spike_indices, "pm10"] += np.random.uniform(
            40, 80, size=len(spike_indices)
        )
        df_sensor.loc[spike_indices, "is_spike"] = 1

        data_list.append(df_sensor)

    df_all = pd.concat(data_list, ignore_index=True)
    return df_all


def tune_contamination_cv(
    X: np.ndarray,
    y: np.ndarray,
    contamination_grid: List[float] = [0.01, 0.03, 0.05, 0.08, 0.10],
) -> float:
    """Trouve la meilleure contamination via K-Fold Cross Validation face aux
    événements labellisés."""
    best_contamination = 0.05
    best_f1 = 0
    kf = KFold(n_splits=3, shuffle=True, random_state=42)

    for c in contamination_grid:
        f1_scores = []
        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_val = y[val_idx]

            model = IsolationForest(contamination=c, random_state=42, n_jobs=-1)
            model.fit(X_train)

            preds = model.predict(X_val)
            binary_preds = np.where(preds == -1, 1, 0)

            f1_scores.append(f1_score(y_val, binary_preds, zero_division=0))

        mean_f1 = np.mean(f1_scores)
        if mean_f1 > best_f1:
            best_f1 = mean_f1
            best_contamination = c

    return best_contamination


def run_anomaly_pipeline() -> None:
    """Original synthetic-data demo pipeline. Unchanged — kept as the
    fallback path when no real engineered feature batch is available."""
    logger.info("⚡ Chargement de la matrice de features historique enrichie...")
    df = generate_labeled_spike_data()
    feature_cols = ["pm25", "pm10", "no2"]

    with mlflow.start_run(run_name="Cross_Sensor_Multivariate_IForest"):
        logger.info("🔮 Entraînement du modèle Cross-Sensor global...")
        X = df[feature_cols].values
        y = df["is_spike"].values

        best_c = tune_contamination_cv(X, y)
        logger.info(f"🎯 Meilleure contamination globale trouvée : {best_c}")

        global_model = IsolationForest(contamination=best_c, random_state=42, n_jobs=-1)
        global_model.fit(X)

        preds = global_model.predict(X)
        df["global_anomaly_pred"] = np.where(preds == -1, 1, 0)
        df["global_anomaly_score"] = -global_model.score_samples(X)

        f1 = f1_score(y, df["global_anomaly_pred"])
        precision = precision_score(y, df["global_anomaly_pred"])
        recall = recall_score(y, df["global_anomaly_pred"])

        mlflow.log_param("model_type", "cross_sensor_multivariate")
        mlflow.log_param("tuned_contamination", best_c)
        mlflow.log_metric("F1_Score", f1)
        mlflow.log_metric("Precision", precision)
        mlflow.log_metric("Recall", recall)

        cm = confusion_matrix(y, df["global_anomaly_pred"])
        plt.figure(figsize=(5, 4))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Normal", "Spike"],
            yticklabels=["Normal", "Spike"],
        )
        plt.title("Confusion Matrix - Global Cross-Sensor")
        plt.ylabel("Actual")
        plt.xlabel("Predicted")
        cm_path = "confusion_matrix_global.png"
        plt.savefig(cm_path)
        plt.close()
        mlflow.log_artifact(cm_path)

        sample_station = "LONDON_1"
        df_sample = (
            df[df["sensor_id"] == sample_station].sort_values("timestamp").head(150)
        )

        fig, ax1 = plt.subplots(figsize=(12, 5))
        ax1.plot(
            df_sample["timestamp"],
            df_sample["pm25"],
            label="PM2.5 Real-Time",
            color="teal",
            alpha=0.8,
        )
        ax1.set_ylabel("PM2.5 Concentration (µg/m³)", color="teal")
        ax1.tick_params(axis="y", labelcolor="teal")

        ax2 = ax1.twinx()
        ax2.plot(
            df_sample["timestamp"],
            df_sample["global_anomaly_score"],
            label="Anomaly Score",
            color="crimson",
            linestyle="--",
            alpha=0.7,
        )
        anomalies = df_sample[df_sample["global_anomaly_pred"] == 1]
        ax1.scatter(
            anomalies["timestamp"],
            anomalies["pm25"],
            color="red",
            s=40,
            label="Detected Anomalies",
            zorder=5,
        )

        ax2.set_ylabel("Isolation Forest Anomaly Score", color="crimson")
        ax2.tick_params(axis="y", labelcolor="crimson")

        plt.title(
            f"Real-Time AirSentinel Monitor: Anomaly Scores Overlaid on PM2.5 ({sample_station})"
        )
        fig.tight_layout()
        viz_path = "anomaly_overlay_global.png"
        plt.savefig(viz_path)
        plt.close()
        mlflow.log_artifact(viz_path)

        mlflow.sklearn.log_model(global_model, name="cross_sensor_iforest")
        logger.info(
            "🚀 Modèle Cross-Sensor sauvegardé et suivi dans MLflow/MinIO avec succès."
        )

    logger.info("⚙️ Lancement du traitement Per-Sensor pour isolation locale...")
    for s_id in df["sensor_id"].unique()[:2]:
        with mlflow.start_run(run_name=f"Per_Sensor_IForest_{s_id}", nested=True):
            df_s = df[df["sensor_id"] == s_id]
            X_s = df_s[feature_cols].values
            y_s = df_s["is_spike"].values

            best_c_s = tune_contamination_cv(X_s, y_s)

            local_model = IsolationForest(contamination=best_c_s, random_state=42)
            local_model.fit(X_s)

            mlflow.log_param("sensor_id", s_id)
            mlflow.log_param("model_type", "per_sensor_local")
            mlflow.sklearn.log_model(local_model, name=f"per_sensor_{s_id}")
            logger.info(f"🎯 Modèle local sauvegardé pour {s_id}")

    if os.path.exists(cm_path):
        os.remove(cm_path)
    if os.path.exists(viz_path):
        os.remove(viz_path)


# ===========================================================================
# NEW: real-data training/scoring path, used by the batch CLI pipeline
# ===========================================================================


def train_baseline_on_features(
    df: pd.DataFrame,
    feature_cols: List[str],
    label_col: str = "is_anomaly",
    run_name: str = "Batch_Pipeline_IForest",
) -> Tuple[IsolationForest, float]:
    """Train an Isolation Forest on a real engineered feature matrix.

    Unlike run_anomaly_pipeline() (which trains on synthetic spike data),
    this consumes the actual output of AirQualityFeatureEngineer /
    build_feature_matrix, logging params/metrics/model to the same MLflow
    experiment so both runs are comparable in the MLflow UI.

    Args:
        df: Engineered feature DataFrame (output of the features step).
        feature_cols: Which columns of df to use as model input.
        label_col: Optional ground-truth anomaly column (e.g. 'is_anomaly'
            from label_spike_events()). If absent, contamination defaults
            to 0.05 instead of being cross-validated.
        run_name: MLflow run name for this training run.

    Returns:
        Tuple of (trained IsolationForest, contamination value used).
    """
    loguru_logger.info(f"Training Isolation Forest on {len(df)} real feature rows...")
    X = df[feature_cols].values

    has_labels = label_col in df.columns and df[label_col].notna().any()
    y = df[label_col].values if has_labels else None

    with mlflow.start_run(run_name=run_name):
        if has_labels:
            best_c = tune_contamination_cv(X, y)
        else:
            loguru_logger.warning(
                f"No '{label_col}' column found — skipping CV tuning, defaulting contamination=0.05."
            )
            best_c = 0.05

        model = IsolationForest(contamination=best_c, random_state=42, n_jobs=-1)
        model.fit(X)

        preds = model.predict(X)
        binary_preds = np.where(preds == -1, 1, 0)

        mlflow.log_param("model_type", "batch_pipeline_iforest")
        mlflow.log_param("feature_cols", ",".join(feature_cols))
        mlflow.log_param("tuned_contamination", best_c)
        mlflow.log_param("n_rows", len(df))

        if has_labels:
            f1 = f1_score(y, binary_preds, zero_division=0)
            precision = precision_score(y, binary_preds, zero_division=0)
            recall = recall_score(y, binary_preds, zero_division=0)
            mlflow.log_metric("F1_Score", f1)
            mlflow.log_metric("Precision", precision)
            mlflow.log_metric("Recall", recall)
            loguru_logger.info(
                f"F1={f1:.3f} | Precision={precision:.3f} | Recall={recall:.3f}"
            )

        mlflow.sklearn.log_model(model, name="batch_pipeline_iforest")
        loguru_logger.success("Model logged to MLflow.")

    return model, best_c


def score_batch(
    df: pd.DataFrame, feature_cols: List[str], model: IsolationForest
) -> pd.DataFrame:
    """Score a feature batch with a trained Isolation Forest.

    Args:
        df: Engineered feature DataFrame to score.
        feature_cols: Same columns the model was trained on, in the same order.
        model: A fitted IsolationForest (e.g. from train_baseline_on_features()).

    Returns:
        Copy of df with two new columns: anomaly_pred (0/1) and anomaly_score
        (higher = more anomalous).
    """
    result = df.copy()
    X = result[feature_cols].values
    preds = model.predict(X)
    result["anomaly_pred"] = np.where(preds == -1, 1, 0)
    result["anomaly_score"] = -model.score_samples(X)

    n_flagged = int(result["anomaly_pred"].sum())
    loguru_logger.info(f"Scored {len(result)} rows — {n_flagged} flagged as anomalies.")
    return result


if __name__ == "__main__":
    run_anomaly_pipeline()
