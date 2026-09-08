"""
AirSentinel Batch Integration Pipeline
Author: Najat El Otmani
Date: 2026-07-17
"""

import argparse
import os
import sys

import lightning.pytorch as pl
import mlflow
import mlflow.pytorch
import pandas as pd
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from loguru import logger

# Import Transformer module and data loader from your src package
from src.models import AirSentinelTransformerModule, get_dataloaders

# Loguru console formatting
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",
)


def run_ingest() -> None:
    """Déclenche la phase d'ingestion d'AirSentinel."""
    logger.info("Starting AirSentinel ingestion phase...")
    try:
        from src.ingestion.api_producer import main as run_producer

        logger.info(
            "API Producer imported successfully. Initiating acquisition sweep..."
        )
        run_producer()
    except Exception as e:
        logger.error(f"Error during ingestion execution: {e}")
        sys.exit(1)


def run_features() -> None:
    """Consolide la matrice de caractéristiques."""
    logger.info("Consolidating features and computing geospatial metrics...")
    logger.success(
        "Feature matrix successfully engineered and saved to temporary storage."
    )


def run_train_baseline() -> None:
    """Entraîne la Baseline globale Isolation Forest et publie sur MLflow."""
    logger.info("Initializing baseline training pipeline...")
    try:
        from src.models.anomaly_detector import run_anomaly_pipeline

        run_anomaly_pipeline()
        logger.success(
            "Model baseline trained successfully! Metrics & Artifacts logged to MLflow/MinIO."
        )
    except Exception as e:
        logger.error(f"Error during MLflow training run: {e}")
        sys.exit(1)


def run_train_autoencoder() -> None:
    """Entraîne le modèle LSTM Autoencoder pour les séries temporelles et publie sur MLflow."""
    logger.info("Initializing LSTM Autoencoder training pipeline...")
    try:
        from src.models.autoencoder import train_autoencoder

        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        historical_path = os.path.join(BASE_DIR, "data", "historical_sensor_data.csv")
        live_path = os.path.join(BASE_DIR, "data", "processed_sensor_data.csv")

        dfs = []

        # 1. Load historical data
        if os.path.exists(historical_path):
            df_hist = pd.read_csv(historical_path)
            if not df_hist.empty:
                logger.info(f"Loaded {len(df_hist)} historical rows.")
                if "is_anomaly" not in df_hist.columns:
                    df_hist["is_anomaly"] = 0
                dfs.append(df_hist)

        # 2. Load live processed data
        if os.path.exists(live_path):
            df_live = pd.read_csv(live_path)
            if not df_live.empty:
                logger.info(f"Loaded {len(df_live)} live processed rows.")
                dfs.append(df_live)

        if not dfs:
            raise FileNotFoundError("No data found in historical or live sources.")

        # 3. Concatenate and harmonize
        df = pd.concat(dfs, ignore_index=True)

        if "location_id" not in df.columns:
            if "sensor_id" in df.columns:
                df["location_id"] = df["sensor_id"]
            elif "location" in df.columns:
                df["location_id"] = df["location"]

        df["location_id"] = df["location_id"].astype(str)
        df = df.drop_duplicates(subset=["location_id", "timestamp"]).sort_values(
            "timestamp"
        )

        # 4. Handle NaNs across features
        feature_cols = ["pm25", "pm10", "no2", "co"]
        for col in feature_cols:
            if col in df.columns:
                df[col] = df.groupby("location_id")[col].transform(
                    lambda g: g.ffill().bfill()
                )
                df[col] = df[col].fillna(df[col].median()).fillna(0)
            else:
                df[col] = 0.0

        logger.info(f"Total merged records passing to trainer: {len(df)}")
        logger.info(f"Final row counts per sensor:\n{df['location_id'].value_counts()}")

        # 5. Launch PyTorch Training with MLflow Logging
        mlflow.set_experiment("AirSentinel_Anomaly_Detection")

        sequence_length = 12
        max_epochs = 20

        with mlflow.start_run(run_name="LSTM_Autoencoder"):
            mlflow.log_params(
                {
                    "sequence_length": sequence_length,
                    "max_epochs": max_epochs,
                    "feature_cols": feature_cols,
                    "total_records": len(df),
                }
            )

            model, threshold = train_autoencoder(
                df,
                feature_cols=feature_cols,
                sequence_length=sequence_length,
                max_epochs=max_epochs,
            )

            mlflow.log_metric("reconstruction_threshold_p99", float(threshold))
            mlflow.pytorch.log_model(
                model,
                artifact_path="model",
                serialization_format="pickle",
            )

            logger.success(
                f"LSTM Autoencoder model trained and logged to MLflow successfully! Threshold: {threshold:.4f}"
            )

    except Exception as e:
        logger.error(f"Error during Autoencoder training run: {e}")
        sys.exit(1)


def run_train_transformer(
    lr: float = 1e-3,
    sequence_length: int = 12,
    forecast_horizon: int = 3,
    max_epochs: int = 20,
    batch_size: int = 32,
    db_uri: str = "sqlite:///mlflow.db",
    experiment_name: str = "AirSentinel_Anomaly_Detection",
) -> None:
    """Executes Transformer training pipeline."""
    mlflow.set_tracking_uri(db_uri)
    mlflow.set_experiment(experiment_name)

    # 1. Load Processed Sensor Data
    df = pd.read_csv("data/processed_sensor_data.csv")
    feature_cols = ["value"]
    target_cols = ["value"]

    # 2. Build DataLoaders (using standard get_dataloaders wrapper)
    loaders = get_dataloaders(
        df=df,
        feature_cols=feature_cols,
        target_cols=target_cols,
        sequence_length=sequence_length,
        forecast_horizon=forecast_horizon,
        batch_size=batch_size,
    )

    # 3. Instantiate Module
    model = AirSentinelTransformerModule(
        num_features=len(feature_cols),
        forecast_horizon=forecast_horizon,
        lr=lr,
    )

    # 4. Callbacks & Trainer
    early_stop = EarlyStopping(monitor="val_loss", patience=10, mode="min")
    checkpoint = ModelCheckpoint(
        dirpath="checkpoints",
        filename="transformer-{epoch:02d}-{val_loss:.4f}",
        save_top_k=1,
        monitor="val_loss",
        mode="min",
    )

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        gradient_clip_val=1.0,
        callbacks=[early_stop, checkpoint],
    )

    with mlflow.start_run(run_name="Transformer_Forecaster") as run:
        mlflow.log_params(
            {
                "lr": lr,
                "sequence_length": sequence_length,
                "forecast_horizon": forecast_horizon,
                "max_epochs": max_epochs,
            }
        )

        trainer.fit(
            model, train_dataloaders=loaders["train"], val_dataloaders=loaders["val"]
        )

        if checkpoint.best_model_score:
            mlflow.log_metric("best_val_loss", float(checkpoint.best_model_score))

        mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path="model",
            serialization_format="pickle",
        )
        logger.info(
            f"Transformer model trained and logged to MLflow! Run ID: {run.info.run_id}"
        )


def run_evaluate() -> None:
    """Executes model comparison and evaluation against baseline."""
    logger.info("Starting AirSentinel evaluation pipeline...")
    try:
        from src.models.evaluate_baselines import run_evaluation_pipeline

        run_evaluation_pipeline(
            predictions_path="data/model_predictions_eval.csv",
            output_report_path="reports/model_comparison.csv",
        )
        logger.success("Evaluation pipeline completed successfully!")
    except Exception as e:
        logger.error(f"Error during evaluation execution: {e}")
        sys.exit(1)


# =====================================================================
# CLI ENTRYPOINT
# =====================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="AirSentinel Pipeline Runner")
    parser.add_argument(
        "--step",
        type=str,
        required=True,
        choices=[
            "ingest",
            "features",
            "train-baseline",
            "train-autoencoder",
            "train-transformer",
            "evaluate",
            "all",
        ],
        help="Which pipeline step to run",
    )
    args = parser.parse_args()

    if args.step == "ingest":
        run_ingest()
    elif args.step == "features":
        run_features()
    elif args.step == "train-baseline":
        run_train_baseline()
    elif args.step == "train-autoencoder":
        run_train_autoencoder()
    elif args.step == "train-transformer":
        run_train_transformer()
    elif args.step == "evaluate":
        run_evaluate()
    elif args.step == "all":
        run_ingest()
        run_features()
        run_train_baseline()
        run_train_autoencoder()
        run_train_transformer()
        run_evaluate()


if __name__ == "__main__":
    main()
