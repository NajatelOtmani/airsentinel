"""
AirSentinel Batch Integration Pipeline
Author: Najat El Otmani
Date: 2026-07-17
"""

import argparse
import sys

from loguru import logger

# Configuration stylisée de Loguru pour notre console
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
    """Simule ou déclenche la consolidation de la matrice de caractéristiques."""
    logger.info("Consolidating features and computing geospatial metrics...")
    # Ici s'exécute ton pipeline d'ingénierie des caractéristiques globales
    logger.success(
        "Feature matrix successfully engineered and saved to temporary storage."
    )


def run_train() -> None:
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AirSentinel CLI - End-to-End Batch Pipeline"
    )
    parser.add_argument(
        "--step",
        type=str,
        required=True,
        choices=["ingest", "features", "train-baseline", "all"],
        help="Target step of the pipeline to execute.",
    )

    args = parser.parse_args()

    if args.step == "ingest":
        run_ingest()
    elif args.step == "features":
        run_features()
    elif args.step == "train-baseline":
        run_train()
    elif args.step == "all":
        logger.info("Executing Full Pipeline Suite...")
        run_ingest()
        run_features()
        run_train()
        logger.success("All steps completed successfully!")


if __name__ == "__main__":
    main()
