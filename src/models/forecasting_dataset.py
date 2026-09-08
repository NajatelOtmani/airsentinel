"""
AirSentinel: Air Quality Anomaly Detection & Forecasting Pipeline.

This module provides dataset constructs and DataLoader generators for multi-step
time-series forecasting using sliding windows across grouped sensor locations.
"""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from loguru import logger
from torch.utils.data import DataLoader, Dataset


class ForecastingDataset(Dataset):
    """PyTorch Dataset for multi-step time-series forecasting.

    Generates (X, Y) sequence pairs across multi-sensor data while strictly enforcing
    sensor boundaries so sliding windows never bridge distinct `location_id`s.

    Args:
        df (pd.DataFrame): Input dataframe containing sensor telemetry.
        feature_cols (List[str]): Columns to use as input features (X).
        target_cols (List[str]): Columns to predict for future steps (Y).
        sequence_length (int): Historical window size (L).
        forecast_horizon (int): Future prediction window size (K).
        location_col (str): Column name identifying distinct sensors. Defaults to "location_id".
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_cols: List[str],
        sequence_length: int = 12,
        forecast_horizon: int = 3,
        location_col: str = "location_id",
    ) -> None:
        super().__init__()
        self.feature_cols = feature_cols
        self.target_cols = target_cols
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.location_col = location_col

        self.x_samples: List[np.ndarray] = []
        self.y_samples: List[np.ndarray] = []

        self._build_windows(df)

    def _build_windows(self, df: pd.DataFrame) -> None:
        """Iterates over each location_id independently to extract valid window pairs."""
        if self.location_col not in df.columns:
            raise KeyError(
                f"Location column '{self.location_col}' missing from DataFrame."
            )

        missing_features = [col for col in self.feature_cols if col not in df.columns]
        missing_targets = [col for col in self.target_cols if col not in df.columns]

        if missing_features:
            raise KeyError(
                f"Feature columns missing from DataFrame: {missing_features}"
            )
        if missing_targets:
            raise KeyError(f"Target columns missing from DataFrame: {missing_targets}")

        total_window = self.sequence_length + self.forecast_horizon
        skipped_sensors = 0

        for location_id, group in df.groupby(self.location_col):
            # Sort chronologically if 'date' column is present
            if "date" in group.columns:
                group = group.sort_values("date")

            features_arr = group[self.feature_cols].to_numpy(dtype=np.float32)
            targets_arr = group[self.target_cols].to_numpy(dtype=np.float32)

            num_rows = len(group)
            if num_rows < total_window:
                logger.warning(
                    f"Sensor '{location_id}' has {num_rows} rows, fewer than required window size "
                    f"({total_window} = {self.sequence_length} + {self.forecast_horizon}). Skipping."
                )
                skipped_sensors += 1
                continue

            # Extract sliding windows for this sensor group
            for i in range(num_rows - total_window + 1):
                x_win = features_arr[i : i + self.sequence_length]
                y_win = targets_arr[i + self.sequence_length : i + total_window]

                self.x_samples.append(x_win)
                self.y_samples.append(y_win)

        logger.info(
            f"ForecastingDataset constructed | Total Samples: {len(self.x_samples)} | "
            f"Input Shape: ({self.sequence_length}, {len(self.feature_cols)}) | "
            f"Target Shape: ({self.forecast_horizon}, {len(self.target_cols)}) | "
            f"Skipped Sensors: {skipped_sensors}"
        )

    def __len__(self) -> int:
        return len(self.x_samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x_tensor = torch.tensor(self.x_samples[idx], dtype=torch.float32)
        y_tensor = torch.tensor(self.y_samples[idx], dtype=torch.float32)
        return x_tensor, y_tensor


def create_forecasting_dataloaders(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_cols: List[str],
    sequence_length: int = 12,
    forecast_horizon: int = 3,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    batch_size: int = 32,
    num_workers: int = 0,
    location_col: str = "location_id",
) -> Dict[str, DataLoader]:
    """Splits sensor data chronologically per sensor, constructs ForecastingDataset instances,
    and returns PyTorch DataLoaders.

    Args:
        df (pd.DataFrame): Sensor dataset.
        feature_cols (List[str]): Input feature columns.
        target_cols (List[str]): Target forecasting columns.
        sequence_length (int, optional): Input sequence window. Defaults to 12.
        forecast_horizon (int, optional): Future horizon window. Defaults to 3.
        train_ratio (float, optional): Proportion of data for training. Defaults to 0.7.
        val_ratio (float, optional): Proportion of data for validation. Defaults to 0.15.
        batch_size (int, optional): Batch size. Defaults to 32.
        num_workers (int, optional): DataLoader workers. Defaults to 0.
        location_col (str, optional): Sensor group column. Defaults to "location_id".

    Returns:
        Dict[str, DataLoader]: Dictionary containing 'train', 'val', and 'test' DataLoaders.
    """
    train_dfs, val_dfs, test_dfs = [], [], []

    for _, group in df.groupby(location_col):
        if "date" in group.columns:
            group = group.sort_values("date")

        n = len(group)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        train_dfs.append(group.iloc[:train_end])
        val_dfs.append(group.iloc[train_end:val_end])
        test_dfs.append(group.iloc[val_end:])

    train_df = pd.concat(train_dfs, ignore_index=True)
    val_df = pd.concat(val_dfs, ignore_index=True)
    test_df = pd.concat(test_dfs, ignore_index=True)

    logger.info(
        f"Chronological split completed | Train rows: {len(train_df)} | "
        f"Val rows: {len(val_df)} | Test rows: {len(test_df)}"
    )

    train_dataset = ForecastingDataset(
        df=train_df,
        feature_cols=feature_cols,
        target_cols=target_cols,
        sequence_length=sequence_length,
        forecast_horizon=forecast_horizon,
        location_col=location_col,
    )

    val_dataset = ForecastingDataset(
        df=val_df,
        feature_cols=feature_cols,
        target_cols=target_cols,
        sequence_length=sequence_length,
        forecast_horizon=forecast_horizon,
        location_col=location_col,
    )

    test_dataset = ForecastingDataset(
        df=test_df,
        feature_cols=feature_cols,
        target_cols=target_cols,
        sequence_length=sequence_length,
        forecast_horizon=forecast_horizon,
        location_col=location_col,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,  # Ensured for Batch Norm / Stability
        num_workers=num_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
    }
