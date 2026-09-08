"""
src/models/lstm_autoencoder.py
================================
Day 6 — Sequence-based anomaly detection via an LSTM autoencoder.

Where this sits relative to the Week 1 Isolation Forest baseline
(src/models/anomaly_detector.py): the IForest baseline scores each row
independently using engineered rolling/spatial/STL features. This module
instead looks at raw *sequences* of length 30 across all engineered sensor
features, learns to reconstruct "normal" sequences, and flags a window as
anomalous when reconstruction error is unusually high. The two approaches
are complementary and both get logged to MLflow under the same experiment.

Pipeline:
    1. SequenceDataset — slides a length-30 window across a per-sensor
       feature matrix, producing (n_windows, 30, n_features) tensors.
    2. LSTMAutoencoder (nn.Module) — 2-layer LSTM encoder (hidden=128,
       dropout=0.2) -> linear bottleneck -> 2-layer LSTM decoder mirroring
       the encoder -> linear output projection back to n_features.
    3. LSTMAutoencoderModule (pl.LightningModule) — wraps the above with
       training/validation steps (MSE reconstruction loss) and optimizer
       config.
    4. train_autoencoder() — trains on NORMAL data only (is_anomaly == 0),
       then computes a reconstruction-error threshold at the 99th
       percentile of training errors.
"""

from typing import List, Tuple

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from loguru import logger
from torch.utils.data import DataLoader, Dataset

SEQUENCE_LENGTH = 30


# ===========================================================================
# 1. Windowed Dataset
# ===========================================================================


class SequenceDataset(Dataset):
    """Slides a fixed-length window across a per-sensor feature matrix.

    Windows are built independently PER SENSOR (never crossing sensor
    boundaries) so a sequence never mixes two different stations' history.

    Args:
        df: Engineered feature DataFrame, must contain `sensor_col` and
            `timestamp_col` plus all columns in `feature_cols`.
        feature_cols: Which columns form the multivariate sequence at each
            timestep.
        sensor_col: Column identifying which sensor a row belongs to.
        timestamp_col: Column used to sort each sensor's rows chronologically
            before windowing.
        sequence_length: Number of consecutive timesteps per window.

    Raises:
        ValueError: if any sensor has fewer than `sequence_length` rows and
            no sensor produces a usable window (dataset would be empty).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        sensor_col: str = "location_id",
        timestamp_col: str = "timestamp",
        sequence_length: int = SEQUENCE_LENGTH,
    ):
        self.feature_cols = feature_cols
        self.sequence_length = sequence_length
        self.windows: List[np.ndarray] = []
        self.window_meta: List[
            Tuple[str, pd.Timestamp]
        ] = []  # (sensor_id, window_end_ts)

        for sensor_id, group in df.groupby(sensor_col):
            group = group.sort_values(timestamp_col)
            values = group[feature_cols].to_numpy(dtype=np.float32)

            if len(values) < sequence_length:
                logger.debug(
                    f"Sensor {sensor_id}: only {len(values)} rows, "
                    f"need >= {sequence_length} — skipped."
                )
                continue

            timestamps = group[timestamp_col].to_numpy()
            for start in range(0, len(values) - sequence_length + 1):
                end = start + sequence_length
                self.windows.append(values[start:end])
                self.window_meta.append((sensor_id, timestamps[end - 1]))

        if not self.windows:
            raise ValueError(
                f"No sensor had >= {sequence_length} rows — SequenceDataset is empty. "
                f"Either lower sequence_length or provide more history."
            )

        logger.info(
            f"SequenceDataset built: {len(self.windows)} windows of length {sequence_length}."
        )

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Returns a (sequence_length, n_features) float32 tensor."""
        return torch.from_numpy(self.windows[idx].copy())


def make_dataloader(
    dataset: SequenceDataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """Wrap a SequenceDataset in a DataLoader.

    drop_last=True is required: BatchNorm/LSTM batch statistics and the
    Lightning module's fixed-shape assumptions break on a final partial
    batch of size 1, which is a real (not hypothetical) failure mode here.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=True,
    )


# ===========================================================================
# 2. Encoder -> bottleneck -> decoder network
# ===========================================================================


class LSTMAutoencoder(nn.Module):
    """2-layer LSTM encoder -> linear bottleneck -> 2-layer LSTM decoder.

    The decoder mirrors the encoder: same hidden size and depth, but reads
    the bottleneck vector repeated across the output sequence length rather
    than the original input, since at inference time we don't have future
    timesteps to feed it.

    Args:
        n_features: Number of input features per timestep.
        hidden_size: LSTM hidden dimension (both encoder and decoder).
        bottleneck_size: Dimension of the compressed sequence representation.
        num_layers: Number of stacked LSTM layers (both encoder and decoder).
        dropout: Dropout applied between stacked LSTM layers.
        sequence_length: Length of the input/output sequence.
    """

    def __init__(
        self,
        n_features: int,
        hidden_size: int = 128,
        bottleneck_size: int = 32,
        num_layers: int = 2,
        dropout: float = 0.2,
        sequence_length: int = SEQUENCE_LENGTH,
    ):
        super().__init__()
        self.sequence_length = sequence_length

        self.encoder_lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.to_bottleneck = nn.Linear(hidden_size, bottleneck_size)

        self.from_bottleneck = nn.Linear(bottleneck_size, hidden_size)
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.output_projection = nn.Linear(hidden_size, n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, sequence_length, n_features)

        Returns:
            Reconstructed sequence, same shape as x.
        """
        x.size(0)

        # Encode: take the final hidden state as the sequence summary
        _, (h_n, _) = self.encoder_lstm(x)
        last_hidden = h_n[-1]  # (batch, hidden_size) — top layer's final hidden state

        bottleneck = self.to_bottleneck(last_hidden)  # (batch, bottleneck_size)

        # Decode: repeat the bottleneck across every timestep since we have
        # no ground-truth future input to feed the decoder autoregressively.
        decoder_input = self.from_bottleneck(bottleneck)  # (batch, hidden_size)
        decoder_input = decoder_input.unsqueeze(1).repeat(1, self.sequence_length, 1)

        decoded, _ = self.decoder_lstm(decoder_input)
        reconstruction = self.output_projection(decoded)  # (batch, seq_len, n_features)

        return reconstruction


# ===========================================================================
# 3. Lightning wrapper
# ===========================================================================


class LSTMAutoencoderModule(pl.LightningModule):
    """LightningModule wrapping LSTMAutoencoder with MSE reconstruction loss."""

    def __init__(
        self,
        n_features: int,
        hidden_size: int = 128,
        bottleneck_size: int = 32,
        num_layers: int = 2,
        dropout: float = 0.2,
        sequence_length: int = SEQUENCE_LENGTH,
        learning_rate: float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = LSTMAutoencoder(
            n_features=n_features,
            hidden_size=hidden_size,
            bottleneck_size=bottleneck_size,
            num_layers=num_layers,
            dropout=dropout,
            sequence_length=sequence_length,
        )
        self.loss_fn = nn.MSELoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        reconstruction = self(batch)
        loss = self.loss_fn(reconstruction, batch)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        reconstruction = self(batch)
        loss = self.loss_fn(reconstruction, batch)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    def reconstruction_errors(self, x: torch.Tensor) -> torch.Tensor:
        """Per-window mean-squared reconstruction error, shape (batch,)."""
        self.eval()
        with torch.no_grad():
            reconstruction = self(x)
            errors = torch.mean((reconstruction - x) ** 2, dim=(1, 2))
        return errors


# ===========================================================================
# 4. Training entrypoint: normal-only training + 99th percentile threshold
# ===========================================================================


def validate_dataset_readiness(
    df: pd.DataFrame,
    sensor_col: str = "location_id",
    sequence_length: int = 12,
    min_sequences_per_sensor: int = 5,
) -> bool:
    """Validates if sensors have accumulated enough consecutive rows for sequences."""
    if df.empty or sensor_col not in df.columns:
        logger.error(f"Dataset is empty or missing '{sensor_col}' column.")
        return False

    min_required_rows = sequence_length + min_sequences_per_sensor
    counts = df[sensor_col].value_counts()
    ready_sensors = counts[counts >= min_required_rows]
    insufficient_sensors = counts[counts < min_required_rows]

    logger.info("=== Pre-flight Dataset Readiness Check ===")
    logger.info(f"Target sequence_length: {sequence_length} steps")
    logger.info(f"Sensors ready: {len(ready_sensors)} / {len(counts)}")

    if len(ready_sensors) == 0:
        logger.error(
            f"❌ Insufficient time-series history! No individual sensor has reached "
            f"the minimum threshold of {min_required_rows} rows (sequence_length={sequence_length} + buffer).\n"
            f"Top sensor row counts currently:\n{counts.head(5).to_string()}\n"
            f"-> Keep the producer/consumer running to collect more telemetry before re-training."
        )
        return False

    if len(insufficient_sensors) > 0:
        logger.warning(
            f"⚠️ {len(insufficient_sensors)} sensor(s) have fewer than {min_required_rows} rows "
            f"and will be skipped during window generation."
        )

    return True


def train_autoencoder(
    df: pd.DataFrame,
    feature_cols: List[str],
    label_col: str = "is_anomaly",
    sensor_col: str = "location_id",
    timestamp_col: str = "timestamp",
    sequence_length: int = SEQUENCE_LENGTH,
    batch_size: int = 32,
    max_epochs: int = 20,
    val_split: float = 0.1,
) -> Tuple[LSTMAutoencoderModule, float]:
    """Train the LSTM autoencoder on NORMAL windows only, then compute a
    reconstruction-error anomaly threshold at the 99th percentile.
    """
    # Fallback column mapping if needed
    if sensor_col not in df.columns and "location" in df.columns:
        df[sensor_col] = df["location"]

    # Pre-flight readiness check
    if not validate_dataset_readiness(
        df, sensor_col=sensor_col, sequence_length=sequence_length
    ):
        raise ValueError(
            "Dataset pre-flight check failed: Not enough telemetry rows per sensor."
        )

    if label_col in df.columns:
        normal_df = df[df[label_col] == 0].copy()
        logger.info(
            f"Training on normal-only data: {len(normal_df)}/{len(df)} rows "
            f"({((len(df) - len(normal_df)) / len(df) * 100):.1f}% flagged anomalous)."
        )
    else:
        normal_df = df.copy()
        logger.warning(
            f"'{label_col}' not found in df — training on all rows as if normal."
        )

        # ... rest of your existing function continues here ...

        normal_df = df.copy()

    dataset = SequenceDataset(
        normal_df,
        feature_cols=feature_cols,
        sensor_col=sensor_col,
        timestamp_col=timestamp_col,
        sequence_length=sequence_length,
    )

    n_val = max(1, int(len(dataset) * val_split))
    n_train = len(dataset) - n_val
    train_subset, val_subset = torch.utils.data.random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42)
    )

    train_loader = make_dataloader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = make_dataloader(val_subset, batch_size=batch_size, shuffle=False)

    module = LSTMAutoencoderModule(
        n_features=len(feature_cols), sequence_length=sequence_length
    )

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        enable_progress_bar=True,
        logger=False,  # MLflow logging is handled by the caller (pipeline.py)
        enable_checkpointing=False,
    )
    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # Compute the 99th-percentile threshold over ALL normal training windows
    # (train + val), since the threshold characterizes "what normal looks
    # like," not model generalization — that's what val_loss is for.
    full_loader = make_dataloader(dataset, batch_size=batch_size, shuffle=False)
    all_errors = []
    module.eval()
    with torch.no_grad():
        for batch in full_loader:
            errors = module.reconstruction_errors(batch)
            all_errors.append(errors.numpy())
    all_errors = np.concatenate(all_errors)

    threshold = float(np.percentile(all_errors, 99))
    logger.success(
        f"Training complete. Reconstruction error threshold (99th pct): {threshold:.6f} "
        f"(min={all_errors.min():.6f}, max={all_errors.max():.6f})"
    )

    return module, threshold


def score_sequences(
    module: LSTMAutoencoderModule,
    df: pd.DataFrame,
    feature_cols: List[str],
    threshold: float,
    sensor_col: str = "location_id",
    timestamp_col: str = "timestamp",
    sequence_length: int = SEQUENCE_LENGTH,
    batch_size: int = 32,
) -> pd.DataFrame:
    """Score new windows against a trained autoencoder + threshold.

    Returns:
        DataFrame with one row per window: sensor_col, timestamp_col
        (window end), reconstruction_error, is_anomaly (error > threshold).
    """
    dataset = SequenceDataset(
        df,
        feature_cols=feature_cols,
        sensor_col=sensor_col,
        timestamp_col=timestamp_col,
        sequence_length=sequence_length,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    module.eval()
    all_errors = []
    with torch.no_grad():
        for batch in loader:
            errors = module.reconstruction_errors(batch)
            all_errors.append(errors.numpy())
    all_errors = np.concatenate(all_errors)

    results = pd.DataFrame(dataset.window_meta, columns=[sensor_col, timestamp_col])
    results["reconstruction_error"] = all_errors
    results["is_anomaly"] = (results["reconstruction_error"] > threshold).astype(int)

    n_flagged = int(results["is_anomaly"].sum())
    logger.info(
        f"Scored {len(results)} windows — {n_flagged} above the {threshold:.6f} threshold."
    )

    return results
