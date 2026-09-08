"""
AirSentinel: Air Quality Anomaly Detection & Forecasting Pipeline.

This module implements a Transformer-based multi-step time-series forecasting model
and wraps it inside a PyTorch Lightning LightningModule for standardized training,
validation, logging, and model lifecycle management.
"""

import math
from typing import Any, Dict, Tuple

import lightning.pytorch as pl
import torch
import torch.nn as nn
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from loguru import logger
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau


class PositionalEncoding(nn.Module):
    """Injects positional information into input embeddings using sinusoidal functions.

    Args:
        d_model (int): Hidden dimension size of the model embeddings.
        max_len (int, optional): Maximum sequence length supported. Defaults to 5000.
        dropout (float, optional): Dropout probability applied after positional addition. Defaults to 0.1.
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Adds positional encoding to the input tensor.

        Args:
            x (torch.Tensor): Tensor of shape (batch_size, sequence_length, d_model).

        Returns:
            torch.Tensor: Tensor with positional encoding added, same shape as input.
        """
        seq_len = x.size(1)
        if seq_len > self.pe.size(1):
            raise ValueError(
                f"Input sequence length {seq_len} exceeds maximum supported length {self.pe.size(1)}"
            )
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


class TransformerForecaster(nn.Module):
    """Transformer Encoder-based multi-step time series forecaster.

    Maps input multivariate sequences (batch_size, sequence_length, num_features)
    to future forecasts (batch_size, forecast_horizon, num_features).

    Args:
        num_features (int): Number of input sensor features per timestep.
        forecast_horizon (int): Number of future timesteps to predict.
        d_model (int, optional): Dimension of encoder representation. Defaults to 64.
        nhead (int, optional): Number of attention heads. Defaults to 4.
        num_layers (int, optional): Number of Transformer encoder layers. Defaults to 2.
        dim_feedforward (int, optional): Dimension of feedforward network. Defaults to 128.
        dropout (float, optional): Dropout probability. Defaults to 0.1.
    """

    def __init__(
        self,
        num_features: int,
        forecast_horizon: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.forecast_horizon = forecast_horizon
        self.d_model = d_model

        if d_model % nhead != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by nhead ({nhead})."
            )

        # Feature projection into embedding dimension
        self.input_projection = nn.Linear(num_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model=d_model, dropout=dropout)

        # Transformer Encoder Stack
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
        )

        # Multi-step Forecast Head: Projects latent representation to forecast matrix
        self.head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(dim_feedforward, forecast_horizon * num_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for multi-step forecasting.

        Args:
            x (torch.Tensor): Input sequence of shape (batch_size, sequence_length, num_features).

        Returns:
            torch.Tensor: Predicted values of shape (batch_size, forecast_horizon, num_features).
        """
        batch_size = x.size(0)

        # Projection & Positional Encoding: (B, L, F) -> (B, L, d_model)
        h = self.input_projection(x)
        h = self.pos_encoder(h)

        # Transformer Encoding: (B, L, d_model) -> (B, L, d_model)
        encoded = self.transformer_encoder(h)

        # Global Pooling across time sequence (B, d_model)
        latent = encoded.mean(dim=1)

        # Map to forecast output: (B, forecast_horizon * num_features)
        out = self.head(latent)

        # Reshape to (B, forecast_horizon, num_features)
        return out.view(batch_size, self.forecast_horizon, self.num_features)


class AirSentinelTransformerModule(pl.LightningModule):
    """PyTorch LightningModule wrapper for AirSentinel Transformer Forecaster.

    Handles training, validation, metric calculation, optimization, and MLflow logging.

    Args:
        num_features (int): Number of input sensor features per timestep.
        forecast_horizon (int, optional): Number of future timesteps to predict. Defaults to 3.
        d_model (int, optional): Dimension of model embeddings. Defaults to 64.
        nhead (int, optional): Number of multi-head attention heads. Defaults to 4.
        num_layers (int, optional): Number of encoder layers. Defaults to 2.
        dim_feedforward (int, optional): Dimension of feedforward network. Defaults to 128.
        dropout (float, optional): Dropout rate. Defaults to 0.1.
        lr (float, optional): Initial learning rate for Adam optimizer. Defaults to 1e-3.
        lr_factor (float, optional): Factor by which learning rate is reduced on plateau. Defaults to 0.5.
        lr_patience (int, optional): Number of epochs with no improvement before reducing LR. Defaults to 5.
    """

    def __init__(
        self,
        num_features: int,
        forecast_horizon: int = 3,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        lr: float = 1e-3,
        lr_factor: float = 0.5,
        lr_patience: int = 5,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.lr = lr
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience

        self.model = TransformerForecaster(
            num_features=num_features,
            forecast_horizon=forecast_horizon,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

        self.criterion = nn.MSELoss()
        logger.info(
            f"AirSentinelTransformerModule initialized | num_features={num_features}, "
            f"forecast_horizon={forecast_horizon}, d_model={d_model}, nhead={nhead}, lr={lr}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Delegates directly to underlying PyTorch Transformer module."""
        return self.model(x)

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Executes single training step.

        Batch layout:
            x (torch.Tensor): Historical sequence (batch_size, sequence_length, num_features).
            y (torch.Tensor): Ground truth future target (batch_size, forecast_horizon, num_features).
        """
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)

        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Executes single validation step."""
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)

        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configures Adam optimizer and ReduceLROnPlateau scheduler."""
        optimizer = Adam(self.parameters(), lr=self.lr)
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=self.lr_factor,
            patience=self.lr_patience,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def get_mlflow_hyperparams(self) -> Dict[str, Any]:
        """Returns hyperparameter dictionary formatted for MLflow logging."""
        return dict(self.hparams)


def build_trainer_and_callbacks(
    max_epochs: int = 20,
    early_stopping_patience: int = 10,
    gradient_clip_val: float = 1.0,
    checkpoint_dir: str = "checkpoints",
) -> Tuple[pl.Trainer, Dict[str, Any]]:
    """Utility helper to construct a standardized Lightning Trainer with early stopping,
    checkpointing, and gradient clipping as specified in the AirSentinel roadmap.

    Args:
        max_epochs (int, optional): Maximum training epochs. Defaults to 20.
        early_stopping_patience (int, optional): Early stopping patience epochs. Defaults to 10.
        gradient_clip_val (float, optional): Gradient norm clipping limit. Defaults to 1.0.
        checkpoint_dir (str, optional): Output directory for checkpoints. Defaults to "checkpoints".

    Returns:
        Tuple[pl.Trainer, Dict[str, Any]]: Configured Lightning Trainer and callbacks dict.
    """
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=early_stopping_patience,
        mode="min",
        verbose=True,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="transformer-air-sentinel-{epoch:02d}-{val_loss:.4f}",
        save_top_k=1,
        monitor="val_loss",
        mode="min",
    )

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        gradient_clip_val=gradient_clip_val,
        callbacks=[early_stop_callback, checkpoint_callback],
        enable_checkpointing=True,
        logger=True,
    )

    callbacks_info = {
        "early_stopping": early_stop_callback,
        "checkpoint": checkpoint_callback,
    }

    return trainer, callbacks_info
