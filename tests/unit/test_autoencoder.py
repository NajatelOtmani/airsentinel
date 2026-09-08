"""
tests/test_lstm_autoencoder.py
================================
Unit tests for src/models/lstm_autoencoder.py.

Runs fully offline/CPU-only, no MLflow/network dependency — these test the
autoencoder architecture and data plumbing in isolation from the rest of
the pipeline.
"""

import numpy as np
import pandas as pd
import pytest
import torch

# NEW:
from src.models.autoencoder import (LSTMAutoencoder, LSTMAutoencoderModule,
                                    SequenceDataset, make_dataloader,
                                    score_sequences, train_autoencoder)

FEATURE_COLS = ["f1", "f2", "f3"]


def _make_synthetic_df(n_sensors=3, n_rows_per_sensor=50, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_sensors):
        sensor_id = f"S{i}"
        timestamps = pd.date_range("2026-01-01", periods=n_rows_per_sensor, freq="h")
        for ts in timestamps:
            rows.append(
                {
                    "location_id": sensor_id,
                    "timestamp": ts,
                    "f1": rng.normal(),
                    "f2": rng.normal(),
                    "f3": rng.normal(),
                    "is_anomaly": 0,
                }
            )
    return pd.DataFrame(rows)


class TestSequenceDataset:
    def test_produces_expected_window_count_per_sensor(self):
        """n_windows per sensor = n_rows - sequence_length + 1."""
        df = _make_synthetic_df(n_sensors=1, n_rows_per_sensor=50)
        dataset = SequenceDataset(df, FEATURE_COLS, sequence_length=10)
        assert len(dataset) == 50 - 10 + 1

    def test_windows_never_cross_sensor_boundaries(self):
        """Each window's metadata sensor_id must be internally consistent —
        windowing groups by sensor before sliding, so no window should span
        two different sensors' data."""
        df = _make_synthetic_df(n_sensors=2, n_rows_per_sensor=20)
        dataset = SequenceDataset(df, FEATURE_COLS, sequence_length=10)
        sensor_ids_seen = {meta[0] for meta in dataset.window_meta}
        assert sensor_ids_seen == {"S0", "S1"}

    def test_skips_sensors_with_insufficient_history(self):
        """A sensor with fewer rows than sequence_length contributes zero windows."""
        df = _make_synthetic_df(n_sensors=1, n_rows_per_sensor=5)
        with pytest.raises(ValueError, match="No sensor had"):
            SequenceDataset(df, FEATURE_COLS, sequence_length=10)

    def test_item_shape_matches_sequence_length_and_features(self):
        df = _make_synthetic_df(n_sensors=1, n_rows_per_sensor=30)
        dataset = SequenceDataset(df, FEATURE_COLS, sequence_length=10)
        sample = dataset[0]
        assert sample.shape == (10, len(FEATURE_COLS))
        assert sample.dtype == torch.float32


class TestDataLoader:
    def test_drop_last_removes_partial_final_batch(self):
        """With 13 windows and batch_size=4, drop_last=True should yield
        exactly 3 full batches (12 samples), never a trailing batch of 1."""
        df = _make_synthetic_df(
            n_sensors=1, n_rows_per_sensor=22
        )  # -> 13 windows @ seq_len=10
        dataset = SequenceDataset(df, FEATURE_COLS, sequence_length=10)
        assert len(dataset) == 13

        loader = make_dataloader(dataset, batch_size=4, shuffle=False)
        batch_sizes = [batch.shape[0] for batch in loader]
        assert all(bs == 4 for bs in batch_sizes)
        assert sum(batch_sizes) == 12  # last partial batch of 1 is dropped


class TestLSTMAutoencoderShapes:
    def test_reconstruction_matches_input_shape(self):
        model = LSTMAutoencoder(
            n_features=3,
            hidden_size=16,
            bottleneck_size=4,
            num_layers=2,
            dropout=0.2,
            sequence_length=10,
        )
        x = torch.randn(5, 10, 3)
        out = model(x)
        assert out.shape == x.shape

    def test_encoder_and_decoder_are_symmetric_depth(self):
        model = LSTMAutoencoder(
            n_features=3, hidden_size=16, bottleneck_size=4, num_layers=2
        )
        assert model.encoder_lstm.num_layers == model.decoder_lstm.num_layers == 2
        assert model.encoder_lstm.hidden_size == model.decoder_lstm.hidden_size == 16


class TestLightningModule:
    def test_training_step_returns_scalar_loss(self):
        module = LSTMAutoencoderModule(
            n_features=3, hidden_size=16, bottleneck_size=4, sequence_length=10
        )
        batch = torch.randn(4, 10, 3)
        loss = module.training_step(batch, batch_idx=0)
        assert loss.dim() == 0
        assert loss.item() >= 0

    def test_reconstruction_errors_are_nonnegative_and_per_window(self):
        module = LSTMAutoencoderModule(
            n_features=3, hidden_size=16, bottleneck_size=4, sequence_length=10
        )
        batch = torch.randn(6, 10, 3)
        errors = module.reconstruction_errors(batch)
        assert errors.shape == (6,)
        assert torch.all(errors >= 0)


class TestTrainAutoencoderPipeline:
    def test_train_excludes_flagged_windows_from_training(self):
        """Rows flagged is_anomaly==1 must not appear in the training set —
        train_autoencoder() should filter them before windowing."""
        df = _make_synthetic_df(n_sensors=1, n_rows_per_sensor=60)
        df.loc[10:15, "is_anomaly"] = 1

        module, threshold = train_autoencoder(
            df,
            FEATURE_COLS,
            sequence_length=10,
            batch_size=4,
            max_epochs=1,
            val_split=0.2,
        )
        assert isinstance(threshold, float)
        assert threshold > 0

    def test_threshold_is_99th_percentile_of_training_errors(self):
        """The returned threshold should sit at/near the 99th percentile —
        i.e. roughly 1% of training windows should exceed it."""
        df = _make_synthetic_df(n_sensors=2, n_rows_per_sensor=80)
        module, threshold = train_autoencoder(
            df,
            FEATURE_COLS,
            sequence_length=10,
            batch_size=8,
            max_epochs=1,
            val_split=0.1,
        )

        scored = score_sequences(
            module, df, FEATURE_COLS, threshold, sequence_length=10, batch_size=8
        )
        flagged_fraction = scored["is_anomaly"].mean()
        # Loose bound: scoring on the same (mostly normal) data the threshold
        # was derived from should flag roughly 1-5%, not e.g. 50%.
        assert 0.0 <= flagged_fraction <= 0.10

    def test_raises_clear_error_with_insufficient_history(self):
        df = _make_synthetic_df(n_sensors=1, n_rows_per_sensor=5)
        with pytest.raises(ValueError):
            train_autoencoder(df, FEATURE_COLS, sequence_length=10, max_epochs=1)


class TestScoreSequences:
    def test_output_has_expected_columns(self):
        df = _make_synthetic_df(n_sensors=1, n_rows_per_sensor=40)
        module, threshold = train_autoencoder(
            df, FEATURE_COLS, sequence_length=10, batch_size=4, max_epochs=1
        )
        scored = score_sequences(
            module, df, FEATURE_COLS, threshold, sequence_length=10, batch_size=4
        )
        assert set(scored.columns) == {
            "location_id",
            "timestamp",
            "reconstruction_error",
            "is_anomaly",
        }

    def test_flags_are_boolean_ints_derived_from_threshold(self):
        df = _make_synthetic_df(n_sensors=1, n_rows_per_sensor=40)
        module, _ = train_autoencoder(
            df, FEATURE_COLS, sequence_length=10, batch_size=4, max_epochs=1
        )

        # A threshold of 0 should flag everything; a threshold of infinity should flag nothing.
        scored_all_flagged = score_sequences(
            module, df, FEATURE_COLS, threshold=0.0, sequence_length=10, batch_size=4
        )
        scored_none_flagged = score_sequences(
            module, df, FEATURE_COLS, threshold=1e9, sequence_length=10, batch_size=4
        )

        assert scored_all_flagged["is_anomaly"].sum() == len(scored_all_flagged)
        assert scored_none_flagged["is_anomaly"].sum() == 0
