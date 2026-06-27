"""Tests for bah2026.models module (nowcasting + forecasting)."""

import numpy as np
import pytest

from bah2026.models.nowcasting import (
    background_subtract_simple,
    detect_flares_threshold,
    detect_flares_bayesian_blocks,
    classify_flare_goes,
)
from bah2026.config import (
    NOWCAST_THRESHOLD_SIGMA,
    NOWCAST_MIN_DURATION_SEC,
    NOWCAST_BAYESIAN_BLOCKS_SIGMA,
    NOWCAST_BACKGROUND_WINDOW_SEC,
)

# Calibration: solexs_counts_to_irradiance_simple(counts) = counts * 2.5e-8 W/m²
# GOES thresholds: X >= 1e-4, M >= 1e-5, C >= 1e-6, B >= 1e-7
# So: X at 4000 cts, M at 400 cts, C at 40 cts, B at 4 cts
_CAL = 2.5e-8  # FLUX_PER_COUNT in calibration.py (GOES-validated)


# ── Nowcasting tests ────────────────────────────────────────────────────


def test_background_subtract_simple():
    counts = np.ones(1000) * 10.0
    counts[500:520] = 100.0
    bg, residual = background_subtract_simple(counts, window=100)
    assert bg.shape == counts.shape
    assert residual[510] > 20.0


def test_background_subtract_simple_default():
    counts = np.random.poisson(10, 1000).astype(float)
    bg, residual = background_subtract_simple(counts)
    assert bg.shape == counts.shape


def test_detect_flares_threshold():
    rng = np.random.RandomState(42)
    counts = rng.normal(0, 1, 10000).astype(float)
    counts[5000:5060] = 50.0
    time = np.arange(10000, dtype=float)
    events = detect_flares_threshold(counts, time, sigma=5.0, min_duration_sec=5)
    assert len(events) >= 1
    assert events[0]["peak_flux"] > 30.0


def test_detect_flares_threshold_no_event():
    rng = np.random.RandomState(42)
    counts = rng.normal(0, 0.1, 1000)
    time = np.arange(1000, dtype=float)
    events = detect_flares_threshold(counts, time, sigma=10.0, min_duration_sec=100)
    assert len(events) == 0


def test_detect_flares_threshold_config_defaults():
    rng = np.random.RandomState(42)
    counts = rng.normal(0, 1, 10000).astype(float)
    counts[5000:5050] = 50.0
    time = np.arange(10000, dtype=float)
    events = detect_flares_threshold(counts, time)
    assert isinstance(events, list)


def test_detect_flares_threshold_config_sigma():
    rng = np.random.RandomState(42)
    counts = rng.normal(0, 1, 10000).astype(float)
    counts[5000:5030] = 30.0
    time = np.arange(10000, dtype=float)
    events_high = detect_flares_threshold(counts, time, sigma=6.0, min_duration_sec=5)
    events_low = detect_flares_threshold(counts, time, sigma=2.0, min_duration_sec=5)
    assert len(events_low) >= len(events_high)


def test_detect_flares_bayesian_blocks():
    rng = np.random.RandomState(42)
    counts = rng.normal(0, 0.1, 200).astype(float)
    counts[50:150] = 100.0
    time = np.arange(200, dtype=float)
    events = detect_flares_bayesian_blocks(counts, time, threshold_sigma=1.0)
    assert isinstance(events, list)


def test_detect_flares_bayesian_blocks_no_event():
    rng = np.random.RandomState(42)
    counts = rng.normal(0, 0.1, 200).astype(float)
    time = np.arange(200, dtype=float)
    events = detect_flares_bayesian_blocks(counts, time, threshold_sigma=10.0)
    assert len(events) == 0


def test_classify_flare_goes():
    # Calibrated (2.5e-8): ~4000 cts → X, ~400 → M, ~40 → C, ~4 → B
    # Add small margins to avoid float-boundary issues (4000*2.5e-8 ≈ 9.999...e-5)
    assert classify_flare_goes(4100.0) == "X"
    assert classify_flare_goes(410.0) == "M"
    assert classify_flare_goes(41.0) == "C"
    assert classify_flare_goes(5.0) == "B"
    assert classify_flare_goes(1.0) == "A"


def test_classify_flare_goes_boundary():
    flux_x = 1e-4 / _CAL
    assert classify_flare_goes(flux_x) == "X"
    flux_m = 1e-5 / _CAL
    assert classify_flare_goes(flux_m) == "M"


# ── Forecasting model tests ─────────────────────────────────────────────


@pytest.fixture
def sample_data():
    rng = np.random.RandomState(42)
    X = rng.randn(200, 10).astype(np.float32)
    y = rng.randint(0, 2, 200)
    y[:30] = 1
    return X, y


def test_lgbm_forecaster(sample_data):
    X, y = sample_data
    from bah2026.models.forecasting import FlareForecasterLightGBM

    model = FlareForecasterLightGBM(n_estimators=50, scale_pos_weight=5.0)
    model.fit(X[:150], y[:150])
    prob = model.predict_proba(X[150:])
    assert prob.shape == (50,)
    assert np.all(prob >= 0)
    assert np.all(prob <= 1)
    imp = model.feature_importance()
    assert len(imp) == 10


def test_xgb_forecaster(sample_data):
    X, y = sample_data
    from bah2026.models.forecasting import FlareForecasterXGBoost

    model = FlareForecasterXGBoost(n_estimators=50, scale_pos_weight=5.0)
    model.fit(X[:150], y[:150])
    prob = model.predict_proba(X[150:])
    assert prob.shape == (50,)
    assert np.all(prob >= 0)
    assert np.all(prob <= 1)


def test_catboost_forecaster(sample_data):
    X, y = sample_data
    from bah2026.models.forecasting import FlareForecasterCatBoost

    model = FlareForecasterCatBoost(iterations=50)
    model.fit(X[:150], y[:150])
    prob = model.predict_proba(X[150:])
    assert prob.shape == (50,)
    assert np.all(prob >= 0)
    assert np.all(prob <= 1)


def test_cnnlstm_forecaster(sample_data):
    X, y = sample_data
    X_seq = np.random.randn(50, 12, 100).astype(np.float32)
    y_seq = np.random.randint(0, 2, 50).astype(np.float32)
    from bah2026.models.forecasting import FlareForecasterCNNLSTM

    model = FlareForecasterCNNLSTM(input_len=100, n_channels=12)
    model.fit(X_seq, y_seq, epochs=2, batch_size=16)
    prob = model.predict_proba(X_seq)
    assert prob.shape == (50,)
    assert np.all(prob >= 0)
    assert np.all(prob <= 1)
