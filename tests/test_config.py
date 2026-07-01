"""Tests for bah2026.config module."""

import json
import os
from pathlib import Path

from bah2026.config import (
    PROJECT_ROOT,
    DATA_ROOT,
    OUTPUT_ROOT,
    SOLEXS_ENERGY_KEV,
    SOLEXS_CHANNELS,
    SOLEXS_CADENCE_SEC,
    SOLEXS_ROWS_PER_DAY,
    CZT_BANDS,
    CDTE_BANDS,
    NOWCAST_THRESHOLD_SIGMA,
    NOWCAST_MIN_DURATION_SEC,
    NOWCAST_BAYESIAN_BLOCKS_SIGMA,
    NOWCAST_WAVELET_SIGMA,
    NOWCAST_BACKGROUND_WINDOW_SEC,
    FEATURE_LOOKBACK_SEC,
    FEATURE_STEP_SEC,
    FEATURE_FORECAST_WINDOW_SEC,
    FEATURE_SPECTRAL_ENTROPY_NPERSEG,
    FEATURE_AUTOCORR_LAGS,
    FEATURE_PERCENTILES,

    LGBM_N_ESTIMATORS,
    LGBM_LEARNING_RATE,
    LGBM_MAX_DEPTH,
    XGB_N_ESTIMATORS,
    XGB_LEARNING_RATE,
    XGB_MAX_DEPTH,
    CNNLSTM_N_FEATURES,
    CNNLSTM_INPUT_LEN,
    CNNLSTM_N_CHANNELS,
    CNNLSTM_LR,
    N_WORKERS,
    ensure_output_dirs,
    load_config,
    save_default_config,
)


def test_project_root():
    assert PROJECT_ROOT.exists()
    assert (PROJECT_ROOT / "pyproject.toml").exists()
    assert (PROJECT_ROOT / "src" / "bah2026").exists()


def test_data_root():
    assert "processed" in str(DATA_ROOT)


def test_solexs_constants():
    assert SOLEXS_ENERGY_KEV == (2.0, 22.0)
    assert SOLEXS_CHANNELS == 340
    assert SOLEXS_CADENCE_SEC == 1
    assert SOLEXS_ROWS_PER_DAY == 86_400
    # SOLEXS_TO_GOES_SCALE was removed — use data.calibration instead


def test_czt_bands():
    assert len(CZT_BANDS) == 5
    for name, (elo, ehi) in CZT_BANDS.items():
        assert "CZT1" in name
        assert elo < ehi
        assert elo >= 18
        assert ehi <= 160


def test_cdte_bands():
    assert len(CDTE_BANDS) == 5
    for name, (elo, ehi) in CDTE_BANDS.items():
        assert "CDTE1" in name
        assert elo < ehi
        assert elo >= 1.8
        assert ehi <= 90


def test_nowcast_params():
    assert NOWCAST_THRESHOLD_SIGMA > 0
    assert NOWCAST_MIN_DURATION_SEC >= 5
    assert NOWCAST_BAYESIAN_BLOCKS_SIGMA > 0
    assert NOWCAST_WAVELET_SIGMA > 0
    assert NOWCAST_BACKGROUND_WINDOW_SEC >= 60


def test_feature_params():
    assert FEATURE_LOOKBACK_SEC > 0
    assert FEATURE_STEP_SEC > 0
    assert FEATURE_FORECAST_WINDOW_SEC > 0
    assert FEATURE_SPECTRAL_ENTROPY_NPERSEG >= 64
    assert len(FEATURE_AUTOCORR_LAGS) > 0
    assert len(FEATURE_PERCENTILES) > 0


def test_lgbm_params():
    assert LGBM_N_ESTIMATORS >= 100
    assert 0 < LGBM_LEARNING_RATE <= 0.5
    assert LGBM_MAX_DEPTH >= 3


def test_xgb_params():
    assert XGB_N_ESTIMATORS >= 100
    assert 0 < XGB_LEARNING_RATE <= 0.5
    assert XGB_MAX_DEPTH >= 3


def test_cnnlstm_params():
    assert CNNLSTM_N_FEATURES > 0
    assert CNNLSTM_INPUT_LEN > 0
    assert CNNLSTM_N_CHANNELS > 0
    assert CNNLSTM_LR > 0


def test_n_workers():
    assert N_WORKERS >= 1


def test_ensure_output_dirs():
    ensure_output_dirs()
    for sub in ["overview", "spectral", "nowcast", "forecast", "statistics"]:
        assert (OUTPUT_ROOT / "plots" / sub).exists()


def test_save_and_load_config():
    from bah2026.config import PROJECT_ROOT

    save_default_config()
    cfg_path = PROJECT_ROOT / "bah2026_config.json"
    assert cfg_path.exists()

    cfg = json.loads(cfg_path.read_text())
    assert "data_root" in cfg
    assert "n_workers" in cfg
    assert "nowcast" in cfg
    assert "features" in cfg
    assert "forecasting" in cfg

    cfg_path.unlink()
