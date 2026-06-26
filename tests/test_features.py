"""Tests for bah2026.features module."""

import numpy as np

from bah2026.features.engineering import extract_features_window, build_feature_matrix


def test_extract_features_solexs_only():
    rng = np.random.RandomState(42)
    counts = rng.poisson(20, 3600).astype(float)
    feat = extract_features_window(counts, window=3600)
    assert feat is not None
    assert "sxr_mean" in feat
    assert "sxr_std" in feat
    assert "sxr_max" in feat
    assert "sxr_skew" in feat
    assert "sxr_spec_entropy" in feat
    assert feat["sxr_mean"] > 0


def test_extract_features_with_hel1os():
    rng = np.random.RandomState(42)
    counts = rng.poisson(20, 3600).astype(float)
    hxr = rng.poisson(5, (3600, 5)).astype(float)
    feat = extract_features_window(counts, hxr, window=3600)
    assert feat is not None
    assert "hxr_b0_mean" in feat
    assert "hxr_hardness_ratio" in feat
    assert "soft_hard_ratio" in feat


def test_extract_features_too_short():
    counts = np.array([1.0, 2.0])
    feat = extract_features_window(counts, window=3600)
    assert feat is None


def test_extract_features_nan_handling():
    counts = np.ones(3600) * 10.0
    counts[100] = np.nan
    feat = extract_features_window(counts, window=3600)
    assert feat is not None
    assert np.isfinite(feat["sxr_mean"])


def test_extract_features_config_params():
    from bah2026.config import FEATURE_AUTOCORR_LAGS, FEATURE_PERCENTILES
    counts = np.random.poisson(20, 3600).astype(float)
    feat = extract_features_window(counts, window=3600)
    for lag in FEATURE_AUTOCORR_LAGS:
        assert f"sxr_acf_{lag}s" in feat
    for pct in FEATURE_PERCENTILES:
        assert f"sxr_p{pct}" in feat


def test_build_feature_matrix():
    rng = np.random.RandomState(42)
    counts = rng.poisson(20, 86400).astype(float)
    X, names = build_feature_matrix(counts, lookback=3600, step=3600)
    assert X.shape[0] >= 1
    assert X.shape[1] == len(names)
    assert len(names) > 10


def test_build_feature_matrix_with_hxr():
    rng = np.random.RandomState(42)
    counts = rng.poisson(20, 86400).astype(float)
    hxr = rng.poisson(5, (86400, 5)).astype(float)
    X, names = build_feature_matrix(counts, hxr, lookback=3600, step=3600)
    assert X.shape[0] >= 1
    assert any("hxr" in n for n in names)


def test_build_feature_matrix_short_input():
    counts = np.ones(100) * 10.0
    X, names = build_feature_matrix(counts, lookback=3600, step=300)
    assert X.shape[0] == 0


def test_build_feature_matrix_no_nan():
    counts = np.random.poisson(20, 86400).astype(float)
    X, _ = build_feature_matrix(counts, lookback=3600, step=3600)
    assert np.all(np.isfinite(X))
