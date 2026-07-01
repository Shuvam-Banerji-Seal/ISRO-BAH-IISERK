"""Tests for bah2026.features module."""

import numpy as np

from bah2026.features.engineering import (
    extract_features_window,
    build_feature_matrix,
    get_canonical_feature_names,
    pad_features_to_canonical,
)


# ── Engineering ─────────────────────────────────────────────────────────


def test_extract_features_solexs_only():
    """SXR-only features are extracted correctly."""
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
    """SXR + HXR features are extracted correctly."""
    rng = np.random.RandomState(42)
    counts = rng.poisson(20, 3600).astype(float)
    hxr = rng.poisson(5, (3600, 5)).astype(float)
    feat = extract_features_window(counts, hxr, window=3600)
    assert feat is not None
    assert "hxr_b0_mean" in feat
    assert "hxr_hardness_ratio" in feat
    assert "soft_hard_ratio" in feat


def test_extract_features_too_short():
    """Too-short input returns None."""
    counts = np.array([1.0, 2.0])
    feat = extract_features_window(counts, window=3600)
    assert feat is None


def test_extract_features_nan_handling():
    """NaN values are handled gracefully."""
    counts = np.ones(3600) * 10.0
    counts[100] = np.nan
    feat = extract_features_window(counts, window=3600)
    assert feat is not None
    assert np.isfinite(feat["sxr_mean"])


def test_extract_features_config_params():
    """ACF and percentile features are present."""
    from bah2026.config import FEATURE_AUTOCORR_LAGS, FEATURE_PERCENTILES

    counts = np.random.poisson(20, 3600).astype(float)
    feat = extract_features_window(counts, window=3600)
    for lag in FEATURE_AUTOCORR_LAGS:
        assert f"sxr_acf_{lag}s" in feat
    for pct in FEATURE_PERCENTILES:
        assert f"sxr_p{pct}" in feat


def test_extract_features_with_precomputed():
    """Precomputed (day-level) features are integrated."""
    rng = np.random.RandomState(42)
    counts = rng.poisson(20, 3600).astype(float)
    hxr = rng.poisson(5, (3600, 10)).astype(float)
    precomputed = {
        "hk_czt1temp": 17.0,
        "hk_czt2temp": 22.0,
        "hk_cdte1temp": -40.0,
        "hk_cdte2temp": -28.0,
        "hk_czthvmon": 634.0,
        "hk_cdtehvmon": 954.0,
        "hk_czt1satctr": 5,
        "hk_cdte1pilectr": 12,
        "hxr_spectral_index_gamma": 3.5,
        "hxr_gamma_czt2": 3.2,
        "hxr_gamma_cdte1": 2.8,
        "hxr_gamma_cdte2": 2.9,
        "nonthermal_gamma": 4.1,
        "nonthermal_ec": 10.0,
        "nonthermal_n_nth": 1e4,
        "thermal_fraction": 0.65,
        "goes_xrsb_flux": 1e-6,
        "goes_xrsa_flux": 1e-7,
        "goes_xrsa_xrsb_ratio": 0.1,
        "deadtime_max_pct": 5.0,
        "bg_fraction_pct": 52.0,
        "sxr_temperature_mk": 15.0,
        "sxr_emission_measure": 1e48,
        "sxr_chi2_red": 1.2,
        "czt2_total_mean": 45.0,
        "czt2_total_max": 80.0,
        "czt2_total_std": 10.0,
        "cdte2_total_mean": 1.5,
        "cdte2_total_max": 3.0,
        "cdte2_total_std": 0.5,
    }
    feat = extract_features_window(counts, hxr, window=3600, precomputed=precomputed)
    assert feat is not None
    assert feat["hk_czt1temp"] == 17.0
    assert feat["nonthermal_gamma"] == 4.1
    assert feat["goes_xrsa_flux"] == 1e-7
    assert feat["deadtime_max_pct"] == 5.0


def test_extract_features_new_categories():
    """All new feature categories are present."""
    rng = np.random.RandomState(42)
    counts = rng.poisson(100, 3600).astype(float)
    hxr = rng.poisson(50, (3600, 10)).astype(float)
    feat = extract_features_window(counts, hxr, window=3600)
    assert feat is not None
    # Information theory
    assert "transfer_entropy_hxr_to_sxr" in feat
    assert "mutual_information_sxr_hxr" in feat
    assert "sample_entropy_sxr" in feat
    assert "sample_entropy_hxr" in feat
    assert "lagged_cross_corr" in feat
    assert "lagged_cross_corr_lag" in feat
    # Neupert
    assert "neupert_rho_mean" in feat
    assert "neupert_rho_std" in feat
    # Hardness evolution
    assert "hardness_ratio_slope" in feat
    assert "hardness_ratio_mean" in feat
    assert "hardness_ratio_std" in feat
    # QPP
    assert "qpp_detected" in feat
    assert "qpp_period" in feat
    assert "qpp_amplitude" in feat
    assert "qpp_significance" in feat


def test_canonical_feature_count():
    """Canonical feature set has expected number of features (v3: 117 basic + 62 advanced)."""
    names = get_canonical_feature_names()
    assert len(names) == 179
    # All names are unique
    assert len(names) == len(set(names))
    # All names are sorted
    assert names == sorted(names)


def test_pad_features_to_canonical():
    """Padding fills missing features with 0."""
    feat = {"sxr_mean": 10.0, "sxr_std": 2.0}
    canonical = get_canonical_feature_names()
    padded = pad_features_to_canonical(feat, canonical)
    assert len(padded) == len(canonical)
    idx_mean = canonical.index("sxr_mean")
    assert padded[idx_mean] == 10.0
    idx_missing = canonical.index("sxr_max")
    assert padded[idx_missing] == 0.0


def test_build_feature_matrix():
    """Feature matrix has correct shape and no NaN."""
    rng = np.random.RandomState(42)
    counts = rng.poisson(20, 86400).astype(float)
    X, names = build_feature_matrix(counts, lookback=3600, step=3600)
    assert X.shape[0] >= 1
    assert X.shape[1] == len(names)
    assert len(names) >= 100
    assert np.all(np.isfinite(X))


def test_build_feature_matrix_with_hxr():
    """Feature matrix with HXR has HXR features."""
    rng = np.random.RandomState(42)
    counts = rng.poisson(20, 86400).astype(float)
    hxr = rng.poisson(5, (86400, 5)).astype(float)
    X, names = build_feature_matrix(counts, hxr, lookback=3600, step=3600)
    assert X.shape[0] >= 1
    assert X.shape[1] >= 100


def test_build_feature_matrix_with_precomputed():
    """Feature matrix with precomputed features."""
    rng = np.random.RandomState(42)
    counts = rng.poisson(20, 86400).astype(float)
    hxr = rng.poisson(5, (86400, 10)).astype(float)
    precomputed = {"hk_czt1temp": 17.0, "nonthermal_gamma": 4.0}
    X, names = build_feature_matrix(
        counts, hxr, lookback=3600, step=3600, precomputed=precomputed
    )
    assert X.shape[0] >= 1
    idx_hk = names.index("hk_czt1temp")
    assert np.all(X[:, idx_hk] == 17.0)


def test_build_feature_matrix_short_input():
    """Short input produces empty matrix."""
    counts = np.ones(100) * 10.0
    X, names = build_feature_matrix(counts, lookback=3600, step=300)
    assert X.shape[0] == 0


def test_build_feature_matrix_no_nan():
    """Feature matrix has no NaN or Inf."""
    counts = np.random.poisson(20, 86400).astype(float)
    X, _ = build_feature_matrix(counts, lookback=3600, step=3600)
    assert np.all(np.isfinite(X))
