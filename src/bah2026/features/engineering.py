"""Feature engineering: statistical, spectral, and cross-instrument features.

v2: Expanded with information-theory, Neupert, hardness evolution, HK,
GOES XRS-A, all 4 spectra, non-thermal, QPP, and correction features.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import skew, kurtosis
from scipy.signal import welch as scipy_welch

from bah2026.config import (
    FEATURE_AUTOCORR_LAGS,
    FEATURE_PERCENTILES,
    FEATURE_SPECTRAL_ENTROPY_NPERSEG,
)

# ── Canonical feature sets ──────────────────────────────────────────────

# SXR statistical features (15)
_SXR_FEATURES = [
    "sxr_abs_slope",
    "sxr_cv",
    "sxr_fall_rate",
    "sxr_iqr",
    "sxr_kurtosis",
    "sxr_max",
    "sxr_mean",
    "sxr_median",
    "sxr_min",
    "sxr_peak_freq",
    "sxr_range",
    "sxr_rise_rate",
    "sxr_skew",
    "sxr_spec_entropy",
    "sxr_std",
]
# HXR band features (30: 10 bands × 3 stats)
_HXR_FEATURES = [
    # CZT1 bands (20-160 keV)
    "hxr_b0_max",
    "hxr_b0_mean",
    "hxr_b0_std",
    "hxr_b1_max",
    "hxr_b1_mean",
    "hxr_b1_std",
    "hxr_b2_max",
    "hxr_b2_mean",
    "hxr_b2_std",
    "hxr_b3_max",
    "hxr_b3_mean",
    "hxr_b3_std",
    "hxr_b4_max",
    "hxr_b4_mean",
    "hxr_b4_std",
    # CdTe1 bands (1.8-90 keV)
    "hxr_b5_max",
    "hxr_b5_mean",
    "hxr_b5_std",
    "hxr_b6_max",
    "hxr_b6_mean",
    "hxr_b6_std",
    "hxr_b7_max",
    "hxr_b7_mean",
    "hxr_b7_std",
    "hxr_b8_max",
    "hxr_b8_mean",
    "hxr_b8_std",
    "hxr_b9_max",
    "hxr_b9_mean",
    "hxr_b9_std",
    # Derived
    "hxr_hardness_ratio",
    "hxr_total_mean",
    "soft_hard_ratio",
    "cdte_thermal_ratio",
    "cdte_boundary_ratio",
]
# SoLEXS PI spectral features (3)
_PI_FEATURES = [
    "sxr_temperature_mk",
    "sxr_emission_measure",
    "sxr_chi2_red",
]
# HEL1OS spectral features from all 4 detectors (4)
_HEL1OS_SPEC_FEATURES = [
    "hxr_spectral_index_gamma",  # CZT1
    "hxr_gamma_czt2",  # CZT2
    "hxr_gamma_cdte1",  # CdTe1
    "hxr_gamma_cdte2",  # CdTe2
]
# GOES features (3: XRS-B, XRS-A, ratio)
_GOES_FEATURES = [
    "goes_xrsb_flux",
    "goes_xrsa_flux",
    "goes_xrsa_xrsb_ratio",
]
# CZT2 / CdTe2 aggregated (6)
_CZT2_FEATURES = [
    "czt2_total_mean",
    "czt2_total_max",
    "czt2_total_std",
]
_CDTE2_FEATURES = [
    "cdte2_total_mean",
    "cdte2_total_max",
    "cdte2_total_std",
]
# Information-theory features (6)
_INFO_THEORY_FEATURES = [
    "transfer_entropy_hxr_to_sxr",
    "mutual_information_sxr_hxr",
    "sample_entropy_sxr",
    "sample_entropy_hxr",
    "lagged_cross_corr",
    "lagged_cross_corr_lag",
]
# Neupert correlation (2)
_NEUPERT_FEATURES = [
    "neupert_rho_mean",
    "neupert_rho_std",
]
# Hardness ratio evolution (3)
_HARDNESS_EVOLUTION_FEATURES = [
    "hardness_ratio_slope",
    "hardness_ratio_mean",
    "hardness_ratio_std",
]
# HK features (8)
_HK_FEATURES = [
    "hk_czt1temp",
    "hk_czt2temp",
    "hk_cdte1temp",
    "hk_cdte2temp",
    "hk_czthvmon",
    "hk_cdtehvmon",
    "hk_czt1satctr",
    "hk_cdte1pilectr",
]
# Non-thermal features (4)
_NONTHERMAL_FEATURES = [
    "nonthermal_gamma",
    "nonthermal_ec",
    "nonthermal_n_nth",
    "thermal_fraction",
]
# QPP features (4)
_QPP_FEATURES = [
    "qpp_detected",
    "qpp_period",
    "qpp_amplitude",
    "qpp_significance",
]
# Correction stats (2)
_CORRECTION_FEATURES = [
    "deadtime_max_pct",
    "bg_fraction_pct",
]
# Meta (1)
_META_FEATURES = ["window_len"]


def extract_features_window(
    solexs_counts: np.ndarray,
    hel1os_ctr: np.ndarray | None = None,
    window: int = 3600,
    precomputed: dict | None = None,
) -> dict[str, float] | None:
    """Extract feature vector from a single time window (v2).

    Parameters
    ----------
    solexs_counts : ndarray
        SoLEXS SDD2 count rates for the window.
    hel1os_ctr : ndarray, optional
        HEL1OS count rates (n_times × n_bands) for the window.
    window : int
        Window size in seconds.
    precomputed : dict, optional
        Day-level precomputed features:
          - hk_czt1temp, hk_czt2temp, hk_cdte1temp, hk_cdte2temp
          - hk_czthvmon, hk_cdtehvmon, hk_czt1satctr, hk_cdte1pilectr
          - hxr_gamma_czt2, hxr_gamma_cdte1, hxr_gamma_cdte2
          - nonthermal_gamma, nonthermal_ec, nonthermal_n_nth, thermal_fraction
          - goes_xrsa_flux, goes_xrsa_xrsb_ratio
          - deadtime_max_pct, bg_fraction_pct
    """
    valid = solexs_counts[np.isfinite(solexs_counts)]
    if len(valid) < 5:
        return None

    f: dict[str, float] = {}

    # ── SXR statistical features ──────────────────────────────────
    f["sxr_mean"] = float(np.mean(valid))
    f["sxr_std"] = float(np.std(valid))
    f["sxr_max"] = float(np.max(valid))
    f["sxr_min"] = float(np.min(valid))
    f["sxr_median"] = float(np.median(valid))
    f["sxr_skew"] = float(skew(valid)) if len(valid) > 2 else 0.0
    f["sxr_kurtosis"] = float(kurtosis(valid)) if len(valid) > 3 else 0.0
    f["sxr_range"] = f["sxr_max"] - f["sxr_min"]
    f["sxr_cv"] = f["sxr_std"] / max(f["sxr_mean"], 1e-6)

    diff = np.diff(valid)
    f["sxr_rise_rate"] = float(np.mean(diff[diff > 0])) if np.any(diff > 0) else 0.0
    f["sxr_fall_rate"] = float(np.mean(diff[diff < 0])) if np.any(diff < 0) else 0.0
    f["sxr_abs_slope"] = float(np.mean(np.abs(diff)))

    q75, q25 = np.percentile(valid, [75, 25])
    f["sxr_iqr"] = float(q75 - q25)

    for pct in FEATURE_PERCENTILES:
        f[f"sxr_p{pct}"] = float(np.percentile(valid, pct))

    if len(valid) > 20:
        ac = np.correlate(valid - np.mean(valid), valid - np.mean(valid), mode="full")
        ac = ac[len(ac) // 2 :]
        ac /= ac[0] + 1e-10
        for lag in FEATURE_AUTOCORR_LAGS:
            f[f"sxr_acf_{lag}s"] = float(ac[lag]) if lag < len(ac) else 0.0

    if len(valid) > 100:
        try:
            nperseg = min(FEATURE_SPECTRAL_ENTROPY_NPERSEG, len(valid) // 2)
            _, psd = scipy_welch(valid, fs=1.0, nperseg=nperseg)
            total = np.sum(psd) + 1e-10
            p = psd / total
            f["sxr_spec_entropy"] = float(-np.sum(p * np.log(p + 1e-30)))
            f["sxr_peak_freq"] = float(np.argmax(psd) / (len(psd) + 1e-10))
        except Exception:
            f["sxr_spec_entropy"] = 0.0
            f["sxr_peak_freq"] = 0.0

    # ── HXR band features ─────────────────────────────────────────
    if hel1os_ctr is not None and hel1os_ctr.size > 0:
        hxr = hel1os_ctr if hel1os_ctr.ndim == 2 else hel1os_ctr[:, np.newaxis]
        nbands = hxr.shape[1]

        for b in range(min(nbands, 10)):
            bv = hxr[:, b][np.isfinite(hxr[:, b])]
            if len(bv) < 3:
                continue
            f[f"hxr_b{b}_mean"] = float(np.mean(bv))
            f[f"hxr_b{b}_std"] = float(np.std(bv))
            f[f"hxr_b{b}_max"] = float(np.max(bv))

        if nbands >= 2:
            lo, hi = hxr[:, 0], hxr[:, 1]
            hr = np.where(
                np.isfinite(lo) & np.isfinite(hi), hi / np.maximum(lo, 1e-6), 0.0
            )
            f["hxr_hardness_ratio"] = float(np.nanmean(hr))

        if nbands >= 5:
            tot = np.nansum(hxr[:, :5], axis=1)
            f["hxr_total_mean"] = float(np.nanmean(tot))

        if len(valid) >= len(hxr[:, 0]):
            hxr_sum = np.nansum(hxr[:, : min(nbands, 5)], axis=1)
            ml = min(len(valid), len(hxr_sum))
            ratio = np.where(hxr_sum[:ml] > 0, valid[:ml] / (hxr_sum[:ml] + 1e-6), 0.0)
            f["soft_hard_ratio"] = float(np.mean(ratio))

        # CdTe-specific features (bands 5-9 in combined array)
        if nbands >= 10:
            thermal_lo = hxr[:, 5]  # CdTe 5-20 keV
            czt_full = hxr[:, 4]  # CZT 18-160 keV
            tr = np.where(
                np.isfinite(thermal_lo) & np.isfinite(czt_full),
                thermal_lo / np.maximum(czt_full, 1e-6),
                0.0,
            )
            f["cdte_thermal_ratio"] = float(np.nanmean(tr))

            bd_lo = hxr[:, 5]  # CdTe 5-20 keV
            bd_hi = hxr[:, 6]  # CdTe 20-30 keV
            br = np.where(
                np.isfinite(bd_lo) & np.isfinite(bd_hi),
                bd_hi / np.maximum(bd_lo, 1e-6),
                0.0,
            )
            f["cdte_boundary_ratio"] = float(np.nanmean(br))
        else:
            f["cdte_thermal_ratio"] = 0.0
            f["cdte_boundary_ratio"] = 0.0

        # ── Hardness ratio evolution ──────────────────────────────
        hxr_full = np.nansum(hxr[:, : min(nbands, 5)], axis=1)
        if nbands >= 2:
            hr_series = np.where(
                hxr[:, 1] > 0,
                hxr[:, 1] / np.maximum(hxr[:, 0], 1e-6),
                0.0,
            )
            hr_valid = hr_series[np.isfinite(hr_series)]
            if len(hr_valid) > 10:
                f["hardness_ratio_mean"] = float(np.mean(hr_valid))
                f["hardness_ratio_std"] = float(np.std(hr_valid))
                # Slope: linear regression
                t_hr = np.arange(len(hr_valid))
                try:
                    A = np.vstack([t_hr, np.ones_like(t_hr)]).T
                    coeffs, *_ = np.linalg.lstsq(A, hr_valid, rcond=None)
                    f["hardness_ratio_slope"] = float(coeffs[0])
                except Exception:
                    f["hardness_ratio_slope"] = 0.0
            else:
                f["hardness_ratio_mean"] = 0.0
                f["hardness_ratio_std"] = 0.0
                f["hardness_ratio_slope"] = 0.0
        else:
            f["hardness_ratio_mean"] = 0.0
            f["hardness_ratio_std"] = 0.0
            f["hardness_ratio_slope"] = 0.0

        # ── Information-theory features ───────────────────────────
        ml = min(len(valid), len(hxr_full))
        try:
            from bah2026.features.information_theory import (
                transfer_entropy,
                mutual_information,
                sample_entropy,
                lagged_cross_correlation,
            )

            # Downsample for speed (60s bins)
            step_ds = 60
            sxr_ds = valid[:ml][::step_ds]
            hxr_ds = hxr_full[:ml][::step_ds]

            if len(sxr_ds) > 20:
                f["transfer_entropy_hxr_to_sxr"] = float(
                    transfer_entropy(hxr_ds, sxr_ds, k=1, bins=8)
                )
                f["mutual_information_sxr_hxr"] = float(
                    mutual_information(sxr_ds, hxr_ds, bins=8)
                )
            else:
                f["transfer_entropy_hxr_to_sxr"] = 0.0
                f["mutual_information_sxr_hxr"] = 0.0

            if len(sxr_ds) > 50:
                f["sample_entropy_sxr"] = float(
                    sample_entropy(sxr_ds[:200], m=2, r_factor=0.2)
                )
            else:
                f["sample_entropy_sxr"] = 0.0

            if len(hxr_ds) > 50:
                f["sample_entropy_hxr"] = float(
                    sample_entropy(hxr_ds[:200], m=2, r_factor=0.2)
                )
            else:
                f["sample_entropy_hxr"] = 0.0

            # Lagged cross-correlation
            if ml > 200:
                corr, lag = lagged_cross_correlation(
                    hxr_full[:ml], valid[:ml], max_lag=100
                )
                f["lagged_cross_corr"] = float(corr)
                f["lagged_cross_corr_lag"] = float(lag)
            else:
                f["lagged_cross_corr"] = 0.0
                f["lagged_cross_corr_lag"] = 0.0
        except Exception:
            f["transfer_entropy_hxr_to_sxr"] = 0.0
            f["mutual_information_sxr_hxr"] = 0.0
            f["sample_entropy_sxr"] = 0.0
            f["sample_entropy_hxr"] = 0.0
            f["lagged_cross_corr"] = 0.0
            f["lagged_cross_corr_lag"] = 0.0

        # ── Neupert correlation ──────────────────────────────────
        try:
            from bah2026.features.spectral_fitting import neupert_correlation

            ml_n = min(len(valid), len(hxr_full))
            rho = neupert_correlation(
                valid[:ml_n], hxr_full[:ml_n], window_sec=300, step_sec=60
            )
            v = rho[np.isfinite(rho)]
            if len(v) > 0:
                f["neupert_rho_mean"] = float(np.mean(v))
                f["neupert_rho_std"] = float(np.std(v))
            else:
                f["neupert_rho_mean"] = 0.0
                f["neupert_rho_std"] = 0.0
        except Exception:
            f["neupert_rho_mean"] = 0.0
            f["neupert_rho_std"] = 0.0

        # ── QPP detection (per-window, fast with reduced LS freqs) ─
        try:
            from bah2026.features.qpp import detect_qpp

            qpp = detect_qpp(hxr_full[:ml], dt=1.0, min_period=10, max_period=300)
            f["qpp_detected"] = 1.0 if qpp["detected"] else 0.0
            f["qpp_period"] = float(qpp["period"])
            f["qpp_amplitude"] = float(qpp["amplitude"])
            f["qpp_significance"] = float(qpp["significance"])
        except Exception:
            f["qpp_detected"] = 0.0
            f["qpp_period"] = 0.0
            f["qpp_amplitude"] = 0.0
            f["qpp_significance"] = 0.0
    else:
        # No HXR: fill all HXR-dependent features with 0
        for k in (
            _HXR_FEATURES
            + _INFO_THEORY_FEATURES
            + _NEUPERT_FEATURES
            + _HARDNESS_EVOLUTION_FEATURES
            + _QPP_FEATURES
        ):
            f[k] = 0.0

    # ── Precomputed (day-level) features ──────────────────────────
    if precomputed:
        # HK features
        for k in _HK_FEATURES:
            f[k] = float(precomputed.get(k, 0.0))

        # Spectral indices from all 4 detectors
        f["hxr_spectral_index_gamma"] = float(
            precomputed.get("hxr_spectral_index_gamma", 0.0)
        )
        f["hxr_gamma_czt2"] = float(precomputed.get("hxr_gamma_czt2", 0.0))
        f["hxr_gamma_cdte1"] = float(precomputed.get("hxr_gamma_cdte1", 0.0))
        f["hxr_gamma_cdte2"] = float(precomputed.get("hxr_gamma_cdte2", 0.0))

        # Non-thermal features
        f["nonthermal_gamma"] = float(precomputed.get("nonthermal_gamma", 0.0))
        f["nonthermal_ec"] = float(precomputed.get("nonthermal_ec", 0.0))
        f["nonthermal_n_nth"] = float(precomputed.get("nonthermal_n_nth", 0.0))
        f["thermal_fraction"] = float(precomputed.get("thermal_fraction", 0.0))

        # GOES features
        f["goes_xrsb_flux"] = float(precomputed.get("goes_xrsb_flux", 0.0))
        f["goes_xrsa_flux"] = float(precomputed.get("goes_xrsa_flux", 0.0))
        f["goes_xrsa_xrsb_ratio"] = float(precomputed.get("goes_xrsa_xrsb_ratio", 0.0))

        # Correction stats
        f["deadtime_max_pct"] = float(precomputed.get("deadtime_max_pct", 0.0))
        f["bg_fraction_pct"] = float(precomputed.get("bg_fraction_pct", 0.0))

        # PI features
        f["sxr_temperature_mk"] = float(precomputed.get("sxr_temperature_mk", 0.0))
        f["sxr_emission_measure"] = float(precomputed.get("sxr_emission_measure", 0.0))
        f["sxr_chi2_red"] = float(precomputed.get("sxr_chi2_red", 0.0))

        # CZT2 / CdTe2
        for k in _CZT2_FEATURES + _CDTE2_FEATURES:
            f[k] = float(precomputed.get(k, 0.0))
    else:
        # Fill all precomputed features with 0
        for k in (
            _PI_FEATURES
            + _HEL1OS_SPEC_FEATURES
            + _GOES_FEATURES
            + _CZT2_FEATURES
            + _CDTE2_FEATURES
            + _HK_FEATURES
            + _NONTHERMAL_FEATURES
            + _CORRECTION_FEATURES
        ):
            f[k] = 0.0

    f["window_len"] = float(len(valid))
    return f


def get_canonical_feature_names() -> list[str]:
    """Return the full sorted list of canonical feature names (v2)."""
    all_feats = (
        _SXR_FEATURES
        + _HXR_FEATURES
        + _PI_FEATURES
        + _HEL1OS_SPEC_FEATURES
        + _GOES_FEATURES
        + _CZT2_FEATURES
        + _CDTE2_FEATURES
        + _INFO_THEORY_FEATURES
        + _NEUPERT_FEATURES
        + _HARDNESS_EVOLUTION_FEATURES
        + _HK_FEATURES
        + _NONTHERMAL_FEATURES
        + _QPP_FEATURES
        + _CORRECTION_FEATURES
        + _META_FEATURES
    )
    for lag in FEATURE_AUTOCORR_LAGS:
        all_feats.append(f"sxr_acf_{lag}s")
    for pct in FEATURE_PERCENTILES:
        all_feats.append(f"sxr_p{pct}")
    return sorted(all_feats)


def pad_features_to_canonical(
    feat: dict[str, float], canonical: list[str]
) -> list[float]:
    """Pad a feature dict to match the canonical feature list."""
    return [feat.get(k, 0.0) for k in canonical]


def build_feature_matrix(
    solexs_counts: np.ndarray,
    hel1os_ctr: np.ndarray | None = None,
    lookback: int = 3600,
    step: int = 300,
) -> tuple[np.ndarray, list[str]]:
    """Build feature matrix by sliding a window over a full day."""
    from bah2026.config import FEATURE_LOOKBACK_SEC, FEATURE_STEP_SEC

    if lookback == 3600 and step == 300:
        lookback = FEATURE_LOOKBACK_SEC
        step = FEATURE_STEP_SEC

    n = len(solexs_counts)
    rows: list[list[float]] = []
    names: list[str] | None = None

    for i in range(lookback, n, step):
        sxr_win = solexs_counts[i - lookback : i]
        hxr_win = None
        if hel1os_ctr is not None and hel1os_ctr.size > 0:
            h_len = len(hel1os_ctr)
            hxr_win = hel1os_ctr[max(0, i - lookback) : min(h_len, i)]

        feat = extract_features_window(sxr_win, hxr_win, window=lookback)
        if feat is None:
            continue
        if names is None:
            names = sorted(feat.keys())
        rows.append([feat[k] for k in names])

    if not rows:
        return np.empty((0, 0)), []
    X = np.array(rows, dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, names or []
