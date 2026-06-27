"""Feature engineering: statistical, spectral, and cross-instrument features."""

from __future__ import annotations

import numpy as np
from scipy.stats import skew, kurtosis
from scipy.signal import welch as scipy_welch

from bah2026.config import (
    FEATURE_AUTOCORR_LAGS, FEATURE_PERCENTILES, FEATURE_SPECTRAL_ENTROPY_NPERSEG,
)

# Canonical feature set — always returned, even if HXR is absent
_SXR_FEATURES = [
    "sxr_abs_slope", "sxr_cv", "sxr_fall_rate", "sxr_iqr", "sxr_kurtosis",
    "sxr_max", "sxr_mean", "sxr_median", "sxr_min", "sxr_peak_freq",
    "sxr_range", "sxr_rise_rate", "sxr_skew", "sxr_spec_entropy", "sxr_std",
]
_HXR_FEATURES = [
    # CZT1 bands (20-160 keV)
    "hxr_b0_max", "hxr_b0_mean", "hxr_b0_std",
    "hxr_b1_max", "hxr_b1_mean", "hxr_b1_std",
    "hxr_b2_max", "hxr_b2_mean", "hxr_b2_std",
    "hxr_b3_max", "hxr_b3_mean", "hxr_b3_std",
    "hxr_b4_max", "hxr_b4_mean", "hxr_b4_std",
    # CdTe1 bands (1.8-90 keV)
    "hxr_b5_max", "hxr_b5_mean", "hxr_b5_std",
    "hxr_b6_max", "hxr_b6_mean", "hxr_b6_std",
    "hxr_b7_max", "hxr_b7_mean", "hxr_b7_std",
    "hxr_b8_max", "hxr_b8_mean", "hxr_b8_std",
    "hxr_b9_max", "hxr_b9_mean", "hxr_b9_std",
    # Derived
    "hxr_hardness_ratio", "hxr_total_mean",
    "soft_hard_ratio",
    # CdTe-specific
    "cdte_thermal_ratio", "cdte_boundary_ratio",
]
_META_FEATURES = ["window_len"]


def extract_features_window(
    solexs_counts: np.ndarray,
    hel1os_ctr: np.ndarray | None = None,
    window: int = 3600,
) -> dict[str, float] | None:
    """Extract feature vector from a single time window."""
    valid = solexs_counts[np.isfinite(solexs_counts)]
    if len(valid) < 5:
        return None

    f: dict[str, float] = {}

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
        ac = ac[len(ac) // 2:]
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
            hr = np.where(np.isfinite(lo) & np.isfinite(hi), hi / np.maximum(lo, 1e-6), 0.0)
            f["hxr_hardness_ratio"] = float(np.nanmean(hr))

        if nbands >= 5:
            tot = np.nansum(hxr[:, :5], axis=1)
            f["hxr_total_mean"] = float(np.nanmean(tot))

        if len(valid) >= len(hxr[:, 0]):
            hxr_sum = np.nansum(hxr[:, :min(nbands, 5)], axis=1)
            ml = min(len(valid), len(hxr_sum))
            ratio = np.where(hxr_sum[:ml] > 0, valid[:ml] / (hxr_sum[:ml] + 1e-6), 0.0)
            f["soft_hard_ratio"] = float(np.mean(ratio))

        # CdTe-specific features (bands 5-9 in combined array)
        if nbands >= 10:
            # Thermal ratio: CdTe low (5-20 keV) / CZT full (18-160 keV)
            thermal_lo = hxr[:, 5]  # CdTe 5-20 keV
            czt_full = hxr[:, 4]    # CZT 18-160 keV
            tr = np.where(np.isfinite(thermal_lo) & np.isfinite(czt_full),
                         thermal_lo / np.maximum(czt_full, 1e-6), 0.0)
            f["cdte_thermal_ratio"] = float(np.nanmean(tr))

            # Boundary ratio: CdTe 20-30 keV / CdTe 5-20 keV
            bd_lo = hxr[:, 5]  # CdTe 5-20 keV
            bd_hi = hxr[:, 6]  # CdTe 20-30 keV
            br = np.where(np.isfinite(bd_lo) & np.isfinite(bd_hi),
                         bd_hi / np.maximum(bd_lo, 1e-6), 0.0)
            f["cdte_boundary_ratio"] = float(np.nanmean(br))
        else:
            f["cdte_thermal_ratio"] = 0.0
            f["cdte_boundary_ratio"] = 0.0

    f["window_len"] = float(len(valid))
    return f


def get_canonical_feature_names() -> list[str]:
    """Return the full sorted list of canonical feature names."""
    all_feats = _SXR_FEATURES + _HXR_FEATURES + _META_FEATURES
    for lag in FEATURE_AUTOCORR_LAGS:
        all_feats.append(f"sxr_acf_{lag}s")
    for pct in FEATURE_PERCENTILES:
        all_feats.append(f"sxr_p{pct}")
    return sorted(all_feats)


def pad_features_to_canonical(feat: dict[str, float], canonical: list[str]) -> list[float]:
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
        sxr_win = solexs_counts[i - lookback:i]
        hxr_win = None
        if hel1os_ctr is not None and hel1os_ctr.size > 0:
            h_len = len(hel1os_ctr)
            hxr_win = hel1os_ctr[max(0, i - lookback):min(h_len, i)]

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
