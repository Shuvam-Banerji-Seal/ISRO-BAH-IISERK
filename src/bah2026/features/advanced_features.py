"""Advanced feature extraction for solar flare forecasting.

62 features across 5 groups:
  1. Temporal derivatives (12) — dSXR/dt, d2SXR/dt2, dHR/dt, Neupert efficiency
  2. Multi-scale temporal (24) — statistics at 5/15/30 min scales + cross-scale ratios
  3. GOES time series (8) — derivatives, rolling stats, flare class
  4. Per-window spectral (8) — temperature, EM, spectral index, SHS, non-thermal fraction
  5. Wavelet scalogram (10) — energy in period bands, peak period, spectral entropy
"""

from __future__ import annotations

import numpy as np

from bah2026.features.spectral_fitting import fit_temperature, fit_spectral_index
from bah2026.features.qpp import wavelet_power_auto
from bah2026.features.non_thermal import separate_thermal_non_thermal


# ── Helpers ────────────────────────────────────────────────────────────


def _f(x) -> float:
    """Convert to Python float with NaN/inf sanitization."""
    return float(np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0))


def _hxr_full(hxr: np.ndarray) -> np.ndarray:
    """Extract 1D full-band HXR from 1D or 2D input."""
    hxr = np.asarray(hxr, dtype=np.float64)
    if hxr.ndim == 2:
        return np.nansum(hxr, axis=1)
    return hxr


def _hxr_hardness_ratio(hxr: np.ndarray) -> np.ndarray:
    """Compute hardness ratio HR = hi/lo from 1D or 2D HXR."""
    hxr = np.asarray(hxr, dtype=np.float64)
    if hxr.ndim == 2 and hxr.shape[1] >= 2:
        lo = np.maximum(hxr[:, 0], 1e-10)
        return np.where(hxr[:, 0] > 0, hxr[:, 1] / lo, 0.0)
    if hxr.ndim == 2:
        return hxr[:, 0].copy()
    return hxr.copy()


def _rolling_mean(arr: np.ndarray, w: int) -> np.ndarray:
    """Rolling mean with uniform window of size w."""
    n = len(arr)
    if n == 0:
        return np.array([0.0])
    w = min(w, n)
    if w < 1:
        return np.array([0.0])
    c = np.cumsum(np.insert(arr, 0, 0.0))
    return (c[w:] - c[:-w]) / w


def _rolling_std(arr: np.ndarray, w: int) -> np.ndarray:
    """Rolling standard deviation with uniform window of size w."""
    n = len(arr)
    if n == 0:
        return np.array([0.0])
    w = min(w, n)
    if w < 2:
        return np.array([0.0])
    c1 = np.cumsum(np.insert(arr, 0, 0.0))
    c2 = np.cumsum(np.insert(arr * arr, 0, 0.0))
    mean = (c1[w:] - c1[:-w]) / w
    var = (c2[w:] - c2[:-w]) / w - mean * mean
    return np.sqrt(np.maximum(var, 0.0))


def _linear_slope(arr: np.ndarray) -> float:
    """Linear regression slope of array vs index."""
    n = len(arr)
    if n < 2:
        return 0.0
    t = np.arange(n, dtype=np.float64)
    A = np.vstack([t, np.ones(n)]).T
    coeffs, *_ = np.linalg.lstsq(A, arr, rcond=None)
    return float(coeffs[0])


def _last_n(arr: np.ndarray, n: int) -> np.ndarray:
    """Return the last n elements (or full array if shorter)."""
    if len(arr) <= n:
        return arr
    return arr[-n:]


# ── Group 1: Temporal Derivatives (12 features) ─────────────────────────


def extract_temporal_derivatives(
    sxr_counts: np.ndarray,
    hxr_counts: np.ndarray | None = None,
) -> dict[str, float]:
    """Extract temporal derivative features.

    Parameters:
        sxr_counts: (N,) array of SoLEXS SXR count rates (1s cadence)
        hxr_counts: (N,) array of HXR full-band count rates (1s cadence, aligned)

    Returns dict with keys:
        dsxr_dt_mean, dsxr_dt_std, dsxr_dt_max, dsxr_dt_min,
        d2sxr_dt2_mean, d2sxr_dt2_std,
        dhxr_dt_mean, dhxr_dt_std, dhxr_dt_max,
        dhr_dt_mean, dhr_dt_max,
        dsxr_dhxr_ratio_mean  # Neupert efficiency η = mean(dSXR/dt) / mean(HXR)
    """
    keys = (
        "dsxr_dt_mean",
        "dsxr_dt_std",
        "dsxr_dt_max",
        "dsxr_dt_min",
        "d2sxr_dt2_mean",
        "d2sxr_dt2_std",
        "dhxr_dt_mean",
        "dhxr_dt_std",
        "dhxr_dt_max",
        "dhr_dt_mean",
        "dhr_dt_max",
        "dsxr_dhxr_ratio_mean",
    )

    sxr = np.asarray(sxr_counts, dtype=np.float64)
    sxr = sxr[np.isfinite(sxr)]

    if len(sxr) < 2:
        return {k: 0.0 for k in keys}

    dsxr_dt = np.diff(sxr)
    d2sxr_dt2 = np.diff(dsxr_dt) if len(dsxr_dt) >= 2 else np.array([0.0])

    result: dict[str, float] = {}
    result["dsxr_dt_mean"] = _f(np.mean(dsxr_dt))
    result["dsxr_dt_std"] = _f(np.std(dsxr_dt))
    result["dsxr_dt_max"] = _f(np.max(dsxr_dt))
    result["dsxr_dt_min"] = _f(np.min(dsxr_dt))
    result["d2sxr_dt2_mean"] = _f(np.mean(d2sxr_dt2)) if len(d2sxr_dt2) > 0 else 0.0
    result["d2sxr_dt2_std"] = _f(np.std(d2sxr_dt2)) if len(d2sxr_dt2) > 0 else 0.0

    if hxr_counts is not None:
        hxr_raw = np.asarray(hxr_counts, dtype=np.float64)
        hxr_full = _hxr_full(hxr_raw)
        hxr_full = hxr_full[np.isfinite(hxr_full)]
        hr = _hxr_hardness_ratio(hxr_raw)
        hr = hr[np.isfinite(hr)]

        if len(hxr_full) >= 2:
            dhxr_dt = np.diff(hxr_full)
            result["dhxr_dt_mean"] = _f(np.mean(dhxr_dt))
            result["dhxr_dt_std"] = _f(np.std(dhxr_dt))
            result["dhxr_dt_max"] = _f(np.max(dhxr_dt))
        else:
            result["dhxr_dt_mean"] = 0.0
            result["dhxr_dt_std"] = 0.0
            result["dhxr_dt_max"] = 0.0

        if len(hr) >= 2:
            dhr_dt = np.diff(hr)
            result["dhr_dt_mean"] = _f(np.mean(dhr_dt))
            result["dhr_dt_max"] = _f(np.max(dhr_dt))
        else:
            result["dhr_dt_mean"] = 0.0
            result["dhr_dt_max"] = 0.0

        dsxr_mean = float(np.mean(dsxr_dt)) if len(dsxr_dt) > 0 else 0.0
        hxr_mean = float(np.mean(hxr_full)) if len(hxr_full) > 0 else 0.0
        result["dsxr_dhxr_ratio_mean"] = _f(dsxr_mean / (hxr_mean + 1e-10))
    else:
        result["dhxr_dt_mean"] = 0.0
        result["dhxr_dt_std"] = 0.0
        result["dhxr_dt_max"] = 0.0
        result["dhr_dt_mean"] = 0.0
        result["dhr_dt_max"] = 0.0
        result["dsxr_dhxr_ratio_mean"] = 0.0

    return result


# ── Group 2: Multi-Scale Temporal (24 features) ────────────────────────


def extract_multiscale_features(
    sxr_counts: np.ndarray,
    hxr_counts: np.ndarray | None = None,
) -> dict[str, float]:
    """Extract features at multiple temporal scales.

    Computes statistics over 5min (300s), 15min (900s), 30min (1800s) windows,
    plus cross-scale ratios and trends.

    Returns dict with 24 keys:
        sxr_mean_5m, sxr_std_5m, sxr_max_5m, hxr_mean_5m, hxr_std_5m, hxr_max_5m,
        sxr_mean_15m, sxr_std_15m, sxr_max_15m, hxr_mean_15m, hxr_std_15m, hxr_max_15m,
        sxr_mean_30m, sxr_std_30m, sxr_max_30m, hxr_mean_30m, hxr_std_30m, hxr_max_30m,
        sxr_5m_to_60m_ratio, hxr_5m_to_60m_ratio,
        sxr_acceleration_trend, hxr_acceleration_trend,
        sxr_15m_slope, hxr_15m_slope
    """
    result: dict[str, float] = {}

    sxr = np.asarray(sxr_counts, dtype=np.float64)
    sxr = sxr[np.isfinite(sxr)]

    hxr: np.ndarray | None = None
    if hxr_counts is not None:
        hxr = _hxr_full(np.asarray(hxr_counts, dtype=np.float64))
        hxr = hxr[np.isfinite(hxr)]

    scales: list[tuple[str, int]] = [
        ("5m", 300),
        ("15m", 900),
        ("30m", 1800),
    ]

    for label, n in scales:
        sxr_win = _last_n(sxr, n) if len(sxr) > 0 else np.array([])
        if len(sxr_win) > 0:
            result[f"sxr_mean_{label}"] = _f(np.mean(sxr_win))
            result[f"sxr_std_{label}"] = _f(np.std(sxr_win))
            result[f"sxr_max_{label}"] = _f(np.max(sxr_win))
        else:
            result[f"sxr_mean_{label}"] = 0.0
            result[f"sxr_std_{label}"] = 0.0
            result[f"sxr_max_{label}"] = 0.0

        if hxr is not None and len(hxr) > 0:
            hxr_win = _last_n(hxr, n)
            result[f"hxr_mean_{label}"] = _f(np.mean(hxr_win))
            result[f"hxr_std_{label}"] = _f(np.std(hxr_win))
            result[f"hxr_max_{label}"] = _f(np.max(hxr_win))
        else:
            result[f"hxr_mean_{label}"] = 0.0
            result[f"hxr_std_{label}"] = 0.0
            result[f"hxr_max_{label}"] = 0.0

    sxr_mean_full = _f(np.mean(sxr)) if len(sxr) > 0 else 0.0
    result["sxr_5m_to_60m_ratio"] = _f(result["sxr_mean_5m"] / (sxr_mean_full + 1e-10))

    if hxr is not None and len(hxr) > 0:
        hxr_mean_full = _f(np.mean(hxr))
        result["hxr_5m_to_60m_ratio"] = _f(
            result["hxr_mean_5m"] / (hxr_mean_full + 1e-10)
        )
    else:
        result["hxr_5m_to_60m_ratio"] = 0.0

    # Acceleration trend: (mean_5m - mean_15m) / (mean_15m + eps)
    result["sxr_acceleration_trend"] = _f(
        (result["sxr_mean_5m"] - result["sxr_mean_15m"])
        / (result["sxr_mean_15m"] + 1e-10)
    )
    result["hxr_acceleration_trend"] = _f(
        (result["hxr_mean_5m"] - result["hxr_mean_15m"])
        / (result["hxr_mean_15m"] + 1e-10)
    )

    # 15m slope: linear regression slope of the last 900 samples
    sxr_15m = _last_n(sxr, 900) if len(sxr) > 0 else np.array([])
    result["sxr_15m_slope"] = _f(_linear_slope(sxr_15m)) if len(sxr_15m) >= 2 else 0.0

    if hxr is not None and len(hxr) > 0:
        hxr_15m = _last_n(hxr, 900)
        result["hxr_15m_slope"] = (
            _f(_linear_slope(hxr_15m)) if len(hxr_15m) >= 2 else 0.0
        )
    else:
        result["hxr_15m_slope"] = 0.0

    return result


# ── Group 3: GOES Time Series (8 features) ─────────────────────────────


def extract_goes_timeseries_features(
    goes_xrsb: np.ndarray | None,
    goes_xrsa: np.ndarray | None,
) -> dict[str, float]:
    """Extract GOES time series features.

    Returns dict with 8 keys:
        goes_xrsb_ddt_max, goes_xrsb_rolling_std_300s,
        goes_xrsb_rolling_std_1800s, goes_xrsa_rolling_mean_300s,
        goes_class_current, goes_xrsb_gradient_1h,
        goes_flare_history_24h, goes_xrsb_prev_peak_ratio
    """
    keys = (
        "goes_xrsb_ddt_max",
        "goes_xrsb_rolling_std_300s",
        "goes_xrsb_rolling_std_1800s",
        "goes_xrsa_rolling_mean_300s",
        "goes_class_current",
        "goes_xrsb_gradient_1h",
        "goes_flare_history_24h",
        "goes_xrsb_prev_peak_ratio",
    )

    if goes_xrsb is None and goes_xrsa is None:
        return {k: 0.0 for k in keys}

    result: dict[str, float] = {}

    if goes_xrsb is not None:
        xrsb = np.asarray(goes_xrsb, dtype=np.float64)
        xrsb = xrsb[np.isfinite(xrsb)]

        if len(xrsb) >= 2:
            dxrsb = np.diff(xrsb)
            result["goes_xrsb_ddt_max"] = _f(np.max(dxrsb))
        else:
            result["goes_xrsb_ddt_max"] = 0.0

        rs_300 = _rolling_std(xrsb, 300)
        result["goes_xrsb_rolling_std_300s"] = (
            _f(rs_300[-1]) if len(rs_300) > 0 else 0.0
        )

        rs_1800 = _rolling_std(xrsb, 1800)
        result["goes_xrsb_rolling_std_1800s"] = (
            _f(rs_1800[-1]) if len(rs_1800) > 0 else 0.0
        )

        result["goes_class_current"] = (
            _f(np.log10(np.max(xrsb) + 1e-10)) if len(xrsb) > 0 else 0.0
        )

        result["goes_xrsb_gradient_1h"] = (
            _f(_linear_slope(xrsb)) if len(xrsb) >= 2 else 0.0
        )
    else:
        result["goes_xrsb_ddt_max"] = 0.0
        result["goes_xrsb_rolling_std_300s"] = 0.0
        result["goes_xrsb_rolling_std_1800s"] = 0.0
        result["goes_class_current"] = 0.0
        result["goes_xrsb_gradient_1h"] = 0.0

    if goes_xrsa is not None:
        xrsa = np.asarray(goes_xrsa, dtype=np.float64)
        xrsa = xrsa[np.isfinite(xrsa)]
        rm_300 = _rolling_mean(xrsa, 300)
        result["goes_xrsa_rolling_mean_300s"] = (
            _f(rm_300[-1]) if len(rm_300) > 0 else 0.0
        )
    else:
        result["goes_xrsa_rolling_mean_300s"] = 0.0

    result["goes_flare_history_24h"] = 0.0
    result["goes_xrsb_prev_peak_ratio"] = 0.0

    return result


# ── Group 4: Per-Window Spectral (8 features) ──────────────────────────


def extract_per_window_spectral(
    pi_counts: np.ndarray | None,
    hxr_spectra_czt: np.ndarray | None,
    hxr_spectra_cdte: np.ndarray | None,
    channel_energies: tuple[np.ndarray, np.ndarray] | None = None,
    prev_gamma: float = 0.0,
) -> dict[str, float]:
    """Extract per-window spectral features.

    Returns dict with 8 keys:
        sxr_temp_window, sxr_em_window, sxr_gamma_window,
        hxr_gamma_window_czt1, hxr_gamma_window_cdte1,
        shs_index, spectral_hardening_rate, nonthermal_fraction_window
    """
    keys = (
        "sxr_temp_window",
        "sxr_em_window",
        "sxr_gamma_window",
        "hxr_gamma_window_czt1",
        "hxr_gamma_window_cdte1",
        "shs_index",
        "spectral_hardening_rate",
        "nonthermal_fraction_window",
    )

    if pi_counts is None and hxr_spectra_czt is None and hxr_spectra_cdte is None:
        return {k: 0.0 for k in keys}

    result: dict[str, float] = {}

    # ── SoLEXS PI: temperature, emission measure, spectral index ──
    sxr_temp = 0.0
    sxr_em = 0.0
    sxr_gamma = 0.0
    summed_pi: np.ndarray | None = None
    pi_centroids: np.ndarray | None = None

    if pi_counts is not None:
        pi = np.asarray(pi_counts, dtype=np.float64)
        if pi.ndim == 2 and pi.shape[0] > 0:
            summed_pi = np.nansum(pi, axis=0)
        elif pi.ndim == 1:
            summed_pi = pi.copy()
        else:
            summed_pi = None

        if summed_pi is not None and np.sum(summed_pi) > 0:
            n_ch = summed_pi.shape[0]
            if channel_energies is not None:
                emin, emax = channel_energies
                pi_centroids = (np.asarray(emin) + np.asarray(emax)) / 2.0
            else:
                pi_centroids = np.linspace(2.0, 22.0, n_ch)

            try:
                t_fit, em_fit, _ = fit_temperature(summed_pi, channel_energies)
                sxr_temp = float(t_fit)
                sxr_em = float(em_fit)
            except Exception:
                sxr_temp = 0.0
                sxr_em = 0.0

            try:
                sxr_gamma = float(fit_spectral_index(summed_pi, pi_centroids))
            except Exception:
                sxr_gamma = 0.0

    result["sxr_temp_window"] = _f(sxr_temp)
    result["sxr_em_window"] = _f(sxr_em)
    result["sxr_gamma_window"] = _f(sxr_gamma)

    # ── HEL1OS CZT: spectral index ──
    hxr_gamma_czt = 0.0
    summed_czt: np.ndarray | None = None

    if hxr_spectra_czt is not None:
        czt = np.asarray(hxr_spectra_czt, dtype=np.float64)
        if czt.ndim == 2 and czt.shape[0] > 0:
            summed_czt = np.nansum(czt, axis=0)
        elif czt.ndim == 1:
            summed_czt = czt.copy()

        if summed_czt is not None and np.sum(summed_czt) > 0:
            n_czt = summed_czt.shape[0]
            czt_centroids = np.linspace(20.0, 150.0, n_czt)
            try:
                hxr_gamma_czt = float(fit_spectral_index(summed_czt, czt_centroids))
            except Exception:
                hxr_gamma_czt = 0.0

    result["hxr_gamma_window_czt1"] = _f(hxr_gamma_czt)

    # ── HEL1OS CdTe: spectral index ──
    hxr_gamma_cdte = 0.0
    summed_cdte: np.ndarray | None = None

    if hxr_spectra_cdte is not None:
        cdte = np.asarray(hxr_spectra_cdte, dtype=np.float64)
        if cdte.ndim == 2 and cdte.shape[0] > 0:
            summed_cdte = np.nansum(cdte, axis=0)
        elif cdte.ndim == 1:
            summed_cdte = cdte.copy()

        if summed_cdte is not None and np.sum(summed_cdte) > 0:
            n_cdte = summed_cdte.shape[0]
            cdte_centroids = np.linspace(1.8, 90.0, n_cdte)
            try:
                hxr_gamma_cdte = float(fit_spectral_index(summed_cdte, cdte_centroids))
            except Exception:
                hxr_gamma_cdte = 0.0

    result["hxr_gamma_window_cdte1"] = _f(hxr_gamma_cdte)

    # ── SHS index and hardening rate ──
    # SHS (Soft-Hard-Soft): positive = hardening. Use HXR gamma as primary.
    if hxr_gamma_czt > 0:
        current_gamma = hxr_gamma_czt
    elif hxr_gamma_cdte > 0:
        current_gamma = hxr_gamma_cdte
    elif sxr_gamma > 0:
        current_gamma = sxr_gamma
    else:
        current_gamma = 0.0

    shs = current_gamma - prev_gamma
    result["shs_index"] = _f(shs)

    # prev_gamma is from 15 minutes ago → window_seconds = 900
    window_seconds = 900.0
    result["spectral_hardening_rate"] = _f(shs / window_seconds)

    # ── Non-thermal fraction ──
    nonthermal_fraction = 0.0

    # Prefer CZT spectrum (covers 20-150 keV, includes non-thermal range)
    if summed_czt is not None and np.sum(summed_czt) > 0:
        n_czt = summed_czt.shape[0]
        czt_energies = np.linspace(20.0, 150.0, n_czt)
        t_init = sxr_temp if sxr_temp > 0 else 10.0
        # Estimate EM from CZT data directly — SoLEXS PI EM (1e52) is in different
        # units and produces thermal model that dominates CZT counts (~5e4)
        czt_total = float(np.sum(summed_czt))
        em_init = max(czt_total * 10.0, 1e3) if czt_total > 0 else 1e3
        try:
            sep = separate_thermal_non_thermal(
                czt_energies, summed_czt, t_init, em_init
            )
            nt = 1.0 - float(sep.get("thermal_fraction", 1.0))
            nonthermal_fraction = float(np.clip(nt, 0.0, 1.0))
        except Exception:
            nonthermal_fraction = 0.0
    elif summed_pi is not None and np.sum(summed_pi) > 0 and pi_centroids is not None:
        t_init = sxr_temp if sxr_temp > 0 else 10.0
        em_init = sxr_em if sxr_em > 0 else float(np.max(summed_pi)) * 10.0
        try:
            sep = separate_thermal_non_thermal(pi_centroids, summed_pi, t_init, em_init)
            nt = 1.0 - float(sep.get("thermal_fraction", 1.0))
            nonthermal_fraction = float(np.clip(nt, 0.0, 1.0))
        except Exception:
            nonthermal_fraction = 0.0

    result["nonthermal_fraction_window"] = _f(nonthermal_fraction)

    return result


# ── Group 5: Wavelet Scalogram (10 features) ───────────────────────────


def extract_wavelet_scalogram_features(
    signal: np.ndarray,
    dt: float = 1.0,
    hxr_signal: np.ndarray | None = None,
) -> dict[str, float]:
    """Extract wavelet scalogram features.

    Uses the existing bah2026.features.qpp.wavelet_power_auto() function
    to compute the Morlet wavelet power spectrum, then extracts features.

    Returns dict with 10 keys:
        wavelet_energy_10_30s, wavelet_energy_30_60s,
        wavelet_energy_60_120s, wavelet_energy_120_300s,
        wavelet_energy_300_600s, wavelet_peak_period,
        wavelet_peak_significance, wavelet_spectral_entropy,
        wavelet_hxr_energy_30_120s, wavelet_cross_power_sxr_hxr
    """
    keys = (
        "wavelet_energy_10_30s",
        "wavelet_energy_30_60s",
        "wavelet_energy_60_120s",
        "wavelet_energy_120_300s",
        "wavelet_energy_300_600s",
        "wavelet_peak_period",
        "wavelet_peak_significance",
        "wavelet_spectral_entropy",
        "wavelet_hxr_energy_30_120s",
        "wavelet_cross_power_sxr_hxr",
    )

    sig = np.asarray(signal, dtype=np.float64)
    sig = sig[np.isfinite(sig)]

    if len(sig) < 100:
        return {k: 0.0 for k in keys}

    power, scales, periods = wavelet_power_auto(sig, dt=dt)

    if power.size == 0 or len(periods) == 0:
        return {k: 0.0 for k in keys}

    result: dict[str, float] = {}

    # ── Energy in period bands ──
    bands = [
        ("wavelet_energy_10_30s", 10.0, 30.0),
        ("wavelet_energy_30_60s", 30.0, 60.0),
        ("wavelet_energy_60_120s", 60.0, 120.0),
        ("wavelet_energy_120_300s", 120.0, 300.0),
        ("wavelet_energy_300_600s", 300.0, 600.0),
    ]

    for name, lo, hi in bands:
        mask = (periods >= lo) & (periods <= hi)
        if np.any(mask):
            result[name] = _f(np.sum(power[mask]))
        else:
            result[name] = 0.0

    # ── Global wavelet spectrum (mean over time) ──
    gws = np.mean(power, axis=1)

    # Peak period
    if len(gws) > 0 and np.any(gws > 0):
        peak_idx = int(np.argmax(gws))
        result["wavelet_peak_period"] = _f(periods[peak_idx])
    else:
        result["wavelet_peak_period"] = 0.0

    # Peak significance: max power / median power (SNR)
    med_power = float(np.median(power))
    max_power = float(np.max(power))
    result["wavelet_peak_significance"] = _f(max_power / (med_power + 1e-10))

    # Spectral entropy: -sum(p * log(p)) where p = normalized GWS
    gws_sum = float(np.sum(gws))
    if gws_sum > 0:
        p = gws / gws_sum
        entropy = -np.sum(p * np.log(p + 1e-30))
        result["wavelet_spectral_entropy"] = _f(entropy)
    else:
        result["wavelet_spectral_entropy"] = 0.0

    # ── Cross-wavelet with HXR ──
    cross_mask = (periods >= 30.0) & (periods <= 120.0)

    if hxr_signal is not None and np.any(cross_mask):
        hxr = np.asarray(hxr_signal, dtype=np.float64)
        hxr = hxr[np.isfinite(hxr)]

        if len(hxr) >= 100:
            n = min(len(sig), len(hxr))
            p_sxr, _, per_sxr = wavelet_power_auto(sig[:n], dt=dt)
            p_hxr, _, per_hxr = wavelet_power_auto(hxr[:n], dt=dt)

            if p_sxr.size > 0 and p_hxr.size > 0 and len(per_sxr) == len(per_hxr):
                hxr_mask = (per_hxr >= 30.0) & (per_hxr <= 120.0)
                if p_sxr.shape == p_hxr.shape and np.any(hxr_mask):
                    cross_power = p_sxr * p_hxr
                    result["wavelet_cross_power_sxr_hxr"] = _f(
                        np.sum(cross_power[hxr_mask])
                    )
                    result["wavelet_hxr_energy_30_120s"] = _f(np.sum(p_hxr[hxr_mask]))
                else:
                    result["wavelet_cross_power_sxr_hxr"] = 0.0
                    result["wavelet_hxr_energy_30_120s"] = _f(
                        np.sum(p_hxr[hxr_mask]) if np.any(hxr_mask) else 0.0
                    )
            else:
                result["wavelet_cross_power_sxr_hxr"] = 0.0
                result["wavelet_hxr_energy_30_120s"] = 0.0
        else:
            result["wavelet_cross_power_sxr_hxr"] = 0.0
            result["wavelet_hxr_energy_30_120s"] = 0.0
    else:
        result["wavelet_cross_power_sxr_hxr"] = 0.0
        result["wavelet_hxr_energy_30_120s"] = 0.0

    return result


# ── Master functions ───────────────────────────────────────────────────


def get_advanced_feature_names() -> list[str]:
    """Return the sorted list of all 62 advanced feature names."""
    return sorted(
        [
            # Group 1: Temporal Derivatives (12)
            "dsxr_dt_mean",
            "dsxr_dt_std",
            "dsxr_dt_max",
            "dsxr_dt_min",
            "d2sxr_dt2_mean",
            "d2sxr_dt2_std",
            "dhxr_dt_mean",
            "dhxr_dt_std",
            "dhxr_dt_max",
            "dhr_dt_mean",
            "dhr_dt_max",
            "dsxr_dhxr_ratio_mean",
            # Group 2: Multi-Scale Temporal (24)
            "sxr_mean_5m",
            "sxr_std_5m",
            "sxr_max_5m",
            "hxr_mean_5m",
            "hxr_std_5m",
            "hxr_max_5m",
            "sxr_mean_15m",
            "sxr_std_15m",
            "sxr_max_15m",
            "hxr_mean_15m",
            "hxr_std_15m",
            "hxr_max_15m",
            "sxr_mean_30m",
            "sxr_std_30m",
            "sxr_max_30m",
            "hxr_mean_30m",
            "hxr_std_30m",
            "hxr_max_30m",
            "sxr_5m_to_60m_ratio",
            "hxr_5m_to_60m_ratio",
            "sxr_acceleration_trend",
            "hxr_acceleration_trend",
            "sxr_15m_slope",
            "hxr_15m_slope",
            # Group 3: GOES Time Series (8)
            "goes_xrsb_ddt_max",
            "goes_xrsb_rolling_std_300s",
            "goes_xrsb_rolling_std_1800s",
            "goes_xrsa_rolling_mean_300s",
            "goes_class_current",
            "goes_xrsb_gradient_1h",
            "goes_flare_history_24h",
            "goes_xrsb_prev_peak_ratio",
            # Group 4: Per-Window Spectral (8)
            "sxr_temp_window",
            "sxr_em_window",
            "sxr_gamma_window",
            "hxr_gamma_window_czt1",
            "hxr_gamma_window_cdte1",
            "shs_index",
            "spectral_hardening_rate",
            "nonthermal_fraction_window",
            # Group 5: Wavelet Scalogram (10)
            "wavelet_energy_10_30s",
            "wavelet_energy_30_60s",
            "wavelet_energy_60_120s",
            "wavelet_energy_120_300s",
            "wavelet_energy_300_600s",
            "wavelet_peak_period",
            "wavelet_peak_significance",
            "wavelet_spectral_entropy",
            "wavelet_hxr_energy_30_120s",
            "wavelet_cross_power_sxr_hxr",
        ]
    )


def extract_all_advanced_features(
    sxr_counts: np.ndarray,
    hxr_counts: np.ndarray | None = None,
    hxr_bands: np.ndarray | None = None,
    pi_counts: np.ndarray | None = None,
    hxr_spectra_czt: np.ndarray | None = None,
    hxr_spectra_cdte: np.ndarray | None = None,
    goes_xrsb: np.ndarray | None = None,
    goes_xrsa: np.ndarray | None = None,
    prev_gamma: float = 0.0,
) -> dict[str, float]:
    """Extract all 62 advanced features at once.

    Calls each group function and merges results.
    """
    # For Groups 1 & 2: prefer multi-band HXR for hardness ratio computation
    hxr_for_groups = hxr_bands if hxr_bands is not None else hxr_counts

    result: dict[str, float] = {}

    result.update(extract_temporal_derivatives(sxr_counts, hxr_for_groups))
    result.update(extract_multiscale_features(sxr_counts, hxr_for_groups))
    result.update(extract_goes_timeseries_features(goes_xrsb, goes_xrsa))
    result.update(
        extract_per_window_spectral(
            pi_counts,
            hxr_spectra_czt,
            hxr_spectra_cdte,
            channel_energies=None,
            prev_gamma=prev_gamma,
        )
    )

    # For wavelet: use 1D HXR signal (full-band)
    hxr_for_wavelet: np.ndarray | None = None
    if hxr_counts is not None:
        hxr_for_wavelet = np.asarray(hxr_counts, dtype=np.float64)
    elif hxr_bands is not None:
        hxr_for_wavelet = _hxr_full(np.asarray(hxr_bands, dtype=np.float64))

    result.update(
        extract_wavelet_scalogram_features(
            sxr_counts, dt=1.0, hxr_signal=hxr_for_wavelet
        )
    )

    return result
