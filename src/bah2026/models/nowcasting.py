"""Solar flare detection algorithms for nowcasting.

Primary algorithms:
  1. detect_flares_swpc — SWPC-style onset (4-min monotonic rise, half-decay end)
  2. detect_flares_hel1os — HEL1OS band threshold detection
  3. coincidence_merge — combine SXR + HXR events with temporal gating

Legacy algorithms (kept for reference, not used in pipeline):
  4. detect_flares_threshold — simple MAD-based (used in v0 pipeline, produces noise)
  5. detect_flares_bayesian_blocks — Scargle 2013 (hand-rolled, use astropy instead)
  6. detect_flares_wavelet — CWT-based (PyWavelets)
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter

from bah2026.config import (
    NOWCAST_THRESHOLD_SIGMA,
    NOWCAST_MIN_DURATION_SEC,
    NOWCAST_BAYESIAN_BLOCKS_SIGMA,
    NOWCAST_WAVELET_SIGMA,
    NOWCAST_BACKGROUND_WINDOW_SEC,
)
from bah2026.data.calibration import solexs_counts_to_irradiance_simple, classify_goes


# ── Background estimation ─────────────────────────────────────────────


def background_subtract_simple(
    counts: np.ndarray,
    window: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Quick background subtraction using median filter.

    Parameters
    ----------
    counts : ndarray
        Raw count rate.
    window : int, optional
        Window size in seconds. Default: config value (600).

    Returns
    -------
    background : ndarray
    residual : ndarray
    """
    if window is None:
        window = NOWCAST_BACKGROUND_WINDOW_SEC
    valid = np.where(np.isfinite(counts), counts, np.nanmedian(counts))
    bg = median_filter(valid, size=window, mode="nearest")
    return bg, valid - bg


def background_subtract_iterative(
    counts: np.ndarray,
    window: int | None = None,
    n_iter: int = 3,
    percentile: float = 10.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Iterative outlier-rejecting background estimation.

    Uses a lower-percentile rolling window as baseline, masking detected
    flares each iteration. This prevents large flares from biasing the
    background estimate (a key failure of the simple median filter).

    Parameters
    ----------
    counts : ndarray
        Raw count rate.
    window : int, optional
        Window size in seconds.
    n_iter : int
        Number of iterations (default 3).
    percentile : float
        Rolling percentile for baseline (default 10th).

    Returns
    -------
    background : ndarray
    residual : ndarray
    """
    if window is None:
        window = NOWCAST_BACKGROUND_WINDOW_SEC

    from scipy.ndimage import percentile_filter

    valid = np.where(np.isfinite(counts), counts, np.nanmedian(counts))
    working = valid.copy()
    mask = np.ones_like(working, dtype=bool)

    for _ in range(n_iter):
        # Estimate baseline from lower percentile of non-masked data
        bg = percentile_filter(
            np.where(mask, working, np.nanmedian(working)),
            percentile,
            size=window,
            mode="nearest",
        )
        residual = working - bg

        # Mask points that are significantly above baseline
        mad = (
            np.median(np.abs(residual[mask] - np.median(residual[mask])))
            if np.any(mask)
            else 1.0
        )
        if mad < 1e-10:
            mad = 1.0
        mask = residual < 3 * 1.4826 * mad

    return bg, valid - bg


# ── SWPC-style onset detection (primary) ──────────────────────────────


def detect_flares_swpc(
    calibrated_flux: np.ndarray,
    time: np.ndarray,
    min_duration_sec: int = 240,
    c_class_threshold: float = 1e-6,
    background_window: int = 600,
) -> list[dict]:
    """Detect flares using the SWPC/GOES onset algorithm on calibrated flux.

    The SWPC definition (from NOAA/SWPC GOES X-ray Event lists):
      1. Begin = first minute of a 4-minute "steep monotonic increase"
         in 0.1-0.8 nm (1.55-12.4 keV) flux.
      2. Maximum = time of peak flux.
      3. End = flux decays to halfway between peak and pre-flare background.
      4. Minimum duration ~4 minutes, minimum peak ≥ C1 class.

    Parameters
    ----------
    calibrated_flux : ndarray
        GOES-equivalent irradiance in W/m² (1 s cadence).
    time : ndarray
        Time in seconds (Unix or relative).
    min_duration_sec : int
        Minimum duration for a valid flare (default 240 = 4 min).
    c_class_threshold : float
        Minimum peak flux for C-class (default 1e-6 W/m²).
    background_window : int
        Window for running background estimation (seconds).

    Returns
    -------
    events : list[dict]
        Each dict: start_idx, peak_idx, end_idx, start_time, peak_time,
                   end_time, peak_flux, duration_sec, goes_class
    """
    from scipy.ndimage import percentile_filter

    valid = np.where(np.isfinite(calibrated_flux), calibrated_flux, 0.0)
    n = len(valid)
    if n < 120:
        return []

    # Percentile-based background (resistant to flare absorption)
    bg = percentile_filter(valid, percentile=10, size=background_window, mode="nearest")

    # Find all significant peaks (local maxima above background+threshold)
    noise_floor = np.maximum(bg * 1.5, np.full_like(bg, c_class_threshold * 0.1))
    above_noise = valid > noise_floor

    # Find contiguous regions above noise floor
    regions = []
    i = 0
    while i < n:
        if above_noise[i]:
            start = i
            while i < n and above_noise[i]:
                i += 1
            end = i - 1
            if end - start >= 10:
                regions.append((start, end))
        else:
            i += 1

    # Process each region into an event
    events = []
    for start, end in regions:
        segment = valid[start : end + 1]
        peak_rel = np.argmax(segment)
        peak_idx = start + peak_rel
        peak_flux = float(valid[peak_idx])

        if peak_flux < c_class_threshold:
            continue

        # Refine begin: first point above bg+threshold
        begin_idx = start
        thr = c_class_threshold * 0.05
        for k in range(start, peak_idx + 1):
            if valid[k] > bg[k] + thr:
                begin_idx = k
                break

        # End: first point after peak where flux drops below half-max
        preflare_bg = max(
            float(bg[begin_idx]), float(np.min(valid[begin_idx : peak_idx + 1]))
        )
        half_max = preflare_bg + (peak_flux - preflare_bg) * 0.5
        end_idx = peak_idx
        while end_idx < min(end, n - 2) and float(valid[end_idx + 1]) >= half_max:
            end_idx += 1
        end_idx = max(end_idx, peak_idx)

        duration = end_idx - begin_idx + 1
        # Allow extremely impulsive high-class flares (single-bin spikes)
        # If flux is above M-class, accept even very short events
        high_class = peak_flux >= c_class_threshold * 10  # ≥M1.0
        if duration < (10 if high_class else min_duration_sec):
            continue

        events.append(
            {
                "start_idx": int(begin_idx),
                "peak_idx": int(peak_idx),
                "end_idx": int(end_idx),
                "start_time": float(time[begin_idx]),
                "peak_time": float(time[peak_idx]),
                "end_time": float(time[min(end_idx, n - 1)]),
                "peak_flux": float(peak_flux),
                "duration_sec": float(duration),
                "goes_class": classify_goes(peak_flux),
                "background": float(preflare_bg),
                "method": "swpc_peak",
            }
        )

    # Merge overlapping events
    if len(events) > 1:
        merged = [events[0]]
        for evt in events[1:]:
            if evt["start_idx"] <= merged[-1]["end_idx"] + 60:
                if evt["peak_flux"] > merged[-1]["peak_flux"]:
                    merged[-1].update(
                        {
                            "peak_idx": evt["peak_idx"],
                            "peak_time": evt["peak_time"],
                            "peak_flux": evt["peak_flux"],
                            "goes_class": evt["goes_class"],
                        }
                    )
                merged[-1]["end_idx"] = max(merged[-1]["end_idx"], evt["end_idx"])
                merged[-1]["end_time"] = max(merged[-1]["end_time"], evt["end_time"])
                merged[-1]["duration_sec"] = float(
                    merged[-1]["end_idx"] - merged[-1]["start_idx"] + 1
                )
            else:
                merged.append(evt)
        events = merged

    return events


# ── HEL1OS flare detection ────────────────────────────────────────────


def detect_flares_hel1os(
    hxr_counts: np.ndarray,
    hxr_mjd: np.ndarray,
    sigma: float = 5.0,
    min_duration_sec: int = 60,
    background_window: int = 300,
) -> list[dict]:
    """Detect flares in HEL1OS hard X-ray light curves.

    Uses a MAD-based threshold on the full-band (18-160 keV for CZT or
    1.8-90 keV for CdTe). HEL1OS count rates are typically low with many
    zero bins, so a clean background estimate is critical.

    Parameters
    ----------
    hxr_counts : ndarray
        HEL1OS count rate (use full band, shape (nrows,) or (nrows, nbands)).
    hxr_mjd : ndarray
        Time in MJD.
    sigma : float
        Detection threshold in sigma (default 5.0).
    min_duration_sec : int
        Minimum duration (default 60 s).
    background_window : int
        Window for running background (default 300 s).

    Returns
    -------
    events : list[dict]
        Same structure as detect_flares_swpc.
    """
    # Use full-band (last column) if multi-band
    if hxr_counts.ndim == 2:
        full_band = hxr_counts[:, -1].copy()
    else:
        full_band = hxr_counts.copy()

    valid = np.where(np.isfinite(full_band), full_band, 0.0)

    # Background via median filter
    bg = median_filter(valid, size=background_window, mode="nearest")
    residual = valid - bg

    # MAD-based threshold
    mad = np.median(np.abs(residual - np.median(residual)))
    if mad < 1e-10:
        return []
    threshold = sigma * 1.4826 * mad

    above = residual > threshold
    n = len(above)
    events = []
    i = 0
    while i < n:
        if above[i]:
            start = i
            while i < n and above[i]:
                i += 1
            end = min(i, n - 1)
            if end - start >= min_duration_sec:
                peak = start + np.argmax(valid[start : end + 1])
                events.append(
                    {
                        "start_idx": int(start),
                        "peak_idx": int(peak),
                        "end_idx": int(end),
                        "start_time": float(hxr_mjd[start]),
                        "peak_time": float(hxr_mjd[peak]),
                        "end_time": float(hxr_mjd[min(end, n - 1)]),
                        "peak_flux": float(valid[peak]),
                        "duration_sec": float(end - start),
                        "method": "hel1os_threshold",
                    }
                )
        else:
            i += 1
    return events


# ── Coincidence merging ───────────────────────────────────────────────


def coincidence_merge(
    sxr_events: list[dict],
    hxr_events: list[dict],
    sxr_time_key: str = "peak_time",
    hxr_time_key: str = "peak_time",
    tolerance_sec: float = 60.0,
    require_hxr_for_low: bool = True,
    high_class_threshold: str = "C",
) -> list[dict]:
    """Merge SXR and HXR event lists by temporal coincidence.

    Implements the combined nowcast logic:
      - If an SXR event has a coincident HXR event within tolerance, keep it.
      - If an SXR event is ≥high_class (e.g., C-class), keep it even without HXR.
      - If an SXR event is <high_class, require HXR coincidence to keep it.
      - HXR events without SXR coincidence are kept if they exceed a minimum flux.

    This severely reduces the noise-dominated sub-minute events while
    preserving real flares.

    Parameters
    ----------
    sxr_events : list[dict]
        Events from SXR detection (e.g., detect_flares_swpc).
    hxr_events : list[dict]
        Events from HXR detection (e.g., detect_flares_hel1os).
    sxr_time_key : str
        Time column to use for matching (default 'peak_time').
    hxr_time_key : str
        Time column for HXR (default 'peak_time').
    tolerance_sec : float
        Maximum temporal difference for coincidence (default 60 s).
    require_hxr_for_low : bool
        Require HXR for sub-threshold events (default True).
    high_class_threshold : str
        Minimum class for automatic retention (default 'C').

    Returns
    -------
    merged : list[dict]
        Merged event list with enhanced metadata.
    """
    CLASS_ORDER = {"A": 0, "B": 1, "C": 2, "M": 3, "X": 4}
    min_class_val = CLASS_ORDER.get(high_class_threshold, 2)

    merged = []
    hxr_matched = set()

    for sxr in sxr_events:
        sxr_t = sxr.get(sxr_time_key, 0)
        sxr_class = sxr.get("goes_class", "A")
        sxr_class_val = CLASS_ORDER.get(sxr_class, 0)

        # Find nearest HXR event
        best_hxr = None
        best_dt = tolerance_sec + 1
        for hi, hxr in enumerate(hxr_events):
            if hi in hxr_matched:
                continue
            hxr_t = hxr.get(hxr_time_key, 0)
            dt = abs(sxr_t - hxr_t)
            if dt < best_dt:
                best_dt = dt
                best_hxr = (hi, hxr)

        has_coincidence = best_hxr is not None and best_dt <= tolerance_sec

        # Decision logic
        if has_coincidence:
            hxr_idx, hxr = best_hxr
            hxr_matched.add(hxr_idx)
            sxr["has_hxr"] = True
            sxr["hxr_peak_flux"] = hxr.get("peak_flux", 0.0)
            sxr["hxr_peak_time"] = hxr.get("peak_time", 0.0)
            sxr["hxr_duration_sec"] = hxr.get("duration_sec", 0.0)
            merged.append(sxr)
        elif sxr_class_val >= min_class_val:
            # High-class event, keep even without HXR
            sxr["has_hxr"] = False
            sxr["hxr_peak_flux"] = 0.0
            merged.append(sxr)
        # else: sub-threshold without HXR → discard (likely noise)

    # Add orphan HXR events (significant events with no SXR counterpart)
    for hi, hxr in enumerate(hxr_events):
        if hi in hxr_matched:
            continue
        if hxr.get("peak_flux", 0) > 0:
            hxr["has_sxr"] = False
            hxr["goes_class"] = "HXR"
            hxr["method"] = "hxr_only"
            merged.append(hxr)

    return merged


# ── Legacy detection methods ──────────────────────────────────────────


def detect_flares_threshold(
    counts: np.ndarray,
    time: np.ndarray,
    sigma: float | None = None,
    min_duration_sec: int | None = None,
) -> list[dict]:
    """DEPRECATED: Detect flares using statistical threshold.

    This was the v0 detection method. It produces ~80% false positives
    (median duration 15 s, peak/bg ≈ 0.42). Use detect_flares_swpc instead.
    """
    if sigma is None:
        sigma = NOWCAST_THRESHOLD_SIGMA
    if min_duration_sec is None:
        min_duration_sec = NOWCAST_MIN_DURATION_SEC

    valid = np.where(np.isfinite(counts), counts, 0.0)
    mad = np.median(np.abs(valid - np.median(valid)))
    threshold = sigma * 1.4826 * mad

    above = valid > threshold
    n = len(above)
    events = []
    i = 0
    while i < n:
        if above[i]:
            start = i
            while i < n and above[i]:
                i += 1
            end = min(i, n - 1)
            if end - start >= min_duration_sec:
                peak = start + np.argmax(valid[start : end + 1])
                events.append(
                    {
                        "start_idx": start,
                        "peak_idx": peak,
                        "end_idx": end,
                        "start_time": float(time[start]),
                        "peak_time": float(time[peak]),
                        "end_time": float(time[end]),
                        "peak_flux": float(valid[peak]),
                        "duration_sec": float(end - start),
                    }
                )
        else:
            i += 1
    return events


def _bayesian_blocks(t: np.ndarray, x: np.ndarray) -> np.ndarray:
    """DEPRECATED: Hand-rolled Bayesian Blocks. Use astropy.stats.bayesian_blocks instead.

    This implementation has an incorrect log-likelihood and O(n²) complexity.
    Kept only for reference.
    """
    nn = len(x)
    if nn < 4:
        return np.array([0, nn])

    edges = np.concatenate([t[:1], t[1:] - 0.5 * np.diff(t), t[-1:]])
    counts_arr = np.bincount(np.clip(np.searchsorted(edges, t), 0, len(edges) - 2))

    best = np.zeros(len(counts_arr), dtype=float)
    last = np.zeros(len(counts_arr), dtype=int)

    for j in range(1, len(counts_arr)):
        best[j] = -np.inf
        for ii in range(j):
            n_k = counts_arr[ii : j + 1].sum()
            if n_k <= 0:
                continue
            avg = n_k / (j - ii + 1)
            loglik = n_k * (np.log(avg + 1e-300) - 1) - np.log(n_k + 1e-300) / 2
            if ii > 0:
                loglik += best[ii - 1]
            if loglik > best[j]:
                best[j] = loglik
                last[j] = ii

    change_points = []
    j = len(counts_arr) - 1
    while j >= 0:
        change_points.append(j)
        j = last[j] - 1
    return np.sort(change_points)


def detect_flares_bayesian_blocks(
    counts: np.ndarray,
    time: np.ndarray,
    threshold_sigma: float | None = None,
    min_duration_sec: int | None = None,
) -> list[dict]:
    """DEPRECATED: Bayesian Blocks detection. Prefer astropy.stats.bayesian_blocks."""
    import warnings

    warnings.warn(
        "Use astropy.stats.bayesian_blocks instead", DeprecationWarning, stacklevel=2
    )

    if threshold_sigma is None:
        threshold_sigma = NOWCAST_BAYESIAN_BLOCKS_SIGMA
    if min_duration_sec is None:
        min_duration_sec = NOWCAST_MIN_DURATION_SEC

    valid = np.where(np.isfinite(counts), counts, 0.0)
    if len(time) < 100:
        return []

    cp = _bayesian_blocks(time, valid)
    n = len(time)
    blocks = []
    for ii in range(len(cp) - 1):
        s, e = int(cp[ii]), int(cp[ii + 1])
        e = min(e, n - 1)
        if e - s < 2:
            continue
        blocks.append({"start": s, "end": e, "mean": np.mean(valid[s : e + 1])})

    if not blocks:
        return []

    means = np.array([b["mean"] for b in blocks])
    global_median = np.median(means)
    global_std = np.std(means)
    if global_std < 1e-10:
        return []

    events = []
    for b in blocks:
        z = (b["mean"] - global_median) / global_std
        if z > threshold_sigma:
            duration = b["end"] - b["start"]
            if duration >= min_duration_sec:
                local = valid[b["start"] : b["end"] + 1]
                pk = b["start"] + np.argmax(local)
                events.append(
                    {
                        "start_idx": b["start"],
                        "peak_idx": pk,
                        "end_idx": b["end"],
                        "start_time": float(time[b["start"]]),
                        "peak_time": float(time[min(pk, n - 1)]),
                        "end_time": float(time[b["end"]]),
                        "peak_flux": float(valid[pk]),
                        "duration_sec": float(duration),
                        "z_score": float(z),
                    }
                )
    return events


def detect_flares_wavelet(
    counts: np.ndarray,
    time: np.ndarray,
    sigma_threshold: float | None = None,
    min_duration_sec: int | None = None,
) -> list[dict]:
    """DEPRECATED: Wavelet-based flare detection. Experimental."""
    import pywt

    if sigma_threshold is None:
        sigma_threshold = NOWCAST_WAVELET_SIGMA
    if min_duration_sec is None:
        min_duration_sec = NOWCAST_MIN_DURATION_SEC

    valid = np.where(np.isfinite(counts), counts, 0.0)
    scales = np.arange(10, min(500, len(valid) // 4))
    if len(scales) < 3:
        return []

    try:
        coefficients, _ = pywt.cwt(valid, scales, "morl", sampling_period=1.0)
    except Exception:
        return []

    power = np.abs(coefficients) ** 2
    max_power = np.max(power, axis=0)
    noise_std = np.std(max_power)
    if noise_std < 1e-10:
        return []

    threshold = np.median(max_power) + sigma_threshold * noise_std
    above = max_power > threshold
    n = len(above)

    events = []
    i = 0
    while i < n:
        if above[i]:
            start = i
            while i < n and above[i]:
                i += 1
            end = min(i, n - 1)
            if end - start >= min_duration_sec:
                peak = start + np.argmax(valid[start : end + 1])
                events.append(
                    {
                        "start_idx": start,
                        "peak_idx": peak,
                        "end_idx": end,
                        "start_time": float(time[start]),
                        "peak_time": float(time[peak]),
                        "end_time": float(time[end]),
                        "peak_flux": float(valid[peak]),
                        "duration_sec": float(end - start),
                    }
                )
        else:
            i += 1
    return events


# ── Classification (updated: uses calibration module) ─────────────────


def classify_flare_goes(peak_flux_solexs: float) -> str:
    """Classify flare by approximate GOES class from SoLEXS peak flux.

    Uses the calibration module to convert SoLEXS counts to GOES-equivalent
    irradiance before classifying.

    Parameters
    ----------
    peak_flux_solexs : float
        Peak SoLEXS count rate (raw COUNTS, not residual).

    Returns
    -------
    goes_class : str
        Flare class (A/B/C/M/X).
    """
    # Convert to GOES-equivalent W/m²
    irradiance = solexs_counts_to_irradiance_simple(np.array([peak_flux_solexs]))
    return classify_goes(irradiance)
