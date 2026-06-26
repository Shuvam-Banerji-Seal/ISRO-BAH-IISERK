"""Solar flare detection algorithms for nowcasting."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter

from bah2026.config import (
    NOWCAST_THRESHOLD_SIGMA, NOWCAST_MIN_DURATION_SEC,
    NOWCAST_BAYESIAN_BLOCKS_SIGMA, NOWCAST_WAVELET_SIGMA,
    NOWCAST_BACKGROUND_WINDOW_SEC, SOLEXS_TO_GOES_SCALE,
)


def background_subtract_simple(
    counts: np.ndarray, window: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Quick background subtraction using median filter."""
    if window is None:
        window = NOWCAST_BACKGROUND_WINDOW_SEC
    valid = np.where(np.isfinite(counts), counts, np.nanmedian(counts))
    bg = median_filter(valid, size=window, mode="nearest")
    return bg, valid - bg


def detect_flares_threshold(
    counts: np.ndarray,
    time: np.ndarray,
    sigma: float | None = None,
    min_duration_sec: int | None = None,
) -> list[dict]:
    """Detect flares using statistical threshold above running background."""
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
                peak = start + np.argmax(valid[start:end + 1])
                events.append({
                    "start_idx": start,
                    "peak_idx": peak,
                    "end_idx": end,
                    "start_time": float(time[start]),
                    "peak_time": float(time[peak]),
                    "end_time": float(time[end]),
                    "peak_flux": float(valid[peak]),
                    "duration_sec": float(end - start),
                })
        else:
            i += 1
    return events


def _bayesian_blocks(t: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Bayesian blocks algorithm (Scargle et al. 2013)."""
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
            n_k = counts_arr[ii:j + 1].sum()
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
    """Detect flares using Bayesian Blocks adaptive binning."""
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
        blocks.append({"start": s, "end": e, "mean": np.mean(valid[s:e + 1])})

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
                local = valid[b["start"]:b["end"] + 1]
                pk = b["start"] + np.argmax(local)
                events.append({
                    "start_idx": b["start"],
                    "peak_idx": pk,
                    "end_idx": b["end"],
                    "start_time": float(time[b["start"]]),
                    "peak_time": float(time[min(pk, n - 1)]),
                    "end_time": float(time[b["end"]]),
                    "peak_flux": float(valid[pk]),
                    "duration_sec": float(duration),
                    "z_score": float(z),
                })
    return events


def detect_flares_wavelet(
    counts: np.ndarray,
    time: np.ndarray,
    sigma_threshold: float | None = None,
    min_duration_sec: int | None = None,
) -> list[dict]:
    """Detect flares using continuous wavelet transform."""
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
                peak = start + np.argmax(valid[start:end + 1])
                events.append({
                    "start_idx": start,
                    "peak_idx": peak,
                    "end_idx": end,
                    "start_time": float(time[start]),
                    "peak_time": float(time[peak]),
                    "end_time": float(time[end]),
                    "peak_flux": float(valid[peak]),
                    "duration_sec": float(end - start),
                })
        else:
            i += 1
    return events


def classify_flare_goes(peak_flux_solexs: float) -> str:
    """Classify flare by approximate GOES class from SoLEXS peak flux."""
    approx = peak_flux_solexs * SOLEXS_TO_GOES_SCALE
    if approx >= 1e-4:
        return "X"
    elif approx >= 1e-5:
        return "M"
    elif approx >= 1e-6:
        return "C"
    elif approx >= 1e-7:
        return "B"
    return "A"
