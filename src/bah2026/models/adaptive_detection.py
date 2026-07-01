#!/usr/bin/env python3
"""Adaptive flare detection and classification using SoLEXS + HEL1OS data."""

import numpy as np
from scipy.ndimage import median_filter, percentile_filter
from scipy.signal import find_peaks


def adaptive_threshold_sxrx(counts, window=600, sigma=3.0):
    """Adaptive threshold based on global statistics.

    Uses running median background and global MAD to set threshold.

    Parameters
    ----------
    counts : ndarray
        SoLEXS count rate (1s cadence).
    window : int
        Background estimation window (seconds).
    sigma : float
        Threshold in sigma units (default 3.0).

    Returns
    -------
    threshold : float
        Fixed threshold based on global statistics.
    residual : ndarray
        Count rate minus background.
    bg : ndarray
        Running background.
    """
    bg = median_filter(counts, size=window, mode="nearest")
    residual = counts - bg

    # Use global MAD (robust to flares)
    mad = np.median(np.abs(residual - np.median(residual)))
    threshold = sigma * 1.4826 * mad

    return threshold, residual, bg


def detect_flares_adaptive(counts, time_s, min_duration=60, min_peak=100, sigma=3.0):
    """Detect flares using adaptive threshold.

    Parameters
    ----------
    counts : ndarray
        SoLEXS count rate (1s cadence).
    time_s : ndarray
        Time array (seconds from mission start).
    min_duration : int
        Minimum flare duration (seconds).
    min_peak : float
        Minimum peak count rate to be considered a flare.
    sigma : float
        Threshold sigma (default 3.0).

    Returns
    -------
    flares : list[dict]
        Detected flare events with timing, intensity, and classification.
    """
    threshold, residual, bg = adaptive_threshold_sxrx(counts, sigma=sigma)

    # Find contiguous regions above threshold
    above = residual > threshold
    regions = []
    i = 0
    while i < len(above):
        if above[i]:
            start = i
            while i < len(above) and above[i]:
                i += 1
            end = i - 1
            if end - start >= min_duration:
                regions.append((start, end))
        else:
            i += 1

    # Extract flare properties
    flares = []
    for start, end in regions:
        segment = counts[start : end + 1]
        peak_idx = start + np.argmax(segment)
        peak_flux = float(counts[peak_idx])

        if peak_flux < min_peak:
            continue

        # Classify by SoLEXS intensity
        if peak_flux > 10000:
            sxr_class = "X-equiv"
        elif peak_flux > 1000:
            sxr_class = "M-equiv"
        elif peak_flux > 100:
            sxr_class = "C-equiv"
        else:
            sxr_class = "B-equiv"

        # Duration
        duration = float(time_s[min(end, len(time_s) - 1)] - time_s[start])

        # Rise and fall rates
        rise_mask = np.arange(start, peak_idx + 1)
        fall_mask = np.arange(peak_idx, end + 1)
        rise_rate = (
            float(np.mean(np.diff(counts[rise_mask]))) if len(rise_mask) > 1 else 0
        )
        fall_rate = (
            float(np.mean(np.diff(counts[fall_mask]))) if len(fall_mask) > 1 else 0
        )

        flares.append(
            {
                "start_idx": start,
                "peak_idx": peak_idx,
                "end_idx": end,
                "peak_flux": peak_flux,
                "sxr_class": sxr_class,
                "duration_sec": duration,
                "rise_rate": rise_rate,
                "fall_rate": fall_rate,
            }
        )

    return flares


def classify_solexs_helios(flares, hxr_czt1, hxr_cdte1, time_s, tolerance_sec=60):
    """Classify flares using both SoLEXS and HEL1OS data.

    Combines SXR intensity with HXR confirmation and spectral analysis
    for a more robust classification.

    Parameters
    ----------
    flares : list[dict]
        Flares detected from SoLEXS.
    hxr_czt1 : ndarray
        HEL1OS CZT1 full band (18-160 keV).
    hxr_cdte1 : ndarray
        HEL1OS CdTe1 full band (1.8-90 keV).
    time_s : ndarray
        Time array (seconds).
    tolerance_sec : int
        Time tolerance for HXR coincidence (seconds).

    Returns
    -------
    classified : list[dict]
        Flares with HEL1OS confirmation and combined classification.
    """
    for flare in flares:
        peak_idx = flare["peak_idx"]

        # Check HEL1OS HXR coincidence
        hxr_window_czt = hxr_czt1[
            max(0, peak_idx - tolerance_sec) : min(
                len(hxr_czt1), peak_idx + tolerance_sec
            )
        ]
        hxr_window_cdte = hxr_cdte1[
            max(0, peak_idx - tolerance_sec) : min(
                len(hxr_cdte1), peak_idx + tolerance_sec
            )
        ]

        hxr_peak_czt = float(np.max(hxr_window_czt)) if hxr_window_czt.size > 0 else 0
        hxr_peak_cdte = (
            float(np.max(hxr_window_cdte)) if hxr_window_cdte.size > 0 else 0
        )

        flare["hxr_peak_czt"] = hxr_peak_czt
        flare["hxr_peak_cdte"] = hxr_peak_cdte
        flare["hxr_confirmed"] = hxr_peak_czt > 10 or hxr_peak_cdte > 50

        # Combined classification
        if flare["sxr_class"] == "X-equiv" and flare["hxr_confirmed"]:
            flare["combined_class"] = "X"
        elif flare["sxr_class"] == "M-equiv" and flare["hxr_confirmed"]:
            flare["combined_class"] = "M"
        elif flare["sxr_class"] == "C-equiv" and flare["hxr_confirmed"]:
            flare["combined_class"] = "C"
        elif flare["sxr_class"] == "X-equiv":
            flare["combined_class"] = "X-uncertain"
        elif flare["sxr_class"] == "M-equiv":
            flare["combined_class"] = "M-uncertain"
        else:
            flare["combined_class"] = "C-uncertain"

    return flares
