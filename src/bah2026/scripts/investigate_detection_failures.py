#!/usr/bin/env python3
"""
Investigate why big flares are missed:
  1. 2025-07-29: NaN saturation after peak (1.45M cts/s)
  2. 2025-02-26: X-class not found by SWPC detector

Test different NaN handling strategies and detection thresholds.
"""

from __future__ import annotations

import sys, os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))
os.environ["BAH2026_DATA"] = str(REPO_ROOT / "data" / "processed")

import numpy as np
from datetime import date
from scipy.ndimage import percentile_filter

from bah2026.data.reader import load_solexs_lc
from bah2026.data.calibration import solexs_counts_to_irradiance_simple, classify_goes
from bah2026.models.nowcasting import detect_flares_swpc


def section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def analyze_nan_saturation():
    """Analyze NaN pattern around the 2025-07-29 flare peak."""
    section("NaN Saturation — 2025-07-29 (1.45M cts/s peak)")

    d = date(2025, 7, 29)
    sx = load_solexs_lc(d)
    counts = sx["counts"]
    time_h = sx["time"] / 3600.0

    # Find the max finite value
    valid_mask = np.isfinite(counts)
    peak_idx = np.nanargmax(counts)
    peak_val = float(counts[peak_idx])

    print(f"Peak index: {peak_idx}")
    print(f"Peak value: {peak_val:,.0f} cts/s")
    print(f"Peak time: {time_h[peak_idx]:.2f} h")

    # NaN pattern around peak
    window = slice(max(0, peak_idx - 60), min(len(counts), peak_idx + 120))
    w_counts = counts[window]
    w_time = time_h[window]

    print(f"\nNaN pattern ±60s around peak:")
    for i in range(len(w_counts)):
        marker = " NaN" if not np.isfinite(w_counts[i]) else ""
        if i == 60:
            marker += " <-- PEAK"
        print(f"  t={w_time[i]:.4f}h  val={w_counts[i]:>10.0f}{marker}")

    # After-peak NaN duration
    after_peak = counts[peak_idx:]
    nan_start = None
    for i, v in enumerate(after_peak):
        if not np.isfinite(v) and nan_start is None:
            nan_start = i
        if nan_start is not None and np.isfinite(v):
            nan_duration = i - nan_start
            print(f"\nNaN block at +{nan_start}s after peak, duration: {nan_duration}s")
            break

    # Test different fill strategies
    print(f"\n--- Fill Strategy Comparison ---")

    # Strategy 1: median fill (current)
    c1 = np.where(np.isfinite(counts), counts, np.nanmedian(counts))
    f1 = solexs_counts_to_irradiance_simple(c1)
    print(f"\n1. Median-fill: peak={f1[peak_idx]:.3e}")
    print(f"   SWPC events: {len(detect_flares_swpc(f1, sx['time']))}")

    # Strategy 2: forward fill
    c2 = counts.copy()
    last_valid = np.nanmedian(counts)
    for i in range(len(c2)):
        if np.isfinite(c2[i]):
            last_valid = c2[i]
        else:
            c2[i] = last_valid
    f2 = solexs_counts_to_irradiance_simple(c2)
    print(f"\n2. Forward-fill: peak={f2[peak_idx]:.3e}")
    evts2 = detect_flares_swpc(f2, sx["time"])
    print(f"   SWPC events: {len(evts2)}")
    for evt in evts2:
        print(
            f"     {classify_goes(float(evt['peak_flux']))}: flux={evt['peak_flux']:.3e}, dur={evt['duration_sec']:.0f}s"
        )

    # Strategy 3: Gaussian smooth over NaN region
    from scipy.ndimage import gaussian_filter1d

    c3 = np.where(np.isfinite(counts), counts, 0.0)
    smooth = gaussian_filter1d(c3, sigma=5, mode="constant", cval=0.0)
    # Only use smoothed values where original was NaN
    c3 = np.where(np.isfinite(counts), counts, smooth)
    f3 = solexs_counts_to_irradiance_simple(c3)
    print(f"\n3. Gauss-smooth fill (σ=5): peak={f3[peak_idx]:.3e}")
    evts3 = detect_flares_swpc(f3, sx["time"])
    print(f"   SWPC events: {len(evts3)}")
    for evt in evts3:
        print(
            f"     {classify_goes(float(evt['peak_flux']))}: flux={evt['peak_flux']:.3e}, dur={evt['duration_sec']:.0f}s"
        )


def analyze_feb26_failure():
    """Analyze why 2025-02-26 X-class is missed."""
    section("2025-02-26 — X-class Detection Failure")

    d = date(2025, 2, 26)
    sx = load_solexs_lc(d)
    c = np.where(np.isfinite(sx["counts"]), sx["counts"], np.nanmedian(sx["counts"]))
    time_h = sx["time"] / 3600.0

    peak_idx = int(np.argmax(c))
    peak_val = float(c[peak_idx])
    print(f"Peak index: {peak_idx}")
    print(f"Peak raw: {peak_val:,.0f} cts/s")

    flux = solexs_counts_to_irradiance_simple(c)
    print(
        f"Peak calibrated: {flux[peak_idx]:.3e} W/m2 -> {classify_goes(float(flux[peak_idx]))}"
    )

    # Check profile around peak
    print(f"\nProfile around peak (t={time_h[peak_idx]:.2f}h):")
    window = slice(max(0, peak_idx - 180), min(len(c), peak_idx + 60))
    for i in range(window.start, window.stop, 10):
        marker = " <-- PEAK" if i == peak_idx else ""
        print(f"  t={time_h[i]:.4f}h  raw={c[i]:>8.0f}  flux={flux[i]:.3e}{marker}")

    # Run SWPC with lower min_duration
    for min_dur in [120, 60, 30, 10]:
        evts = detect_flares_swpc(flux, sx["time"], min_duration_sec=min_dur)
        n_x = sum(1 for e in evts if classify_goes(float(e["peak_flux"])) == "X")
        print(f"\nSWPC with min_dur={min_dur}s: {len(evts)} events, X-class: {n_x}")
        for evt in evts[:3]:
            cls = classify_goes(float(evt["peak_flux"]))
            print(
                f"  {cls}: flux={evt['peak_flux']:.3e}, dur={evt['duration_sec']:.0f}s, peak_idx={evt['peak_idx']}"
            )


def main():
    print(f"{'=' * 60}")
    print(f"  Detection Failure Investigation")
    print(f"{'=' * 60}")

    analyze_nan_saturation()
    analyze_feb26_failure()

    print(f"\n{'=' * 60}")
    print(f"  Investigation Complete")
    print(f"  Recommended fixes identified")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
