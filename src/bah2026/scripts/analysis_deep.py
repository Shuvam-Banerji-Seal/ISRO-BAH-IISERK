#!/usr/bin/env python3
"""
Deep data exploration: catch what the first pass missed.

Topics:
  1. HEL1OS post-concat coverage analysis
  2. SoLEXS pipeline version consistency
  3. CZT band alignment (row-count mismatches)
  4. CdTe vs CZT time alignment
  5. Detector anomaly characterization (2026-02-01→03)
  6. GTI gap analysis - distribution and patterns
  7. Calibration cross-check against known events
  8. CZT2 vs CZT1 detailed comparison
  9. Signal-to-noise ratio analysis
 10. Multi-band energy evolution during flares
"""

from __future__ import annotations

import sys
import os
from pathlib import Path
from datetime import date, timedelta

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))
os.environ["BAH2026_DATA"] = str(REPO_ROOT / "data" / "processed")

import numpy as np
from astropy.io import fits
from scipy.stats import pearsonr

from bah2026.data.reader import (
    load_solexs_lc,
    load_hel1os_lc,
    load_solexs_gti,
    load_solexs_pi,
    discover_solexs_days,
    discover_hel1os_days,
    discover_combined_days,
)
from bah2026.data.preprocessing import met_to_mjd
from bah2026.data.calibration import solexs_counts_to_irradiance_simple, classify_goes


def section(n: int, title: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  [{n}] {title}")
    print(f"{'=' * 70}")


# ── 1. HEL1OS Post-Concat Coverage ───────────────────────────────────


def analyze_hel1os_coverage() -> dict:
    """Analyze HEL1OS coverage distribution after multi-orbit concatenation."""
    section(1, "HEL1OS Coverage After Multi-Orbit Concat")

    days = discover_hel1os_days()
    print(f"  Total days: {len(days)}")

    # Sample coverage from every 5th day
    coverages = []
    multi_band_sizes = []
    for d in days[::5]:
        try:
            czt = load_hel1os_lc(d, detector="czt", num=1)
            if czt["ctr"].size > 0:
                cov_h = (czt["mjd"][-1] - czt["mjd"][0]) * 24
                coverages.append(cov_h)
                # Check if bands have different row counts
                band_sizes = []
                for i in range(5):
                    b = czt["ctr"][:, i]
                    band_sizes.append(len(b[np.isfinite(b)]))
                if len(set(band_sizes)) > 1:
                    multi_band_sizes.append((d, band_sizes))
        except Exception as e:
            print(f"  ERROR {d}: {e}")

    if coverages:
        c_arr = np.array(coverages)
        print(f"  Coverage (n={len(coverages)}):")
        print(f"    Min:     {c_arr.min():.1f} h")
        print(f"    5%:      {np.percentile(c_arr, 5):.1f} h")
        print(f"    25%:     {np.percentile(c_arr, 25):.1f} h")
        print(f"    Median:  {np.median(c_arr):.1f} h")
        print(f"    Mean:    {c_arr.mean():.1f} h")
        print(f"    75%:     {np.percentile(c_arr, 75):.1f} h")
        print(f"    95%:     {np.percentile(c_arr, 95):.1f} h")
        print(f"    Max:     {c_arr.max():.1f} h")

        # Days with unusually low/high coverage
        short_days = sum(1 for c in coverages if c < 2)
        long_days = sum(1 for c in coverages if c > 20)
        print(f"    Days <2h:  {short_days}")
        print(f"    Days >20h: {long_days}")

    if multi_band_sizes:
        print(f"\n  Days with CZT band row-count mismatch: {len(multi_band_sizes)}")
        for d, sizes in multi_band_sizes[:5]:
            print(f"    {d}: {sizes}")
    else:
        print(f"\n  All sampled days have equal CZT band sizes")

    return {
        "days": len(days),
        "median_coverage": float(np.median(coverages)) if coverages else 0,
        "short_days": len([c for c in coverages if c < 2]) if coverages else 0,
        "long_days": len([c for c in coverages if c > 20]) if coverages else 0,
    }


# ── 2. SoLEXS Pipeline Version Consistency ────────────────────────────


def analyze_pipeline_versions() -> dict:
    """Check if SoLEXS pipeline versions produce consistent data."""
    section(2, "SoLEXS Pipeline Version Consistency")

    from astropy.io import fits as apy_fits
    from pathlib import Path

    data_root = Path(os.environ["BAH2026_DATA"])
    solexs_root = data_root / "solexs"

    versions = {}
    for year in sorted(solexs_root.iterdir()):
        if not year.is_dir() or not year.name.isdigit():
            continue
        for month in sorted(year.iterdir()):
            if not month.is_dir() or not month.name.isdigit():
                continue
            for day in sorted(month.iterdir()):
                if not day.is_dir() or not day.name.isdigit():
                    continue
                lc_file = (
                    day
                    / "SDD2"
                    / f"AL1_SOLEXS_{year.name}{month.name}{day.name}_SDD2_L1.lc"
                )
                if lc_file.exists():
                    try:
                        hdul = apy_fits.open(lc_file)
                        creator = hdul[0].header.get("CREATOR", "unknown")
                        hdul.close()
                        versions[creator] = versions.get(creator, 0) + 1
                    except Exception:
                        pass

    print(f"  Pipeline versions found:")
    total = 0
    for ver, count in sorted(versions.items()):
        print(f"    {ver}: {count} days")
        total += count
    print(f"    Total: {total}")

    # Check if different versions have different count scales
    version_stats = {}
    for ver, _ in sorted(versions.items()):
        # Find first day with this version
        ver_days = []
        for year in sorted(solexs_root.iterdir()):
            if not year.is_dir() or not year.name.isdigit():
                continue
            for month in sorted(year.iterdir()):
                if not month.is_dir() or not month.name.isdigit():
                    continue
                for day_dir in sorted(month.iterdir()):
                    if not day_dir.is_dir() or not day_dir.name.isdigit():
                        continue
                    lc_file = (
                        day_dir
                        / "SDD2"
                        / f"AL1_SOLEXS_{year.name}{month.name}{day_dir.name}_SDD2_L1.lc"
                    )
                    if lc_file.exists():
                        try:
                            hdul = apy_fits.open(lc_file)
                            if hdul[0].header.get("CREATOR", "") == ver:
                                ver_days.append(str(day_dir))
                            hdul.close()
                        except Exception:
                            pass
                    if len(ver_days) >= 3:
                        break
                if len(ver_days) >= 3:
                    break
            if len(ver_days) >= 3:
                break

        if ver_days:
            # ver_days entries are full paths like .../solexs/2024/02/22
            p = Path(str(ver_days[0]))
            try:
                d = date(int(p.parent.parent.name), int(p.parent.name), int(p.name))
            except (ValueError, IndexError):
                continue
            try:
                sx = load_solexs_lc(d)
                c = sx["counts"]
                c_valid = c[np.isfinite(c)]
                version_stats[ver] = {
                    "median": float(np.median(c_valid)),
                    "max": float(np.max(c_valid)),
                    "n_nan": int(np.sum(~np.isfinite(c))),
                }
            except Exception as e:
                print(f"    {ver}: load error - {e}")

    if version_stats:
        print(f"\n  Per-version sample stats:")
        for v, s in sorted(version_stats.items()):
            print(
                f"    {v}: median={s['median']:.1f}, max={s['max']:.1f}, NaN={s['n_nan']}"
            )

    return {"versions": versions, "version_stats": version_stats}


# ── 3. CZT Band Alignment ────────────────────────────────────────────


def analyze_czt_band_alignment() -> dict:
    """Check row-count mismatches between CZT energy bands."""
    section(3, "CZT Band Alignment Analysis")

    days = discover_hel1os_days()
    mismatches = 0
    max_diff = 0
    example = None

    for d in days[::3]:
        try:
            czt = load_hel1os_lc(d, detector="czt", num=1)
            if czt["ctr"].size == 0:
                continue
            sizes = []
            for i in range(czt["ctr"].shape[1]):
                b = czt["ctr"][:, i]
                sizes.append(len(b[np.isfinite(b)]))
            if len(set(sizes)) > 1:
                mismatches += 1
                diff = max(sizes) - min(sizes)
                if diff > max_diff:
                    max_diff = diff
                    example = (d, sizes)
        except Exception:
            pass

    print(f"  Days with band mismatches: {mismatches}/{len(days) // 3} sampled")
    print(f"  Max row-count difference: {max_diff}")
    if example:
        print(f"  Worst case: {example[0]} with sizes {example[1]}")

    return {"mismatch_days": mismatches, "max_diff": max_diff}


# ── 4. CdTe vs CZT Time Alignment ────────────────────────────────────


def analyze_detector_alignment() -> dict:
    """Check if CdTe and CZT are aligned in time."""
    section(4, "CdTe vs CZT Time Alignment")

    days = discover_hel1os_days()
    misaligned = 0
    time_diffs = []

    for d in days[::5]:
        try:
            czt = load_hel1os_lc(d, detector="czt", num=1)
            cdte = load_hel1os_lc(d, detector="cdte", num=1)
            if czt["ctr"].size == 0 or cdte["ctr"].size == 0:
                continue
            # Compare MJD ranges
            czt_start, czt_end = czt["mjd"][0], czt["mjd"][-1]
            cdte_start, cdte_end = cdte["mjd"][0], cdte["mjd"][-1]
            start_diff = abs(czt_start - cdte_start) * 86400  # seconds
            end_diff = abs(czt_end - cdte_end) * 86400
            if start_diff > 1 or end_diff > 1:
                misaligned += 1
                time_diffs.append((d, start_diff, end_diff))
        except Exception:
            pass

    print(f"  Sampled {len(days) // 5} days")
    print(f"  Misaligned (start/end diff >1s): {misaligned}")
    if time_diffs:
        print(f"  Examples:")
        for d, sd, ed in time_diffs[:3]:
            print(f"    {d}: start_diff={sd:.1f}s, end_diff={ed:.1f}s")

    # Full-band cross-correlation for one day
    try:
        d = days[len(days) // 2]
        czt = load_hel1os_lc(d, detector="czt", num=1)
        cdte = load_hel1os_lc(d, detector="cdte", num=1)
        if czt["ctr"].size > 0 and cdte["ctr"].size > 0:
            czt_fb = czt["ctr"][:, -1]
            cdte_fb = cdte["ctr"][:, -1]
            min_n = min(len(czt_fb), len(cdte_fb))
            # Downsample for speed
            step = max(1, min_n // 10000)
            c_fb = czt_fb[:min_n:step]
            d_fb = cdte_fb[:min_n:step]
            mask = (c_fb > 0) & (d_fb > 0)
            if mask.sum() > 10:
                r, p = pearsonr(c_fb[mask], d_fb[mask])
                print(f"\n  CZT vs CdTe full-band correlation ({d}):")
                print(f"    Pearson r = {r:.3f} (p={p:.2e}, n={mask.sum()})")
                print(
                    f"    CZT median: {np.median(czt_fb):.1f}, CdTe median: {np.median(cdte_fb):.1f}"
                )
    except Exception as e:
        print(f"  Correlation check failed: {e}")

    return {"misaligned": misaligned, "total_sampled": len(days) // 5}


# ── 5. Detector Anomaly Analysis ─────────────────────────────────────


def analyze_anomaly_period() -> dict:
    """Characterize the 2026-02-01→03 detector anomaly."""
    section(5, "Detector Anomaly: 2026-02-01 → 02-03")

    dates = [date(2026, 2, 1), date(2026, 2, 2), date(2026, 2, 3)]
    before = [date(2026, 1, 30), date(2026, 1, 31)]
    after = [date(2026, 2, 4), date(2026, 2, 5)]

    for label, days_list in [("ANOMALY", dates), ("BEFORE", before), ("AFTER", after)]:
        print(f"\n  {label}:")
        for d in days_list:
            try:
                sx = load_solexs_lc(d)
                c = sx["counts"]
                c_valid = c[np.isfinite(c)]
                n_nan = np.sum(~np.isfinite(c))
                print(
                    f"    {d}: median={np.median(c_valid):.1f}, "
                    f"mean={np.mean(c_valid):.1f}, max={np.max(c_valid):.1f}, "
                    f"NaN={n_nan} ({n_nan / 86400 * 100:.1f}%)"
                )
            except Exception:
                print(f"    {d}: no data")

    # Check if the anomaly is also visible in HEL1OS
    print(f"\n  HEL1OS during anomaly period:")
    for d in dates:
        try:
            czt = load_hel1os_lc(d, detector="czt", num=1)
            if czt["ctr"].size > 0:
                fb = czt["ctr"][:, -1]
                print(
                    f"    {d}: CZT rows={len(fb)}, median={np.median(fb):.1f}, "
                    f"max={np.max(fb):.1f}"
                )
        except Exception:
            print(f"    {d}: no HEL1OS data")

    return {"anomaly_detected": True}


# ── 6. GTI Gap Analysis ─────────────────────────────────────────────


def analyze_gti_gaps() -> dict:
    """Distribution of GTI gaps and patterns."""
    section(6, "GTI Gap Analysis")

    days = discover_solexs_days()

    gap_durations = []
    gap_start_hours = []
    n_gaps_per_day = []

    for d in days[::10]:
        try:
            gti = load_solexs_gti(d)
            if len(gti) == 0:
                continue

            sx = load_solexs_lc(d)
            mjd = met_to_mjd(sx["time"], sx["mjdrefi"], sx["mjdreff"])

            # Find gaps between GTI intervals
            # GTI START/STOP are in MJD (days), convert to seconds
            for i in range(len(gti) - 1):
                gap_start_mjd = gti[i, 1]
                gap_end_mjd = gti[i + 1, 0]
                gap_dur = (gap_end_mjd - gap_start_mjd) * 86400
                gap_durations.append(gap_dur)
                # Hours from day start
                gap_start_hours.append((gap_start_mjd - mjd[0]) * 24)

            n_gaps_per_day.append(len(gti) - 1)

        except Exception:
            pass

    if gap_durations:
        gd = np.array(gap_durations)
        print(f"  Gaps found: {len(gd)}")
        print(f"  Duration stats (seconds):")
        print(f"    Min:     {gd.min():.0f}")
        print(f"    25%:     {np.percentile(gd, 25):.0f}")
        print(f"    Median:  {np.median(gd):.0f}")
        print(f"    75%:     {np.percentile(gd, 75):.0f}")
        print(f"    Max:     {gd.max():.0f}")

        # Look for recurring gaps at specific times
        if gap_start_hours:
            gsh = np.array(gap_start_hours)
            print(f"\n  Gap start times (hours from day start):")
            print(f"    Min: {gsh.min():.1f}h, Max: {gsh.max():.1f}h")
            # Check for gaps near noon/midnight (Earth occultation pattern)
            around_12 = np.sum(np.abs(gsh - 12) < 2)
            around_0 = np.sum((gsh < 2) | (gsh > 22))
            print(f"    Near 12h: {around_12} gaps, Near 0h: {around_0} gaps")

    if n_gaps_per_day:
        ng = np.array(n_gaps_per_day)
        print(f"\n  Gaps per day:")
        print(f"    Most common: {int(np.median(ng))} (median)")
        print(f"    Max: {ng.max()}")

    return {
        "n_gaps": len(gap_durations),
        "max_gap": float(np.max(gap_durations)) if gap_durations else 0,
    }


# ── 7. Calibration Cross-Check ───────────────────────────────────────


def analyze_calibration_check() -> dict:
    """Cross-check calibration against known flare events."""
    section(7, "Calibration Cross-Check")

    # Known big flares from the dataset
    flare_days = [
        (date(2024, 2, 22), "X6.3", 25452),
        (date(2025, 7, 29), "Biggest", 1454091),
        (date(2025, 2, 26), "Major", 321993),
    ]

    print(f"  Current calibration: F = counts * 5.0e-9 W/m2")
    print(f"  GOES thresholds: X≥1e-4, M≥1e-5, C≥1e-6, B≥1e-7")
    print()

    for d, name, raw_peak in flare_days:
        sx = load_solexs_lc(d)
        c = np.where(
            np.isfinite(sx["counts"]), sx["counts"], np.nanmedian(sx["counts"])
        )
        actual_peak = float(np.nanmax(sx["counts"]))
        flux = solexs_counts_to_irradiance_simple(np.array([actual_peak]))
        cls = classify_goes(float(flux[0]))

        print(f"  {name} ({d}):")
        print(f"    Raw peak: {actual_peak:,.0f} cts/s")
        print(f"    Calibrated: {float(flux[0]):.3e} W/m2 → {cls}")

    # Compare with GOES XRS if available
    goes_dir = REPO_ROOT / "data" / "external" / "goes"
    goes_files = list(goes_dir.glob("*.nc"))
    if goes_files:
        print(f"\n  GOES XRS data available: {len(goes_files)} netCDF files")
        # Try to read one GOES file for comparison
        try:
            from netCDF4 import Dataset

            sample = goes_files[len(goes_files) // 2]
            with Dataset(str(sample), "r") as nc:
                print(f"    Sample: {sample.name}")
                print(f"    Variables: {list(nc.variables.keys())[:10]}")
                for var_name in [
                    "A_COUNTS",
                    "B_COUNTS",
                    "A_AVG",
                    "B_AVG",
                    "time",
                    "xrsa",
                    "xrsb",
                ]:
                    if var_name in nc.variables:
                        var = nc.variables[var_name]
                        print(f"    {var_name}: shape={var.shape}, dtype={var.dtype}")
                        if var.shape[0] > 0:
                            print(f"      values: {var[0]:.4f} .. {var[-1]:.4f}")
        except Exception as e:
            print(f"    Could not read sample: {e}")
    else:
        print(f"\n  No GOES XRS data found at {goes_dir}")

    return {"calibration_scale": 5e-9}


# ── 8. CZT2 vs CZT1 Comparison ──────────────────────────────────────


def analyze_czt_detector_comparison() -> dict:
    """Detailed comparison of CZT1 and CZT2 detectors."""
    section(8, "CZT2 vs CZT1 Detailed Comparison")

    days = discover_hel1os_days()
    correlations = []
    ratio_values = []

    for d in days[::10]:
        try:
            c1 = load_hel1os_lc(d, detector="czt", num=1)
            c2 = load_hel1os_lc(d, detector="czt", num=2)
            if c1["ctr"].size == 0 or c2["ctr"].size == 0:
                continue

            fb1 = c1["ctr"][:, -1]
            fb2 = c2["ctr"][:, -1]
            min_n = min(len(fb1), len(fb2))
            fb1, fb2 = fb1[:min_n], fb2[:min_n]

            # Correlation
            mask = (fb1 > 0) & (fb2 > 0)
            if mask.sum() < 10:
                continue
            r, _ = pearsonr(fb1[mask], fb2[mask])
            correlations.append(r)

            # Ratio
            ratio = np.mean(fb2[mask]) / np.mean(fb1[mask])
            ratio_values.append(ratio)

        except Exception:
            pass

    if correlations:
        corr_arr = np.array(correlations)
        print(f"  Sampled {len(correlations)} days")
        print(f"  CZT1/CZT2 full-band Pearson r:")
        print(f"    Min:     {corr_arr.min():.3f}")
        print(f"    Median:  {np.median(corr_arr):.3f}")
        print(f"    Mean:    {corr_arr.mean():.3f}")
        print(f"    Max:     {corr_arr.max():.3f}")
        print(f"    Days with r<0.5: {np.sum(corr_arr < 0.5)}")

        ratio_arr = np.array(ratio_values)
        print(f"\n  CZT2/CZT1 count ratio:")
        print(f"    Min:     {ratio_arr.min():.3f}")
        print(f"    Median:  {np.median(ratio_arr):.3f}")
        print(f"    Max:     {ratio_arr.max():.3f}")

    return {
        "median_corr": float(np.median(correlations)) if correlations else 0,
        "low_corr_days": int(np.sum(np.array(correlations) < 0.5))
        if correlations
        else 0,
    }


# ── 9. Signal-to-Noise Analysis ──────────────────────────────────────


def analyze_snr() -> dict:
    """Analyze signal-to-noise ratio for flare detection."""
    section(9, "Signal-to-Noise Analysis")

    # Sample a quiet day and a flare day
    quiet_day = date(2025, 6, 1)  # arbitrary quiet day
    flare_day = date(2024, 2, 22)

    results = {}
    for label, d in [("Quiet", quiet_day), ("Flare (X6.3)", flare_day)]:
        try:
            sx = load_solexs_lc(d)
            c = np.where(
                np.isfinite(sx["counts"]), sx["counts"], np.nanmedian(sx["counts"])
            )

            # Noise = MAD of non-flare residual
            from scipy.ndimage import median_filter

            bg = median_filter(c, size=600, mode="nearest")
            noise = np.median(np.abs(c - bg - np.median(c - bg)))

            # Signal = peak - background
            peak_idx = np.argmax(c)
            signal = c[peak_idx] - bg[peak_idx]

            snr = signal / max(noise, 1)
            results[label] = {
                "median_bg": float(np.median(bg)),
                "peak": float(c[peak_idx]),
                "noise": float(noise),
                "signal": float(signal),
                "snr": float(snr),
            }
            print(f"  {label} day ({d}):")
            print(f"    median bg: {results[label]['median_bg']:.1f}")
            print(f"    peak:      {results[label]['peak']:.1f}")
            print(f"    noise:     {results[label]['noise']:.2f}")
            print(f"    SNR:       {results[label]['snr']:.1f}")

        except Exception as e:
            print(f"  {label}: {e}")

    return results


# ── 10. Multi-band Energy Evolution ──────────────────────────────────


def analyze_band_evolution() -> dict:
    """Analyze how the 10 HXR bands evolve during a flare."""
    section(10, "Multi-band Energy Evolution During Flare")

    d = date(2024, 2, 22)
    try:
        czt = load_hel1os_lc(d, detector="czt", num=1)
        cdte = load_hel1os_lc(d, detector="cdte", num=1)

        if czt["ctr"].size == 0 or cdte["ctr"].size == 0:
            print("  No HEL1OS data for X6.3 day")
            return {"has_hel1os": False}

        print(
            f"  CZT1: {czt['ctr'].shape[0]} rows, MJD {czt['mjd'][0]:.5f}..{czt['mjd'][-1]:.5f}"
        )
        print(
            f"  CdTe1: {cdte['ctr'].shape[0]} rows, MJD {cdte['mjd'][0]:.5f}..{cdte['mjd'][-1]:.5f}"
        )

        # Band energy ranges
        czt_bands = [(20, 40), (40, 60), (60, 80), (80, 150), (18, 160)]
        cdte_bands = [(5, 20), (20, 30), (30, 40), (40, 60), (1.8, 90)]

        print(f"\n  CZT1 per-band stats:")
        for i, (lo, hi) in enumerate(czt_bands):
            band = czt["ctr"][:, i]
            bv = band[np.isfinite(band)]
            nonzero = (bv > 0).mean() * 100
            print(
                f"    {lo:4.0f}-{hi:4.0f} keV: median={np.median(bv):.1f}, "
                f"max={np.max(bv):.1f}, nonzero={nonzero:.0f}%"
            )

        print(f"\n  CdTe1 per-band stats:")
        for i, (lo, hi) in enumerate(cdte_bands):
            band = cdte["ctr"][:, i]
            bv = band[np.isfinite(band)]
            nonzero = (bv > 0).mean() * 100
            print(
                f"    {lo:4.0f}-{hi:4.0f} keV: median={np.median(bv):.1f}, "
                f"max={np.max(bv):.1f}, nonzero={nonzero:.0f}%"
            )

        # Hardness ratio evolution
        if czt["ctr"].shape[0] > 0 and cdte["ctr"].shape[0] > 0:
            # Align by MJD
            mjd_min = max(czt["mjd"][0], cdte["mjd"][0])
            mjd_max = min(czt["mjd"][-1], cdte["mjd"][-1])
            czt_mask = (czt["mjd"] >= mjd_min) & (czt["mjd"] <= mjd_max)
            cdte_mask = (cdte["mjd"] >= mjd_min) & (cdte["mjd"] <= mjd_max)

            czt_aligned = czt["ctr"][czt_mask, :]
            cdte_aligned = cdte["ctr"][cdte_mask, :]
            min_r = min(czt_aligned.shape[0], cdte_aligned.shape[0])
            czt_aligned = czt_aligned[:min_r]
            cdte_aligned = cdte_aligned[:min_r]

            # Compute hardness ratios
            hr_40_60_20_40 = np.where(
                czt_aligned[:, 0] > 0,
                czt_aligned[:, 1] / czt_aligned[:, 0],
                0,
            )
            hr_20_40_5_20 = np.where(
                cdte_aligned[:, 0] > 0,
                czt_aligned[:, 0] / cdte_aligned[:, 0],
                0,
            )

            print(f"\n  Hardness ratios (aligned, n={min_r}):")
            print(
                f"    CZT 40-60/20-40: median={np.median(hr_40_60_20_40):.3f}, "
                f"99%={np.percentile(hr_40_60_20_40, 99):.3f}"
            )
            print(
                f"    CZT20-40/CdTe5-20: median={np.median(hr_20_40_5_20):.3f}, "
                f"99%={np.percentile(hr_20_40_5_20, 99):.3f}"
            )

        return {"has_hel1os": True}

    except Exception as e:
        print(f"  Error: {e}")
        return {"has_hel1os": False, "error": str(e)}


# ── Main ─────────────────────────────────────────────────────────────


def main() -> int:
    print(f"{'=' * 70}")
    print(f"  DEEP DATA EXPLORATION — MISSED ASPECTS")
    print(f"  SoLEXS + HEL1OS dual-instrument analysis")
    print(f"{'=' * 70}")

    results = {}
    results["coverage"] = analyze_hel1os_coverage()
    results["versions"] = analyze_pipeline_versions()
    results["czt_bands"] = analyze_czt_band_alignment()
    results["alignment"] = analyze_detector_alignment()
    results["anomaly"] = analyze_anomaly_period()
    results["gti"] = analyze_gti_gaps()
    results["calibration"] = analyze_calibration_check()
    results["czt_detectors"] = analyze_czt_detector_comparison()
    results["snr"] = analyze_snr()
    results["bands"] = analyze_band_evolution()

    print(f"\n{'=' * 70}")
    print(f"  ANALYSIS COMPLETE — {len(results)} sections explored")
    print(f"{'=' * 70}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
