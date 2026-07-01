"""
Stage 1: Preprocessing & Calibration
====================================
Input:  master_dataset_YYYYMMDD.npz (from Stage 0)
Output: stage1_YYYYMMDD.npz (to Stage 2 Nowcasting)

Pipeline:
  1.2 Quality flag consolidation  → single master_flag (0-7)
  1.3 Particle event detection   → cross-validated SXR/HXR
  1.4 Saturation detection        → histogram ceiling
  1.5 Multi-timescale background  → long trend + short residual
  1.6 Background subtraction      → excess, SNR, anomaly flags
  1.7 GOES cross-calibration      → linregress on quiet periods
  1.8 CZT diagnostic / fallback   → CdTe primary if CZT broken
  1.9 Output assembly             → stage1_YYYYMMDD.npz
"""

import numpy as np
from scipy import ndimage, signal
from scipy.ndimage import median_filter
from scipy.stats import linregress
from datetime import datetime, timezone
from pathlib import Path
import sys
import os

# Ensure src/ is on the path for sibling imports
_src_dir = os.path.dirname(os.path.abspath(__file__))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)
from goes_utils import _cls_str

import time as time_module
import warnings

# ================================================================
# FLAG SCHEMA
# ================================================================
FLAGS = {
    0: "GOOD",
    1: "GTI_GAP",
    2: "HXR_NO_DATA",
    3: "SAA",
    4: "PARTICLE_EVENT",
    5: "SATURATED",
    6: "INSTRUMENTAL",
    7: "MARGINAL",
}


def load_stage0(path=None):
    if path is None:
        path = "data/processed/master_dataset_20260623.npz"
    elif not Path(path).exists():
        # Try pipeline data/processed/
        path = f"data/processed/{Path(path).name}"
    ds = np.load(path, allow_pickle=True)
    meta = ds["__metadata__"].item()
    return ds, meta


def consolidate_flags(ds):
    """1.2 — Merge sxr_quality + hxr_quality + gap_type into master_flag.

    Schema:
      0 GOOD          — SXR in GTI, HXR present, no gap
      1 GTI_GAP       — SXR NaN (outside GTI)
      2 HXR_NO_DATA   — HXR NaN but SXR is good
      3 SAA           — isolated short gap
      4 PARTICLE_EVENT
      5 SATURATED
      6 INSTRUMENTAL
      7 MARGINAL

    Returns master_flag (int16 array), decoded_flag (str array).
    """
    N = len(ds["time"])
    f = np.zeros(N, dtype=np.int16)

    sxr_q = ds["sxr_quality"].astype(int)
    hxr_q = ds["hxr_quality"].astype(int)
    gt = ds["gap_type"].astype(int)
    is_saa = ds["is_saa"]

    # SAA
    f[is_saa] = 3

    # GTI gap: SXR bad (outside GTI)
    f[(sxr_q == 0) & (f == 0)] = 1

    # HXR no data: SXR good but HXR not available
    f[(hxr_q == 0) & (sxr_q >= 1) & (f == 0)] = 2

    # Instrumental gap (gap_type==3, HXR off)
    f[(gt == 3) & (f == 0)] = 1  # treat as GTI_GAP too

    # Default: GOOD
    f[(f == 0)] = 0

    decoded = np.array([FLAGS.get(int(v), "UNKNOWN") for v in f])
    return f, decoded


def detect_particle_events(sxr_flux, hxr_flux, master_flag, t, n_sigma=5, width_max=3):
    """1.3 — Detect particle events via SXR gradient + cross-validate with HXR.

    Returns boolean mask of affected bins (±30s buffer).
    """
    N = len(t)
    particle_mask = np.zeros(N, dtype=bool)

    good = master_flag == 0

    # 1.3a: SXR gradient outlier
    sxr_clean = np.where(np.isnan(sxr_flux), 0, sxr_flux)
    grad = np.gradient(sxr_clean)
    grad_std = np.nanstd(grad[good])
    grad_outliers = np.abs(grad) > n_sigma * grad_std

    # 1.3b: narrow-width HXR peaks (width < 3s)
    hxr_clean = np.where(np.isnan(hxr_flux), 0, hxr_flux)
    peaks, props = signal.find_peaks(hxr_clean, prominence=5)
    narrow_peaks = np.zeros(N, dtype=bool)
    if len(peaks) > 0 and "widths" in props:
        for pi, wi in zip(peaks, props["widths"]):
            if wi < width_max:
                narrow_peaks[max(0, pi - 1) : min(N, pi + 2)] = True

    # Combined candidates: SXR gradient outlier AND (HXR flat or no HXR)
    hxr_present = ~np.isnan(hxr_flux)
    hxr_grad = np.gradient(hxr_clean)
    hxr_grad_std = np.nanstd(hxr_grad[good]) if good.sum() > 10 else 1.0
    hxr_flat = np.abs(hxr_grad) < hxr_grad_std * 2

    # Candidate = SXR extreme gradient AND (HXR missing OR HXR flat)
    candidates = grad_outliers & (~hxr_present | hxr_flat) & good

    # 1.3d: Expand ±30s around each candidate
    for i in np.where(candidates)[0]:
        s = max(0, i - 30)
        e = min(N, i + 31)
        particle_mask[s:e] = True

    # Cross-check: bins where both SXR and HXR spike together = flare, not particle
    both_spike = grad_outliers & (np.abs(hxr_grad) > hxr_grad_std * 3) & hxr_present
    for i in np.where(both_spike)[0]:
        s = max(0, i - 5)
        e = min(N, i + 6)
        particle_mask[s:e] = False

    return particle_mask


def detect_saturation(flux, name="", quantile_thresh=0.99):
    """1.4 — Find saturation ceiling via log-histogram.

    Returns (ceiling, saturated_mask).
    """
    valid = flux[~np.isnan(flux) & (flux > 0)]
    if len(valid) < 100:
        return np.nanmax(flux), np.zeros(len(flux), dtype=bool)

    max_val = np.nanmax(flux)
    nbins = 60
    bins = np.logspace(np.log10(np.nanmin(valid)), np.log10(max_val * 1.05), nbins)
    hist, edges = np.histogram(valid, bins=bins)

    # Find last bin that has >1% of peak bin count
    peak_count = np.max(hist)
    threshold = peak_count * 0.01

    # Find where histogram drops below threshold (near max)
    ceiling_idx = None
    for i in range(len(hist) - 1, 0, -1):
        if hist[i] >= threshold:
            ceiling_idx = i
            break

    if ceiling_idx is None or ceiling_idx < 2:
        ceiling = max_val
    else:
        ceiling = edges[ceiling_idx]

    sat_threshold = 0.95 * ceiling
    saturated = flux >= sat_threshold
    sat_count = np.sum(saturated & ~np.isnan(flux))

    return ceiling, saturated


def estimate_background_multiscale(
    flux, master_flag, t, long_window=3600, short_window=600
):
    """1.5 — Multi-timescale background estimation.

    Long-term trend: rolling median on low-flux GOOD samples (~1hr window).
    Short-term residual: rolling 10th percentile on detrended data (~10min window).

    Robust to NaN and fragmented good segments — uses ndimage.median_filter
    which handles zeros in place of NaN, then masks non-GOOD bins.

    Returns (background, bg_sigma).
    """
    N = len(t)
    good = (master_flag == 0) & ~np.isnan(flux)
    bg = np.full(N, np.nan, dtype=np.float64)
    bg_sigma = np.full(N, np.nan, dtype=np.float64)

    if good.sum() < 100:
        return bg, bg_sigma

    flux_clean = np.where(np.isnan(flux), 0, flux)

    # 1.5a: Long-term trend — rolling median (handles NaN-filled data robustly)
    half_long = long_window // 2
    long_trend = median_filter(flux_clean, size=long_window, mode="reflect").astype(
        np.float64
    )
    # Replace trend with NaN where not GOOD, then interpolate
    long_trend[~good] = np.nan
    # Fill NaN gaps with nearest good value
    good_idx = np.where(good)[0]
    if len(good_idx) > 0:
        for i in range(N):
            if np.isnan(long_trend[i]):
                nearest = good_idx[np.argmin(np.abs(good_idx - i))]
                long_trend[i] = long_trend[nearest]

    # 1.5b: Short-term rolling 10th percentile on detrended residual
    residual = flux_clean - long_trend
    short_bg = np.full(N, np.nan, dtype=np.float64)

    half_short = short_window // 2
    for i in range(half_short, N - half_short):
        seg = residual[i - half_short : i + half_short + 1]
        seg_good = good[i - half_short : i + half_short + 1]
        if seg_good.sum() >= 5:
            short_bg[i] = np.percentile(seg[seg_good], 10)

    # Fill edges
    if np.any(~np.isnan(short_bg)):
        first_valid = int(np.where(~np.isnan(short_bg))[0][0])
        last_valid = int(np.where(~np.isnan(short_bg))[0][-1])
        short_bg[:first_valid] = short_bg[first_valid]
        short_bg[last_valid + 1 :] = short_bg[last_valid]
    short_bg = np.where(np.isnan(short_bg), 0, short_bg)

    # Combine
    bg[:] = long_trend + short_bg
    bg[~good] = np.nan

    # bg_sigma from quiet-sample scatter around combined background
    quiet = (
        good
        & ~np.isnan(bg)
        & (flux_clean < bg)
        & ((bg - flux_clean) < 0.05 * np.nanmean(bg[good]) + 1e-10)
    )
    if quiet.sum() > 10:
        scatter = flux_clean[quiet] - bg[quiet]
        sigma_val = np.nanstd(scatter)
    else:
        sigma_val = np.nanstd(flux_clean[good] - bg[good]) if good.sum() > 10 else 1.0

    bg_sigma[good] = max(float(sigma_val), 1e-10)

    return bg.astype(np.float32), bg_sigma.astype(np.float32)


def goes_cross_calibration(sxr_counts, goes_flux, master_flag, t):
    """1.7 — Fit SoLEXS→GOES calibration via linregress on quiet overlapping periods.

    Quiet = GOES < C1 (1e-6), GTI-verified, no flare, no particle.

    Returns (calibrated_flux, slope, intercept, r2).
    """
    good = (master_flag == 0) & ~np.isnan(sxr_counts) & ~np.isnan(goes_flux)
    quiet = good & (goes_flux < 1e-6) & (goes_flux > 1e-8)

    if quiet.sum() < 100:
        # Fallback: use median ratio
        r = np.nanmedian(goes_flux[good] / np.maximum(sxr_counts[good], 1))
        return (sxr_counts * r).astype(np.float32), r, 0.0, 0.0

    sxr_q = sxr_counts[quiet]
    goes_q = goes_flux[quiet]

    slope, intercept, r_val, p_val, se = linregress(sxr_q, goes_q)
    r2 = r_val**2

    # Regression: goes_flux = slope * sxr_counts + intercept
    # So calibrated SoLEXS flux = slope * sxr_counts + intercept
    calibrated = sxr_counts * slope + intercept
    calibrated = np.where(np.isnan(calibrated), 0, calibrated)
    calibrated = np.where(calibrated < 0, 0, calibrated)

    return calibrated.astype(np.float32), float(slope), float(intercept), float(r2)


def run_stage1(date_str="20260623", master_path=None):
    """Run full Stage 1 pipeline for a given date."""
    if master_path is None:
        master_path = f"data/processed/master_dataset_{date_str}.npz"

    print(f"Stage 1: Preprocessing & Calibration — {date_str}")
    print("=" * 60)
    T0 = time_module.time()

    # Load Stage 0
    ds, meta = load_stage0(master_path)
    t = ds["time"].astype(np.float64)
    N = len(t)

    sxr_flux = ds["sxr_flux"].astype(np.float64)
    sxr_counts = ds["sxr_counts"].astype(np.float64)
    hxr_flux = ds["hxr_flux"].astype(np.float64)
    goes_flux = ds["goes_flux"].astype(np.float64)

    # Sub-band loaded for per-band background
    hxr_bands = {
        "cdte1": ds["hxr_cdte_band1"].astype(np.float64),
        "cdte2": ds["hxr_cdte_band2"].astype(np.float64),
        "cdte3": ds["hxr_cdte_band3"].astype(np.float64),
        "cdte4": ds["hxr_cdte_band4"].astype(np.float64),
        "czt_full": ds["hxr_czt_full"].astype(np.float64),
    }

    # ================================================================
    # 1.2 Quality Flag Consolidation
    # ================================================================
    print("\n[1.2] Quality flag consolidation...", end=" ")
    master_flag, flag_str = consolidate_flags(ds)
    n_good = np.sum(master_flag == 0)
    print(f"GOOD: {n_good}/{N} ({n_good / N * 100:.1f}%)")

    # ================================================================
    # 1.3 Particle Event Detection
    # ================================================================
    print("[1.3] Particle event detection...", end=" ")
    particle_mask = detect_particle_events(sxr_flux, hxr_flux, master_flag, t)
    n_particle = int(np.sum(particle_mask))
    master_flag[particle_mask] = np.maximum(master_flag[particle_mask], 4)
    print(f"{n_particle} bins flagged ({n_particle / N * 100:.2f}%)")

    # ================================================================
    # 1.4 Saturation Detection
    # ================================================================
    print("[1.4] Saturation detection...")
    sxr_ceiling, sxr_sat = detect_saturation(sxr_counts, "SXR")
    hxr_ceiling, hxr_sat = detect_saturation(hxr_flux, "HXR CdTe")
    saturated = sxr_sat | hxr_sat
    n_sat = int(np.sum(saturated))
    master_flag[saturated] = np.maximum(master_flag[saturated], 5)
    print(
        f"  SXR ceiling: {sxr_ceiling:.0f} cts/s  HXR ceiling: {hxr_ceiling:.1f} cts/s"
    )
    print(f"  Saturated: {n_sat} bins ({n_sat / N * 100:.2f}%)")

    # ================================================================
    # 1.5 Background Estimation (Multi-timescale)
    # ================================================================
    print("[1.5] Background estimation (multi-timescale)...")

    # SXR background
    print("  SXR...", end=" ")
    bg_sxr, bg_sigma_sxr = estimate_background_multiscale(
        sxr_flux, master_flag, t, long_window=3600, short_window=600
    )
    print(f"done  (bg range: [{np.nanmin(bg_sxr):.3e}, {np.nanmax(bg_sxr):.3e}])")

    # HXR CdTe broadband background
    print("  HXR CdTe...", end=" ")
    # HXR bg: use 10th percentile of hxr itself during HXR-on periods
    good_hxr = (master_flag == 0) & ~np.isnan(hxr_flux)
    hxr_clean = np.where(np.isnan(hxr_flux), 0, hxr_flux)
    bg_hxr = np.full(N, np.nan, dtype=np.float32)
    bg_sigma_hxr = np.full(N, np.nan, dtype=np.float32)

    half_win = 150  # 5 min HWHM
    for i in range(N):
        s = max(0, i - half_win)
        e = min(N, i + half_win + 1)
        seg = hxr_flux[s:e]
        seg_good = ~np.isnan(seg)
        if seg_good.sum() >= 5:
            bg_hxr[i] = np.percentile(seg[seg_good], 10)

    # Fill edges
    bg_hxr = np.where(np.isnan(bg_hxr), 0, bg_hxr)
    bg_hxr[~good_hxr] = np.nan

    # Sigma from quiet scatter
    quiet_hxr = good_hxr & (hxr_flux < bg_hxr) & (bg_hxr > 0)
    if quiet_hxr.sum() > 10:
        hxr_scatter = hxr_flux[quiet_hxr] - bg_hxr[quiet_hxr]
        hxr_sigma_val = max(np.nanstd(hxr_scatter), 1.0)
    else:
        hxr_sigma_val = 1.0
    bg_sigma_hxr[good_hxr] = hxr_sigma_val
    print(f"done  (bg sigma: {hxr_sigma_val:.2f})")

    # HXR sub-band backgrounds
    print("  HXR sub-bands...", end=" ")
    bg_hxr_bands = {}
    bg_sigma_hxr_bands = {}
    for bname, bdata in hxr_bands.items():
        nb = len(bdata)
        bgb = np.full(N, np.nan, dtype=np.float32)
        bsigma = np.full(N, np.nan, dtype=np.float32)
        gb = ~np.isnan(bdata)
        for i in range(N):
            s = max(0, i - half_win)
            e = min(N, i + half_win + 1)
            seg = bdata[s:e]
            seg_good = ~np.isnan(seg)
            if seg_good.sum() >= 5:
                bgb[i] = np.percentile(seg[seg_good], 10)
        bgb = np.where(np.isnan(bgb), 0, bgb)
        bgb[~gb] = np.nan
        bg_hxr_bands[bname] = bgb
        bg_sigma_hxr_bands[bname] = bsigma
    print("done")

    # ================================================================
    # 1.6 Background Subtraction & Excess
    # ================================================================
    print("[1.6] Background subtraction...")

    good = master_flag == 0

    # SXR excess
    sxr_excess = np.where(good, sxr_flux - bg_sxr, np.nan).astype(np.float32)
    sxr_excess_clipped = np.copy(sxr_excess)
    sxr_excess_clipped[sxr_excess_clipped < 0] = 0  # Clip negative to zero

    sxr_snr = np.where(
        good & (bg_sigma_sxr > 0) & ~np.isnan(bg_sxr),
        (sxr_flux - bg_sxr) / bg_sigma_sxr,
        np.nan,
    ).astype(np.float32)

    # HXR excess
    hxr_excess = np.where(good, hxr_flux - bg_hxr, np.nan).astype(np.float32)
    hxr_excess_clipped = np.copy(hxr_excess)
    hxr_excess_clipped[hxr_excess_clipped < 0] = 0

    hxr_snr = np.where(
        good & (bg_sigma_hxr > 0) & ~np.isnan(bg_hxr),
        (hxr_flux - bg_hxr) / bg_sigma_hxr,
        np.nan,
    ).astype(np.float32)

    print(
        f"  SXR excess range: [{np.nanmin(sxr_excess):.3e}, {np.nanmax(sxr_excess):.3e}]"
    )
    print(f"  SXR SNR range:    [{np.nanmin(sxr_snr):.2f}, {np.nanmax(sxr_snr):.2f}]")
    print(f"  HXR SNR range:    [{np.nanmin(hxr_snr):.2f}, {np.nanmax(hxr_snr):.2f}]")

    # Anomaly flag: high SXR SNR with flat HXR → INSTRUMENTAL
    hxr_available = ~np.isnan(hxr_flux) & good
    anomaly_high_sxr = (sxr_snr > 5) & (~hxr_available | (hxr_snr < 2))
    # But real flares will also have high SXR SNR — only flag if HXR is completely absent
    anomaly_flag = anomaly_high_sxr & (~hxr_available)
    master_flag[anomaly_flag] = np.maximum(master_flag[anomaly_flag], 6)
    n_anom = int(np.sum(anomaly_flag))
    print(f"  Anomalies flagged (SXR>5σ, no HXR): {n_anom}")

    # ================================================================
    # 1.7 GOES Cross-Calibration
    # ================================================================
    print("[1.7] GOES cross-calibration...")
    sxr_flux_cal, cal_slope, cal_intercept, cal_r2 = goes_cross_calibration(
        sxr_counts, goes_flux, master_flag, t
    )
    print(
        f"  linregress: slope={cal_slope:.6e}, intercept={cal_intercept:.6e}, r²={cal_r2:.4f}"
    )
    print(
        f"  Calibrated flux range: [{np.nanmin(sxr_flux_cal):.3e}, {np.nanmax(sxr_flux_cal):.3e}] W/m²"
    )

    # ================================================================
    # 1.8 CZT Diagnostic
    # ================================================================
    print("[1.8] CZT diagnostic...")
    czt_full = ds["hxr_czt_full"].astype(np.float64)
    czt_valid = czt_full[~np.isnan(czt_full)]
    czt_zero_frac = np.sum(czt_valid == 0) / max(len(czt_valid), 1)
    print(f"  CZT zero-fraction: {czt_zero_frac * 100:.1f}%")

    czt_status = None
    if czt_zero_frac > 0.1:
        czt_status = "ZERO_INFLATED"
        print(
            f"  → Zero-inflated ({czt_zero_frac * 100:.0f}% zeros). Using CdTe broadband as primary HXR."
        )
    else:
        czt_status = "OK"
        print("  → CZT background acceptable.")

    # Separate CZT quality flag (does NOT pollute master_flag — that's for CdTe path)
    czt_zero_mask = ~np.isnan(czt_full) & (czt_full == 0)

    # ================================================================
    # 1.9 Output Assembly
    # ================================================================
    print("\n[1.9] Assembling output...")

    stage1 = {
        "time": t.astype(np.float64),
        "sxr_flux": sxr_flux_cal,
        "hxr_flux": hxr_flux.astype(np.float32),
        "bg_sxr": bg_sxr,
        "bg_hxr": bg_hxr,
        "bg_sigma_sxr": bg_sigma_sxr,
        "bg_sigma_hxr": bg_sigma_hxr,
        "sxr_excess": sxr_excess_clipped,
        "hxr_excess": hxr_excess_clipped,
        "sxr_snr": sxr_snr,
        "hxr_snr": hxr_snr,
        "master_flag": master_flag,
        "master_flag_str": flag_str,
        "particle_mask": particle_mask,
        "saturated_mask": saturated,
        "anomaly_flag": anomaly_flag,
        "czt_zero_mask": czt_zero_mask,
        # Per-band backgrounds (sub-band flagged states)
        "hxr_cdte_band1": ds["hxr_cdte_band1"],
        "hxr_cdte_band2": ds["hxr_cdte_band2"],
        "hxr_cdte_band3": ds["hxr_cdte_band3"],
        "hxr_cdte_band4": ds["hxr_cdte_band4"],
        "bg_hxr_band1": bg_hxr_bands.get("cdte1", np.full(N, np.nan)),
        "bg_hxr_band2": bg_hxr_bands.get("cdte2", np.full(N, np.nan)),
        "bg_hxr_band3": bg_hxr_bands.get("cdte3", np.full(N, np.nan)),
        "bg_hxr_band4": bg_hxr_bands.get("cdte4", np.full(N, np.nan)),
        # CZT data (diagnostic, not for primary use)
        "hxr_czt_full": ds["hxr_czt_full"],
        "hxr_czt_band1": ds["hxr_czt_band1"],
        "hxr_czt_band2": ds["hxr_czt_band2"],
        "hxr_czt_band3": ds["hxr_czt_band3"],
        "hxr_czt_band4": ds["hxr_czt_band4"],
        # Supplementary (carried from Stage 0 for completeness)
        "goes_flux": ds["goes_flux"],
        "goes_flux_a": ds["goes_flux_a"],
        "goes_class": ds["goes_class"],
        "flare_id": ds["flare_id"],
        "flare_label": ds["flare_label"],
        "neupert_rho": ds["neupert_rho"],
        "sxr_goes_equiv": ds["sxr_goes_equiv"],
        "sxr_quality": ds["sxr_quality"],
        "hxr_quality": ds["hxr_quality"],
        "gap_type": ds["gap_type"],
    }

    stage1_meta = {
        "stage": "1",
        "version": "v1.0.0",
        "date_created": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "input": str(Path(master_path).resolve()),
        "date_obs": date_str,
        "n_samples": N,
        "n_good": int(n_good),
        "n_particle": n_particle,
        "n_saturated": n_sat,
        "n_anomaly": n_anom,
        "czt_status": czt_status,
        "czt_zero_fraction": float(czt_zero_frac),
        "cal_method": "GOES_linregress",
        "cal_slope": cal_slope,
        "cal_intercept": cal_intercept,
        "cal_r2": cal_r2,
        "bg_method_sxr": "multiscale (SavGol 1h + rolling 10pct 10min)",
        "bg_method_hxr": "rolling_10pct_5min",
        "flags_schema": FLAGS,
    }

    stage1["__metadata__"] = stage1_meta

    out_path = f"data/processed/stage1_{date_str}.npz"
    np.savez_compressed(out_path, **stage1)

    elapsed = time_module.time() - T0
    fsize = Path(out_path).stat().st_size
    print(f"\n{'=' * 60}")
    print(f"✓ {out_path} saved ({fsize / 1024:.0f} KB, {elapsed:.1f}s)")
    print(f"{'=' * 60}")

    # Summary table
    print(f"\nFlag distribution:")
    for fval, fname in sorted(FLAGS.items()):
        count = int(np.sum(master_flag == fval))
        print(f"  {fval}: {fname:20s} → {count:6d} bins ({count / N * 100:5.1f}%)")

    return stage1, stage1_meta


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stage1, meta = run_stage1()
