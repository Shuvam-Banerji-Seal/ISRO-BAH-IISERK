"""
Stage 2 — Phase 10: Extended nowcasting/forecasting features.
Tier 1 (pointwise) + Tier 2 (windowed rolling stats) from the user's feature list.
"""
import numpy as np
from pathlib import Path
from scipy.ndimage import uniform_filter1d, maximum_filter1d

STAGE0 = Path("data/processed/master_dataset_20260623.npz")
STAGE1 = Path("data/processed/stage1_20260623.npz")
PHASE2 = Path("dist/features/phase2_perflare.npz")
OUT_DIR = Path("dist/features")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "phase10_extended.npz"
OUT_PATH.unlink(missing_ok=True)

N = 86400
WINDOWS = {"3min": 180, "5min": 300, "10min": 600}


def _rolling_mean(x, w):
    out = uniform_filter1d(np.nan_to_num(x, nan=0.0), size=w, mode="constant")
    out *= w / min(w, N)
    nan_mask = uniform_filter1d(np.isnan(x).astype(np.float32), size=w, mode="constant") > 0.99
    out[nan_mask] = np.nan
    return out


def _rolling_max(x, w):
    nan_mask = np.isnan(x)
    xf = np.where(nan_mask, -np.inf, x)
    out = maximum_filter1d(xf, size=w, mode="constant")
    all_nan = uniform_filter1d(nan_mask.astype(np.float32), size=w, mode="constant") > 0.99
    out[all_nan] = np.nan
    out[np.isinf(out)] = np.nan
    return out


def _rolling_spearman(x, y, w):
    """Rolling Spearman correlation of x and y over window w (in bins)."""
    n = len(x)
    rho = np.full(n, np.nan, dtype=np.float32)
    x_r = np.nan_to_num(x, nan=0.0)
    y_r = np.nan_to_num(y, nan=0.0)
    from scipy.stats import rankdata
    for i in range(w, n):
        xw = x_r[i - w : i]
        yw = y_r[i - w : i]
        ux = len(np.unique(xw))
        uy = len(np.unique(yw))
        if ux < 3 or uy < 3:
            continue
        rx = rankdata(xw)
        ry = rankdata(yw)
        c = np.corrcoef(rx, ry)
        if c.size >= 4 and not np.isnan(c[0, 1]):
            rho[i] = c[0, 1]
    return rho


def extract():
    s0 = np.load(STAGE0, allow_pickle=True)
    s1 = np.load(STAGE1, allow_pickle=True)
    p2 = np.load(PHASE2, allow_pickle=True)

    features = {}

    sxr_flux = s1["sxr_flux"].astype(np.float32)
    hxr_flux = s1["hxr_flux"].astype(np.float32)
    f_id = s1["flare_id"]
    master_flag = s1["master_flag"].astype(np.int16)

    # ── Tier 1: Pointwise features ──────────────────────────────────
    features["sxr_flux"] = sxr_flux
    features["hxr_flux"] = hxr_flux
    features["sxr_snr"] = s1["sxr_snr"].astype(np.float32)
    features["hxr_snr"] = s1["hxr_snr"].astype(np.float32)

    # First derivatives (from Stage 0, or computed from flux)
    if "sxr_deriv1" in s0:
        dSXR = s0["sxr_deriv1"].astype(np.float32)
    else:
        dSXR = np.gradient(np.nan_to_num(sxr_flux, nan=0.0)).astype(np.float32)
    if "hxr_deriv1" in s0:
        dHXR = s0["hxr_deriv1"].astype(np.float32)
    else:
        dHXR = np.gradient(np.nan_to_num(hxr_flux, nan=0.0)).astype(np.float32)
    features["dSXR_dt"] = dSXR
    features["dHXR_dt"] = dHXR

    # Second derivative
    features["d2SXR_dt2"] = np.gradient(np.nan_to_num(dSXR, nan=0.0)).astype(np.float32)

    # Broadband hardness ratio (CdTe band4 / band1)
    b4 = s1["hxr_cdte_band4"].astype(np.float32)
    b1 = s1["hxr_cdte_band1"].astype(np.float32)
    hr = np.full(N, np.nan, dtype=np.float32)
    ok = b1 > 1e-10
    hr[ok] = b4[ok] / b1[ok]
    features["hardness_ratio"] = hr
    features["d_hardness_dt"] = np.gradient(np.nan_to_num(hr, nan=0.0)).astype(np.float32)

    # HXR availability flag
    features["hxr_available"] = (~np.isnan(hxr_flux)).astype(np.int8)

    # ── Neupert effect features ──────────────────────────────────────
    # Residual: rolling mean(dSXR/dt) - rolling mean(hxr_flux) over 5min
    # Only compute where we have meaningful signal
    dSXR_smooth = _rolling_mean(dSXR, 300)
    hxr_smooth = _rolling_mean(hxr_flux, 300)
    residual = np.where(
        np.isfinite(dSXR_smooth) & np.isfinite(hxr_smooth),
        dSXR_smooth - hxr_smooth,
        np.nan,
    )
    features["neupert_residual"] = residual.astype(np.float32)

    # Normalized deviation: compare z-scored dSXR vs z-scored hxr
    def _zscore(x):
        m = np.nanmean(x)
        s = np.nanstd(x)
        return (x - m) / max(s, 1e-10)
    dev = _zscore(dSXR_smooth) - _zscore(hxr_smooth)
    dev[~np.isfinite(dev)] = np.nan
    features["neupert_deviation"] = dev.astype(np.float32)

    # Rolling 5-min Spearman correlation of dSXR/dt vs hxr_flux
    print("  Computing rolling 5-min Spearman (Neupert correlation)...")
    rho = _rolling_spearman(dSXR, hxr_flux, 300)
    features["neupert_corr_5min"] = rho.astype(np.float32)

    # ── Precursor features ──────────────────────────────────────────
    # accelerating: positive second derivative AND positive first derivative
    acc = (np.nan_to_num(features["d2SXR_dt2"], nan=0.0) > 0) & (dSXR > 0)
    features["accelerating"] = acc.astype(np.int8)

    # precursor_flag: elevated HXR (>3σ) before flare start, not in flare
    hxr_snr = np.nan_to_num(s1["hxr_snr"], nan=0.0)
    precursor = np.zeros(N, dtype=bool)
    # Find flare start indices
    in_flare = f_id > 0
    starts = np.where(np.diff(in_flare.astype(int)) == 1)[0] + 1
    for s in starts:
        t0 = max(0, s - 1800)  # 30 min before flare start
        for i in range(t0, s):
            if hxr_snr[i] > 3 and not in_flare[i]:
                precursor[i] = True
    features["precursor_flag"] = precursor.astype(np.int8)

    # hxr_lead_time: per-flare propagation of dt_peak_hxr_minus_sxr
    ltime = np.full(N, np.nan, dtype=np.float32)
    if "dt_peak_hxr_minus_sxr" in p2:
        dt_p2 = p2["dt_peak_hxr_minus_sxr"].astype(np.float32)
        ltime[:] = dt_p2[:]
    features["hxr_lead_time"] = ltime

    # ── Temporal context ────────────────────────────────────────────
    # flare_count_6h: rolling count of flare starts in 6h (21600s) window
    flare_starts = np.zeros(N, dtype=bool)
    if in_flare.any():
        fs_idx = np.where(np.diff(in_flare.astype(int)) == 1)[0] + 1
        flare_starts[fs_idx] = True
    fc6 = np.zeros(N, dtype=np.int16)
    half_win = 10800  # 3h in each direction for 6h total
    for i in range(N):
        t0 = max(0, i - half_win)
        t1 = min(N, i + half_win)
        fc6[i] = int(flare_starts[t0:t1].sum())
    features["flare_count_6h"] = fc6

    # master_flag as a carry-forward feature
    features["master_flag"] = master_flag

    # ── Tier 2: Windowed rolling statistics ─────────────────────────
    for label, w in WINDOWS.items():
        features[f"sxr_mean_{label}"] = _rolling_mean(sxr_flux, w).astype(np.float32)
        features[f"sxr_max_{label}"] = _rolling_max(sxr_flux, w).astype(np.float32)
        features[f"hxr_mean_{label}"] = _rolling_mean(hxr_flux, w).astype(np.float32)
        features[f"hxr_max_{label}"] = _rolling_max(hxr_flux, w).astype(np.float32)

    # ── Save ────────────────────────────────────────────────────────
    np.savez_compressed(OUT_PATH, **features)
    print(f"  Saved: {OUT_PATH} ({len(features)} features)")
    return features


if __name__ == "__main__":
    extract()
