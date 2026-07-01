"""
Stage 2 — Phase 8: Event-level + auxiliary features.
#20 Windowed trigger, #21 CZT coincidence, #25 Cross-corr with lag, #35 EMD periods.
"""

import numpy as np
from pathlib import Path
from astropy.io import fits
from scipy import signal

STAGE1 = Path("data/processed/stage1_20260623.npz")
# HEL1OS event files — may not exist for all dates
EVT = Path("../data/processed/hel1os/2026/06/23/evt.fits")
OUT_DIR = Path("dist/features")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "phase8_event_aux.npz"


def extract():
    ds1 = np.load(STAGE1, allow_pickle=True)
    t = ds1["time"].astype(np.float64)
    sxr_excess = ds1["sxr_excess"].astype(np.float64)
    hxr_flux = ds1["hxr_flux"].astype(np.float64)
    master_flag = ds1["master_flag"]
    n = 86400
    features = {}

    # ── #20 Windowed trigger (W1/W2 proxy) ────────────────────
    print("  Computing windowed trigger...")
    hxr_smooth = signal.savgol_filter(
        np.nan_to_num(hxr_flux, nan=0), 11, 2, mode="constant"
    )
    short_window = 1
    long_window = 5
    hxr_short = np.convolve(
        np.nan_to_num(hxr_flux, nan=0),
        np.ones(short_window) / short_window,
        mode="same",
    )
    hxr_long = np.convolve(
        np.nan_to_num(hxr_flux, nan=0), np.ones(long_window) / long_window, mode="same"
    )
    trigger_ratio = np.full(n, np.nan, dtype=np.float32)
    ok = hxr_long > 0.1
    trigger_ratio[ok] = hxr_short[ok] / hxr_long[ok]
    trigger_ratio = np.clip(trigger_ratio, 0, 10)
    features["hxr_window_trigger"] = trigger_ratio
    # Binary trigger flag: ratio > 2 (abrupt rise) OR absolute > 5-sigma bg
    hxr_bg = np.nanmedian(hxr_flux[:600])  # first 10 min as bg
    hxr_sigma = np.nanstd(hxr_flux[:600])
    if np.isnan(hxr_sigma) or hxr_sigma == 0:
        hxr_sigma = 1.0
    binary_trigger = (
        (trigger_ratio > 2.0) | (hxr_flux > hxr_bg + 5 * hxr_sigma)
    ).astype(np.int8)
    features["hxr_binary_trigger"] = binary_trigger

    # ── #25 Cross-correlation with lag ─────────────────────────
    print("  Computing rolling cross-correlation...")
    lag_max = np.full(n, np.nan, dtype=np.float32)
    corr_max = np.full(n, np.nan, dtype=np.float32)
    half_w = 300  # ±5 min window
    hxr_filled = np.nan_to_num(hxr_flux, nan=0)
    sxr_filled = np.nan_to_num(sxr_excess, nan=0)
    lags = np.arange(-30, 31)
    for i in range(half_w, n - half_w):
        if master_flag[i] != 0:
            continue
        s = sxr_filled[i - half_w : i + half_w + 1]
        h = hxr_filled[i - half_w : i + half_w + 1]
        corr = np.correlate(s - s.mean(), h - h.mean(), mode="same")
        corr /= np.std(s) * np.std(h) * len(s) + 1e-30
        mid = len(corr) // 2
        max_idx = np.argmax(corr[mid - 30 : mid + 31])
        lag_max[i] = max_idx - 30
        corr_max[i] = corr[mid - 30 + max_idx]
        if i % 21600 == 0:
            print(
                f"    Cross-corr: t={i}s, max_corr={corr_max[i]:.3f} @ lag={lag_max[i]:.0f}s"
            )

    features["xcorr_lag"] = lag_max
    features["xcorr_max"] = corr_max

    # ── #35 EMD-derived periods (simplified) ──────────────────
    print("  Computing EMD proxy (Hilbert-Huang transform)...")
    # Simplified: use detrended fluctuation analysis proxy
    # For each flare window, compute dominant period via autocorrelation
    flare_id = ds1["flare_id"]
    emd_period = np.full(n, np.nan, dtype=np.float32)
    for fid in sorted(set(f for f in flare_id if f > 0)):
        mask = flare_id == fid
        idx = np.where(mask)[0]
        sx = sxr_excess[idx]
        sx_c = sx[~np.isnan(sx)]
        if len(sx_c) < 50:
            continue
        # Autocorrelation to find dominant period
        acf = np.correlate(sx_c - sx_c.mean(), sx_c - sx_c.mean(), mode="full")
        acf = acf[len(acf) // 2 :] / acf.max()
        # Find first significant minimum after the zero-lag peak
        half = len(acf) // 2
        peaks = signal.find_peaks(-acf[:half], height=-0.5)[0]
        if len(peaks) > 0:
            period = peaks[0]
        else:
            # First zero crossing
            zc = np.where(acf[1:] * acf[:-1] <= 0)[0]
            period = zc[0] + 1 if len(zc) > 0 else 100
        emd_period[mask] = max(period, 5)

    features["emd_dominant_period"] = emd_period

    # ── #21 CZT coincidence mask ──────────────────────────────
    print("  Computing CZT coincidence flag...")
    czt_coinc = np.zeros(n, dtype=np.int8)
    try:
        with fits.open(EVT) as h:
            czt1 = h[3].data
            czt2 = h[4].data
        if len(czt1) > 0:
            mjd = np.concatenate(
                [czt1["mjd"].astype(np.float64), czt2["mjd"].astype(np.float64)]
            )
            pix = np.concatenate([czt1["pix"].astype(int), czt2["pix"].astype(int)])
            ener = np.concatenate(
                [czt1["ener"].astype(np.float64), czt2["ener"].astype(np.float64)]
            )
            # Sort by pixel, then by time
            order = np.lexsort((mjd, pix))
            mjd = mjd[order]
            pix = pix[order]
            ener = ener[order]
            # Find events from same pixel within 6 µs
            dt_threshold = 6e-5  # 6 µs in days (MJD units)
            for i in range(1, len(mjd)):
                if pix[i] == pix[i - 1] and (mjd[i] - mjd[i - 1]) < dt_threshold:
                    # Convert to 1s bin
                    unix_t = (mjd[i] - 40587) * 86400
                    bin_idx = int(np.round(unix_t - t[0]))
                    if 0 <= bin_idx < n:
                        czt_coinc[bin_idx] = 1
        print(f"    Found {czt_coinc.sum()} coincidence events")
    except Exception as e:
        print(f"    CZT coincidence failed: {e}")

    features["czt_coincidence_flag"] = czt_coinc

    # ── Metadata ──────────────────────────────────────────────
    metadata = {}
    for k, v in features.items():
        nnan = int(np.isnan(v).sum()) if v.dtype.kind == "f" else 0
        metadata[f"{k}_nan"] = nnan
        if v.dtype.kind == "f":
            vv = v[~np.isnan(v)]
            metadata[f"{k}_min"] = float(vv.min()) if len(vv) > 0 else np.nan
            metadata[f"{k}_max"] = float(vv.max()) if len(vv) > 0 else np.nan
        elif v.dtype.kind in ("i", "b"):
            metadata[f"{k}_min"] = int(v.min())
            metadata[f"{k}_max"] = int(v.max())

    metadata["n_features"] = len(features)
    metadata["feature_names"] = list(features.keys())
    metadata["phase"] = 8
    metadata["source"] = f"{STAGE1.name}, {EVT.name}"

    np.savez_compressed(OUT_PATH, **features, __metadata__=metadata)
    print(f"Phase 8 done -> {OUT_PATH}")
    print(f"  {len(features)} features")
    for k, v in features.items():
        nnan = metadata.get(f"{k}_nan", -1)
        lo = metadata.get(f"{k}_min", "?")
        hi = metadata.get(f"{k}_max", "?")
        print(f"    {k:30s} nan={nnan:<6d}  range=[{lo:.4g},{hi:.4g}]")

    return features


if __name__ == "__main__":
    extract()
