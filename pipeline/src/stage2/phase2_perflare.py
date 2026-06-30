"""
Stage 2 — Phase 2: Per-flare catalog features.
Group by flare_id, compute timing/energetic metrics, broadcast to 1s grid.
"""
import numpy as np
from pathlib import Path
from scipy.ndimage import uniform_filter1d
from datetime import datetime, timezone

STAGE0 = Path("data/processed/master_dataset_20260623.npz")
STAGE1 = Path("data/processed/stage1_20260623.npz")
OUT_DIR = Path("dist/features")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "phase2_perflare.npz"


def _class_to_numeric(goes_class):
    """Convert 'C1.5' -> 1.5, 'B3.8' -> 0.38, etc."""
    if not isinstance(goes_class, str) or len(goes_class) < 2:
        return np.nan
    try:
        prefix = goes_class[0].upper()
        val = float(goes_class[1:])
        if prefix == "A":
            return val / 1000.0
        elif prefix == "B":
            return val / 10.0
        elif prefix == "C":
            return val
        elif prefix == "M":
            return val * 10
        elif prefix == "X":
            return val * 100
        else:
            return np.nan
    except (ValueError, IndexError):
        return np.nan


def _compute_per_flare_metrics(ds0, ds1):
    """Return dict of per-flare (8,) arrays, and dict of broadcast (86400,) arrays."""
    feat = {}

    t = ds0["time"].astype(np.float64)
    goes_flux = ds0["goes_flux"].astype(np.float64)
    sxr_flux = ds0["sxr_flux"].astype(np.float64)
    hxr_flux = ds1["hxr_flux"].astype(np.float64)
    hxr_excess = ds1["hxr_excess"].astype(np.float64)
    sxr_deriv1 = ds0["sxr_deriv1"].astype(np.float64)
    goes_class_arr = ds1["goes_class"]
    flare_id = ds1["flare_id"]

    n = 86400
    all_t_start = np.full(n, np.nan, dtype=np.float64)
    all_t_peak = np.full(n, np.nan, dtype=np.float64)
    all_t_end = np.full(n, np.nan, dtype=np.float64)
    all_rise_time = np.full(n, np.nan, dtype=np.float32)
    all_decay_time = np.full(n, np.nan, dtype=np.float32)
    all_duration = np.full(n, np.nan, dtype=np.float32)
    all_peak_flux = np.full(n, np.nan, dtype=np.float32)
    all_peak_class = np.full(n, np.nan, dtype=np.float32)
    all_bg_flux = np.full(n, np.nan, dtype=np.float32)
    all_max_deriv = np.full(n, np.nan, dtype=np.float32)
    all_dt_peak = np.full(n, np.nan, dtype=np.float32)
    all_hxr_fluence = np.full(n, np.nan, dtype=np.float32)
    all_peak_sxr = np.full(n, np.nan, dtype=np.float32)
    all_peak_hxr = np.full(n, np.nan, dtype=np.float32)

    flare_catalog = []

    for fid in sorted(set(fl_id for fl_id in flare_id if fl_id > 0)):
        mask = flare_id == fid
        idx = np.where(mask)[0]
        t_fl = t[idx]
        t0 = t_fl[0]
        t1 = t_fl[-1]

        # Smooth GOES for peak finding
        smooth = uniform_filter1d(goes_flux[idx], size=11, mode="nearest")

        # Peak: max of smoothed within central 80% of window to avoid edges
        margin = max(1, len(idx) // 5)
        central = slice(margin, -margin) if len(idx) > 2 * margin else slice(0, len(idx))
        peak_idx_local = margin + np.argmax(smooth[central])
        peak_idx_global = idx[peak_idx_local]
        t_peak_val = t_fl[peak_idx_local]
        peak_flux_val = goes_flux[idx][peak_idx_local]

        # Class at peak
        peak_class_str = goes_class_arr[peak_idx_global]
        peak_class_num = _class_to_numeric(peak_class_str)

        # Pre-flare background: median of 10 min before start
        bg_t0 = t0 - 600
        bg_mask = (t >= bg_t0) & (t < t0)
        if bg_mask.sum() > 10:
            bg_flux_val = float(np.nanmedian(goes_flux[bg_mask]))
        else:
            bg_flux_val = float(np.nanmin(goes_flux[idx]))

        # Start: first bin where smoothed > bg + 0.1 * (peak - bg)
        threshold = bg_flux_val + 0.1 * (peak_flux_val - bg_flux_val)
        rise_cross = np.where(smooth >= threshold)[0]
        t_start_val = t_fl[rise_cross[0]] if len(rise_cross) > 0 else t0

        # End: last bin where smoothed > bg + 0.1 * (peak - bg) (search backward from peak)
        decay_cross = np.where(smooth >= threshold)[0]
        t_end_val = t_fl[decay_cross[-1]] if len(decay_cross) > 0 else t1

        rise_time_val = t_peak_val - t_start_val
        decay_time_val = t_end_val - t_peak_val
        duration_val = t_end_val - t_start_val

        # Max SXR derivative in flare
        max_deriv_val = float(np.nanmax(np.abs(sxr_deriv1[idx])))

        # Δt_peak: HXR peak time - SXR peak time
        sxr_peak_local = np.nanargmax(sxr_flux[idx])
        sxr_peak_global_t = t_fl[sxr_peak_local]
        hxr_ok = ~np.isnan(hxr_flux[idx])
        if hxr_ok.sum() > 0:
            hxr_peak_local = np.nanargmax(hxr_flux[idx])
            hxr_peak_global_t = t_fl[hxr_peak_local]
            dt_peak_val = hxr_peak_global_t - sxr_peak_global_t
            peak_sxr_val = float(np.nanmax(sxr_flux[idx]))
            peak_hxr_val = float(np.nanmax(hxr_flux[idx]))
            # HXR fluence: integral of excess over flare window
            hxr_flu_val = float(np.nansum(hxr_excess[idx]) * (t_fl[1] - t_fl[0]))
        else:
            dt_peak_val = np.nan
            peak_sxr_val = float(np.nanmax(sxr_flux[idx]))
            peak_hxr_val = np.nan
            hxr_flu_val = np.nan

        flare_catalog.append({
            "flare_id": fid,
            "t_start": t_start_val,
            "t_peak": t_peak_val,
            "t_end": t_end_val,
            "rise_time": rise_time_val,
            "decay_time": decay_time_val,
            "duration": duration_val,
            "peak_flux": peak_flux_val,
            "peak_class_str": peak_class_str,
            "peak_class_num": peak_class_num,
            "bg_flux": bg_flux_val,
            "max_deriv": max_deriv_val,
            "dt_peak": dt_peak_val,
            "hxr_fluence": hxr_flu_val,
            "peak_sxr": peak_sxr_val,
            "peak_hxr": peak_hxr_val,
        })

        # Broadcast to 86400 grid
        for i in idx:
            all_t_start[i] = t_start_val
            all_t_peak[i] = t_peak_val
            all_t_end[i] = t_end_val
            all_rise_time[i] = rise_time_val
            all_decay_time[i] = decay_time_val
            all_duration[i] = duration_val
            all_peak_flux[i] = peak_flux_val
            all_peak_class[i] = peak_class_num
            all_bg_flux[i] = bg_flux_val
            all_max_deriv[i] = max_deriv_val
            all_dt_peak[i] = dt_peak_val
            all_hxr_fluence[i] = hxr_flu_val
            all_peak_sxr[i] = peak_sxr_val
            all_peak_hxr[i] = peak_hxr_val

    feat["t_start"] = all_t_start.astype(np.float32)
    feat["t_peak"] = all_t_peak.astype(np.float32)
    feat["t_end"] = all_t_end.astype(np.float32)
    feat["rise_time"] = all_rise_time
    feat["decay_time"] = all_decay_time
    feat["duration"] = all_duration
    feat["peak_flux"] = all_peak_flux
    feat["peak_goes_class"] = all_peak_class  # numeric B1.2=1.2, C1.5=1.5, etc
    feat["bg_flux"] = all_bg_flux
    feat["max_deriv"] = all_max_deriv
    feat["dt_peak_hxr_minus_sxr"] = all_dt_peak
    feat["hxr_fluence"] = all_hxr_fluence
    feat["peak_sxr_flux"] = all_peak_sxr
    feat["peak_hxr_flux"] = all_peak_hxr

    return feat, flare_catalog


def extract():
    ds0 = np.load(STAGE0, allow_pickle=True)
    ds1 = np.load(STAGE1, allow_pickle=True)

    features, catalog = _compute_per_flare_metrics(ds0, ds1)

    metadata = {}
    for k, v in features.items():
        nnan = int(np.isnan(v).sum()) if v.dtype.kind == "f" else 0
        metadata[f"{k}_nan"] = nnan
        metadata[f"{k}_min"] = float(np.nanmin(v)) if v.dtype.kind == "f" else int(v.min())
        metadata[f"{k}_max"] = float(np.nanmax(v)) if v.dtype.kind == "f" else int(v.max())

    metadata["n_features"] = len(features)
    metadata["feature_names"] = list(features.keys())
    metadata["phase"] = 2
    metadata["n_flares"] = len(catalog)
    metadata["flare_ids"] = [c["flare_id"] for c in catalog]
    metadata["catalog"] = catalog
    metadata["source"] = f"{STAGE0.name}, {STAGE1.name}"

    np.savez_compressed(OUT_PATH, **features, __metadata__=metadata)
    print(f"Phase 2 done -> {OUT_PATH}")
    print(f"  {len(features)} features, {len(catalog)} flares")
    for c in catalog:
        utc = datetime.fromtimestamp(c["t_peak"], tz=timezone.utc)
        print(f"  flare {c['flare_id']}: {utc}, "
              f"class={c['peak_class_str']:>4s}, "
              f"peak={c['peak_flux']:.3e}, "
              f"rise={c['rise_time']:.0f}s, "
              f"dt_peak={c['dt_peak']:.1f}s")
    for k, v in features.items():
        nnan = metadata.get(f"{k}_nan", -1)
        lo = metadata.get(f"{k}_min", "?")
        hi = metadata.get(f"{k}_max", "?")
        print(f"    {k:30s} nan={nnan:<6d}  range=[{lo:.4g},{hi:.4g}]")

    return features


if __name__ == "__main__":
    extract()
