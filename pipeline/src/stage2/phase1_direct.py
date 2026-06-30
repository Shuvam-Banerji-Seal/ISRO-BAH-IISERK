"""
Stage 2 — Phase 1: Direct extractions from Stage 1 and Stage 0 NPZs.
Features that need zero new data I/O — everything is already computed.
"""
import numpy as np
from pathlib import Path

STAGE0 = Path("data/processed/master_dataset_20260623.npz")
STAGE1 = Path("data/processed/stage1_20260623.npz")
OUT_DIR = Path("dist/features")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "phase1_direct.npz"


def extract():
    ds0 = np.load(STAGE0, allow_pickle=True)
    ds1 = np.load(STAGE1, allow_pickle=True)
    features = {}
    metadata = {}

    # ── #5 Flux time-derivatives (from Stage 0) ────────────────
    for key in ["sxr_deriv1", "sxr_deriv5", "hxr_deriv1", "hxr_deriv5"]:
        features[key] = ds0[key].astype(np.float32)

    # ── #16 Hardness ratios (CdTe bands, from Stage 1) ────────
    b1 = ds1["hxr_cdte_band1"].astype(np.float32)  # 5-20 keV
    b2 = ds1["hxr_cdte_band2"].astype(np.float32)  # 20-30 keV
    b3 = ds1["hxr_cdte_band3"].astype(np.float32)  # 30-40 keV
    b4 = ds1["hxr_cdte_band4"].astype(np.float32)  # 40-60 keV

    def safe_ratio(num, den):
        r = np.full_like(num, np.nan, dtype=np.float32)
        ok = (den > 0) & (num >= 0) & ~np.isnan(num) & ~np.isnan(den)
        r[ok] = num[ok] / den[ok]
        return np.clip(r, 0.001, 1000.0)

    features["hr_cdte_band4_band1"] = safe_ratio(b4, b1)
    features["hr_cdte_band3_band1"] = safe_ratio(b3, b1)
    features["hr_cdte_band2_band1"] = safe_ratio(b2, b1)
    features["hr_cdte_band4_band2"] = safe_ratio(b4, b2)

    # ── #24 Neupert score (in both, use Stage 1) ──────────────
    features["neupert_rho"] = ds1["neupert_rho"].astype(np.float32)

    # ── #11 SXR waiting-time / recency features (from Stage 0) ─
    features["time_since_last_flare"] = ds0["time_last_flare"].astype(np.float32)
    features["time_until_next_flare"] = ds0["time_next_flare"].astype(np.float32)

    # Rolling flare counts
    t = ds0["time"].astype(np.float64)
    flare_id = ds1["flare_id"]  # use Stage 1 (same as Stage 0)
    in_flare = flare_id > 0

    # Find flare start/end times
    diff = np.diff(in_flare.astype(np.int8))
    flare_starts = np.where(diff == 1)[0] + 1
    flare_ends = np.where(diff == -1)[0] + 1
    if in_flare[0]:
        flare_starts = np.concatenate([[0], flare_starts])
    if in_flare[-1]:
        flare_ends = np.concatenate([flare_ends, [86399]])

    flare_windows = list(zip(flare_starts, flare_ends))
    flare_times = [(t[s], t[e]) for s, e in flare_windows]

    n_1hr = np.zeros(86400, dtype=np.float32)
    n_3hr = np.zeros(86400, dtype=np.float32)
    half1 = 3600.0
    half3 = 10800.0
    for i in range(86400):
        ti = t[i]
        n_1hr[i] = sum(1 for fs, fe in flare_times if ti - half1 <= fs <= ti)
        n_3hr[i] = sum(1 for fs, fe in flare_times if ti - half3 <= fs <= ti)
    features["flares_last_1hr"] = n_1hr
    features["flares_last_3hr"] = n_3hr

    # ── #22 HXR waiting-time features ──────────────────────────
    hxr_snr = ds1["hxr_snr"].astype(np.float32)
    hxr_event = (hxr_snr > 3) & ~np.isnan(hxr_snr)
    running = np.zeros(86400, dtype=np.float32)
    count = 0
    for i in range(86400):
        if hxr_event[i]:
            count += 1
        running[i] = count
    features["hxr_event_count"] = running
    # Time since last HXR event
    last_hxr = np.full(86400, np.nan, dtype=np.float32)
    last_idx = -1
    for i in range(86400):
        if hxr_event[i]:
            last_idx = i
        if last_idx >= 0:
            last_hxr[i] = t[i] - t[last_idx]
    features["hxr_time_since_event"] = last_hxr

    # ── Metadata ────────────────────────────────────────────────
    for k, v in features.items():
        nnan = int(np.isnan(v).sum()) if v.dtype.kind == "f" else 0
        metadata[f"{k}_nan"] = nnan
        metadata[f"{k}_min"] = float(np.nanmin(v)) if v.dtype.kind == "f" else int(v.min())
        metadata[f"{k}_max"] = float(np.nanmax(v)) if v.dtype.kind == "f" else int(v.max())

    metadata["n_features"] = len(features)
    metadata["feature_names"] = list(features.keys())
    metadata["phase"] = 1
    metadata["source"] = f"{STAGE0.name}, {STAGE1.name}"

    np.savez_compressed(OUT_PATH, **features, __metadata__=metadata)
    print(f"Phase 1 done -> {OUT_PATH}")
    print(f"  {len(features)} features saved")
    for k, v in features.items():
        nnan = metadata.get(f"{k}_nan", -1)
        lo = metadata.get(f"{k}_min", "?")
        hi = metadata.get(f"{k}_max", "?")
        print(f"    {k:30s} nan={nnan:<6d}  range=[{lo:.4g},{hi:.4g}]")

    return features


if __name__ == "__main__":
    extract()
