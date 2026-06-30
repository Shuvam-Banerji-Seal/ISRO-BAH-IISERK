#!/usr/bin/env python3
"""GPU extraction — loads ALL data to GPU for sustained computation."""

import sys, os, time, warnings, json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
os.environ.setdefault(
    "BAH2026_DATA", os.path.join(os.path.dirname(__file__), "../../../data/processed")
)
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
from bah2026.data import discover_combined_days
from bah2026.config import CATALOGS_DIR, HDF5_DIR

HDF5_DIR.mkdir(parents=True, exist_ok=True)
LOOKBACK, STEP = 3600, 300


def _load_all_data(days, event_times):
    """Load ALL days to CPU arrays, accumulating windowed data."""
    from bah2026.data.reader import load_solexs_lc, load_solexs_pi, load_hel1os_lc
    from bah2026.data.preprocessing import align_hel1os_to_solexs
    from bah2026.data.corrections import (
        correct_solexs_deadtime,
        subtract_hel1os_background,
    )

    all_sxr_windows = []
    all_hxr_windows = []
    all_pi_windows = []
    all_ys = []
    all_precomputed = []

    t0 = time.time()
    for i, d in enumerate(days):
        try:
            sxr = load_solexs_lc(d)
            counts = correct_solexs_deadtime(
                np.where(
                    np.isfinite(sxr["counts"]),
                    sxr["counts"],
                    np.nanmedian(sxr["counts"]),
                )
            )
            time_s = sxr["time"]

            hxr4 = np.zeros((len(counts), 20), dtype=np.float32)
            for idx, (det, num) in enumerate(
                [("czt", 1), ("czt", 2), ("cdte", 1), ("cdte", 2)]
            ):
                try:
                    hx = load_hel1os_lc(d, detector=det, num=num)
                    if hx["ctr"].size > 0:
                        ctr = subtract_hel1os_background(hx["ctr"], det)
                        aligned = align_hel1os_to_solexs(
                            hx["mjd"], ctr, time_s, sxr["mjdrefi"], sxr["mjdreff"]
                        )
                        ml = min(len(counts), aligned.shape[0])
                        hxr4[:ml, idx * 5 : (idx + 1) * 5] = aligned[:ml, :5].astype(
                            np.float32
                        )
                except Exception:
                    pass

            pi_win = None
            try:
                pi = load_solexs_pi(d)
                if pi["counts"].size > 0:
                    pi_raw = pi["counts"].astype(np.float32)
                    n_w = (len(pi_raw) - LOOKBACK) // STEP + 1
                    if n_w > 0:
                        pi_win = np.zeros((n_w, 340), dtype=np.float32)
                        for wi in range(n_w):
                            s = wi * STEP
                            pi_win[wi] = np.nansum(pi_raw[s : s + LOOKBACK], axis=0)
            except Exception:
                pass

            n_w = (len(counts) - LOOKBACK) // STEP + 1
            if n_w < 1:
                continue

            for wi in range(n_w):
                start = wi * STEP
                sxr_win = counts[start : start + LOOKBACK].astype(np.float32)
                hxr_win = hxr4[start : start + LOOKBACK]
                all_sxr_windows.append(sxr_win)
                all_hxr_windows.append(hxr_win)
                if pi_win is not None and wi < len(pi_win):
                    all_pi_windows.append(pi_win[wi])
                else:
                    all_pi_windows.append(np.zeros(340, dtype=np.float32))

                et = event_times.get(str(d), [])
                t = time_s[min(start + LOOKBACK, len(time_s) - 1)]
                y = 1 if any(0 < e - t <= 1800 for e in et) else 0
                all_ys.append(y)

        except Exception:
            pass

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(
                f"  Loaded {i + 1}/{len(days)} days, {len(all_sxr_windows)} windows, {elapsed:.0f}s",
                flush=True,
            )

    print(
        f"  Total: {len(all_sxr_windows)} windows from {len(days)} days in {time.time() - t0:.0f}s",
        flush=True,
    )
    return all_sxr_windows, all_hxr_windows, all_pi_windows, all_ys


def main():
    csv = CATALOGS_DIR / "nowcast_catalogue.csv"
    df = pd.read_csv(csv) if csv.exists() else pd.DataFrame()
    event_times = {}
    if "date" in df.columns and "peak_time" in df.columns:
        for _, row in df.iterrows():
            event_times.setdefault(str(row["date"]), []).append(row["peak_time"])

    days = discover_combined_days()
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"GPU: {gpu_name} | {len(days)} days", flush=True)

    print("Phase 1: Loading ALL data to CPU...", flush=True)
    sxr_wins, hxr_wins, pi_wins, ys = _load_all_data(days, event_times)

    n_windows = len(sxr_wins)
    print(f"  {n_windows} windows total", flush=True)

    print("Phase 2: Transferring ALL to GPU...", flush=True)
    t0 = time.time()

    sxr_tensor = torch.from_numpy(np.array(sxr_wins, dtype=np.float32)).to("cuda")
    hxr_tensor = torch.from_numpy(np.array(hxr_wins, dtype=np.float32)).to("cuda")
    pi_tensor = torch.from_numpy(np.array(pi_wins, dtype=np.float32)).to("cuda")

    torch.cuda.synchronize()
    print(f"  Transfer: {time.time() - t0:.1f}s", flush=True)
    print(
        f"  SXR: {sxr_tensor.shape} = {sxr_tensor.element_size() * sxr_tensor.nelement() / 1024**3:.2f}GB on {sxr_tensor.device}",
        flush=True,
    )
    print(
        f"  HXR: {hxr_tensor.shape} = {hxr_tensor.element_size() * hxr_tensor.nelement() / 1024**3:.2f}GB on {hxr_tensor.device}",
        flush=True,
    )
    print(
        f"  PI:  {pi_tensor.shape} = {pi_tensor.element_size() * pi_tensor.nelement() / 1024**3:.2f}GB on {pi_tensor.device}",
        flush=True,
    )
    print(
        f"  VRAM: {torch.cuda.memory_allocated() / 1024**3:.2f}GB allocated, {torch.cuda.memory_reserved() / 1024**3:.2f}GB reserved",
        flush=True,
    )

    print("Phase 3: Computing ALL features on GPU...", flush=True)
    t0 = time.time()

    from bah2026.features.gpu_features import (
        _batch_stats,
        _batch_acf,
        _batch_spectral_entropy,
        _batch_derivative_features,
        _batch_multiscale,
        _batch_neupert,
        _batch_hxr_features,
        _batch_pi_channel_features,
        FEATURE_AUTOCORR_LAGS,
        get_canonical_feature_names,
    )

    feats = {}
    feats.update(_batch_stats(sxr_tensor))
    feats.update(_batch_acf(sxr_tensor, FEATURE_AUTOCORR_LAGS))
    feats.update(_batch_spectral_entropy(sxr_tensor))
    feats.update(_batch_derivative_features(sxr_tensor, hxr_tensor))
    feats.update(_batch_multiscale(sxr_tensor, hxr_tensor))
    feats.update(_batch_neupert(sxr_tensor, hxr_tensor))
    feats.update(_batch_hxr_features(hxr_tensor))
    feats.update(_batch_pi_channel_features(pi_tensor))

    torch.cuda.synchronize()
    compute_time = time.time() - t0
    print(f"  GPU compute: {compute_time:.1f}s", flush=True)
    print(
        f"  VRAM after compute: {torch.cuda.memory_allocated() / 1024**3:.2f}GB",
        flush=True,
    )

    print("Phase 4: Assembling feature matrix...", flush=True)
    canonical = get_canonical_feature_names()
    n_feat = len(canonical)
    row = torch.zeros(n_windows, n_feat, device="cuda", dtype=torch.float32)
    feat_keys = list(feats.keys())
    for fi, fn in enumerate(canonical):
        if fn in feat_keys:
            val = feats[fn]
            if isinstance(val, torch.Tensor) and val.shape[0] == n_windows:
                row[:, fi] = val

    torch.cuda.synchronize()
    X = row.cpu().numpy().astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.array(ys, dtype=int)

    total_time = time.time() - t0
    print(
        f"X={X.shape}, y={y.shape}, pos={y.sum()} ({100 * y.mean():.2f}%) in {total_time:.0f}s",
        flush=True,
    )
    np.save(HDF5_DIR / "X_features.npy", X)
    np.save(HDF5_DIR / "y_labels.npy", y)
    (HDF5_DIR / "feature_names.json").write_text(json.dumps(canonical))
    print(f"Saved to {HDF5_DIR}", flush=True)


if __name__ == "__main__":
    main()
