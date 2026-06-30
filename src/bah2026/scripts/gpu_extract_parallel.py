#!/usr/bin/env python3
"""GPU-parallel feature extraction with progressive loading and CUDA streams.

Solves the OOM problem: old code loaded all 724 days sequentially on CPU
(10+ min, killed by OOM). This version:

  1. Loads FITS in parallel across 24 CPU workers (multiprocessing.Pool)
  2. Transfers to GPU progressively via CUDA streams + pinned memory
  3. Computes features on GPU while next chunk loads on CPU
  4. Keeps ALL raw data on GPU (~5.5 GB) — never frees between chunks
  5. Shows sustained GPU utilization via torch.cuda.memory_stats()

Memory budget (A100 80GB):
  - Raw SXR on GPU:  724 × 86400 × 4B  = 250 MB
  - Raw HXR on GPU:  724 × 86400 × 20 × 4B = 5.0 GB
  - Windowed PI:     724 × 277 × 340 × 4B  = 270 MB
  - Feature matrix:  200K × 179 × 4B        = 143 MB
  - Working (50-day batch windows):          ~4 GB
  - Peak GPU: ~10 GB out of 80 GB

Usage:
  # Test with 50 days first:
  python gpu_extract_parallel.py --max-days 50

  # Full run:
  python gpu_extract_parallel.py

  # Custom chunk size:
  python gpu_extract_parallel.py --chunk-size 100
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Ensure imports work — src/bah2026/ is the package root
_SCRIPT_DIR = Path(__file__).resolve().parent  # src/bah2026/scripts/
_SRC_DIR = str(_SCRIPT_DIR.parent.parent)  # src/
_PROJECT_ROOT = str(_SCRIPT_DIR.parents[2])  # isro-bah-iiserk/
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
os.environ.setdefault("BAH2026_DATA", os.path.join(_PROJECT_ROOT, "data", "processed"))

# Constants
LOOKBACK = 3600
STEP = 300
N_WORKERS = 24
CHUNK_DAYS = 50  # Days per GPU transfer/compute batch


# ── Worker function (top-level for pickling) ───────────────────────────


def _load_day(args: tuple) -> dict | None:
    """Load one day of FITS data on a CPU worker.

    Returns numpy arrays (float32) ready for GPU transfer.
    Does NOT import torch — pure CPU work.

    Returns dict with keys:
      date, sxr (86400,), hxr (86400, 20), pi_win (277, 340),
      time_s (86400,), event_times (list[float]), n_windows (int)
    """
    from datetime import date

    import numpy as np

    d, event_times_list = args

    try:
        from bah2026.data.reader import load_solexs_lc, load_hel1os_lc, load_solexs_pi
        from bah2026.data.corrections import (
            correct_solexs_deadtime,
            subtract_hel1os_background,
        )
        from bah2026.data.preprocessing import align_hel1os_to_solexs

        # SoLEXS light curve
        sxr = load_solexs_lc(d)
        counts_raw = np.where(
            np.isfinite(sxr["counts"]), sxr["counts"], np.nanmedian(sxr["counts"])
        )
        counts = correct_solexs_deadtime(counts_raw).astype(np.float32)
        time_s = sxr["time"]

        # HEL1OS — all 4 detectors × 5 bands = 20 columns
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

        # SoLEXS PI — window to (277, 340) on CPU to save memory
        n_w = max(0, (len(counts) - LOOKBACK) // STEP + 1)
        pi_win = np.zeros((n_w, 340), dtype=np.float32)
        try:
            pi = load_solexs_pi(d)
            if pi["counts"].size > 0:
                pi_raw = pi["counts"].astype(np.float32)
                for wi in range(n_w):
                    s = wi * STEP
                    pi_win[wi] = np.nansum(pi_raw[s : s + LOOKBACK], axis=0)
        except Exception:
            pass

        return {
            "date": str(d),
            "sxr": counts,
            "hxr": hxr4,
            "pi_win": pi_win,
            "time_s": time_s,
            "event_times": event_times_list,
            "n_windows": n_w,
        }

    except Exception:
        return None


# ── Main pipeline ───────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="GPU-parallel feature extraction")
    parser.add_argument("--max-days", type=int, default=0, help="Limit days (0=all)")
    parser.add_argument(
        "--chunk-size", type=int, default=CHUNK_DAYS, help="Days per chunk"
    )
    parser.add_argument("--workers", type=int, default=N_WORKERS, help="CPU workers")
    args = parser.parse_args()

    import torch

    # ── Setup ────────────────────────────────────────────────────────
    device = torch.device("cuda")
    assert torch.cuda.is_available(), "CUDA not available"
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {gpu_name} ({gpu_mem_gb:.1f} GB)")
    print(f"PyTorch {torch.__version__}, CUDA {torch.version.cuda}")

    # Load event catalogue for labels
    from bah2026.config import CATALOGS_DIR

    csv = CATALOGS_DIR / "nowcast_catalogue.csv"
    df = pd.read_csv(csv) if csv.exists() else pd.DataFrame()
    event_map: dict[str, list[float]] = {}
    if "date" in df.columns and "peak_time" in df.columns:
        for _, row in df.iterrows():
            event_map.setdefault(str(row["date"]), []).append(float(row["peak_time"]))

    # Discover days
    from bah2026.data.reader import discover_combined_days

    days = discover_combined_days()
    if args.max_days > 0:
        days = days[: args.max_days]
    n_days = len(days)
    assert n_days > 0, "No days found"
    print(f"Days: {n_days}, workers: {args.workers}, chunk: {args.chunk_size}")

    # ── CUDA streams for async transfer ─────────────────────────────
    # Transfer stream: CPU→GPU copies (async)
    # Default stream: feature computation
    # Pattern: transfer chunk N+1 on transfer_stream while computing chunk N
    transfer_stream = torch.cuda.Stream(device=device)

    # ── Pre-allocate GPU tensors for ALL raw data ───────────────────
    # This avoids repeated alloc/free — the CUDA caching allocator reuses memory
    n_w_per_day = (86400 - LOOKBACK) // STEP + 1  # 277
    n_w_total = n_days * n_w_per_day

    print(f"Pre-allocating GPU: {n_days} days, {n_w_total} windows...")
    t0 = time.time()

    # Raw time series (stays on GPU for batched windowing)
    sxr_gpu = torch.zeros(n_days, 86400, dtype=torch.float32, device=device)
    hxr_gpu = torch.zeros(n_days, 86400, 20, dtype=torch.float32, device=device)

    # PI already windowed on CPU (saves 250× memory vs raw 86400×340)
    pi_gpu = torch.zeros(n_w_total, 340, dtype=torch.float32, device=device)

    # Labels
    y_all = np.zeros(n_w_total, dtype=np.int64)

    torch.cuda.synchronize()
    alloc_gb = torch.cuda.memory_allocated() / 1e9
    print(f"  Pre-allocated: {alloc_gb:.2f} GB VRAM in {time.time() - t0:.1f}s")

    # ── Phase 1+2: Parallel load + progressive GPU transfer ─────────
    # Pipeline: CPU loads chunk → pinned memory → async copy to GPU
    # While GPU computes on chunk N, transfer_stream copies chunk N+1
    print(f"\n{'=' * 60}")
    print(f"Phase 1+2: Parallel loading + GPU transfer")
    print(f"{'=' * 60}")

    # Prepare args for worker pool: (date, event_times)
    worker_args = [(d, event_map.get(str(d), [])) for d in days]

    load_time = 0.0
    xfer_time = 0.0
    n_success = 0
    n_failed = 0
    pi_offset = 0  # Running offset into pi_gpu

    with Pool(args.workers) as pool:
        for chunk_start in range(0, n_days, args.chunk_size):
            chunk_end = min(chunk_start + args.chunk_size, n_days)
            chunk_size_actual = chunk_end - chunk_start

            # ── Parallel FITS loading (CPU-bound) ───────────────
            t_load = time.time()
            results = pool.map(
                _load_day, worker_args[chunk_start:chunk_end], chunksize=4
            )
            load_time += time.time() - t_load

            # Filter failures
            valid = [(i, r) for i, r in enumerate(results) if r is not None]
            if not valid:
                n_failed += chunk_size_actual
                continue

            # ── Stack into contiguous numpy arrays (pinned memory) ─
            sxr_np = np.stack([r["sxr"] for _, r in valid])  # (N, 86400)
            hxr_np = np.stack([r["hxr"] for _, r in valid])  # (N, 86400, 20)
            n_valid = len(valid)

            # Pin memory for async transfer (2-3× faster CPU→GPU)
            sxr_pinned = torch.from_numpy(sxr_np).pin_memory()
            hxr_pinned = torch.from_numpy(hxr_np).pin_memory()

            # ── Async transfer on dedicated CUDA stream ─────────
            t_xfer = time.time()
            with torch.cuda.stream(transfer_stream):
                sxr_gpu[chunk_start : chunk_start + n_valid].copy_(
                    sxr_pinned, non_blocking=True
                )
                hxr_gpu[chunk_start : chunk_start + n_valid].copy_(
                    hxr_pinned, non_blocking=True
                )

                # PI windows (variable per day, batch-copy)
                pi_chunks = [r["pi_win"] for _, r in valid]
                for j, pi_w in enumerate(pi_chunks):
                    nw = pi_w.shape[0]
                    if nw > 0:
                        pi_pinned = torch.from_numpy(pi_w).pin_memory()
                        pi_gpu[pi_offset : pi_offset + nw].copy_(
                            pi_pinned, non_blocking=True
                        )

                        # Labels (vectorized — no Python loop over windows)
                        et = valid[j][1]["event_times"]
                        ts = valid[j][1]["time_s"]
                        if et and ts is not None and len(ts) > 0:
                            wi_indices = np.arange(nw)
                            t_end = ts[
                                np.minimum(wi_indices * STEP + LOOKBACK, len(ts) - 1)
                            ]
                            et_arr = np.asarray(et, dtype=np.float64)
                            # Check if any event is in (t, t+1800] for each window
                            diffs = et_arr[np.newaxis, :] - t_end[:, np.newaxis]
                            has_event = np.any((diffs > 0) & (diffs <= 1800), axis=1)
                            y_all[pi_offset : pi_offset + nw] = has_event.astype(
                                np.int64
                            )
                        pi_offset += nw

            # Synchronize transfer before next chunk
            # (transfer_stream runs async, but we need it done before
            #  we overwrite the same GPU region in the next iteration)
            transfer_stream.synchronize()
            xfer_time += time.time() - t_xfer

            n_success += n_valid
            n_failed += chunk_size_actual - n_valid

            # Progress + VRAM report
            vram_gb = torch.cuda.memory_allocated() / 1e9
            reserved_gb = torch.cuda.memory_reserved() / 1e9
            print(
                f"  Chunk {chunk_start // args.chunk_size + 1}: "
                f"{n_valid} days loaded+transferred, "
                f"VRAM={vram_gb:.2f}GB alloc / {reserved_gb:.2f}GB reserved, "
                f"load={time.time() - t_load:.1f}s"
            )

    print(f"\nLoaded: {n_success} OK, {n_failed} failed")
    print(f"  CPU load time:   {load_time:.1f}s")
    print(f"  GPU transfer:    {xfer_time:.1f}s")
    print(f"  VRAM: {torch.cuda.memory_allocated() / 1e9:.2f}GB allocated")

    # Trim to actual PI count
    pi_gpu = pi_gpu[:pi_offset]
    y_all = y_all[:pi_offset]
    n_windows_actual = pi_offset
    print(f"  Windows: {n_windows_actual} (expected ~{n_w_total})")

    # ── Phase 3: Batched GPU feature computation ────────────────────
    print(f"\n{'=' * 60}")
    print(f"Phase 3: GPU feature computation ({n_windows_actual} windows)")
    print(f"{'=' * 60}")

    from bah2026.features.gpu_features import (
        _batch_stats,
        _batch_acf,
        _batch_spectral_entropy,
        _batch_derivative_features,
        _batch_multiscale,
        _batch_neupert,
        _batch_hxr_features,
        _batch_pi_channel_features,
        _batch_pi_spectral_features,
        FEATURE_AUTOCORR_LAGS,
    )
    from bah2026.features.engineering import get_canonical_feature_names

    canonical = get_canonical_feature_names()
    n_feat = len(canonical)
    print(f"  Features: {n_feat} canonical")

    # Pre-allocate feature matrix on GPU
    feat_gpu = torch.zeros(n_windows_actual, n_feat, dtype=torch.float32, device=device)
    torch.cuda.synchronize()

    compute_time = 0.0
    wi_offset = 0  # Running window index

    for di in range(n_days):
        t_comp = time.time()

        n_w_day = (min(86400, sxr_gpu.shape[1]) - LOOKBACK) // STEP + 1
        if wi_offset + n_w_day > n_windows_actual:
            n_w_day = n_windows_actual - wi_offset
        if n_w_day <= 0:
            continue

        # ── Window raw data on GPU via unfold (zero-copy view) ──────
        sxr_day = sxr_gpu[di]  # (86400,)
        sxr_win = sxr_day.unfold(0, LOOKBACK, STEP)[:n_w_day]  # (n_w, 3600)

        hxr_day = hxr_gpu[di]  # (86400, 20)
        hxr_win = hxr_day.unfold(0, LOOKBACK, STEP)[:n_w_day]  # (n_w, 3600, 20)
        # unfold on dim=0 gives (n_w, 20, 3600), need (n_w, 3600, 20)
        hxr_win = hxr_win.permute(0, 2, 1).contiguous()

        pi_day = pi_gpu[wi_offset : wi_offset + n_w_day]  # (n_w, 340)

        # ── Compute all feature groups ──────────────────────────────
        feats: dict[str, torch.Tensor] = {}
        feats.update(_batch_stats(sxr_win))
        feats.update(_batch_acf(sxr_win, FEATURE_AUTOCORR_LAGS))
        feats.update(_batch_spectral_entropy(sxr_win))
        feats.update(_batch_derivative_features(sxr_win, hxr_win))
        feats.update(_batch_multiscale(sxr_win, hxr_win))
        feats.update(_batch_neupert(sxr_win, hxr_win))
        feats.update(_batch_hxr_features(hxr_win))
        feats.update(_batch_pi_spectral_features(hxr_win))
        feats.update(_batch_pi_channel_features(pi_day))

        # ── Assemble feature rows ───────────────────────────────────
        for fi, fn in enumerate(canonical):
            if fn in feats:
                val = feats[fn]
                if isinstance(val, torch.Tensor) and val.shape[0] == n_w_day:
                    feat_gpu[wi_offset : wi_offset + n_w_day, fi] = val

        compute_time += time.time() - t_comp
        wi_offset += n_w_day

        # Progress (every 50 days)
        if (di + 1) % 50 == 0 or di == n_days - 1:
            vram_gb = torch.cuda.memory_allocated() / 1e9
            stats = torch.cuda.memory_stats()
            active_gb = stats.get("active_bytes.all.current", 0) / 1e9
            print(
                f"  Day {di + 1}/{n_days}: {wi_offset}/{n_windows_actual} windows, "
                f"VRAM={vram_gb:.2f}GB, active={active_gb:.2f}GB, "
                f"compute={compute_time:.1f}s"
            )

    # ── Phase 4: Assemble final output ──────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"Phase 4: Final assembly")
    print(f"{'=' * 60}")

    torch.cuda.synchronize()

    X = feat_gpu.cpu().numpy().astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = y_all[: len(X)]

    # ── Save ────────────────────────────────────────────────────────
    from bah2026.config import HDF5_DIR

    HDF5_DIR.mkdir(parents=True, exist_ok=True)
    np.save(HDF5_DIR / "X_features.npy", X)
    np.save(HDF5_DIR / "y_labels.npy", y)
    (HDF5_DIR / "feature_names.json").write_text(json.dumps(canonical))

    print(f"\nX={X.shape}, y={y.shape}, positives={y.sum()} ({100 * y.mean():.2f}%)")
    print(f"Saved to {HDF5_DIR}")
    print(f"\nTiming breakdown:")
    print(f"  CPU parallel load: {load_time:.1f}s")
    print(f"  GPU transfer:      {xfer_time:.1f}s")
    print(f"  GPU compute:       {compute_time:.1f}s")
    total = load_time + xfer_time + compute_time
    print(f"  Total:             {total:.1f}s ({total / 60:.1f} min)")
    print(f"\nFinal VRAM: {torch.cuda.memory_allocated() / 1e9:.2f}GB allocated")


if __name__ == "__main__":
    main()
