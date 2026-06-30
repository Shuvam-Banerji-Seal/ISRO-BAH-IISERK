#!/usr/bin/env python3
"""GPU extraction — parallel CPU loading, GPU compute. Workers do NOT touch CUDA."""

import sys, os, time, warnings, json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
os.environ.setdefault(
    "BAH2026_DATA", os.path.join(os.path.dirname(__file__), "../../../data/processed")
)
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
from multiprocessing import Pool, cpu_count
from bah2026.data import discover_combined_days
from bah2026.config import CATALOGS_DIR, HDF5_DIR

HDF5_DIR.mkdir(parents=True, exist_ok=True)
LOOKBACK, STEP = 3600, 300
N_WORKERS = min(cpu_count(), 24)
CHUNK = 200  # Process 200 days at a time (uses ~16GB GPU VRAM)


def _worker_load(args):
    """Load one day. Pure numpy, NO torch, NO CUDA. Top-level for pickling."""
    d, event_times = args
    from bah2026.data.reader import load_solexs_lc, load_solexs_pi, load_hel1os_lc
    from bah2026.data.preprocessing import align_hel1os_to_solexs
    from bah2026.data.corrections import (
        correct_solexs_deadtime,
        subtract_hel1os_background,
    )

    try:
        sxr = load_solexs_lc(d)
        counts = correct_solexs_deadtime(
            np.where(
                np.isfinite(sxr["counts"]), sxr["counts"], np.nanmedian(sxr["counts"])
            )
        ).astype(np.float32)
        time_s = sxr["time"]

        hxr4 = np.zeros((len(counts), 20), dtype=np.float32)
        for idx, (det, num) in enumerate(
            [("czt", 1), ("czt", 2), ("cdte", 1), ("cdte", 2)]
        ):
            try:
                hx = load_hel1os_lc(d, detector=det, num=num)
                if hx["ctr"].size > 0:
                    from bah2026.data.corrections import subtract_hel1os_background

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
        et = event_times.get(str(d), [])
        ys = np.zeros(n_w, dtype=np.int64)
        for wi in range(n_w):
            t = time_s[min(wi * STEP + LOOKBACK, len(time_s) - 1)]
            ys[wi] = 1 if any(0 < e - t <= 1800 for e in et) else 0

        return {
            "ok": True,
            "sxr": counts,
            "hxr": hxr4,
            "pi": pi_win,
            "y": ys,
            "n_w": n_w,
        }
    except Exception as e:
        return {"ok": False, "d": str(d), "error": str(e)}


def main():
    print(f"[{time.strftime('%H:%M:%S')}] Starting GPU extraction", flush=True)
    print(f"  Workers: {N_WORKERS}, Chunk: {CHUNK} days", flush=True)

    csv = CATALOGS_DIR / "nowcast_catalogue.csv"
    df = pd.read_csv(csv) if csv.exists() else pd.DataFrame()
    event_times = {}
    if "date" in df.columns and "peak_time" in df.columns:
        for _, row in df.iterrows():
            event_times.setdefault(str(row["date"]), []).append(row["peak_time"])

    days = discover_combined_days()
    print(f"  Days: {len(days)}", flush=True)

    chunk_X, chunk_y = [], []
    total_w = 0
    t_pipeline = time.time()

    # CUDA initialized ONLY here — AFTER Pool setup
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

    for ci in range(0, len(days), CHUNK):
        chunk_days = days[ci : ci + CHUNK]
        cnum = ci // CHUNK + 1
        t_chunk = time.time()
        print(
            f"\n[{time.strftime('%H:%M:%S')}] Chunk {cnum}: {len(chunk_days)} days ({ci + 1}-{min(ci + CHUNK, len(days))})",
            flush=True,
        )

        # Phase 1: Parallel CPU loading (NO CUDA)
        print(f"  Parallel load with {N_WORKERS} workers...", flush=True)
        with Pool(N_WORKERS) as pool:
            results = pool.map(_worker_load, [(d, event_times) for d in chunk_days])

        ok_results = [r for r in results if r.get("ok")]
        fail_results = [r for r in results if not r.get("ok")]
        t_load = time.time() - t_chunk
        print(
            f"  Load: {len(ok_results)} ok, {len(fail_results)} fail in {t_load:.1f}s",
            flush=True,
        )

        if not ok_results:
            continue

        total_nw = sum(r["n_w"] for r in ok_results)
        print(f"  Windows: {total_nw}, assembling arrays...", flush=True)

        # Phase 2: Assemble on CPU (numpy)
        sxr_np = np.zeros((total_nw, LOOKBACK), dtype=np.float32)
        hxr_np = np.zeros((total_nw, LOOKBACK, 20), dtype=np.float32)
        pi_np = np.zeros((total_nw, 340), dtype=np.float32)
        y_np = np.zeros(total_nw, dtype=np.int64)

        w_idx = 0
        for r in ok_results:
            n_w, counts, hxr4, pi_win, ys = (
                r["n_w"],
                r["sxr"],
                r["hxr"],
                r["pi"],
                r["y"],
            )
            for wi in range(n_w):
                start = wi * STEP
                sxr_np[w_idx] = counts[start : start + LOOKBACK]
                hxr_np[w_idx] = hxr4[start : start + LOOKBACK]
                if pi_win is not None and wi < len(pi_win):
                    pi_np[w_idx] = pi_win[wi]
                y_np[w_idx] = ys[wi]
                w_idx += 1

        # Phase 3: GPU transfer + compute
        print(f"  GPU transfer...", flush=True)
        t0 = time.time()
        sxr_gpu = torch.from_numpy(sxr_np).to("cuda")
        hxr_gpu = torch.from_numpy(hxr_np).to("cuda")
        pi_gpu = torch.from_numpy(pi_np).to("cuda")
        torch.cuda.synchronize()
        vram = torch.cuda.memory_allocated() / 1024**3
        print(f"  GPU: {sxr_gpu.shape[0]} windows, {vram:.2f}GB VRAM", flush=True)
        print(
            f"    SXR={sxr_gpu.shape} {sxr_gpu.element_size() * sxr_gpu.nelement() / 1024**3:.2f}GB",
            flush=True,
        )
        print(
            f"    HXR={hxr_gpu.shape} {hxr_gpu.element_size() * hxr_gpu.nelement() / 1024**3:.2f}GB",
            flush=True,
        )
        print(
            f"    PI={pi_gpu.shape} {pi_gpu.element_size() * pi_gpu.nelement() / 1024**3:.3f}GB",
            flush=True,
        )

        print(f"  GPU compute...", flush=True)
        t0 = time.time()
        feats = {}
        feats.update(_batch_stats(sxr_gpu))
        feats.update(_batch_acf(sxr_gpu, FEATURE_AUTOCORR_LAGS))
        feats.update(_batch_spectral_entropy(sxr_gpu))
        feats.update(_batch_derivative_features(sxr_gpu, hxr_gpu))
        feats.update(_batch_multiscale(sxr_gpu, hxr_gpu))
        feats.update(_batch_neupert(sxr_gpu, hxr_gpu))
        feats.update(_batch_hxr_features(hxr_gpu))
        feats.update(_batch_pi_channel_features(pi_gpu))
        torch.cuda.synchronize()
        t_comp = time.time() - t0

        print(f"  Assembling feature matrix...", flush=True)
        canonical = get_canonical_feature_names()
        n_feat = len(canonical)
        row = torch.zeros(total_nw, n_feat, device="cuda", dtype=torch.float32)
        for fi, fn in enumerate(canonical):
            if fn in feats:
                val = feats[fn]
                if isinstance(val, torch.Tensor) and val.shape[0] == total_nw:
                    row[:, fi] = val
        torch.cuda.synchronize()

        X_c = row.cpu().numpy().astype(np.float32)
        X_c = np.nan_to_num(X_c, nan=0.0, posinf=0.0, neginf=0.0)
        chunk_X.append(X_c)
        chunk_y.append(y_np)
        total_w += total_nw

        t_chunk_total = time.time() - t_chunk
        print(
            f"  Chunk done: X={X_c.shape}, compute={t_comp:.1f}s, total={t_chunk_total:.1f}s, {vram:.2f}GB VRAM",
            flush=True,
        )

        del sxr_gpu, hxr_gpu, pi_gpu, row, feats
        torch.cuda.empty_cache()

    # Final assembly
    X = np.vstack(chunk_X)
    y = np.concatenate(chunk_y)
    print(
        f"\n[{time.strftime('%H:%M:%S')}] Done: X={X.shape}, y={y.shape}, pos={y.sum()} ({100 * y.mean():.2f}%)",
        flush=True,
    )
    print(f"  Total time: {time.time() - t_pipeline:.0f}s", flush=True)
    np.save(HDF5_DIR / "X_features.npy", X)
    np.save(HDF5_DIR / "y_labels.npy", y)
    (HDF5_DIR / "feature_names.json").write_text(
        json.dumps(get_canonical_feature_names())
    )
    print(f"  Saved to {HDF5_DIR}", flush=True)


if __name__ == "__main__":
    main()
