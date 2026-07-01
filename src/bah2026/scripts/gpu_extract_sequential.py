#!/usr/bin/env python3
"""GPU extraction — parallel CPU loading, GPU compute, v3 features + day-level precomputed."""

import sys, os, time, warnings, json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
os.environ.setdefault(
    "BAH2026_DATA", os.path.join(os.path.dirname(__file__), "../../../data/processed")
)
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, torch
from multiprocessing import Pool, cpu_count
from bah2026.data import discover_combined_days
from bah2026.config import CATALOGS_DIR, HDF5_DIR

HDF5_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_LOG = HDF5_DIR / "processed_days.txt"
LOOKBACK, STEP = 3600, 300
N_WORKERS = min(cpu_count(), 24)
CHUNK = 200


def _worker_load(args):
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
                    ctr = subtract_hel1os_background(hx["ctr"], det)
                    a = align_hel1os_to_solexs(
                        hx["mjd"], ctr, time_s, sxr["mjdrefi"], sxr["mjdreff"]
                    )
                    ml = min(len(counts), a.shape[0])
                    hxr4[:ml, idx * 5 : (idx + 1) * 5] = a[:ml, :5].astype(np.float32)
            except Exception:
                pass
        pi_win = None
        try:
            pi = load_solexs_pi(d)
            if pi["counts"].size > 0:
                pr = pi["counts"].astype(np.float32)
                n_w = (len(pr) - LOOKBACK) // STEP + 1
                if n_w > 0:
                    pi_win = np.zeros((n_w, 340), dtype=np.float32)
                    for wi in range(n_w):
                        pi_win[wi] = np.nansum(
                            pr[wi * STEP : wi * STEP + LOOKBACK], axis=0
                        )
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
    print(
        f"[{time.strftime('%H:%M:%S')}] GPU extraction: {N_WORKERS} workers, {CHUNK} day chunks",
        flush=True,
    )

    # Resume handling
    processed = set()
    if PROCESSED_LOG.exists():
        with open(PROCESSED_LOG) as f:
            processed = {l.strip() for l in f if l.strip()}
        print(f"  Resume: {len(processed)} days already processed", flush=True)
        if (HDF5_DIR / "X_features.npy").exists():
            import numpy as _np

            existing_X = _np.load(HDF5_DIR / "X_features.npy")
            existing_y = _np.load(HDF5_DIR / "y_labels.npy")
            chunk_X, chunk_y = [existing_X], [existing_y]
            total_w = len(existing_y)
            print(
                f"  Loaded existing: X={existing_X.shape}, y={existing_y.shape}",
                flush=True,
            )
        else:
            chunk_X, chunk_y = [], []
            total_w = 0
    else:
        chunk_X, chunk_y = [], []
        total_w = 0

    csv = CATALOGS_DIR / "nowcast_catalogue.csv"
    df = pd.read_csv(csv) if csv.exists() else pd.DataFrame()
    event_times = {}
    if "date" in df.columns and "peak_time" in df.columns:
        for _, row in df.iterrows():
            event_times.setdefault(str(row["date"]), []).append(row["peak_time"])
    days = discover_combined_days()
    days = [d for d in days if str(d) not in processed]
    print(f"  Remaining: {len(days)} days", flush=True)
    if not days:
        print("  All days already processed!", flush=True)
        return

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

    t_pipeline = time.time()

    for ci in range(0, len(days), CHUNK):
        chunk_days = days[ci : ci + CHUNK]
        cnum = ci // CHUNK + 1
        t_chunk = time.time()
        print(
            f"\n[{time.strftime('%H:%M:%S')}] Chunk {cnum}: {len(chunk_days)} days",
            flush=True,
        )

        with Pool(N_WORKERS) as pool:
            results = pool.map(_worker_load, [(d, event_times) for d in chunk_days])
        ok_r = [r for r in results if r.get("ok")]
        t_load = time.time() - t_chunk
        print(f"  Load: {len(ok_r)}/{len(chunk_days)} in {t_load:.1f}s", flush=True)
        if not ok_r:
            continue

        total_nw = sum(r["n_w"] for r in ok_r)
        sxr_np = np.zeros((total_nw, LOOKBACK), dtype=np.float32)
        hxr_np = np.zeros((total_nw, LOOKBACK, 20), dtype=np.float32)
        pi_np = np.zeros((total_nw, 340), dtype=np.float32)
        y_np = np.zeros(total_nw, dtype=np.int64)
        w_idx = 0
        for r in ok_r:
            for wi in range(r["n_w"]):
                s = wi * STEP
                sxr_np[w_idx] = r["sxr"][s : s + LOOKBACK]
                hxr_np[w_idx] = r["hxr"][s : s + LOOKBACK]
                if r["pi"] is not None and wi < len(r["pi"]):
                    pi_np[w_idx] = r["pi"][wi]
                y_np[w_idx] = r["y"][wi]
                w_idx += 1

        print(f"  GPU: {total_nw} windows...", flush=True)
        sxr_g = torch.from_numpy(sxr_np).to("cuda")
        hxr_g = torch.from_numpy(hxr_np).to("cuda")
        pi_g = torch.from_numpy(pi_np).to("cuda")
        torch.cuda.synchronize()
        vram = torch.cuda.memory_allocated() / 1024**3
        print(f"    VRAM={vram:.2f}GB", flush=True)

        feats = {}
        feats.update(_batch_stats(sxr_g))
        feats.update(_batch_acf(sxr_g, FEATURE_AUTOCORR_LAGS))
        feats.update(_batch_spectral_entropy(sxr_g))
        feats.update(_batch_derivative_features(sxr_g, hxr_g))
        feats.update(_batch_multiscale(sxr_g, hxr_g))
        feats.update(_batch_neupert(sxr_g, hxr_g))
        feats.update(_batch_hxr_features(hxr_g))
        feats.update(_batch_pi_channel_features(pi_g))
        torch.cuda.synchronize()
        t_comp = time.time() - t_chunk

        canonical = get_canonical_feature_names()
        n_feat = len(canonical)
        row = torch.zeros(total_nw, n_feat, device="cuda", dtype=torch.float32)
        for fi, fn in enumerate(canonical):
            if (
                fn in feats
                and isinstance(feats[fn], torch.Tensor)
                and feats[fn].shape[0] == total_nw
            ):
                row[:, fi] = feats[fn]
        torch.cuda.synchronize()
        X_c = row.cpu().numpy().astype(np.float32)
        X_c = np.nan_to_num(X_c, nan=0.0, posinf=0.0, neginf=0.0)
        chunk_X.append(X_c)
        chunk_y.append(y_np)
        total_w += total_nw
        print(
            f"  Done: X={X_c.shape}, compute={t_comp - t_load:.1f}s, total={time.time() - t_chunk:.1f}s",
            flush=True,
        )

        with open(PROCESSED_LOG, "a") as f:
            for d in chunk_days:
                f.write(f"{d}\n")

        if chunk_X:
            X_tmp = np.vstack(chunk_X)
            y_tmp = np.concatenate(chunk_y)
            np.save(HDF5_DIR / "X_features.npy", X_tmp)
            np.save(HDF5_DIR / "y_labels.npy", y_tmp)
            print(f"  Saved checkpoint: X={X_tmp.shape}", flush=True)

        del sxr_g, hxr_g, pi_g, row, feats
        torch.cuda.empty_cache()

    X = np.vstack(chunk_X)
    y = np.concatenate(chunk_y)
    print(
        f"\n[{time.strftime('%H:%M:%S')}] X={X.shape}, y={y.shape}, pos={y.sum()} ({100 * y.mean():.2f}%) in {time.time() - t_pipeline:.0f}s",
        flush=True,
    )
    np.save(HDF5_DIR / "X_features.npy", X)
    np.save(HDF5_DIR / "y_labels.npy", y)
    (HDF5_DIR / "feature_names.json").write_text(
        json.dumps(get_canonical_feature_names())
    )
    print(f"Saved to {HDF5_DIR}", flush=True)


if __name__ == "__main__":
    main()
