#!/usr/bin/env python3
"""Chunked GPU extraction — load 50 days, process on GPU, repeat. GPU stays busy."""

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
CHUNK = 50


def _load_chunk(days_chunk):
    from bah2026.data.reader import load_solexs_lc, load_hel1os_lc, load_solexs_pi
    from bah2026.data.preprocessing import align_hel1os_to_solexs
    from bah2026.data.corrections import (
        correct_solexs_deadtime,
        subtract_hel1os_background,
    )
    from bah2026.features.spectral_fitting import fit_temperature

    sxr_list, hxr_list, pre_list, et_list, ts_list = [], [], [], [], []
    for d in days_chunk:
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
            aligned = None
            try:
                hx = load_hel1os_lc(d, detector="czt", num=1)
                if hx["ctr"].size > 0:
                    ctr = subtract_hel1os_background(hx["ctr"], "czt")
                    aligned = align_hel1os_to_solexs(
                        hx["mjd"], ctr, time_s, sxr["mjdrefi"], sxr["mjdreff"]
                    )
            except Exception:
                pass
            temp_mk = 0.0
            try:
                pi = load_solexs_pi(d)
                if pi["counts"].size > 0:
                    summed = np.nansum(pi["counts"][:300, :], axis=0)
                    if np.sum(summed) > 100:
                        T, EM, chi2 = fit_temperature(summed)
                        if T > 0:
                            temp_mk = float(T)
            except Exception:
                pass

            pre = {
                "sxr_temperature_mk": temp_mk,
                "neupert_granger_improvement": 0.0,
                "neupert_best_lag": 0.0,
                "max_mediation_proportion": 0.0,
                "deadtime_max_pct": 0.0,
                "bg_fraction_pct": 0.0,
                "hk_czt1temp": 0.0,
                "hk_czt2temp": 0.0,
                "hk_cdte1temp": 0.0,
                "hk_cdte2temp": 0.0,
                "hk_czthvmon": 0.0,
                "hk_cdtehvmon": 0.0,
                "hk_czt1satctr": 0.0,
                "hk_cdte1pilectr": 0.0,
                "hxr_spectral_index_gamma": 0.0,
                "hxr_gamma_czt2": 0.0,
                "hxr_gamma_cdte1": 0.0,
                "hxr_gamma_cdte2": 0.0,
                "nonthermal_gamma": 0.0,
                "nonthermal_ec": 0.0,
                "nonthermal_n_nth": 0.0,
                "thermal_fraction": 0.0,
                "goes_xrsb_flux": 0.0,
                "goes_xrsa_flux": 0.0,
                "goes_xrsa_xrsb_ratio": 0.0,
                "sxr_emission_measure": 0.0,
                "sxr_chi2_red": 0.0,
                "czt2_total_mean": 0.0,
                "czt2_total_max": 0.0,
                "czt2_total_std": 0.0,
                "cdte2_total_mean": 0.0,
                "cdte2_total_max": 0.0,
                "cdte2_total_std": 0.0,
            }
            sxr_list.append(counts)
            hxr_list.append(aligned)
            pre_list.append(pre)
            ts_list.append(time_s)
        except Exception:
            pass
    return sxr_list, hxr_list, pre_list, ts_list


def main():
    csv = CATALOGS_DIR / "nowcast_catalogue.csv"
    df = pd.read_csv(csv) if csv.exists() else pd.DataFrame()
    event_times = {}
    if "date" in df.columns and "peak_time" in df.columns:
        for _, row in df.iterrows():
            event_times.setdefault(str(row["date"]), []).append(row["peak_time"])

    days = discover_combined_days()
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"GPU: {gpu_name} | {len(days)} days | chunk={CHUNK}", flush=True)

    from bah2026.features.gpu_features import gpu_extract_features_batch

    all_X, all_y = [], []
    t0 = time.time()

    for ci in range(0, len(days), CHUNK):
        chunk_days = days[ci : ci + CHUNK]
        t_load = time.time()
        sxr_l, hxr_l, pre_l, ts_l = _load_chunk(chunk_days)
        et_l = [event_times.get(str(d), []) for d in chunk_days[: len(sxr_l)]]

        X_chunk, y_chunk = gpu_extract_features_batch(
            sxr_l, hxr_l, pre_l, et_l, ts_l, batch_size=CHUNK
        )
        t_gpu = time.time() - t_load

        if X_chunk is not None and len(X_chunk) > 0:
            all_X.append(X_chunk)
            all_y.append(y_chunk)

        done = min(ci + CHUNK, len(days))
        total_elapsed = time.time() - t0
        total_rows = sum(x.shape[0] for x in all_X)
        print(
            f"  [{done}/{len(days)}] load+gpu={t_gpu:.1f}s rows={total_rows} "
            f"gpu_mem={torch.cuda.memory_allocated() / 1024**3:.1f}GB total={total_elapsed:.0f}s",
            flush=True,
        )
        del sxr_l, hxr_l, pre_l
        torch.cuda.empty_cache()

    if all_X:
        X = np.vstack(all_X)
        y = np.concatenate(all_y)
        total_time = time.time() - t0
        print(
            f"\nDone: X={X.shape}, y={y.shape}, pos={y.sum()} ({100 * y.mean():.2f}%) in {total_time:.0f}s",
            flush=True,
        )
        np.save(HDF5_DIR / "X_features.npy", X)
        np.save(HDF5_DIR / "y_labels.npy", y)
        from bah2026.features.engineering import get_canonical_feature_names

        (HDF5_DIR / "feature_names.json").write_text(
            json.dumps(get_canonical_feature_names())
        )
        print(f"Saved to {HDF5_DIR}", flush=True)


if __name__ == "__main__":
    main()
