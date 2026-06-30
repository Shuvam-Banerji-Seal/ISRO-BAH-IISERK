#!/usr/bin/env python3
"""
BAH 2026 — Complete Pipeline Runner (v2).

Runs all three phases sequentially:
  1. Nowcast  — SXR+HXR flare detection with corrections
  2. Features — 117-feature extraction with multiprocessing
  3. Forecast — GPU-accelerated CatBoost/LightGBM/XGBoost training

Usage:
    python -m bah2026.scripts.run_full_pipeline
    python -m bah2026.scripts.run_full_pipeline --skip-nowcast
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
os.environ.setdefault(
    "BAH2026_DATA", str(Path(__file__).resolve().parents[3] / "data" / "processed")
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(
            Path(__file__).resolve().parents[3]
            / "logs"
            / f"pipeline_v2_{time.strftime('%Y%m%d_%H%M%S')}.log"
        ),
    ],
)
log = logging.getLogger("bah2026")


def _feature_worker(args):
    d, event_times, hdf5_dir = args
    try:
        return _feature_worker_inner(d, event_times, hdf5_dir)
    except Exception:
        return None


def _feature_worker_inner(d, event_times, hdf5_dir):
    from bah2026.data.reader import (
        load_solexs_lc,
        load_solexs_pi,
        load_hel1os_lc,
        load_hel1os_spectra,
        load_hel1os_hk,
    )
    from bah2026.data.preprocessing import align_hel1os_to_solexs
    from bah2026.data.corrections import (
        correct_solexs_deadtime,
        subtract_hel1os_background,
    )
    from bah2026.features.engineering import (
        extract_features_window,
        get_canonical_feature_names,
        pad_features_to_canonical,
    )
    from bah2026.features.spectral_fitting import fit_temperature, fit_spectral_index
    from bah2026.features.non_thermal import separate_thermal_non_thermal
    from scipy.interpolate import interp1d
    import numpy as np
    import warnings

    warnings.filterwarnings("ignore", category=RuntimeWarning)

    try:
        sxr = load_solexs_lc(d)
    except FileNotFoundError:
        return None

    counts_raw = np.where(
        np.isfinite(sxr["counts"]), sxr["counts"], np.nanmedian(sxr["counts"])
    )
    counts_corr = correct_solexs_deadtime(counts_raw)
    deadtime_pct = float(
        np.max(
            np.where(counts_raw > 0, (counts_corr - counts_raw) / counts_raw * 100, 0)
        )
    )
    counts = counts_corr
    time_s = sxr["time"]

    aligned, bg_frac = {}, 0.0
    for det, num in [("czt", 1), ("czt", 2), ("cdte", 1), ("cdte", 2)]:
        try:
            hx = load_hel1os_lc(d, detector=det, num=num)
            if hx["ctr"].size > 0:
                ctr = subtract_hel1os_background(hx["ctr"], det)
                a = align_hel1os_to_solexs(
                    hx["mjd"], ctr, time_s, sxr["mjdrefi"], sxr["mjdreff"]
                )
                aligned[f"{det}{num}"] = a
                if det == "czt" and num == 1:
                    raw = hx["ctr"][:, 4]
                    bg_frac = (
                        float(np.mean(np.where(raw > 0, 70.0 / raw * 100, 0)))
                        if np.any(raw > 0)
                        else 0.0
                    )
        except Exception:
            pass

    hxr_parts = [aligned[k] for k in ["czt1", "czt2", "cdte1", "cdte2"] if k in aligned]
    combined_hxr = None
    if hxr_parts:
        ml = min(a.shape[0] for a in hxr_parts)
        combined_hxr = np.hstack([a[:ml] for a in hxr_parts])

    precomputed = {
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
        "deadtime_max_pct": deadtime_pct,
        "bg_fraction_pct": bg_frac,
        "sxr_temperature_mk": 0.0,
        "sxr_emission_measure": 0.0,
        "sxr_chi2_red": 999.0,
    }

    try:
        hk = load_hel1os_hk(d)
        for k, n in [
            ("hk_czt1temp", "czt1temp"),
            ("hk_czt2temp", "czt2temp"),
            ("hk_cdte1temp", "cdte1temp"),
            ("hk_cdte2temp", "cdte2temp"),
            ("hk_czthvmon", "czthvmon"),
            ("hk_cdtehvmon", "cdtehvmon"),
            ("hk_czt1satctr", "czt1satctr1"),
            ("hk_cdte1pilectr", "cdte1pilectr"),
        ]:
            if n in hk:
                precomputed[k] = float(np.mean(hk[n]))
    except Exception:
        pass

    try:
        pi = load_solexs_pi(d)
        if pi["counts"].size > 0:
            summed = np.nansum(pi["counts"][:300, :], axis=0)
            if np.sum(summed) > 100:
                T, EM, chi2 = fit_temperature(summed)
                if T > 0:
                    precomputed["sxr_temperature_mk"] = float(T)
                    precomputed["sxr_emission_measure"] = float(EM)
                    precomputed["sxr_chi2_red"] = float(chi2)
    except Exception:
        pass

    for det, num, key in [
        ("czt", 1, "hxr_spectral_index_gamma"),
        ("czt", 2, "hxr_gamma_czt2"),
        ("cdte", 1, "hxr_gamma_cdte1"),
        ("cdte", 2, "hxr_gamma_cdte2"),
    ]:
        try:
            spec = load_hel1os_spectra(d, detector=det, num=num)
            if spec["counts"].size > 0:
                summed = np.nansum(spec["counts"][:100, :], axis=0)
                if np.sum(summed) > 10:
                    nch = len(summed)
                    bp = max(nch // 4, 1)
                    centroids = np.array(
                        [30, 50, 70, 115] if det == "czt" else [12, 25, 35, 50],
                        dtype=float,
                    )
                    rates = np.array(
                        [np.sum(summed[i * bp : (i + 1) * bp]) for i in range(4)],
                        dtype=float,
                    )
                    precomputed[key] = fit_spectral_index(
                        np.maximum(rates, 1e-10), centroids
                    )
        except Exception:
            pass

    try:
        spec = load_hel1os_spectra(d, detector="czt", num=1)
        if spec["counts"].size > 0:
            summed = np.nansum(spec["counts"][:100, :], axis=0)
            if np.sum(summed) > 50:
                nch = len(summed)
                sep = separate_thermal_non_thermal(
                    np.linspace(20, 150, nch),
                    summed,
                    precomputed["sxr_temperature_mk"] or 15.0,
                    max(precomputed["sxr_emission_measure"], 1e3),
                    thermal_range_kev=(20, 40),
                    nonthermal_range_kev=(40, 150),
                )
                precomputed["nonthermal_gamma"] = sep["gamma"]
                precomputed["nonthermal_ec"] = sep["ec"]
                precomputed["nonthermal_n_nth"] = sep["n_nth"]
                precomputed["thermal_fraction"] = sep["thermal_fraction"]
    except Exception:
        pass

    try:
        gdir = Path(hdf5_dir).parents[2] / "data" / "external" / "goes"
        for nc_file in gdir.glob(f"*g16_d{d.strftime('%Y%m%d')}_v*.nc"):
            from netCDF4 import Dataset

            with Dataset(str(nc_file), "r") as nc:
                gt = nc.variables["time"][:].astype(np.float64)
                gt_mjd = 51544.5 + gt / 86400.0
                solexs_mjd = (sxr["mjdrefi"] + sxr["mjdreff"]) + time_s / 86400.0
                for band, key in [
                    ("xrsb_flux", "goes_xrsb_flux"),
                    ("xrsa_flux", "goes_xrsa_flux"),
                ]:
                    gf = np.where(
                        nc.variables[band][:] < 0, np.nan, nc.variables[band][:]
                    ).astype(np.float64)
                    fi = interp1d(gt_mjd, gf, bounds_error=False, fill_value=np.nan)
                    interp = fi(solexs_mjd[:3600])
                    precomputed[key] = (
                        float(np.nanmean(interp))
                        if np.any(np.isfinite(interp))
                        else 0.0
                    )
                if precomputed["goes_xrsb_flux"] > 0:
                    precomputed["goes_xrsa_xrsb_ratio"] = (
                        precomputed["goes_xrsa_flux"] / precomputed["goes_xrsb_flux"]
                    )
            break
    except Exception:
        pass

    if combined_hxr is not None and combined_hxr.shape[1] >= 10:
        for prefix, start, end in [("czt2", 5, 10), ("cdte2", 15, 20)]:
            if combined_hxr.shape[1] > start:
                v = combined_hxr[:, start : min(end, combined_hxr.shape[1])][
                    np.isfinite(
                        combined_hxr[:, start : min(end, combined_hxr.shape[1])]
                    )
                ]
                if len(v) > 0:
                    precomputed[f"{prefix}_total_mean"] = float(np.mean(v))
                    precomputed[f"{prefix}_total_max"] = float(np.max(v))
                    precomputed[f"{prefix}_total_std"] = float(np.std(v))

    lookback, step = 3600, 300
    canonical = get_canonical_feature_names()
    rows, y_list = [], []
    for i in range(lookback, len(counts), step):
        sxr_win = counts[i - lookback : i]
        hxr_win = (
            combined_hxr[max(0, i - lookback) : min(combined_hxr.shape[0], i)]
            if combined_hxr is not None
            else None
        )
        feat = extract_features_window(
            sxr_win, hxr_win, window=lookback, precomputed=precomputed
        )
        if feat is None:
            continue
        rows.append(pad_features_to_canonical(feat, canonical))
        t = time_s[i]
        y_list.append(1 if any(0 < et - t <= 1800 for et in event_times) else 0)

    if not rows:
        return None
    X = np.nan_to_num(np.array(rows, dtype=np.float32), nan=0.0)
    y = np.array(y_list, dtype=int)
    return X, y


def phase_features(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    from bah2026.data import discover_combined_days
    from bah2026.config import HDF5_DIR, N_WORKERS
    from bah2026.features.engineering import get_canonical_feature_names

    days = discover_combined_days()
    event_times: dict[str, list[float]] = {}
    if "date" in df.columns and "peak_time" in df.columns:
        for _, row in df.iterrows():
            event_times.setdefault(str(row["date"]), []).append(row["peak_time"])

    canonical = get_canonical_feature_names()
    log.info(f"Features: {len(canonical)} days={len(days)} workers={N_WORKERS}")

    HDF5_DIR.mkdir(parents=True, exist_ok=True)
    work = [(d, event_times.get(str(d), []), str(HDF5_DIR)) for d in days]
    all_X, all_y = [], []

    t0 = time.time()
    with Pool(N_WORKERS) as pool:
        for j, result in enumerate(
            pool.imap_unordered(_feature_worker, work, chunksize=4)
        ):
            if result is not None:
                all_X.append(result[0])
                all_y.append(result[1])
            if j % 50 == 49:
                elapsed = time.time() - t0
                rate = (j + 1) / (elapsed / 60)
                eta = (len(work) - j - 1) / max(rate, 0.01)
                log.info(
                    f"  {j + 1}/{len(work)} days ({rate:.0f} d/min, ETA {eta:.0f} min)"
                )
    elapsed = time.time() - t0
    log.info(
        f"Features done: {len(work)} days in {elapsed:.0f}s ({elapsed / 60:.1f} min, {elapsed / max(len(work), 1):.1f}s/day)"
    )

    if not all_X:
        log.error("No features extracted!")
        return np.empty((0, 0)), np.array([]), canonical
    X = np.vstack(all_X)
    y = np.concatenate(all_y)
    log.info(f"X: {X.shape}, y: {y.shape}, pos: {y.sum()} ({100 * y.mean():.2f}%)")
    np.save(HDF5_DIR / "X_features.npy", X)
    np.save(HDF5_DIR / "y_labels.npy", y)
    (HDF5_DIR / "feature_names.json").write_text(json.dumps(canonical))
    return X, y, canonical


def phase_forecast(X, y, fnames):
    from bah2026.config import (
        CATALOGS_DIR,
        FORECAST_TRAIN_RATIO,
        FORECAST_VAL_RATIO,
        detect_gpu,
    )
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import (
        roc_auc_score,
        average_precision_score,
        f1_score,
        precision_score,
        recall_score,
        confusion_matrix,
        balanced_accuracy_score,
        matthews_corrcoef,
        cohen_kappa_score,
    )
    from bah2026.models import (
        FlareForecasterLightGBM,
        FlareForecasterXGBoost,
        FlareForecasterCatBoost,
    )
    import json, numpy as np

    detect_gpu()
    n = len(X)
    tr = int(n * FORECAST_TRAIN_RATIO)
    va = int(n * (FORECAST_TRAIN_RATIO + FORECAST_VAL_RATIO))
    Xtr, ytr = X[:tr], y[:tr]
    Xva, yva = X[tr:va], y[tr:va]
    Xte, yte = X[va:], y[va:]

    sc = StandardScaler()
    Xtr_s, Xva_s, Xte_s = sc.fit_transform(Xtr), sc.transform(Xva), sc.transform(Xte)
    pw = max(1.0, (ytr == 0).sum() / max((ytr == 1).sum(), 1))
    log.info(
        f"Train {len(Xtr)} ({ytr.sum()} pos) | Val {len(Xva)} ({yva.sum()} pos) | Test {len(Xte)} ({yte.sum()} pos)"
    )

    models = {
        "LightGBM": FlareForecasterLightGBM(scale_pos_weight=pw),
        "XGBoost": FlareForecasterXGBoost(scale_pos_weight=pw),
        "CatBoost": FlareForecasterCatBoost(),
    }
    results = {}
    for name, model in models.items():
        t0 = time.time()
        model.fit(Xtr_s, ytr, Xva_s, yva)
        prob = model.predict_proba(Xte_s)
        best_thr = 0.5
        if yva.sum() > 0:
            val_prob = model.predict_proba(Xva_s)
            best_tss = -1.0
            for thr in np.linspace(0.01, 0.99, 99):
                vp = (val_prob > thr).astype(int)
                tn, fp, fn, tp = confusion_matrix(yva, vp).ravel()
                tss = tp / max(tp + fn, 1) - fp / max(fp + tn, 1)
                if tss > best_tss:
                    best_tss, best_thr = tss, thr
        pred = (prob > best_thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(yte, pred).ravel()
        tss = tp / max(tp + fn, 1) - fp / max(fp + tn, 1)
        hss_num, hss_den = (
            2 * (tp * tn - fp * fn),
            (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn),
        )
        r = {
            "auc_roc": float(roc_auc_score(yte, prob)) if yte.sum() > 0 else 0.0,
            "auc_pr": float(average_precision_score(yte, prob))
            if yte.sum() > 0
            else 0.0,
            "tss": float(tss),
            "hss": float(hss_num / max(hss_den, 1)),
            "f1": float(f1_score(yte, pred, zero_division=0)),
            "precision": float(precision_score(yte, pred, zero_division=0)),
            "recall": float(recall_score(yte, pred, zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(yte, pred)),
            "mcc": float(matthews_corrcoef(yte, pred)),
            "kappa": float(cohen_kappa_score(yte, pred)),
            "best_threshold": float(best_thr),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
            "train_time_s": time.time() - t0,
        }
        results[name] = r
        log.info(
            f"  {name}: TSS={tss:.3f} HSS={hss_num / max(hss_den, 1):.3f} AUC={r['auc_roc']:.3f} F1={r['f1']:.3f} ({(time.time() - t0):.0f}s)"
        )

    CATALOGS_DIR.mkdir(parents=True, exist_ok=True)
    (CATALOGS_DIR / "forecast_results.json").write_text(
        json.dumps(
            {k: {kk: vv for kk, vv in v.items()} for k, v in results.items()}, indent=2
        )
    )
    return results


def main():
    parser = argparse.ArgumentParser(description="BAH 2026 v2 — Complete Pipeline")
    parser.add_argument("--skip-forecast", action="store_true")
    parser.add_argument("--skip-nowcast", action="store_true")
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("BAH 2026 v2 Pipeline — 117 features, all corrections")
    log.info("=" * 60)

    from bah2026.config import CATALOGS_DIR, ensure_output_dirs

    ensure_output_dirs()

    df = pd.DataFrame()
    if not args.skip_nowcast:
        log.info("[1/3] Nowcast")
        from bah2026.main import cmd_nowcast
        from bah2026.data import discover_combined_days

        df = cmd_nowcast(discover_combined_days())
    else:
        csv = CATALOGS_DIR / "nowcast_catalogue.csv"
        if csv.exists():
            df = pd.read_csv(csv)
            log.info(f"Nowcast: loaded {len(df)} events from cache")

    log.info("[2/3] Features")
    X, y, fnames = phase_features(df)

    if not args.skip_forecast and X.size > 0:
        log.info("[3/3] Forecast")
        phase_forecast(X, y, fnames)

    log.info("=" * 60)
    log.info("Pipeline complete!")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
