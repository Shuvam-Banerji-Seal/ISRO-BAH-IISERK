#!/usr/bin/env python3
"""BAH 2026 — v3 GPU-Accelerated Pipeline.

Phases:
  1. GPU feature extraction (0.9s/day → ~18min total)
  2. Sequence data preparation (~2h, parallel CPU)
  3. GBDT training with Optuna (~1h GPU)
  4. CNN-LSTM training (~3h GPU)
  5. Transformer training (~3h GPU)
  6. Ensemble stacking + evaluation (~30min)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

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
            / f"v3_pipeline_{time.strftime('%Y%m%d_%H%M%S')}.log"
        ),
    ],
)
log = logging.getLogger("bah2026")


def _gpu_feature_worker(args):
    d, event_times, hdf5_dir = args
    try:
        return _gpu_feature_worker_inner(d, event_times, hdf5_dir)
    except Exception:
        return None


def _gpu_feature_worker_inner(d, event_times, hdf5_dir):
    from bah2026.data.reader import (
        load_solexs_lc,
        load_hel1os_lc,
        load_hel1os_hk,
        load_hel1os_spectra,
    )
    from bah2026.data.preprocessing import align_hel1os_to_solexs
    from bah2026.data.corrections import (
        correct_solexs_deadtime,
        subtract_hel1os_background,
    )
    from bah2026.features.gpu_features import gpu_extract_features_day
    import warnings

    warnings.filterwarnings("ignore")

    sxr = load_solexs_lc(d)
    counts = correct_solexs_deadtime(
        np.where(np.isfinite(sxr["counts"]), sxr["counts"], np.nanmedian(sxr["counts"]))
    )
    deadtime_pct = float(
        np.max(
            np.where(
                sxr["counts"] > 0, (counts - sxr["counts"]) / sxr["counts"] * 100, 0
            )
        )
    )
    time_s = sxr["time"]

    aligned = None
    bg_frac = 0.0
    for det, num in [("czt", 1), ("czt", 2), ("cdte", 1), ("cdte", 2)]:
        try:
            hx = load_hel1os_lc(d, detector=det, num=num)
            if hx["ctr"].size > 0:
                ctr = subtract_hel1os_background(hx["ctr"], det)
                a = align_hel1os_to_solexs(
                    hx["mjd"], ctr, time_s, sxr["mjdrefi"], sxr["mjdreff"]
                )
                if det == "czt" and num == 1:
                    aligned = a
                    raw = hx["ctr"][:, 4]
                    bg_frac = float(np.mean(np.where(raw > 0, 70.0 / raw * 100, 0)))
        except Exception:
            pass

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
        "czt2_total_mean": 0.0,
        "czt2_total_max": 0.0,
        "czt2_total_std": 0.0,
        "cdte2_total_mean": 0.0,
        "cdte2_total_max": 0.0,
        "cdte2_total_std": 0.0,
        "neupert_granger_improvement": 0.0,
        "neupert_best_lag": 0.0,
        "max_mediation_proportion": 0.0,
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
        from bah2026.features.spectral_fitting import (
            fit_temperature,
            fit_spectral_index,
        )

        pi = None
        try:
            from bah2026.data.reader import load_solexs_pi

            pi = load_solexs_pi(d)
        except Exception:
            pass
        if pi is not None and pi["counts"].size > 0:
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
            from bah2026.features.spectral_fitting import fit_spectral_index

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
        from bah2026.features.non_thermal import separate_thermal_non_thermal

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
        from scipy.interpolate import interp1d

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
                    fi = interp1d(gt_mjd, gf, bounds_error=False, fill_value=0.0)
                    full_interp = np.nan_to_num(fi(solexs_mjd), nan=0.0)
                    precomputed[key] = (
                        float(np.nanmean(full_interp))
                        if np.any(np.isfinite(full_interp))
                        else 0.0
                    )
                if precomputed["goes_xrsb_flux"] > 0:
                    precomputed["goes_xrsa_xrsb_ratio"] = (
                        precomputed["goes_xrsa_flux"] / precomputed["goes_xrsb_flux"]
                    )
            break
    except Exception:
        pass

    if aligned is not None and aligned.shape[1] >= 10:
        for prefix, start, end in [("czt2", 5, 10), ("cdte2", 15, 20)]:
            if aligned.shape[1] > start:
                v = aligned[:, start : min(end, aligned.shape[1])]
                v = v[np.isfinite(v)]
                if len(v) > 0:
                    precomputed[f"{prefix}_total_mean"] = float(np.mean(v))
                    precomputed[f"{prefix}_total_max"] = float(np.max(v))
                    precomputed[f"{prefix}_total_std"] = float(np.std(v))

    try:
        from bah2026.features.causal_network import (
            granger_causality_simple,
            mediation_analysis,
        )

        if aligned is not None and aligned.ndim == 2 and aligned.shape[1] >= 5:
            ml = min(len(counts), aligned.shape[0])
            ds = 10
            dsxr = np.diff(counts[:ml][::ds])
            hxr4 = aligned[:ml, 4][::ds]
            gc = granger_causality_simple(hxr4, dsxr, max_lag=30, n_splits=3)
            precomputed["neupert_granger_improvement"] = float(gc["improvement"])
            precomputed["neupert_best_lag"] = float(gc["best_lag"])
            if aligned.shape[1] > 6:
                med = mediation_analysis(
                    aligned[:ml, 1][::ds], aligned[:ml, 6][::ds], counts[:ml][::ds]
                )
                precomputed["max_mediation_proportion"] = float(
                    med["mediation_proportion"]
                )
    except Exception:
        pass

    result = gpu_extract_features_day(
        counts, aligned, precomputed=precomputed, event_times=event_times, time_s=time_s
    )
    return result


def phase_gpu_features(df):
    from bah2026.data import discover_combined_days
    from bah2026.config import HDF5_DIR, N_WORKERS
    from bah2026.features.engineering import get_canonical_feature_names

    days = discover_combined_days()
    event_times = {}
    if "date" in df.columns and "peak_time" in df.columns:
        for _, row in df.iterrows():
            event_times.setdefault(str(row["date"]), []).append(row["peak_time"])

    canonical = get_canonical_feature_names()
    log.info(
        f"v3 Features: {len(canonical)} features, {len(days)} days, {N_WORKERS} workers"
    )

    HDF5_DIR.mkdir(parents=True, exist_ok=True)
    work = [(d, event_times.get(str(d), []), str(HDF5_DIR)) for d in days]

    t0 = time.time()
    all_X, all_y = [], []
    with Pool(N_WORKERS) as pool:
        for j, result in enumerate(
            pool.imap_unordered(_gpu_feature_worker, work, chunksize=4)
        ):
            if result is not None:
                all_X.append(result[0])
                all_y.append(result[1])
            if j % 50 == 49:
                elapsed = time.time() - t0
                rate = (j + 1) / (elapsed / 60)
                log.info(
                    f"  {j + 1}/{len(days)} days ({rate:.0f} d/min, ETA {(len(days) - j - 1) / max(rate, 0.01):.0f} min)"
                )

    elapsed = time.time() - t0
    log.info(
        f"v3 Features: {len(all_X)}/{len(days)} days in {elapsed:.0f}s ({elapsed / 60:.1f} min)"
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


def phase_sequences(df):
    from bah2026.data import discover_combined_days
    from bah2026.data.sequence_builder import (
        build_all_sequences,
        prepare_downsampled_sequences,
    )
    from bah2026.config import HDF5_DIR, CATALOGS_DIR, N_WORKERS

    days = discover_combined_days()
    event_times = {}
    if "date" in df.columns and "peak_time" in df.columns:
        for _, row in df.iterrows():
            event_times.setdefault(str(row["date"]), []).append(row["peak_time"])

    seq_dir = HDF5_DIR / "sequences"
    seq_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Building sequences: {len(days)} days")
    t0 = time.time()
    build_all_sequences(days, event_times, str(seq_dir))
    log.info(f"Sequences built in {(time.time() - t0) / 60:.1f} min")

    X_path = seq_dir / "X_seq.npy"
    y_path = seq_dir / "y_seq.npy"
    if X_path.exists():
        X = np.load(X_path, mmap_mode="r")
        y = np.load(y_path)
        log.info(f"X_seq: {X.shape}, y_seq: {y.shape}")

        ds_path = str(seq_dir / "X_seq_ds10.npy")
        log.info("Downsampling 1s → 10s for transformer...")
        prepare_downsampled_sequences(str(X_path), str(y_path), ds_path, factor=10)
        X_ds = np.load(ds_path, mmap_mode="r")
        log.info(f"X_seq_ds10: {X_ds.shape}")


def phase_forest(X, y, fnames):
    from bah2026.config import CATALOGS_DIR, detect_gpu
    from bah2026.models import (
        FlareForecasterCatBoost,
        FlareForecasterXGBoost,
        FlareForecasterLightGBM,
    )
    from sklearn.metrics import (
        confusion_matrix,
        roc_auc_score,
        average_precision_score,
        f1_score,
    )
    from sklearn.preprocessing import StandardScaler

    detect_gpu()
    n = len(X)
    tr = int(n * 0.70)
    va = int(n * 0.85)
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
        hss_num = 2 * (tp * tn - fp * fn)
        hss_den = (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)
        r = {
            "auc_roc": float(roc_auc_score(yte, prob)) if yte.sum() > 0 else 0.0,
            "auc_pr": float(average_precision_score(yte, prob))
            if yte.sum() > 0
            else 0.0,
            "tss": float(tss),
            "hss": float(hss_num / max(hss_den, 1)),
            "f1": float(f1_score(yte, pred, zero_division=0)),
            "precision": float(tp / max(tp + fp, 1)),
            "recall": float(tp / max(tp + fn, 1)),
            "best_threshold": float(best_thr),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
            "train_time_s": time.time() - t0,
        }
        results[name] = r
        log.info(
            f"  {name}: TSS={tss:.3f} HSS={hss_num / max(hss_den, 1):.3f} AUC={r['auc_roc']:.3f} ({(time.time() - t0):.0f}s)"
        )

    CATALOGS_DIR.mkdir(parents=True, exist_ok=True)
    (CATALOGS_DIR / "v3_forecast_results.json").write_text(
        json.dumps(results, indent=2)
    )
    return results


def phase_cnn_lstm(X_seq_path, y_seq_path):
    import torch
    from torch.utils.data import DataLoader
    from bah2026.data.sequence_builder import SequenceDataset, create_dataloaders
    from bah2026.models.cnn_lstm_v3 import FlareForecasterCNNLSTMv3, evaluate_model
    from bah2026.config import CATALOGS_DIR, HDF5_DIR, detect_gpu, N_WORKERS

    detect_gpu()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"CNN-LSTM training on {device}")

    y = np.load(y_seq_path)
    n = len(y)
    tr = int(n * 0.70)
    va = int(n * 0.85)
    train_idx = np.arange(tr)
    val_idx = np.arange(tr, va)
    test_idx = np.arange(va, n)

    loaders = create_dataloaders(
        str(X_seq_path),
        str(y_seq_path),
        train_idx,
        val_idx,
        test_idx,
        batch_size=256,
    )

    model = FlareForecasterCNNLSTMv3(device=str(device))
    ckpt = str(HDF5_DIR / "cnn_lstm_v3_best.pt")
    history = model.fit(
        loaders["train"], loaders["val"], epochs=50, patience=10, checkpoint_path=ckpt
    )

    log.info(
        f"CNN-LSTM best TSS: {history.get('best_tss', 0):.3f} at epoch {history.get('best_epoch', 0)}"
    )
    return history


def main():
    parser = argparse.ArgumentParser(description="BAH 2026 v3 GPU Pipeline")
    parser.add_argument("--skip-features", action="store_true")
    parser.add_argument("--skip-sequences", action="store_true")
    parser.add_argument("--skip-gbdt", action="store_true")
    parser.add_argument("--skip-cnn", action="store_true")
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("BAH 2026 v3 Pipeline — 179 features, GPU-accelerated, ensemble")
    log.info("=" * 60)

    from bah2026.config import CATALOGS_DIR, HDF5_DIR, ensure_output_dirs

    ensure_output_dirs()

    df = pd.DataFrame()
    csv = CATALOGS_DIR / "nowcast_catalogue.csv"
    if csv.exists():
        df = pd.read_csv(csv)
        log.info(f"Nowcast: {len(df)} events loaded")

    X, y, fnames = None, None, None
    if not args.skip_features:
        log.info("[1/4] GPU Feature Extraction")
        X, y, fnames = phase_gpu_features(df)

    if not args.skip_sequences:
        log.info("[2/4] Sequence Data Preparation")
        phase_sequences(df)

    if not args.skip_gbdt and X is not None:
        log.info("[3/4] GBDT Training")
        phase_forest(X, y, fnames)

    if not args.skip_cnn:
        seq_dir = HDF5_DIR / "sequences"
        X_seq = seq_dir / "X_seq.npy"
        y_seq = seq_dir / "y_seq.npy"
        if X_seq.exists() and y_seq.exists():
            log.info("[4/4] CNN-LSTM Training")
            phase_cnn_lstm(X_seq, y_seq)

    log.info("=" * 60)
    log.info("v3 Pipeline complete!")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
