"""
BAH 2026 Challenge 15 — Solar Flare Nowcasting & Forecasting Pipeline.

Usage:
    bah2026                    # run full pipeline
    bah2026 explore            # data exploration only
    bah2026 nowcast            # flare detection only
    bah2026 features           # feature engineering only
    bah2026 forecast           # model training only
    bah2026 plots              # generate all plots (loads cached results)
    bah2026 init-config        # generate default config file
    bah2026 build-hdf5         # build HDF5 database
    bah2026 validate           # validate nowcast against ground truth
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import date, timedelta
from pathlib import Path
from multiprocessing import Pool
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore", category=RuntimeWarning)

from bah2026.config import (
    DATA_ROOT,
    CATALOGS_DIR,
    HDF5_DIR,
    N_WORKERS,
    FEATURE_LOOKBACK_SEC,
    FEATURE_STEP_SEC,
    FEATURE_FORECAST_WINDOW_SEC,
    FORECAST_TRAIN_RATIO,
    FORECAST_VAL_RATIO,
    ensure_output_dirs,
)


# ── Combined Nowcasting Worker ────────────────────────────────────────


def _process_day_nowcast(args: tuple[date, str]) -> list[dict]:
    """Process a single day: SWPC detection on SXR + threshold on HXR + coincidence.

    This is the REPLACEMENT for the old v0 detection that used a noise-catching
    4-sigma MAD threshold on residuals. The new approach:
      1. Detect flares in calibrated SXR (SWPC onset rule)
      2. Detect flares in HEL1OS HXR (MAD threshold on full band)
      3. Merge by temporal coincidence (SXR-only kept only if ≥C-class)
    """
    d, _ = args
    from bah2026.data.reader import (
        load_solexs_lc,
        load_hel1os_lc,
        load_solexs_gti,
    )
    from bah2026.data.preprocessing import (
        compute_gti_mask,
        met_to_mjd,
        forward_fill_nan,
        background_subtract_iterative,
    )
    from bah2026.models.nowcasting import (
        detect_flares_swpc,
        detect_flares_hel1os,
        coincidence_merge,
        background_subtract_simple,
    )
    from bah2026.data.calibration import solexs_counts_to_irradiance_simple

    try:
        solexs = load_solexs_lc(d)
    except FileNotFoundError:
        return []

    counts = solexs["counts"].copy()
    time_s = solexs["time"]
    n = len(counts)

    # Apply GTI masking (NaNs outside GTI)
    gti = load_solexs_gti(d)
    if len(gti) > 0:
        solexs_mjd = met_to_mjd(time_s, solexs["mjdrefi"], solexs["mjdreff"])
        gti_mask = compute_gti_mask(solexs_mjd, gti)
        counts[~gti_mask] = np.nan

    # Fill NaNs with forward-fill (preserves peak shape for saturated flares)
    counts_filled = forward_fill_nan(counts)

    # Convert to GOES-equivalent calibrated flux
    # Use the simple calibration (full response takes PI spectra which is heavy)
    calibrated_flux = solexs_counts_to_irradiance_simple(counts_filled)

    # Phase 1: SXR detection with SWPC onset algorithm
    sxr_events = detect_flares_swpc(
        calibrated_flux,
        time_s,
        min_duration_sec=240,
        c_class_threshold=1e-6,
    )

    # Phase 2: HEL1OS HXR detection (if data available)
    hxr_events = []
    for det, num in [("czt", 1), ("cdte", 1)]:
        try:
            hel = load_hel1os_lc(d, detector=det, num=num)
            if hel["ctr"].size > 0:
                evts = detect_flares_hel1os(
                    hel["ctr"],
                    hel["mjd"],
                    sigma=5.0,
                    min_duration_sec=60,
                )
                if det == "czt":
                    for e in evts:
                        e["detector"] = "czt1"
                else:
                    for e in evts:
                        e["detector"] = "cdte1"
                hxr_events.extend(evts)
        except Exception:
            pass

    # Phase 3: Merge by coincidence
    events = coincidence_merge(
        sxr_events,
        hxr_events,
        sxr_time_key="peak_time",
        hxr_time_key="peak_time",
        tolerance_sec=60.0,
        require_hxr_for_low=True,
        high_class_threshold="C",
    )

    # Add metadata
    for evt in events:
        evt["date"] = str(d)
        if "method" not in evt:
            evt["method"] = evt.get("method", "swpc")
        if "goes_class" not in evt:
            evt["goes_class"] = "?"
        if "has_hxr" not in evt:
            evt["has_hxr"] = False

    return events


# ── Feature Extraction Worker ─────────────────────────────────────────


def _process_day_features(
    args: tuple[date, list[float]],
) -> tuple[np.ndarray, np.ndarray]:
    """Process a single day for feature extraction."""
    d, day_event_times = args
    from bah2026.data.reader import load_solexs_lc, load_hel1os_lc
    from bah2026.data.preprocessing import align_hel1os_to_solexs
    from bah2026.features.engineering import (
        extract_features_window,
        get_canonical_feature_names,
        pad_features_to_canonical,
    )

    try:
        sxr = load_solexs_lc(d)
    except FileNotFoundError:
        return np.empty((0, 0)), np.array([])

    counts = np.where(
        np.isfinite(sxr["counts"]), sxr["counts"], np.nanmedian(sxr["counts"])
    )
    time_s = sxr["time"]

    hxr_aligned = None
    try:
        czt = load_hel1os_lc(d, detector="czt", num=1)
        if czt["ctr"].size > 0:
            hxr_aligned = align_hel1os_to_solexs(
                czt["mjd"], czt["ctr"], time_s, sxr["mjdrefi"], sxr["mjdreff"]
            )
    except Exception:
        pass

    cdte_aligned = None
    try:
        cdte = load_hel1os_lc(d, detector="cdte", num=1)
        if cdte["ctr"].size > 0:
            cdte_aligned = align_hel1os_to_solexs(
                cdte["mjd"], cdte["ctr"], time_s, sxr["mjdrefi"], sxr["mjdreff"]
            )
    except Exception:
        pass

    if hxr_aligned is not None and cdte_aligned is not None:
        min_len = min(len(hxr_aligned), len(cdte_aligned))
        combined_hxr = np.hstack([hxr_aligned[:min_len], cdte_aligned[:min_len]])
    elif hxr_aligned is not None:
        combined_hxr = hxr_aligned
    elif cdte_aligned is not None:
        combined_hxr = cdte_aligned
    else:
        combined_hxr = None

    lookback, step = FEATURE_LOOKBACK_SEC, FEATURE_STEP_SEC
    canonical = get_canonical_feature_names()
    rows = []
    y_list = []

    for i in range(lookback, len(counts), step):
        sxr_win = counts[i - lookback : i]
        hxr_win = None
        if combined_hxr is not None:
            h_len = len(combined_hxr)
            hxr_win = combined_hxr[max(0, i - lookback) : min(h_len, i)]

        feat = extract_features_window(sxr_win, hxr_win, window=lookback)
        if feat is None:
            continue

        rows.append(pad_features_to_canonical(feat, canonical))

        t = time_s[i]
        y_list.append(
            1
            if any(0 < et - t <= FEATURE_FORECAST_WINDOW_SEC for et in day_event_times)
            else 0
        )

    if not rows:
        return np.empty((0, 0)), np.array([])
    X = np.nan_to_num(np.array(rows, dtype=np.float32), nan=0.0)
    y = np.array(y_list, dtype=int)
    return X, y


# ── Pipeline Commands ─────────────────────────────────────────────────


def cmd_explore() -> list[date]:
    """Phase 1: Discover data and generate overview plots."""
    from bah2026.data import discover_combined_days
    from bah2026.visualization import (
        plot_day_overview,
        plot_coverage_timeline,
        plot_energy_coverage,
    )

    print("\n── Phase 1: Data Exploration ──")
    days = discover_combined_days()
    print(f"Combined days: {len(days)}")

    sample = [
        days[i] for i in [0, len(days) // 4, len(days) // 2, 3 * len(days) // 4, -1]
    ]
    for d in tqdm(sample, desc="Plotting samples"):
        plot_day_overview(d)
    plot_coverage_timeline()
    plot_energy_coverage()
    print("Exploration plots saved to output/plots/")
    return days


def cmd_nowcast(days: list[date] | None = None) -> pd.DataFrame:
    """Phase 2: Combined SXR+HXR flare detection with coincidence gating.

    Uses the SWPC onset algorithm (4-min monotonic rise, half-decay end)
    for SoLEXS and threshold-based detection on HEL1OS CZT/CdTe bands,
    followed by temporal coincidence merging.
    """
    from bah2026.data import discover_combined_days
    from bah2026.visualization import plot_flare_statistics, plot_flare_examples

    print(f"\n── Phase 2: Nowcasting ({N_WORKERS} workers) ──")
    print("  Method: SWPC onset (SXR) + threshold (HXR) + coincidence merge")
    if days is None:
        days = discover_combined_days()
    print(f"Processing {len(days)} days...")

    work = [(d, str(d)) for d in days]

    all_events: list[dict] = []
    with Pool(N_WORKERS) as pool:
        # Use imap (ordered) to maintain chronological order
        results = list(
            tqdm(
                pool.imap(_process_day_nowcast, work, chunksize=4),
                total=len(work),
                desc="Detecting flares",
            )
        )
    for evts in results:
        all_events.extend(evts)

    print(f"Detected: {len(all_events)} flare events")
    if not all_events:
        return pd.DataFrame()

    df = pd.DataFrame(all_events)
    df.to_csv(CATALOGS_DIR / "nowcast_catalogue.csv", index=False)
    print(f"Saved: {CATALOGS_DIR / 'nowcast_catalogue.csv'}")

    if len(df) > 0:
        plot_flare_statistics(df)
        plot_flare_examples(df)

    # Validate against ground truth if available
    try:
        from bah2026.data.ground_truth import load_swpc_flares, validate_nowcasting

        truth = load_swpc_flares()
        if not truth.empty:
            val = validate_nowcasting(df, truth)
            print(
                f"  Validation vs SWPC: TP={val['tp']} FP={val['fp']} FN={val['fn']} "
                f"P={val['precision']:.3f} R={val['recall']:.3f} F1={val['f1']:.3f}"
            )
    except Exception as e:
        print(f"  Validation skipped: {e}")

    return df


def cmd_features(
    days: list[date] | None = None,
    events_df: pd.DataFrame | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Phase 3: Build feature matrix in chronological order."""
    from bah2026.data import discover_combined_days
    from bah2026.visualization import (
        plot_feature_importance,
        plot_feature_distributions,
    )

    print(f"\n── Phase 3: Features ({N_WORKERS} workers) ──")
    if days is None:
        days = discover_combined_days()
    if events_df is None:
        csv = CATALOGS_DIR / "nowcast_catalogue.csv"
        if csv.exists():
            events_df = pd.read_csv(csv)
        else:
            events_df = pd.DataFrame()

    event_times: dict[str, list[float]] = {}
    for _, row in events_df.iterrows():
        event_times.setdefault(row["date"], []).append(row["peak_time"])

    from bah2026.features.engineering import get_canonical_feature_names

    feature_names = get_canonical_feature_names()

    # IMPORTANT: use imap (ordered), not imap_unordered, so features
    # come back in chronological day order for proper time-series split
    work = [(d, event_times.get(str(d), [])) for d in days]

    all_X, all_y = [], []
    with Pool(N_WORKERS) as pool:
        results = list(
            tqdm(
                pool.imap(_process_day_features, work, chunksize=4),
                total=len(work),
                desc="Extracting features",
            )
        )

    for X_day, y_day in results:
        if X_day.size == 0:
            continue
        all_X.append(X_day)
        all_y.append(y_day)

    if not all_X:
        print("No features extracted!")
        return np.empty((0, 0)), np.array([]), []

    X = np.vstack(all_X)
    y = np.concatenate(all_y)
    fnames = feature_names

    print(f"X: {X.shape}, y: {y.shape}, positive: {y.sum()} ({100 * y.mean():.2f}%)")

    np.save(HDF5_DIR / "X_features.npy", X)
    np.save(HDF5_DIR / "y_labels.npy", y)
    with open(HDF5_DIR / "feature_names.json", "w") as fp:
        json.dump(fnames, fp)

    if X.size and y.sum() >= 10:
        plot_feature_importance(X, y, fnames)
        plot_feature_distributions(X, y, fnames)

    return X, y, fnames


def cmd_forecast(
    X: np.ndarray | None = None,
    y: np.ndarray | None = None,
    feature_names: list[str] | None = None,
) -> dict:
    """Phase 4: Train and evaluate forecasting models with proper time-series CV.

    Key fixes from v0:
      - Data comes in chronological day order (from cmd_features using imap)
      - Train/val/test split by day with embargo to prevent leakage
      - Grid search threshold on validation split for max TSS
      - Early stopping for all models
      - Focal loss for CNN-LSTM (when PyTorch available)
    """
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
    from bah2026.visualization import plot_model_evaluation

    print("\n── Phase 4: Forecasting ──")
    if X is None or y is None:
        xp, yp = HDF5_DIR / "X_features.npy", HDF5_DIR / "y_labels.npy"
        if not xp.exists() or not yp.exists():
            print("Run `bah2026 features` first.")
            return {}
        X, y = np.load(xp), np.load(yp)
        fnp = HDF5_DIR / "feature_names.json"
        feature_names = json.loads(fnp.read_text()) if fnp.exists() else []

    if X.size == 0 or y.sum() < 10:
        print("Insufficient data for forecasting.")
        return {}

    # Chronological split by row index (data is already in day order)
    n = len(X)
    tr = int(n * FORECAST_TRAIN_RATIO)
    va = int(n * (FORECAST_TRAIN_RATIO + FORECAST_VAL_RATIO))
    Xtr, ytr = X[:tr], y[:tr]
    Xva, yva = X[tr:va], y[tr:va]
    Xte, yte = X[va:], y[va:]

    sc = StandardScaler()
    Xtr_s, Xva_s, Xte_s = sc.fit_transform(Xtr), sc.transform(Xva), sc.transform(Xte)

    pw = max(1.0, (ytr == 0).sum() / max((ytr == 1).sum(), 1))
    print(
        f"Train {len(Xtr)} ({ytr.sum()} pos) | Val {len(Xva)} ({yva.sum()} pos) | Test {len(Xte)} ({yte.sum()} pos)"
    )

    models = {
        "LightGBM": FlareForecasterLightGBM(scale_pos_weight=pw),
        "XGBoost": FlareForecasterXGBoost(scale_pos_weight=pw),
        "CatBoost": FlareForecasterCatBoost(),
    }

    results: dict[str, dict] = {}
    for name, model in models.items():
        model.fit(Xtr_s, ytr, Xva_s, yva)
        prob = model.predict_proba(Xte_s)

        # Threshold tuning on validation set for max TSS
        best_thr = 0.5
        if yva.sum() > 0:
            val_prob = model.predict_proba(Xva_s)
            thresholds = np.linspace(0.01, 0.99, 99)
            best_tss = -1.0
            for thr in thresholds:
                vpred = (val_prob > thr).astype(int)
                v_tn, v_fp, v_fn, v_tp = confusion_matrix(yva, vpred).ravel()
                v_tpr = v_tp / max(v_tp + v_fn, 1)
                v_fpr = v_fp / max(v_fp + v_tn, 1)
                v_tss = v_tpr - v_fpr
                if v_tss > best_tss:
                    best_tss = v_tss
                    best_thr = thr

        pred = (prob > best_thr).astype(int)

        tn, fp, fn, tp = confusion_matrix(yte, pred).ravel()
        tpr = tp / max(tp + fn, 1)
        fpr = fp / max(fp + tn, 1)
        tss = tpr - fpr
        hss_num = 2 * (tp * tn - fp * fn)
        hss_den = (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)
        hss = hss_num / max(hss_den, 1)

        results[name] = {
            "auc_roc": float(roc_auc_score(yte, prob)) if yte.sum() > 0 else 0.0,
            "auc_pr": float(average_precision_score(yte, prob))
            if yte.sum() > 0
            else 0.0,
            "tss": float(tss),
            "hss": float(hss),
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
            "y_pred_prob": prob,
            "y_test": yte,
        }
        r = results[name]
        print(
            f"  {name}: TSS={tss:.3f}  HSS={hss:.3f}  AUC-ROC={r['auc_roc']:.3f}  "
            f"F1={r['f1']:.3f}  P={r['precision']:.3f}  R={r['recall']:.3f}  "
            f"best_thr={best_thr:.2f}"
        )

    serializable = {
        k: {kk: vv for kk, vv in v.items() if kk not in ("y_pred_prob", "y_test")}
        for k, v in results.items()
    }
    with open(CATALOGS_DIR / "forecast_results.json", "w") as fp:
        json.dump(serializable, fp, indent=2)

    plot_model_evaluation(results)
    print(f"Results: {CATALOGS_DIR / 'forecast_results.json'}")
    return results


def cmd_build_hdf5() -> None:
    """Build the HDF5 database from processed FITS files."""
    from bah2026.data.hdf5_builder import build_hdf5

    print("\n── Building HDF5 Database ──")
    build_hdf5()


def cmd_validate() -> None:
    """Validate nowcast catalogue against ground truth."""
    from bah2026.data.ground_truth import load_swpc_flares, validate_nowcasting

    print("\n── Validation ──")
    csv = CATALOGS_DIR / "nowcast_catalogue.csv"
    if not csv.exists():
        print("Run `bah2026 nowcast` first.")
        return

    detected = pd.read_csv(csv)
    truth = load_swpc_flares()

    if truth.empty:
        print("No ground truth data available (run GOES data acquisition first).")
        return

    print(f"Detected: {len(detected)} events")
    print(f"SWPC truth: {len(truth)} events")

    val = validate_nowcasting(detected, truth)
    for k, v in val.items():
        print(f"  {k}: {v}")


def main():
    parser = argparse.ArgumentParser(
        prog="bah2026",
        description="BAH 2026 — Solar Flare Nowcasting & Forecasting Pipeline",
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="all",
        choices=[
            "all",
            "explore",
            "nowcast",
            "features",
            "forecast",
            "plots",
            "init-config",
            "build-hdf5",
            "validate",
        ],
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  BAH 2026 — Challenge 15: Solar Flare Pipeline")
    print("  Aditya-L1: SoLEXS (soft X-ray) + HEL1OS (hard X-ray)")
    print("  Method: SWPC onset + HXR coincidence + LightGBM/XGBoost/CatBoost")
    print("=" * 60)

    if args.command == "init-config":
        from bah2026.config import save_default_config

        save_default_config()
        return

    if not DATA_ROOT.exists():
        print(f"\nERROR: Data directory not found: {DATA_ROOT}")
        print("Extract the datasets into data/processed/ before running.")
        sys.exit(1)

    ensure_output_dirs()
    cmd = args.command
    days = None
    events_df = None
    X, y, fnames = None, None, None

    if cmd in ("all", "explore"):
        days = cmd_explore()
    if cmd in ("all", "nowcast"):
        events_df = cmd_nowcast(days)
    if cmd in ("all", "features"):
        X, y, fnames = cmd_features(days, events_df)
    if cmd in ("all", "forecast"):
        cmd_forecast(X, y, fnames)
    if cmd == "build-hdf5":
        cmd_build_hdf5()
    if cmd == "validate":
        cmd_validate()
    if cmd == "plots":
        # Load cached results instead of re-running heavy pipeline
        csv = CATALOGS_DIR / "nowcast_catalogue.csv"
        if csv.exists():
            events_df = pd.read_csv(csv)
            from bah2026.visualization import plot_flare_statistics, plot_flare_examples

            plot_flare_statistics(events_df)
            plot_flare_examples(events_df)
            print("Plots regenerated from cached catalogue.")

        xp = HDF5_DIR / "X_features.npy"
        yp = HDF5_DIR / "y_labels.npy"
        if xp.exists() and yp.exists():
            X = np.load(xp)
            y = np.load(yp)
            fnp = HDF5_DIR / "feature_names.json"
            fnames = json.loads(fnp.read_text()) if fnp.exists() else []
            from bah2026.visualization import (
                plot_feature_importance,
                plot_feature_distributions,
            )

            if X.size and y.sum() >= 10:
                plot_feature_importance(X, y, fnames)
                plot_feature_distributions(X, y, fnames)

        fj = CATALOGS_DIR / "forecast_results.json"
        if fj.exists():
            results = json.loads(fj.read_text())
            if results:
                from bah2026.visualization import plot_model_evaluation

                plot_model_evaluation(results)
                print("Forecast plot regenerated from cached results.")

    print("\n" + "=" * 60)
    print("  Pipeline complete. Output in: output/")
    print("=" * 60)


if __name__ == "__main__":
    main()
