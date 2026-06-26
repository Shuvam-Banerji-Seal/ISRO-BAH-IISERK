"""
BAH 2026 Challenge 15 — Solar Flare Nowcasting & Forecasting Pipeline.

Usage:
    bah2026                    # run full pipeline
    bah2026 explore            # data exploration only
    bah2026 nowcast            # flare detection only
    bah2026 features           # feature engineering only
    bah2026 forecast           # model training only
    bah2026 plots              # generate all plots
    bah2026 init-config        # generate default config file
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import date
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore", category=RuntimeWarning)

from bah2026.config import (
    DATA_ROOT, CATALOGS_DIR, HDF5_DIR, N_WORKERS,
    NOWCAST_THRESHOLD_SIGMA, NOWCAST_MIN_DURATION_SEC,
    FEATURE_LOOKBACK_SEC, FEATURE_STEP_SEC, FEATURE_FORECAST_WINDOW_SEC,
    FORECAST_TRAIN_RATIO, FORECAST_VAL_RATIO,
    ensure_output_dirs,
)


def _process_day_nowcast(args: tuple[date, str]) -> list[dict]:
    """Process a single day for nowcasting (worker function for multiprocessing)."""
    d, _ = args
    from bah2026.data.reader import load_solexs_lc, load_hel1os_lc
    from bah2026.models.nowcasting import (
        detect_flares_threshold, classify_flare_goes, background_subtract_simple,
    )

    try:
        solexs = load_solexs_lc(d)
    except FileNotFoundError:
        return []

    counts = np.where(np.isfinite(solexs["counts"]), solexs["counts"], np.nanmedian(solexs["counts"]))
    bg, residual = background_subtract_simple(counts)
    raw_events = detect_flares_threshold(residual, solexs["time"])

    hxr_full, hxr_mjd = None, None
    try:
        hxr = load_hel1os_lc(d, detector="czt", num=1)
        if hxr["ctr"].size > 0:
            hxr_full = hxr["ctr"][:, -1]
            hxr_mjd = hxr["mjd"]
    except Exception:
        pass

    events = []
    seen: set[float] = set()
    for evt in raw_events:
        pt = evt["peak_time"]
        if pt in seen:
            continue
        seen.add(pt)
        evt["date"] = str(d)
        evt["method"] = "threshold"
        evt["goes_class"] = classify_flare_goes(evt["peak_flux"])
        evt["background"] = float(bg[evt["peak_idx"]])

        if hxr_full is not None and hxr_mjd is not None:
            mjd_ref = solexs["mjdrefi"] + solexs["mjdreff"]
            hxr_s = (hxr_mjd - mjd_ref) * 86400.0
            near = np.abs(hxr_s - pt) < 30
            evt["hxr_flux"] = float(np.max(hxr_full[near])) if np.any(near) else 0.0
            evt["has_hxr"] = bool(np.any(near))
        else:
            evt["hxr_flux"] = 0.0
            evt["has_hxr"] = False
        events.append(evt)
    return events


def _process_day_features(args: tuple[date, list[float]]) -> tuple[np.ndarray, np.ndarray]:
    """Process a single day for feature extraction (worker function)."""
    d, day_event_times = args
    from bah2026.data.reader import load_solexs_lc, load_hel1os_lc
    from bah2026.data.preprocessing import align_hel1os_to_solexs
    from bah2026.features.engineering import extract_features_window, get_canonical_feature_names, pad_features_to_canonical

    try:
        sxr = load_solexs_lc(d)
    except FileNotFoundError:
        return np.empty((0, 0)), np.array([])

    counts = np.where(np.isfinite(sxr["counts"]), sxr["counts"], np.nanmedian(sxr["counts"]))
    time_s = sxr["time"]

    hxr_aligned = None
    try:
        hxr = load_hel1os_lc(d, detector="czt", num=1)
        if hxr["ctr"].size > 0:
            hxr_aligned = align_hel1os_to_solexs(
                hxr["mjd"], hxr["ctr"], time_s, sxr["mjdrefi"], sxr["mjdreff"])
    except Exception:
        pass

    lookback, step = FEATURE_LOOKBACK_SEC, FEATURE_STEP_SEC
    canonical = get_canonical_feature_names()
    rows = []
    y_list = []

    for i in range(lookback, len(counts), step):
        sxr_win = counts[i - lookback:i]
        hxr_win = None
        if hxr_aligned is not None:
            h_len = len(hxr_aligned)
            hxr_win = hxr_aligned[max(0, i - lookback):min(h_len, i)]

        feat = extract_features_window(sxr_win, hxr_win, window=lookback)
        if feat is None:
            continue

        rows.append(pad_features_to_canonical(feat, canonical))

        t = time_s[i]
        y_list.append(1 if any(0 < et - t <= FEATURE_FORECAST_WINDOW_SEC for et in day_event_times) else 0)

    if not rows:
        return np.empty((0, 0)), np.array([])
    X = np.nan_to_num(np.array(rows, dtype=np.float32), nan=0.0)
    y = np.array(y_list, dtype=int)
    return X, y


def cmd_explore() -> list[date]:
    """Phase 1: Discover data and generate overview plots."""
    from bah2026.data import discover_combined_days
    from bah2026.visualization import (
        plot_day_overview, plot_coverage_timeline, plot_energy_coverage,
    )

    print("\n── Phase 1: Data Exploration ──")
    days = discover_combined_days()
    print(f"Combined days: {len(days)}")

    sample = [days[i] for i in [0, len(days)//4, len(days)//2, 3*len(days)//4, -1]]
    for d in tqdm(sample, desc="Plotting samples"):
        plot_day_overview(d)
    plot_coverage_timeline()
    plot_energy_coverage()
    print("Exploration plots saved to output/plots/")
    return days


def cmd_nowcast(days: list[date] | None = None) -> pd.DataFrame:
    """Phase 2: Run flare detection across all days using multiprocessing."""
    from bah2026.data import discover_combined_days
    from bah2026.visualization import plot_flare_statistics, plot_flare_examples

    print(f"\n── Phase 2: Nowcasting ({N_WORKERS} workers) ──")
    if days is None:
        days = discover_combined_days()
    print(f"Processing {len(days)} days...")

    work = [(d, str(d)) for d in days]

    all_events: list[dict] = []
    with Pool(N_WORKERS) as pool:
        results = list(tqdm(
            pool.imap_unordered(_process_day_nowcast, work, chunksize=4),
            total=len(work), desc="Detecting flares",
        ))
    for evts in results:
        all_events.extend(evts)

    print(f"Detected: {len(all_events)} flare events")
    if not all_events:
        return pd.DataFrame()

    df = pd.DataFrame(all_events)
    df.to_csv(CATALOGS_DIR / "nowcast_catalogue.csv", index=False)
    print(f"Saved: {CATALOGS_DIR / 'nowcast_catalogue.csv'}")

    plot_flare_statistics(df)
    plot_flare_examples(df)
    return df


def cmd_features(
    days: list[date] | None = None,
    events_df: pd.DataFrame | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Phase 3: Build feature matrix using multiprocessing."""
    from bah2026.data import discover_combined_days
    from bah2026.visualization import plot_feature_importance, plot_feature_distributions

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

    work = [(d, event_times.get(str(d), [])) for d in days]

    all_X, all_y = [], []
    from bah2026.features.engineering import get_canonical_feature_names
    feature_names = get_canonical_feature_names()

    with Pool(N_WORKERS) as pool:
        results = list(tqdm(
            pool.imap_unordered(_process_day_features, work, chunksize=4),
            total=len(work), desc="Extracting features",
        ))

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

    print(f"X: {X.shape}, y: {y.shape}, positive: {y.sum()} ({100*y.mean():.2f}%)")

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
    """Phase 4: Train and evaluate forecasting models."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import (
        roc_auc_score, average_precision_score, f1_score,
        precision_score, recall_score,
    )
    from bah2026.models import (
        FlareForecasterLightGBM, FlareForecasterXGBoost, FlareForecasterCatBoost,
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

    n = len(X)
    tr = int(n * FORECAST_TRAIN_RATIO)
    va = int(n * (FORECAST_TRAIN_RATIO + FORECAST_VAL_RATIO))
    Xtr, ytr = X[:tr], y[:tr]
    Xva, yva = X[tr:va], y[tr:va]
    Xte, yte = X[va:], y[va:]

    sc = StandardScaler()
    Xtr_s, Xva_s, Xte_s = sc.fit_transform(Xtr), sc.transform(Xva), sc.transform(Xte)

    pw = max(1.0, (ytr == 0).sum() / max((ytr == 1).sum(), 1))
    print(f"Train {len(Xtr)} ({ytr.sum()} pos) | Val {len(Xva)} ({yva.sum()} pos) | Test {len(Xte)} ({yte.sum()} pos)")

    models = {
        "LightGBM": FlareForecasterLightGBM(scale_pos_weight=pw),
        "XGBoost": FlareForecasterXGBoost(scale_pos_weight=pw),
        "CatBoost": FlareForecasterCatBoost(),
    }

    results: dict[str, dict] = {}
    for name, model in models.items():
        model.fit(Xtr_s, ytr, Xva_s, yva)
        prob = model.predict_proba(Xte_s)
        pred = (prob > 0.5).astype(int)

        results[name] = {
            "auc_roc": float(roc_auc_score(yte, prob)) if yte.sum() > 0 else 0.0,
            "auc_pr": float(average_precision_score(yte, prob)) if yte.sum() > 0 else 0.0,
            "f1": float(f1_score(yte, pred, zero_division=0)),
            "precision": float(precision_score(yte, pred, zero_division=0)),
            "recall": float(recall_score(yte, pred, zero_division=0)),
            "y_pred_prob": prob,
            "y_test": yte,
        }
        r = results[name]
        print(f"  {name}: AUC-ROC={r['auc_roc']:.3f}  AUC-PR={r['auc_pr']:.3f}  "
              f"F1={r['f1']:.3f}  P={r['precision']:.3f}  R={r['recall']:.3f}")

    serializable = {
        k: {kk: vv for kk, vv in v.items() if kk not in ("y_pred_prob", "y_test")}
        for k, v in results.items()
    }
    with open(CATALOGS_DIR / "forecast_results.json", "w") as fp:
        json.dump(serializable, fp, indent=2)

    plot_model_evaluation(results)
    print(f"Results: {CATALOGS_DIR / 'forecast_results.json'}")
    return results


def main():
    parser = argparse.ArgumentParser(
        prog="bah2026",
        description="BAH 2026 — Solar Flare Nowcasting & Forecasting Pipeline",
    )
    parser.add_argument(
        "command", nargs="?", default="all",
        choices=["all", "explore", "nowcast", "features", "forecast", "plots", "init-config"],
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  BAH 2026 — Challenge 15: Solar Flare Pipeline")
    print("  Aditya-L1: SoLEXS (soft X-ray) + HEL1OS (hard X-ray)")
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
    if cmd == "plots":
        days = cmd_explore()
        events_df = cmd_nowcast(days)
        X, y, fnames = cmd_features(days, events_df)
        cmd_forecast(X, y, fnames)

    print("\n" + "=" * 60)
    print("  Pipeline complete. Output in: output/")
    print("=" * 60)


if __name__ == "__main__":
    main()
