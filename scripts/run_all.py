#!/usr/bin/env python3
"""Run full pipeline for ALL days: nowcast + GPU features + CSV/interpretation + forecast."""

import sys, os, time, warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ["BAH2026_DATA"] = os.path.abspath("data/processed")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
warnings.filterwarnings("ignore")

import numpy as np, pandas as pd
from bah2026.data import discover_combined_days
from bah2026.main import _process_day_nowcast, _process_day_features_gpu
from bah2026.config import HDF5_DIR, MODELS_DIR, CATALOGS_DIR, ensure_output_dirs


def main():
    ensure_output_dirs()
    t_start = time.time()
    print("=== FULL PIPELINE ===", flush=True)
    days = discover_combined_days()
    print(f"{len(days)} days: {days[0]} to {days[-1]}", flush=True)

    # 1. Nowcast
    print("--- NOWCAST ---", flush=True)
    all_events = []
    n_fail = 0
    for i, d in enumerate(days):
        try:
            evts = _process_day_nowcast((d, str(d)))
            all_events.extend(evts)
        except Exception as e:
            n_fail += 1
            if n_fail <= 5:
                print(f"  FAIL {d}: {e}", flush=True)
        if (i + 1) % 100 == 0:
            print(f"  [{i + 1}/{len(days)}] {time.time() - t_start:.0f}s", flush=True)

    events_df = pd.DataFrame(all_events)
    events_df.to_csv(CATALOGS_DIR / "nowcast_catalogue.csv", index=False)
    print(
        f"{len(all_events)} events ({n_fail} fails) in {time.time() - t_start:.0f}s",
        flush=True,
    )

    event_times = {}
    for _, r in events_df.iterrows():
        event_times.setdefault(str(r["date"]), []).append(r["peak_time"])

    # 2. GPU Features + CSV
    print("--- GPU FEATURES + CSV ---", flush=True)
    all_X, all_y = [], []
    t1 = time.time()
    n_fail = 0
    for i, d in enumerate(days):
        try:
            Xd, yd = _process_day_features_gpu(
                (d, event_times.get(str(d), [])), save_csv=True
            )
            if Xd.size > 0:
                all_X.append(Xd)
                all_y.append(yd)
        except Exception as e:
            n_fail += 1
            if n_fail <= 5:
                print(f"  FAIL {d}: {e}", flush=True)
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t1
            rate = (i + 1) / max(elapsed, 0.01)
            remain = (len(days) - i - 1) / rate
            print(
                f"  [{i + 1}/{len(days)}] {elapsed:.0f}s elap, ~{remain:.0f}s rem ({n_fail} fails)",
                flush=True,
            )

    X = np.vstack(all_X) if all_X else np.empty((0, 179))
    y = np.concatenate(all_y) if all_y else np.array([])
    np.save(HDF5_DIR / "X_features.npy", X)
    np.save(HDF5_DIR / "y_labels.npy", y)
    z = (X == 0).all(axis=0).sum()
    print(f"X={X.shape}, y={y.sum()}/{len(y)} pos, z_cols={z}/{X.shape[1]}", flush=True)

    # 3. Forecast
    if y.sum() >= 10:
        print("--- FORECAST ---", flush=True)
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import confusion_matrix, roc_auc_score
        from catboost import CatBoostClassifier
        from bah2026.config import FORECAST_TRAIN_RATIO, FORECAST_VAL_RATIO
        from bah2026.features.engineering import get_canonical_feature_names
        import json

        n = len(X)
        tr = int(n * FORECAST_TRAIN_RATIO)
        va = int(n * (FORECAST_TRAIN_RATIO + FORECAST_VAL_RATIO))

        scaler = StandardScaler()
        Xtr_s = scaler.fit_transform(X[:tr])
        Xva_s = scaler.transform(X[tr:va])
        Xte_s = scaler.transform(X[va:])
        ytr, yva, yte = y[:tr], y[tr:va], y[va:]

        print(
            f"Tr:{len(Xtr_s)}|{ytr.sum()}p Va:{len(Xva_s)}|{yva.sum()}p Te:{len(Xte_s)}|{yte.sum()}p",
            flush=True,
        )

        model = CatBoostClassifier(
            iterations=1000,
            depth=8,
            learning_rate=0.05,
            verbose=0,
            loss_function="Logloss",
            task_type="GPU",
            devices="0",
        )
        model.fit(
            Xtr_s, ytr, eval_set=(Xva_s, yva), verbose=0, early_stopping_rounds=50
        )
        prob = model.predict_proba(Xte_s)[:, 1]

        best_tss, best_thr = -1, 0.5
        for thr in np.linspace(0.01, 0.99, 99):
            p = (prob > thr).astype(int)
            tn, fp, fn, tp = confusion_matrix(yte, p).ravel()
            tss = tp / max(tp + fn, 1) - fp / max(fp + tn, 1)
            if tss > best_tss:
                best_tss, best_thr = tss, thr

        p = (prob > best_thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(yte, p).ravel()
        tss = tp / max(tp + fn, 1) - fp / max(fp + tn, 1)
        auc = roc_auc_score(yte, prob)

        import joblib

        joblib.dump(model, MODELS_DIR / "catboost_checkpoint.joblib")
        joblib.dump(scaler, MODELS_DIR / "scaler.joblib")
        with open(HDF5_DIR / "feature_names.json", "w") as fp:
            json.dump(get_canonical_feature_names(), fp)

        print(
            f"CatBoost GPU: TSS={tss:.4f}  AUC={auc:.4f}  thr={best_thr:.2f}",
            flush=True,
        )
        print(f"TP={tp} FP={fp} FN={fn} TN={tn}", flush=True)
        print(
            f"Precision={tp / max(tp + fp, 1):.3f}  Recall={tp / max(tp + fn, 1):.3f}",
            flush=True,
        )
        print(f"Model: {MODELS_DIR}/catboost_checkpoint.joblib", flush=True)

    print(f"TOTAL: {time.time() - t_start:.0f}s", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
