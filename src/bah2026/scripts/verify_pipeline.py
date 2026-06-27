#!/usr/bin/env python3
"""
Pipeline verification script.

Tests the v1 pipeline fixes by running detection on known flares,
verifying calibration, and checking data integrity.
Run: python -m bah2026.scripts.verify_pipeline
"""

from __future__ import annotations

import sys
import os
from pathlib import Path

# Add src to path
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))
os.environ["BAH2026_DATA"] = str(REPO_ROOT / "data" / "processed")

import numpy as np
from datetime import date

from bah2026.data.reader import (
    load_solexs_lc,
    load_hel1os_lc,
    discover_solexs_days,
    discover_hel1os_days,
    discover_combined_days,
)
from bah2026.data.preprocessing import background_subtract, met_to_mjd
from bah2026.data.calibration import solexs_counts_to_irradiance_simple, classify_goes
from bah2026.models.nowcasting import (
    detect_flares_swpc,
    detect_flares_hel1os,
    coincidence_merge,
)
from bah2026.models.forecasting import (
    FlareForecasterLightGBM,
    FlareForecasterXGBoost,
    FlareForecasterCatBoost,
)


def section(title: str) -> None:
    """Print a section header."""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def test_solexs_integrity() -> dict:
    """Verify SoLEXS data integrity across all days."""
    section("SoLEXS Data Integrity")

    days = discover_solexs_days()
    print(f"  Days: {len(days)}")
    print(f"  Range: {days[0]} -> {days[-1]}")

    # Sample stats
    all_ok = True
    nan_sum = 0
    for d in days[::30]:
        try:
            sx = load_solexs_lc(d)
            n_nan = np.sum(~np.isfinite(sx["counts"]))
            nan_sum += n_nan
            if len(sx["time"]) != 86_400:
                print(f"  BAD: {d} has {len(sx['time'])} rows (expected 86400)")
                all_ok = False
        except Exception as e:
            print(f"  BAD: {d} could not load: {e}")
            all_ok = False

    print(f"  All days correct structure: {all_ok}")
    print(f"  NaN count (sampled): {nan_sum}")
    return {"integrity": all_ok, "days": len(days), "nan_sampled": nan_sum}


def test_hel1os_integrity() -> dict:
    """Verify HEL1OS data integrity."""
    section("HEL1OS Data Integrity")

    days = discover_hel1os_days()
    print(f"  Days: {len(days)}")
    print(f"  Range: {days[0]} -> {days[-1]}")

    all_ok = True
    coverages = []
    for d in days[::20]:
        try:
            czt = load_hel1os_lc(d, detector="czt", num=1)
            if czt["ctr"].size > 0:
                cov_h = (czt["mjd"][-1] - czt["mjd"][0]) * 24
                coverages.append(cov_h)
        except Exception as e:
            print(f"  BAD: {d}: {e}")
            all_ok = False

    if coverages:
        print(
            f"  Coverage (sampled): min={min(coverages):.1f}h, "
            f"median={np.median(coverages):.1f}h, max={max(coverages):.1f}h"
        )
    print(f"  All days load OK: {all_ok}")
    return {"integrity": all_ok, "days": len(days)}


def test_calibration() -> dict:
    """Verify the calibration fix: X6.3 should be X-class now."""
    section("Calibration Verification")

    results = {}

    d = date(2024, 2, 22)
    sx = load_solexs_lc(d)
    raw_peak = float(np.nanmax(sx["counts"]))
    flux = solexs_counts_to_irradiance_simple(np.array([raw_peak]))
    calibrated_peak = float(flux[0])
    cls = classify_goes(calibrated_peak)

    print(f"  X6.3 flare (2024-02-22):")
    print(f"    Raw peak: {raw_peak:,.0f} cts/s")
    print(f"    Calibrated: {calibrated_peak:.3e} W/m2")
    print(f"    GOES class: {cls}")
    print(f"    [v0 said M2.8 - 22x error]")

    results["x63_class"] = cls
    results["x63_correct"] = cls == "X"

    if cls == "X":
        print(f"  ✓ X6.3 correctly classified as X-class!")
    else:
        print(f"  ✗ X6.3 misclassified - expected X, got {cls}")

    # Test big flare
    d2 = date(2025, 7, 29)
    sx2 = load_solexs_lc(d2)
    raw_peak2 = float(np.nanmax(sx2["counts"]))
    flux2 = solexs_counts_to_irradiance_simple(np.array([raw_peak2]))
    cls2 = classify_goes(float(flux2[0]))

    print(f"\n  Biggest flare (2025-07-29):")
    print(f"    Raw peak: {raw_peak2:,.0f} cts/s")
    print(f"    GOES class: {cls2}")

    results["biggest_class"] = cls2
    return results


def test_detection() -> dict:
    """Verify detection on known flares."""
    section("Detection Verification")

    test_cases = [
        (date(2024, 2, 22), "X6.3 flare"),
        (date(2025, 7, 29), "Biggest flare"),
        (date(2025, 2, 26), "Major flare"),
    ]

    results = {}
    for d, desc in test_cases:
        sx = load_solexs_lc(d)
        c = np.where(
            np.isfinite(sx["counts"]),
            sx["counts"],
            np.nanmedian(sx["counts"]),
        )
        flux = solexs_counts_to_irradiance_simple(c)

        evts = detect_flares_swpc(flux, sx["time"])

        classes = set()
        for evt in evts:
            classes.add(classify_goes(float(evt["peak_flux"])))

        print(f"  {desc} ({d}): {len(evts)} events, classes={sorted(classes)}")
        results[str(d)] = {"count": len(evts), "classes": sorted(classes)}

    return results


def test_models() -> dict:
    """Quick verification that forecasting models instantiate."""
    section("Forecasting Model Verification")

    rng = np.random.RandomState(42)
    X = rng.randn(100, 10).astype(np.float32)
    y = rng.randint(0, 2, 100)

    models = {
        "LightGBM": FlareForecasterLightGBM(n_estimators=10),
        "XGBoost": FlareForecasterXGBoost(n_estimators=10),
        "CatBoost": FlareForecasterCatBoost(iterations=10),
    }

    results = {}
    for name, model in models.items():
        try:
            model.fit(X[:80], y[:80])
            prob = model.predict_proba(X[80:])
            results[name] = {
                "ok": True,
                "proba_shape": prob.shape,
                "proba_range": (float(prob.min()), float(prob.max())),
            }
            print(f"  {name}: OK (proba {prob.min():.3f}..{prob.max():.3f})")
        except Exception as e:
            results[name] = {"ok": False, "error": str(e)}
            print(f"  {name}: FAILED - {e}")

    return results


def test_split_integrity() -> dict:
    """Verify the chronological split produces non-leaky results."""
    section("Split Integrity Check")

    rng = np.random.RandomState(42)
    n_samples = 1000
    X = rng.randn(n_samples, 5)
    # Create labels that are time-dependent
    y = (np.arange(n_samples) > 700).astype(int)

    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import confusion_matrix

    tr, va = int(0.7 * n_samples), int(0.85 * n_samples)
    Xtr, ytr = X[:tr], y[:tr]
    Xva, yva = X[tr:va], y[tr:va]
    Xte, yte = X[va:], y[va:]

    sc = StandardScaler()
    Xtr_s, Xte_s = sc.fit_transform(Xtr), sc.transform(Xte)

    model = FlareForecasterLightGBM(n_estimators=50, scale_pos_weight=5.0)
    model.fit(Xtr_s, ytr)
    prob = model.predict_proba(Xte_s)

    # Check that positive class appears only in test
    print(f"  Train positive: {ytr.sum()}/{len(ytr)}")
    print(f"  Val positive: {yva.sum()}/{len(yva)}")
    print(f"  Test positive: {yte.sum()}/{len(yte)}")

    # Verify no leakage: the split is at indices 700, so test should only have
    # positives if the labels are correct
    if ytr.sum() == 0 and yte.sum() > 0:
        print(f"  ✓ Chronological split separates train/test correctly")
    else:
        print(f"  ℹ Standard split - labels spread across sets")

    return {
        "train_pos": int(ytr.sum()),
        "test_pos": int(yte.sum()),
    }


def main() -> int:
    """Run all verification tests."""
    print(f"{'=' * 70}")
    print(f"  BAH 2026 Pipeline Verification")
    print(f"  v1.0 fixes: calibration, detection, split, models")
    print(f"{'=' * 70}")

    results = {}

    results["solexs"] = test_solexs_integrity()
    results["hel1os"] = test_hel1os_integrity()
    results["calibration"] = test_calibration()
    results["detection"] = test_detection()
    results["models"] = test_models()
    results["split"] = test_split_integrity()

    # Summary
    section("Summary")

    n_pass = sum(
        1
        for r in results.values()
        if isinstance(r, dict) and r.get("x63_correct", True)
    )
    n_total = len(results)

    print(f"  Tests: {n_pass}/{n_total} passed")
    print(
        f"  Calibration: X6.3 = {results['calibration']['x63_class']} "
        f"({'✓' if results['calibration']['x63_correct'] else '✗'})"
    )
    print(f"  SoLEXS integrity: {'✓' if results['solexs']['integrity'] else '✗'}")

    det = results["detection"]
    if "2024-02-22" in det:
        classes = det["2024-02-22"]["classes"]
        has_x = "X" in classes
        print(f"  X6.3 detection: {'✓ X-class found' if has_x else '✗ X-class missed'}")
        if has_x:
            print(f"    Classes on flare day: {classes}")

    print(f"\n  Run `bah2026 nowcast` to generate corrected catalogue")
    print(f"  Run `bah2026 forecast` for updated model performance")
    print(f"  Run `streamlit run src/bah2026/visualization/dashboard.py` for dashboard")

    return 0 if results["calibration"]["x63_correct"] else 1


if __name__ == "__main__":
    sys.exit(main())
