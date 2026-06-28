#!/usr/bin/env python3
"""
Investigate why CZT2 has low correlation with CZT1 on some days.

From the deep analysis: CZT1/CZT2 Pearson r ranges from -0.126 to 0.956,
with median 0.600. 27/91 sampled days have r < 0.5.

Hypotheses to test:
  1. CZT2 is turned off on some days (all zeros → correlation undefined)
  2. CZT2 has different GTI (different observation window)
  3. CZT2 has a different energy band structure
  4. One detector is degraded/noisy on specific dates
  5. Pipeline version differences cause calibration mismatch

Processes every 3rd day for speed (~300 days instead of 902).
"""

from __future__ import annotations

import sys
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))
os.environ["BAH2026_DATA"] = str(REPO_ROOT / "data" / "processed")

import numpy as np
from scipy.stats import pearsonr
from collections import Counter
from tqdm import tqdm

from bah2026.data.reader import load_hel1os_lc, discover_hel1os_days


def section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def investigate():
    section("CZT2 vs CZT1 Correlation Investigation")

    days = discover_hel1os_days()
    print(f"Total HEL1OS days: {len(days)}")
    # Sample every 3rd day for speed
    sample = days[::3]
    print(f"Sampling every 3rd day: {len(sample)} days")

    results = []
    for d in tqdm(sample, desc="Analyzing CZT correlation"):
        try:
            c1 = load_hel1os_lc(d, detector="czt", num=1)
            c2 = load_hel1os_lc(d, detector="czt", num=2)
            if c1["ctr"].size == 0 or c2["ctr"].size == 0:
                continue

            fb1 = c1["ctr"][:, -1]  # Full band
            fb2 = c2["ctr"][:, -1]

            min_n = min(len(fb1), len(fb2))
            fb1, fb2 = fb1[:min_n], fb2[:min_n]

            nz1 = (fb1 > 0).mean() * 100
            nz2 = (fb2 > 0).mean() * 100
            med1 = float(np.median(fb1[fb1 > 0])) if np.any(fb1 > 0) else 0
            med2 = float(np.median(fb2[fb2 > 0])) if np.any(fb2 > 0) else 0
            max1, max2 = float(np.max(fb1)), float(np.max(fb2))
            n1, n2 = len(fb1), len(fb2)
            mjd_start1, mjd_start2 = c1["mjd"][0], c2["mjd"][0]
            mjd_end1, mjd_end2 = c1["mjd"][-1], c2["mjd"][-1]

            mask = (fb1 > 0) & (fb2 > 0)
            if mask.sum() < 5:
                r_val = 0.0
            else:
                r_val, _ = pearsonr(fb1[mask], fb2[mask])

            results.append(
                {
                    "date": d,
                    "r": r_val,
                    "n_czt1": n1,
                    "n_czt2": n2,
                    "nz1_pct": nz1,
                    "nz2_pct": nz2,
                    "med1": med1,
                    "med2": med2,
                    "max1": max1,
                    "max2": max2,
                    "mjd_start1": float(mjd_start1),
                    "mjd_start2": float(mjd_start2),
                    "mjd_end1": float(mjd_end1),
                    "mjd_end2": float(mjd_end2),
                    "n_overlap": int(mask.sum()),
                }
            )
        except Exception as e:
            print(f"  Error {d}: {e}")

    # Sort by correlation
    results.sort(key=lambda x: x["r"])
    r_vals = np.array([r["r"] for r in results])

    print(f"\nAnalyzed {len(results)} days")
    print(f"Correlation range: {r_vals.min():.3f} to {r_vals.max():.3f}")
    print(f"Median: {np.median(r_vals):.3f}, Mean: {np.mean(r_vals):.3f}")
    print(
        f"Days with r < 0.5: {np.sum(r_vals < 0.5)} ({np.mean(r_vals < 0.5) * 100:.1f}%)"
    )
    print(f"Days with r < 0.0: {np.sum(r_vals < 0.0)}")

    # Show worst days
    section("Worst 10 days (r < 0)")
    for r in results[:10]:
        if r["r"] >= 0:
            break
        print(f"  {r['date']}: r={r['r']:.3f}")
        print(
            f"    CZT1: n={r['n_czt1']}, nonzero={r['nz1_pct']:.0f}%, "
            f"median={r['med1']:.1f}, max={r['max1']:.0f}"
        )
        print(
            f"    CZT2: n={r['n_czt2']}, nonzero={r['nz2_pct']:.0f}%, "
            f"median={r['med2']:.1f}, max={r['max2']:.0f}"
        )
        print(
            f"    Rows: CZT1={r['n_czt1']}, CZT2={r['n_czt2']}, overlap={r['n_overlap']}"
        )
        print(f"    MJD: CZT1={r['mjd_start1']:.5f}-{r['mjd_end1']:.5f}")
        print(f"         CZT2={r['mjd_start2']:.5f}-{r['mjd_end2']:.5f}")

    # Show best days
    section("Best 5 days (r > 0.9)")
    best = [r for r in results if r["r"] > 0.9]
    for r in best[:5]:
        print(f"  {r['date']}: r={r['r']:.3f}")
        print(f"    CZT1: nonzero={r['nz1_pct']:.0f}%, median={r['med1']:.1f}")
        print(f"    CZT2: nonzero={r['nz2_pct']:.0f}%, median={r['med2']:.1f}")

    # Check hypothesis: Is CZT2 just all zeros on bad days?
    section("Hypothesis Check: Is CZT2 Disabled?")
    disabled_days = [r for r in results if r["nz2_pct"] < 1.0]
    print(f"Days with CZT2 <1% nonzero: {len(disabled_days)}")
    if disabled_days:
        print(f"  Examples:")
        for r in disabled_days[:5]:
            print(
                f"    {r['date']}: CZT2 nonzero={r['nz2_pct']:.1f}%, "
                f"CZT1 nonzero={r['nz1_pct']:.1f}%, r={r['r']:.3f}"
            )

    # Check hypothesis: different time windows
    section("Hypothesis Check: Different Observation Windows?")
    mismatch_times = [
        r for r in results if abs(r["mjd_start1"] - r["mjd_start2"]) > 0.001
    ]
    print(f"Days with >86s start time mismatch: {len(mismatch_times)}")
    if mismatch_times:
        print(f"  Examples:")
        for r in mismatch_times[:5]:
            diff_s = abs(r["mjd_start1"] - r["mjd_start2"]) * 86400
            print(
                f"    {r['date']}: CZT1 start={r['mjd_start1']:.5f}, "
                f"CZT2 start={r['mjd_start2']:.5f}, diff={diff_s:.0f}s"
            )

    # Check hypothesis: different row counts
    section("Hypothesis Check: Different Row Counts?")
    diff_rows = [r for r in results if r["n_czt1"] != r["n_czt2"]]
    print(f"Days with different row counts: {len(diff_rows)}")
    if diff_rows:
        print(f"  Examples:")
        for r in diff_rows[:5]:
            print(f"    {r['date']}: CZT1={r['n_czt1']}, CZT2={r['n_czt2']}")

    # Monthly pattern
    section("Monthly Correlation Pattern")
    monthly = Counter()
    monthly_sum = Counter()
    for r in results:
        key = f"{r['date'].year}-{r['date'].month:02d}"
        monthly[key] += 1
        monthly_sum[key] += r["r"]

    for key in sorted(monthly):
        avg_r = monthly_sum[key] / monthly[key]
        n_low = sum(
            1
            for r in results
            if f"{r['date'].year}-{r['date'].month:02d}" == key and r["r"] < 0.5
        )
        print(f"  {key}: n={monthly[key]}, avg_r={avg_r:.3f}, low_days={n_low}")

    # Summary
    section("Summary")
    print(f"Key finding: CZT2 low-correlation days are characterized by:")
    zero_days = [r for r in results if r["nz2_pct"] < 5 and r["nz1_pct"] > 10]
    mismatch_days = [r for r in results if abs(r["n_czt1"] - r["n_czt2"]) > 100]
    print(
        f"  1. CZT2 near-zero on {len(zero_days)} days (CZT1 has data but CZT2 doesn't)"
    )
    print(f"  2. Different observation windows: {len(mismatch_times)} days")
    print(f"  3. Different row counts: {len(diff_rows)} days")


if __name__ == "__main__":
    investigate()
