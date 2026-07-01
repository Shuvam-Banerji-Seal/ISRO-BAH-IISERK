#!/usr/bin/env python3
"""
Run the BAH 2026 pipeline: nowcast → features.
All data sources used, GPU accelerated where possible.

Usage:
    python -m bah2026.scripts.run_pipeline              # full run
    python -m bah2026.scripts.run_pipeline --nowcast    # step 1 only
    python -m bah2026.scripts.run_pipeline --features   # step 2 only
    python -m bah2026.scripts.run_pipeline --gpu-bench  # test GPU performance
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# ── Path setup ──────────────────────────────────────────────────────
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

import os

os.environ["BAH2026_DATA"] = str(REPO / "data" / "processed")

# ── Logging ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(
            REPO / "logs" / f"pipeline_{time.strftime('%Y%m%d_%H%M%S')}.log"
        ),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("bah2026")


def section(msg: str) -> None:
    log.info("=" * 60)
    log.info(msg)
    log.info("=" * 60)


def gpu_benchmark() -> dict:
    """Benchmark GPU throughput for model training."""
    section("GPU Benchmark")
    import torch
    import numpy as np

    if not torch.cuda.is_available():
        log.warning("No GPU available, running CPU benchmark instead")
        device = "cpu"
    else:
        device = f"cuda:0 ({torch.cuda.get_device_name(0)})"
        log.info(f"GPU: {torch.cuda.get_device_name(0)}")
        log.info(
            f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        )

    # Benchmark matrix multiply
    sizes = [1000, 5000, 10000]
    results = {}
    for n in sizes:
        x = torch.randn(n, n, device="cuda" if torch.cuda.is_available() else "cpu")
        y = torch.randn(n, n, device="cuda" if torch.cuda.is_available() else "cpu")
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t0 = time.time()
        for _ in range(10):
            z = x @ y
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        elapsed = (time.time() - t0) / 10
        results[n] = elapsed
        log.info(
            f"  {n}x{n} matmul: {elapsed * 1000:.1f} ms ({2 * n**3 / elapsed / 1e12:.1f} TFLOPS)"
        )

    return results


def step_nowcast() -> int:
    """Run flare detection with all 4 HEL1OS detectors + SoLEXS."""
    section("[1/3] Nowcast — Combined SXR+HXR Detection")

    from bah2026.data import discover_combined_days
    from bah2026.main import cmd_nowcast

    log.info("Discovering data...")
    days = discover_combined_days()
    log.info(f"Combined days: {len(days)}")

    log.info("Running nowcast (all 4 HEL1OS detectors + SWPC onset)...")
    t0 = time.time()
    events_df = cmd_nowcast(days)
    elapsed = time.time() - t0

    log.info(f"Events detected: {len(events_df)}")
    log.info(f"Time: {elapsed:.0f}s ({elapsed / len(days):.1f}s/day)")

    if len(events_df) > 0:
        from collections import Counter

        classes = Counter(events_df["goes_class"])
        for c in ["X", "M", "C", "B", "A"]:
            if classes.get(c, 0):
                log.info(f"  {c}: {classes[c]}")

    return len(events_df)


def step_features() -> tuple:
    """Extract features from all data sources."""
    section("[2/3] Features — All 12 Data Sources")

    from bah2026.data import discover_combined_days
    from bah2026.main import cmd_features
    import pandas as pd

    days = discover_combined_days()
    csv_path = REPO / "output" / "catalogs" / "nowcast_catalogue.csv"
    events_df = pd.read_csv(csv_path) if csv_path.exists() else pd.DataFrame()

    log.info(f"Days: {len(days)}, Events loaded: {len(events_df)}")
    log.info("Extracting features (PI spectra, 4xHEL1OS, GOES, T/EM, gamma)...")

    t0 = time.time()
    X, y, fnames = cmd_features(days, events_df)
    elapsed = time.time() - t0

    log.info(f"Feature matrix: {X.shape[0]} samples x {X.shape[1]} features")
    log.info(f"Positive rate: {y.sum()}/{len(y)} = {y.mean() * 100:.1f}%")
    log.info(f"Time: {elapsed:.0f}s ({elapsed / len(days):.1f}s/day)")

    return X, y, fnames




def main():
    parser = argparse.ArgumentParser(
        description="BAH 2026 — Full Pipeline (GPU + all data)"
    )
    parser.add_argument("--nowcast", action="store_true", help="Run nowcast only")
    parser.add_argument(
        "--features", action="store_true", help="Run feature extraction only"
    )
    parser.add_argument("--gpu-bench", action="store_true", help="GPU benchmark only")
    parser.add_argument(
        "--all", action="store_true", default=True, help="Run all steps (default)"
    )
    args = parser.parse_args()

    # If no specific step, run all
    run_all = not (args.nowcast or args.features or args.gpu_bench)

    log.info(f"BAH 2026 Pipeline — Full data (12/12 sources) + GPU")
    from bah2026.config import N_WORKERS

    log.info(f"GPU: deferred (activated during forecast)")
    log.info(f"CPU workers: {N_WORKERS}")

    # Ensure output dirs
    from bah2026.config import ensure_output_dirs

    ensure_output_dirs()

    if args.gpu_bench:
        gpu_benchmark()

    if args.nowcast or run_all:
        step_nowcast()

    if args.features or run_all:
        X, y, fnames = step_features()

    log.info("Pipeline complete. Use run_master_pipeline for DL training.")


if __name__ == "__main__":
    main()
