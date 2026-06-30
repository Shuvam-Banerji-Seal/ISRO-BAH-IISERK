#!/usr/bin/env python3
"""Build 12-channel sequence data for deep learning models."""

from __future__ import annotations

import logging
import os
import sys
import time
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
            / f"seq_build_{time.strftime('%Y%m%d_%H%M%S')}.log"
        ),
    ],
)
log = logging.getLogger("bah2026")


def main():
    from bah2026.data import discover_combined_days
    from bah2026.data.sequence_builder import build_all_sequences
    from bah2026.config import HDF5_DIR, CATALOGS_DIR

    log.info("=" * 60)
    log.info("Sequence Builder — 12-channel × 3600s windows")
    log.info("=" * 60)

    days = discover_combined_days()
    log.info(f"Combined days: {len(days)}")

    csv = CATALOGS_DIR / "nowcast_catalogue.csv"
    df = pd.read_csv(csv) if csv.exists() else pd.DataFrame()

    event_times: dict[str, list[float]] = {}
    if "date" in df.columns and "peak_time" in df.columns:
        for _, row in df.iterrows():
            event_times.setdefault(str(row["date"]), []).append(row["peak_time"])

    output_dir = HDF5_DIR / "sequences"
    output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    build_all_sequences(days, event_times, str(output_dir))

    X_path = output_dir / "X_seq.npy"
    y_path = output_dir / "y_seq.npy"
    if X_path.exists():
        X = np.load(X_path, mmap_mode="r")
        y = np.load(y_path)
        log.info(
            f"X_seq: {X.shape}, y_seq: {y.shape}, pos: {y.sum()} ({100 * y.mean():.2f}%)"
        )
        log.info(f"Total time: {(time.time() - t0) / 60:.1f} min")

        from bah2026.data.sequence_builder import prepare_downsampled_sequences

        ds_path = str(output_dir / "X_seq_ds10.npy")
        log.info("Downsampling 1s → 10s for transformer...")
        prepare_downsampled_sequences(str(X_path), str(y_path), ds_path, factor=10)
        X_ds = np.load(ds_path, mmap_mode="r")
        log.info(f"X_seq_ds10: {X_ds.shape}")

    log.info("=" * 60)
    log.info("Sequence builder complete!")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
