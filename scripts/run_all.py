#!/usr/bin/env python3
"""Run generate_master_csv.py for ALL days sequentially.

Calls the proven generate_master_csv.py for each day to ensure
consistent results (277 columns, full CPU features, proper flare detection).
"""

import sys, os, time, warnings, subprocess

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ["BAH2026_DATA"] = os.path.abspath("data/processed")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
warnings.filterwarnings("ignore")
from bah2026.data import discover_combined_days
from bah2026.config import CATALOGS_DIR, HDF5_DIR, MODELS_DIR, ensure_output_dirs
from tqdm import tqdm
import numpy as np, pandas as pd


def main():
    ensure_output_dirs()
    t_start = time.time()
    days = discover_combined_days()
    print(f"=== GENERATE MASTER CSV: {len(days)} days ===", flush=True)
    print(f"{days[0]} to {days[-1]}", flush=True)

    n_fail = 0
    for i, d in enumerate(tqdm(days, desc="Master CSV")):
        try:
            r = subprocess.run(
                [sys.executable, "src/bah2026/scripts/generate_master_csv.py", str(d)],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if r.returncode != 0:
                n_fail += 1
                if n_fail <= 5:
                    print(f"  FAIL {d}: {r.stderr[:200]}", flush=True)
        except subprocess.TimeoutExpired:
            n_fail += 1
            print(f"  TIMEOUT {d}", flush=True)
        except Exception as e:
            n_fail += 1
            if n_fail <= 5:
                print(f"  FAIL {d}: {e}", flush=True)

    print(
        f"Done: {len(days)} days, {n_fail} fails, {time.time() - t_start:.0f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
