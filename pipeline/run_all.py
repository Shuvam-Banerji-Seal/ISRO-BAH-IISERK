#!/usr/bin/env python3
"""
Stage 2 orchestrator — runs all phases sequentially.

Usage:
    WORKSPACE=/home/alok/codes/Isro python3 src/stage2/run_all.py
"""
import sys
import os
import time
from pathlib import Path

# Pipeline root directory
PIPELINE_DIR = Path(__file__).parent  # pipeline/
# Add src/ to path
SRC_DIR = PIPELINE_DIR / "src"
sys.path.insert(0, str(SRC_DIR))
os.chdir(str(PIPELINE_DIR))  # resolve all relative paths from pipeline/

PHASES = {
    1: ("phase1_direct", "extract"),
    2: ("phase2_perflare", "extract"),
    3: ("phase3_tem_goes", "extract"),
    4: ("phase4_tem_solexs", "extract"),
    5: ("phase5_wavelet", "compute_wavelet_features"),
    6: ("phase6_spectral", "extract"),
    7: ("phase7_nonlinear", "extract"),
    8: ("phase8_event", "extract"),
    9: ("phase9_assemble", "assemble"),
}


def run_phase(phase_num):
    if phase_num not in PHASES:
        print(f"Unknown phase {phase_num}. Options: {list(PHASES.keys())}")
        return False

    mod_name, func_name = PHASES[phase_num]

    print(f"\n{'='*60}")
    print(f"STAGE 2 — Phase {phase_num}: {mod_name}")
    print(f"{'='*60}")

    # Import from the stage2 package
    import importlib
    mod = importlib.import_module(f"stage2.{mod_name}")
    func = getattr(mod, func_name)

    t0 = time.time()
    try:
        result = func()
        elapsed = time.time() - t0
        print(f"  \u2713 Phase {phase_num} completed in {elapsed:.1f}s")
        return True
    except Exception as e:
        elapsed = time.time() - t0
        print(f"  \u2717 Phase {phase_num} FAILED after {elapsed:.1f}s: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("-")]

    if args:
        try:
            phase = int(args[0])
            run_phase(phase)
        except ValueError:
            print(f"Usage: {sys.argv[0]} [phase_number]")
            print(f"  No args = run all phases (1-9)")
    else:
        print("STAGE 2 — Feature Engineering Pipeline")
        t_start = time.time()

        for phase_num in sorted(PHASES.keys()):
            ok = run_phase(phase_num)
            if not ok:
                print(f"\n  Stopping due to Phase {phase_num} failure.")
                break

        t_total = time.time() - t_start
        print(f"\nPipeline finished in {t_total:.1f}s")
        out_path = Path("dist/stage2_feature_matrix_20260623.npz")
        if out_path.exists():
            import numpy as np
            ds = np.load(out_path, allow_pickle=True)
            meta = ds["__metadata__"].item() if "__metadata__" in ds else {}
            n_feat = len([k for k in ds.files if k != "__metadata__"])
            size_mb = out_path.stat().st_size / 1e6
            print(f"Output: {out_path.name} ({size_mb:.1f} MB, {n_feat} features)")


if __name__ == "__main__":
    main()
