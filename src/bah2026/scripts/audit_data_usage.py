#!/usr/bin/env python3
"""Comprehensive data usage audit — checks every file vs pipeline load status."""

from __future__ import annotations
import sys, os
from pathlib import Path
from collections import defaultdict

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
os.environ["BAH2026_DATA"] = str(REPO / "data" / "processed")


def scan_solexs(root: Path) -> dict:
    """Count SoLEXS files: .lc, .pi, .gti in SDD1/SDD2."""
    counts = defaultdict(int)
    sizes = defaultdict(float)
    for year in sorted(root.iterdir()):
        if not year.is_dir() or not year.name.isdigit():
            continue
        for month in sorted(year.iterdir()):
            if not month.is_dir() or not month.name.isdigit():
                continue
            for day in sorted(month.iterdir()):
                if not day.is_dir() or not day.name.isdigit():
                    continue
                for det in ["SDD1", "SDD2"]:
                    det_dir = day / det
                    if not det_dir.is_dir():
                        continue
                    for f in det_dir.iterdir():
                        if not f.is_file() or f.name.startswith("."):
                            continue
                        if f.name.endswith(".lc"):
                            cat = "SXR_LC"
                        elif f.name.endswith(".pi"):
                            cat = "SXR_PI"
                        elif f.name.endswith(".gti"):
                            cat = "GTI"
                        else:
                            cat = "SXR_OTHER"
                        counts[cat] += 1
                        sizes[cat] += f.stat().st_size / 1e6
    return {"counts": dict(counts), "sizes": dict(sizes)}


def scan_hel1os(root: Path) -> dict:
    """Count HEL1OS files: lightcurve_*.fits, spectra_*.fits, dispix.txt."""
    counts = defaultdict(int)
    sizes = defaultdict(float)
    for year in sorted(root.iterdir()):
        if not year.is_dir() or not year.name.isdigit():
            continue
        for month in sorted(year.iterdir()):
            if not month.is_dir() or not month.name.isdigit():
                continue
            for day in sorted(month.iterdir()):
                if not day.is_dir() or not day.name.isdigit():
                    continue
                for f in day.iterdir():
                    if not f.is_file() or f.name.startswith("."):
                        continue
                    name = f.name
                    if "lightcurve_czt1" in name:
                        cat = "LC_CZT1"
                    elif "lightcurve_czt2" in name:
                        cat = "LC_CZT2"
                    elif "lightcurve_cdte1" in name:
                        cat = "LC_CDTE1"
                    elif "lightcurve_cdte2" in name:
                        cat = "LC_CDTE2"
                    elif "spectra_czt1" in name:
                        cat = "SPEC_CZT1"
                    elif "spectra_czt2" in name:
                        cat = "SPEC_CZT2"
                    elif "spectra_cdte1" in name:
                        cat = "SPEC_CDTE1"
                    elif "spectra_cdte2" in name:
                        cat = "SPEC_CDTE2"
                    elif "dispix" in name:
                        cat = "DISPIX"
                    else:
                        cat = "HEL_OTHER"
                    counts[cat] += 1
                    sizes[cat] += f.stat().st_size / 1e6
    return {"counts": dict(counts), "sizes": dict(sizes)}


def check_pipeline_usage(main_py: Path) -> dict:
    """Check which data loaders main.py actually calls.

    Uses code patterns (loop structures, function calls) rather than
    fragile string matching against parameter values.
    """
    code = main_py.read_text()

    # SoLEXS readers
    has_sxr_lc = "load_solexs_lc" in code
    has_sxr_pi = "load_solexs_pi" in code
    has_gti = "load_solexs_gti" in code

    # HEL1OS LC detection: for det in ["czt", "cdte"] for num in [1, 2]
    # This pattern loads ALL 4 detectors
    has_det_loop = 'for det in ["czt", "cdte"]' in code and "for num in [1, 2]" in code
    # Feature extraction: for det, num in [("czt",1),("czt",2),("cdte",1),("cdte",2)]
    has_feat_loop = '"czt", 1), ("czt", 2), ("cdte", 1), ("cdte", 2)' in code

    # Spectra
    has_czt_spec = "load_hel1os_spectra" in code
    has_cdte_spec = has_czt_spec  # same function, different param

    # External data
    has_goes = "xrsb_flux" in code
    has_pi_features = "pi_temp" in code or "pi_em" in code

    return {
        "SXR_LC (SDD2 LC)": has_sxr_lc,
        "SXR_PI (340ch spectra)": has_sxr_pi,
        "GTI": has_gti,
        "LC_CZT1 (detection)": has_det_loop,
        "LC_CZT2 (detection)": has_det_loop,
        "LC_CDTE1 (detection)": has_det_loop,
        "LC_CDTE2 (detection)": has_det_loop,
        "CZT1+2+CdTe1+2 (features)": has_feat_loop,
        "HEL1OS CZT spectra": has_czt_spec,
        "HEL1OS CdTe spectra": has_cdte_spec,
        "GOES XRSB flux": has_goes,
        "PI → T, EM features": has_pi_features,
    }


def main():
    print("=" * 70)
    print("  DATA USAGE AUDIT: disk vs pipeline")
    print("=" * 70)

    sxr_root = Path(os.environ["BAH2026_DATA"]) / "solexs"
    hel_root = Path(os.environ["BAH2026_DATA"]) / "hel1os"

    print("\n── SoLEXS ──")
    sxr = scan_solexs(sxr_root)
    for cat in ["SXR_LC", "SXR_PI", "GTI", "SXR_OTHER"]:
        if sxr["counts"].get(cat, 0):
            print(
                f"  {cat:10s}: {sxr['counts'][cat]:5d} files, {sxr['sizes'][cat]:8.1f} MB"
            )

    print("\n── HEL1OS ──")
    hel = scan_hel1os(hel_root)
    for cat in [
        "LC_CZT1",
        "LC_CZT2",
        "LC_CDTE1",
        "LC_CDTE2",
        "SPEC_CZT1",
        "SPEC_CZT2",
        "SPEC_CDTE1",
        "SPEC_CDTE2",
        "DISPIX",
        "HEL_OTHER",
    ]:
        if hel["counts"].get(cat, 0):
            print(
                f"  {cat:10s}: {hel['counts'][cat]:5d} files, {hel['sizes'][cat]:8.1f} MB"
            )

    print("\n── Pipeline Status (from src/bah2026/main.py) ──")
    main_py = REPO / "src" / "bah2026" / "main.py"
    pipe = check_pipeline_usage(main_py)

    for name, used in pipe.items():
        print(f"  {'✅' if used else '❌'} {name}")

    # Summary
    used_count = sum(1 for v in pipe.values() if v)
    total = len(pipe)
    print(f"\n── Summary: {used_count}/{total} data sources used ──")

    unused = [n for n, u in pipe.items() if not u]
    if unused:
        print(f"\n❌ NOT USED:")
        for n in unused:
            print(f"   - {n}")


if __name__ == "__main__":
    main()
