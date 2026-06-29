#!/usr/bin/env python3
"""Extract HK, GTI, and dispix files from HEL1OS raw zips.

These files were missed by decompress.sh because the unzip patterns
(*.hk.fits, *gti*.fits) didn't match the actual filenames (hk.fits, gticzt1.fits).

Usage:
    python -m bah2026.scripts.extract_aux_files
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from datetime import date
from pathlib import Path
import sys

DATA_ROOT = Path("/store/shuvam/ISRO-BAH-IISERK/data")
RAW_DIR = DATA_ROOT / "raw" / "hel1os"
OUT_DIR = DATA_ROOT / "processed" / "hel1os"


def extract_aux_from_zip(zip_path: Path, out_dir: Path) -> dict:
    """Extract aux files (HK, GTI, dispix) from a single HEL1OS zip."""
    result = {"hk": False, "gti": 0, "dispix": 0, "evt": False}

    with tempfile.TemporaryDirectory() as tmpdir:
        # Extract only aux/ and events/ directories
        cmd = [
            "unzip",
            "-q",
            "-o",
            str(zip_path),
            "*/aux/*",
            "*/events/evt.fits",
            "-d",
            tmpdir,
        ]
        try:
            subprocess.run(cmd, capture_output=True, timeout=60)
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return result

        # Find extracted files
        for f in Path(tmpdir).rglob("*"):
            if not f.is_file():
                continue

            fname = f.name.lower()

            if fname == "hk.fits":
                dest = out_dir / "hk.fits"
                if not dest.exists():
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(f), str(dest))
                    result["hk"] = True

            elif fname.startswith("gti") and fname.endswith(".fits"):
                dest = out_dir / f.name
                if not dest.exists():
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(f), str(dest))
                    result["gti"] += 1

            elif "dispix" in fname:
                dest = out_dir / f.name
                if not dest.exists():
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(f), str(dest))
                    result["dispix"] += 1

            elif fname == "evt.fits":
                dest = out_dir / "evt.fits"
                if not dest.exists():
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(f), str(dest))
                    result["evt"] = True

    return result


def main():
    print("=" * 60)
    print("  Extracting HK, GTI, dispix, evt from HEL1OS raw zips")
    print("=" * 60)

    zip_files = sorted(RAW_DIR.glob("HLS_*.zip"))
    print(f"\n  Found {len(zip_files)} raw zip files")

    # Check how many already have hk.fits
    existing_hk = 0
    existing_gti = 0
    for out_dir in OUT_DIR.rglob("20*"):
        if out_dir.is_dir():
            if (out_dir / "hk.fits").exists():
                existing_hk += 1
            gti_files = list(out_dir.glob("gti*.fits"))
            existing_gti += len(gti_files)

    print(f"  Already extracted: {existing_hk} HK files, {existing_gti} GTI files")
    print(f"  Need to extract from {len(zip_files)} zips\n")

    total = {"hk": 0, "gti": 0, "dispix": 0, "evt": 0, "errors": 0}
    processed = 0

    for i, zip_path in enumerate(zip_files):
        # Extract date from zip filename
        fname = zip_path.stem
        parts = fname.split("_")
        if len(parts) < 2:
            continue
        date_str = parts[1]  # YYYYMMDD
        if len(date_str) != 8:
            continue

        try:
            d = date(int(date_str[:4]), int(date_str[4:6]), int(date_str[6:8]))
        except ValueError:
            continue

        out_dir = OUT_DIR / f"{d.year:04d}" / f"{d.month:02d}" / f"{d.day:02d}"

        # Skip if already has hk.fits
        if (out_dir / "hk.fits").exists():
            continue

        result = extract_aux_from_zip(zip_path, out_dir)

        if result["hk"]:
            total["hk"] += 1
        total["gti"] += result["gti"]
        total["dispix"] += result["dispix"]
        if result["evt"]:
            total["evt"] += 1

        processed += 1
        if processed % 100 == 0:
            print(f"  Progress: {processed}/{len(zip_files)} zips processed")

    print(f"\n{'=' * 60}")
    print(f"  EXTRACTION COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Zips processed: {processed}")
    print(f"  HK files extracted: {total['hk']}")
    print(f"  GTI files extracted: {total['gti']}")
    print(f"  Dispix files extracted: {total['dispix']}")
    print(f"  Event files extracted: {total['evt']}")

    # Verify
    print(f"\n  --- Verification ---")
    hk_count = len(list(OUT_DIR.rglob("hk.fits")))
    gti_count = len(list(OUT_DIR.rglob("gti*.fits")))
    evt_count = len(list(OUT_DIR.rglob("evt.fits")))
    dispix_count = len(list(OUT_DIR.rglob("*dispix*")))
    print(f"  Total HK files on disk: {hk_count}")
    print(f"  Total GTI files on disk: {gti_count}")
    print(f"  Total evt files on disk: {evt_count}")
    print(f"  Total dispix files on disk: {dispix_count}")


if __name__ == "__main__":
    main()
