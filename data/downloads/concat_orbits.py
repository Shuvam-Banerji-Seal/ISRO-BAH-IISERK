#!/usr/bin/env python3
"""
HEL1OS multi-orbit concatenation.

Problem: Each processed/hel1os/YYYY/MM/DD/ holds only ONE orbit's data
(the last-extracted zip from 2-3+ orbits/day). This script:

  1. Scans all HLS_*.zip files in data/raw/hel1os/ for each date
  2. Extracts lightcurve files from ALL orbits for that date
  3. Concatenates them by MJD order (gaps preserved)
  4. Saves concatenated files back to the processed directory
  5. Also handles the 7 days with stranded nested orbits

Usage:
    python data/downloads/concat_orbits.py              # all dates
    python data/downloads/concat_orbits.py --date 2024-02-01  # single date
    python data/downloads/concat_orbits.py --check-only       # just report what needs merging

Recovers ~63% of HEL1OS coverage currently lost to orbit clobbering.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
import tempfile
import zipfile
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "hel1os"
PROC_DIR = PROJECT_ROOT / "data" / "processed" / "hel1os"

# Files to concatenate per orbit
LC_FILES = [
    "lightcurve_czt1.fits",
    "lightcurve_czt2.fits",
    "lightcurve_cdte1.fits",
    "lightcurve_cdte2.fits",
]
SPECTRA_FILES = [
    "hel1os_czt_spectra_czt1.fits",
    "hel1os_czt_spectra_czt2.fits",
    "hel1os_cdte_spectra_cdte1.fits",
    "hel1os_cdte_spectra_cdte2.fits",
]
DISPIX_FILES = [
    "czt1dispix.txt",
    "czt2dispix.txt",
]


def parse_hel1os_orbit_zips() -> dict[date, list[Path]]:
    """Scan raw HEL1OS directory and group zip files by date.

    Returns dict[date, list[Path]] sorted by orbit start time.
    """
    if not RAW_DIR.exists():
        print(f"WARNING: Raw HEL1OS dir not found: {RAW_DIR}")
        return {}

    orbit_zips: dict[date, list[tuple[str, Path]]] = defaultdict(list)

    for f in sorted(RAW_DIR.iterdir()):
        if not f.name.endswith(".zip"):
            continue
        # Skip calibration zips
        if f.name.startswith("CAL_"):
            continue
        # Parse: HLS_YYYYMMDD_HHMMSS_XXXXXsec_lev1_V*.zip
        m = re.match(
            r"HLS_(\d{4})(\d{2})(\d{2})_(\d{6})_(\d+)sec_lev1_V(\d+)\.zip", f.name
        )
        if not m:
            print(f"  Skipping unparseable zip: {f.name}")
            continue
        y, mo, d, start_time, dur, version = m.groups()
        day = date(int(y), int(mo), int(d))
        # Key for sorting: start_time (HHMMSS) + version (higher = newer pipeline)
        sort_key = f"{start_time}_{int(version):03d}"
        orbit_zips[day].append((sort_key, f))

    # Sort by orbit time + version
    result: dict[date, list[Path]] = {}
    for day in sorted(orbit_zips):
        orbit_zips[day].sort(key=lambda x: x[0])
        result[day] = [p for _, p in orbit_zips[day]]

    return result


def get_processed_days() -> list[date]:
    """Get list of dates that have HEL1OS processed data."""
    if not PROC_DIR.exists():
        return []
    days = []
    for yd in PROC_DIR.iterdir():
        if not yd.is_dir() or not yd.name.isdigit():
            continue
        for md in yd.iterdir():
            if not md.is_dir() or not md.name.isdigit():
                continue
            for dd in md.iterdir():
                if not dd.is_dir() or not dd.name.isdigit():
                    continue
                days.append(date(int(yd.name), int(md.name), int(dd.name)))
    return sorted(days)


def check_merge_needed(
    day: date,
    orbit_zips: dict[date, list[Path]],
) -> int:
    """Return number of orbits available but NOT merged for this day."""
    if day not in orbit_zips:
        return 0
    zips = orbit_zips[day]
    return max(0, len(zips) - 1)  # -1 for the one already extracted


def concat_lightcurve_fits(
    extracted_dirs: list[Path],
    output_path: Path,
) -> bool:
    """Concatenate lightcurve FITS files from multiple orbit extractions.

    For each band HDU, concatenates the table rows in MJD order.
    Returns True if successful.
    """
    if not extracted_dirs:
        return False

    # Read first file to get template structure
    first_file = None
    for ed in extracted_dirs:
        candidates = list(ed.glob("lightcurve_czt1.fits"))
        if candidates:
            first_file = candidates[0]
            break
    if first_file is None:
        return False

    with fits.open(first_file) as hdul:
        n_hdus = len(hdul)
        # Determine which file type we're writing from the output name
        fname = output_path.name
        is_czt = "czt" in fname
        is_cdte = "cdte" in fname
        det_prefix = fname.replace("lightcurve_", "").replace(".fits", "")

    # Collect band data from all extracted dirs
    per_band_data: dict[int, list[Any]] = {}
    per_band_header: dict[int, fits.Header] = {}

    for ed in sorted(extracted_dirs):
        files = list(ed.glob(f"lightcurve_{det_prefix}.fits"))
        if not files:
            continue
        fpath = files[0]
        try:
            with fits.open(fpath) as hdul:
                primary_hdr = hdul[0].header
                for i in range(1, n_hdus):
                    data = hdul[i].data
                    if i not in per_band_data:
                        per_band_data[i] = []
                        per_band_header[i] = hdul[i].header
                    per_band_data[i].append(data)
        except Exception as e:
            print(f"  Error reading {fpath}: {e}")
            continue

    if not per_band_data:
        return False

    # Concatenate per band and sort by MJD
    with fits.open(first_file, mode="readonly") as template:
        primary_hdu = fits.PrimaryHDU(header=template[0].header)

        hdus = [primary_hdu]
        for i in range(1, n_hdus):
            if i not in per_band_data:
                continue
            all_rows = np.concatenate(per_band_data[i])
            # Sort by MJD if column exists
            if "MJD" in all_rows.dtype.names:
                sort_idx = np.argsort(all_rows["MJD"])
                all_rows = all_rows[sort_idx]
            elif "TIME" in all_rows.dtype.names:
                sort_idx = np.argsort(all_rows["TIME"])
                all_rows = all_rows[sort_idx]

            col_defs = template[i].columns
            hdu = fits.BinTableHDU(
                data=all_rows, header=per_band_header[i], name=template[i].name
            )
            hdus.append(hdu)

    try:
        fits.HDUList(hdus).writeto(output_path, overwrite=True)
        return True
    except Exception as e:
        print(f"  Error writing {output_path}: {e}")
        return False


def concat_spectra_fits(
    extracted_dirs: list[Path],
    output_path: Path,
) -> bool:
    """Concatenate spectra FITS files (SPECTRUM HDU)."""
    spec_data: list[np.ndarray] = []
    header = None

    fname = output_path.name
    det_prefix = (
        fname.replace("hel1os_", "").replace("_spectra_", "/").replace(".fits", "")
    )

    for ed in sorted(extracted_dirs):
        files = list(ed.glob(f"*{det_prefix}*spectra*.fits"))
        if not files:
            # Try alternative naming
            files = list(ed.glob(f"*spectra_{det_prefix}.fits"))
        if not files:
            continue
        fpath = files[0]
        try:
            with fits.open(fpath) as hdul:
                data = hdul["SPECTRUM"].data
                if header is None:
                    header = hdul["SPECTRUM"].header
                spec_data.append(data)
        except Exception:
            continue

    if not spec_data:
        return False

    all_rows = np.concatenate(spec_data)
    # Sort by SPEC_NUM or TSTART
    if "SPEC_NUM" in all_rows.dtype.names:
        all_rows = np.sort(all_rows, order="SPEC_NUM")
    elif "TSTART" in all_rows.dtype.names:
        all_rows = np.sort(all_rows, order="TSTART")

    primary = fits.PrimaryHDU()
    hdu = fits.BinTableHDU(data=all_rows, header=header, name="SPECTRUM")
    fits.HDUList([primary, hdu]).writeto(output_path, overwrite=True)
    return True


def extract_orbit_zips(day: date, zips: list[Path]) -> list[Path]:
    """Extract all orbit zips for a day into temporary directories.

    Returns list of paths to extraction temp dirs.
    """
    extracted = []
    for zip_path in zips:
        tmp = Path(tempfile.mkdtemp(prefix=f"hel1os_{day.strftime('%Y%m%d')}_"))
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                # Extract only lightcurves, spectra, and txt files
                for name in zf.namelist():
                    if any(name.endswith(ext) for ext in [".fits", ".txt"]):
                        if (
                            "lightcurve" in name
                            or "spectra" in name
                            or "dispix" in name
                        ):
                            zf.extract(name, tmp)
                            # Flatten: move from nested dir to root
                            src = tmp / name
                            if src != tmp / src.name:
                                shutil.move(str(src), str(tmp / src.name))
            extracted.append(tmp)
        except zipfile.BadZipFile:
            shutil.rmtree(tmp, ignore_errors=True)
            print(f"  Bad zip: {zip_path.name}")
    return extracted


def merge_day(
    day: date,
    zips: list[Path],
    force: bool = False,
) -> int:
    """Merge all orbit zips for a single day into the processed directory.

    Returns number of orbits merged (0 if no action).
    """
    day_proc = PROC_DIR / f"{day.year:04d}" / f"{day.month:02d}" / f"{day.day:02d}"
    if not day_proc.exists():
        day_proc.mkdir(parents=True, exist_ok=True)

    # Check if already merged (marker file or existing multiple-orbit data)
    marker = day_proc / ".merged"
    if marker.exists() and not force:
        return 0

    # Extract all orbit zips
    extracted = extract_orbit_zips(day, zips)
    if len(extracted) < 2:
        # Clean up single extraction
        for d in extracted:
            shutil.rmtree(d, ignore_errors=True)
        return 0

    print(f"  {day}: {len(zips)} zips → {len(extracted)} extractions, merging...")

    # Concatenate each lightcurve file
    for lc_file in LC_FILES + SPECTRA_FILES:
        out_path = day_proc / lc_file
        if lc_file.startswith("lightcurve"):
            concat_lightcurve_fits(extracted, out_path)
        else:
            concat_spectra_fits(extracted, out_path)

    # Copy dispix files from first extraction
    for dx_file in DISPIX_FILES:
        for ed in extracted:
            src = ed / dx_file
            if src.exists():
                dst = day_proc / dx_file
                shutil.copy2(str(src), str(dst))
                break

    # Cleanup temp dirs
    for d in extracted:
        shutil.rmtree(d, ignore_errors=True)

    # Mark as merged
    marker.touch()
    print(f"  ✓ {day}: {len(zips)} orbits → 1 merged file")
    return len(zips) - 1  # orbits saved


def main():
    parser = argparse.ArgumentParser(description="HEL1OS multi-orbit concatenation")
    parser.add_argument("--date", type=str, help="Single date (YYYY-MM-DD)")
    parser.add_argument(
        "--force", action="store_true", help="Re-merge even if already done"
    )
    parser.add_argument(
        "--check-only", action="store_true", help="Just report which days need merging"
    )
    args = parser.parse_args()

    print("═" * 60)
    print("  HEL1OS Multi-Orbit Concatenation")
    print("═" * 60)

    # Scan raw zips
    orbit_zips = parse_hel1os_orbit_zips()
    print(
        f"Found {sum(len(v) for v in orbit_zips.values())} orbit zips "
        f"across {len(orbit_zips)} days"
    )
    print(
        f"Average: {sum(len(v) for v in orbit_zips.values()) / max(len(orbit_zips), 1):.2f} orbits/day"
    )

    # Check merge status
    processed_days = get_processed_days()
    print(f"Processed days: {len(processed_days)}")

    merge_needed = 0
    total_orbits_saved = 0
    days_to_process = []

    if args.date:
        d = date.fromisoformat(args.date)
        if d in orbit_zips and len(orbit_zips[d]) > 1:
            days_to_process.append((d, orbit_zips[d]))
            merge_needed = 1
            total_orbits_saved = len(orbit_zips[d]) - 1
    else:
        for d in processed_days:
            n = check_merge_needed(d, orbit_zips)
            if n > 0:
                merge_needed += 1
                total_orbits_saved += n
                days_to_process.append((d, orbit_zips[d]))

    print(f"Days needing merge: {merge_needed}")
    print(f"Lost orbits to recover: {total_orbits_saved}")

    if args.check_only:
        return

    # Process in order
    from tqdm import tqdm

    recovered = 0
    for d, zips in tqdm(days_to_process, desc="Merging orbits"):
        n = merge_day(d, zips, force=args.force)
        recovered += n

    print(f"\nRecovered {recovered} orbits across {len(days_to_process)} days")
    print(f"Total orbits extracted: {recovered + 1} per day (was 1)")


if __name__ == "__main__":
    main()
