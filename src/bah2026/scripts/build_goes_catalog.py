#!/usr/bin/env python3
"""
Build a GOES XRS flare event catalog from the cached netCDF files.

Reads GOES-16 XRSF L2 1-second flux (xrsb_flux = 0.1-0.8 nm channel),
applies the SWPC onset detection algorithm, and saves the events as CSV.

Files are in data/external/goes/sci_xrsf-l2-flx1s_g16_dYYYYMMDD_v2-2-1.nc
"""

from __future__ import annotations

import sys
import os
from pathlib import Path
from datetime import date, timedelta

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import pandas as pd
from scipy.ndimage import percentile_filter
from tqdm import tqdm

GOES_DIR = REPO_ROOT / "data" / "external" / "goes"
OUTPUT = REPO_ROOT / "output" / "catalogs" / "goes_flare_catalog.csv"

GOES_THRESHOLDS = {
    "A": (0.0, 1e-7),
    "B": (1e-7, 1e-6),
    "C": (1e-6, 1e-5),
    "M": (1e-5, 1e-4),
    "X": (1e-4, float("inf")),
}


def classify_goes(flux_wm2: float) -> str:
    for cls, (lo, hi) in GOES_THRESHOLDS.items():
        if lo <= flux_wm2 < hi:
            return cls
    return "X"


def detect_flares_goes(
    flux: np.ndarray,
    time_unix: np.ndarray,
    min_dur: int = 240,
    c_threshold: float = 1e-6,
) -> list[dict]:
    """Detect flares in GOES XRS flux using SWPC-style peak finding."""
    n = len(flux)
    if n < min_dur:
        return []

    valid = np.where(np.isfinite(flux), flux, 0.0)
    bg = percentile_filter(valid, 10, size=600, mode="nearest")
    noise_floor = np.maximum(bg * 1.3, np.full_like(bg, c_threshold * 0.1))
    above = valid > noise_floor

    regions = []
    i = 0
    while i < n:
        if above[i]:
            start = i
            while i < n and above[i]:
                i += 1
            end = i - 1
            if end - start >= 60:
                regions.append((start, end))
        else:
            i += 1

    events = []
    for start, end in regions:
        seg = valid[start : end + 1]
        pk_rel = int(np.argmax(seg))
        pk = start + pk_rel
        pflux = float(valid[pk])
        if pflux < c_threshold:
            continue

        # Begin
        begin = start
        for k in range(start, pk + 1):
            if valid[k] > bg[k] + c_threshold * 0.05:
                begin = k
                break

        # End (half-max decay)
        pref_bg = max(float(bg[begin]), float(np.min(valid[begin : pk + 1])))
        half = pref_bg + (pflux - pref_bg) * 0.5
        e_idx = pk
        while e_idx < min(end, n - 2) and float(valid[e_idx + 1]) >= half:
            e_idx += 1
        e_idx = max(e_idx, pk)

        dur = e_idx - begin + 1
        if dur < min_dur:
            continue

        events.append(
            {
                "start_time": float(time_unix[begin]),
                "peak_time": float(time_unix[pk]),
                "end_time": float(time_unix[min(e_idx, n - 1)]),
                "peak_flux": pflux,
                "goes_class": classify_goes(pflux),
                "duration_sec": dur,
                "background": float(pref_bg),
            }
        )

    # Merge overlapping
    if len(events) > 1:
        merged = [events[0]]
        for e in events[1:]:
            if e["start_time"] <= merged[-1]["end_time"] + 60:
                if e["peak_flux"] > merged[-1]["peak_flux"]:
                    merged[-1].update(
                        peak_time=e["peak_time"],
                        peak_flux=e["peak_flux"],
                        goes_class=e["goes_class"],
                    )
                merged[-1]["end_time"] = max(merged[-1]["end_time"], e["end_time"])
                merged[-1]["duration_sec"] = (
                    merged[-1]["end_time"] - merged[-1]["start_time"]
                )
            else:
                merged.append(e)
        events = merged

    return events


def main():
    print("=" * 60)
    print("  GOES XRS Flare Catalog Builder")
    print("=" * 60)

    nc_files = sorted(GOES_DIR.glob("sci_xrsf-l2-flx1s_g16_d*.nc"))
    print(f"Found {len(nc_files)} GOES-16 XRSF L2 netCDF files")

    all_events = []
    for f in tqdm(nc_files, desc="Processing GOES files"):
        try:
            from netCDF4 import Dataset

            with Dataset(str(f), "r") as nc:
                time_var = nc.variables["time"]
                xrsb = nc.variables["xrsb_flux"]

                t = time_var[:].astype(np.float64)
                flux = xrsb[:].astype(np.float64)

                # Fix fill values
                flux = np.where(flux < 0, np.nan, flux)

                if flux.size < 240 or np.all(~np.isfinite(flux)):
                    continue

                evts = detect_flares_goes(flux, t)
                if evts:
                    import re

                    m = re.search(r"_d(\d{8})_", f.name)
                    date_str = m.group(1) if m else f.stem[-16:-8]
                    for e in evts:
                        e["date"] = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
                    all_events.extend(evts)
        except Exception as e:
            print(f"  Error processing {f.name}: {e}")
            continue

    if not all_events:
        print("No events found!")
        return

    df = pd.DataFrame(all_events)

    # Sort by time
    df = df.sort_values("peak_time").reset_index(drop=True)

    # Date conversion
    from datetime import datetime, timezone

    df["peak_datetime"] = df["peak_time"].apply(
        lambda t: datetime.fromtimestamp(t, tz=timezone.utc).isoformat()
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT, index=False)

    print(f"\nSaved {len(df)} events to {OUTPUT}")

    # Stats
    print(f"\nClass distribution:")
    class_counts = df["goes_class"].value_counts()
    for cls in ["X", "M", "C", "B", "A"]:
        cnt = class_counts.get(cls, 0)
        if cnt > 0:
            print(f"  {cls}: {cnt}")

    print(f"\nDate range: {df['date'].min()} -> {df['date'].max()}")
    print(f"Events per day: {len(df) / df['date'].nunique():.1f}")

    # Cross-reference with known X6.3
    x63 = df[(df["date"] == "2024-02-22")]
    if len(x63) > 0:
        print(f"\nX6.3 flare on 2024-02-22:")
        for _, evt in x63.iterrows():
            print(
                f"  {evt['goes_class']}: flux={evt['peak_flux']:.3e}, dur={evt['duration_sec']:.0f}s"
            )


if __name__ == "__main__":
    main()
