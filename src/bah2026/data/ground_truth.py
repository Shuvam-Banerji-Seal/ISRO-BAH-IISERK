"""Ground-truth flare catalog loader for validation.

Sources (tried in order):
  1. NOAA SWPC GOES X-ray Event Reports (preferred) — from data/external/goes/
  2. HEK (Heliophysics Event Knowledgebase) — SWPC FRM events
  3. Fallback: SWPC JSON endpoints for recent flares

The ground truth is used for:
  - Nowcast validation (precision/recall against real flares)
  - Forecast label generation (positive = SWPC flare within forecast window)
  - GOES calibration cross-check
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

EXTERNAL_DIR = Path(__file__).resolve().parents[3] / "data" / "external" / "goes"
CATALOGS_DIR = Path(__file__).resolve().parents[3] / "output" / "catalogs"

# GOES classification thresholds (W/m², peak 0.1-0.8 nm flux)
GOES_THRESHOLDS = {
    "A": (0.0, 1e-7),
    "B": (1e-7, 1e-6),
    "C": (1e-6, 1e-5),
    "M": (1e-5, 1e-4),
    "X": (1e-4, float("inf")),
}


def parse_goes_class(class_str: str) -> float:
    """Parse 'M6.3' or 'X1.0' → numerical peak flux in W/m².

    Returns the midpoint of the GOES class bin.
    """
    class_str = class_str.strip().upper()
    if not class_str:
        return 0.0

    letter = class_str[0]
    try:
        value = float(class_str[1:])
    except (ValueError, IndexError):
        return 0.0

    if letter == "X":
        return 1e-4 + value * 1e-4
    elif letter == "M":
        return 1e-5 + value * 1e-5
    elif letter == "C":
        return 1e-6 + value * 1e-6
    elif letter == "B":
        return 1e-7 + value * 1e-7
    elif letter == "A":
        return 0.0
    return 0.0


def load_swpc_flares() -> pd.DataFrame:
    """Load SWPC/GOES flare events from cached files.

    Sources (tried in order):
      1. GOES XRSF-derived catalog from build_goes_catalog.py (most reliable)
      2. HEK SWPC CSV
      3. SWPC JSON endpoints

    Returns
    -------
    df : pd.DataFrame
        Columns: date, begin_time, peak_time, end_time, goes_class,
                 peak_flux, duration_sec
    """
    all_events = []

    # Try GOES XRSF-derived catalog first (most complete and reliable)
    goes_catalog = CATALOGS_DIR / "goes_flare_catalog.csv"
    if goes_catalog.exists():
        try:
            df = pd.read_csv(goes_catalog)
            required = {"date", "peak_time", "goes_class", "peak_flux"}
            if required.issubset(set(df.columns)):
                df["begin_time"] = pd.to_datetime(df["start_time"], unit="s", utc=True)
                df["peak_time"] = pd.to_datetime(df["peak_time"], unit="s", utc=True)
                df["end_time"] = pd.to_datetime(df["end_time"], unit="s", utc=True)
                print(f"Loaded {len(df)} events from GOES XRSF catalog")
                return df
        except Exception as e:
            print(f"GOES catalog load failed: {e}")

    # Try HEK SWPC CSV
    hek_file = EXTERNAL_DIR / "hek_swpc_flares.csv"
    if hek_file.exists():
        try:
            df = pd.read_csv(hek_file)
            # Map HEK columns to standard format
            required = {
                "event_starttime",
                "event_peaktime",
                "event_endtime",
                "fl_goescls",
            }
            if required.issubset(set(df.columns)):
                events = pd.DataFrame(
                    {
                        "date": df["event_starttime"].str[:10],
                        "begin_time": pd.to_datetime(df["event_starttime"]),
                        "peak_time": pd.to_datetime(df["event_peaktime"]),
                        "end_time": pd.to_datetime(df["event_endtime"]),
                        "goes_class": df["fl_goescls"],
                        "active_region": df.get("ar_noaa", np.nan),
                    }
                )
                events["peak_flux"] = events["goes_class"].apply(parse_goes_class)
                all_events.append(events)
                print(f"Loaded {len(events)} flares from HEK SWPC")
        except Exception as e:
            print(f"HEK load failed: {e}")

    # Try JSON files from SWPC
    for json_file in EXTERNAL_DIR.glob("*.json"):
        try:
            import json

            data = json.loads(json_file.read_text())
            if isinstance(data, list) and len(data) > 0 and "begin" in data[0]:
                rows = []
                for evt in data:
                    rows.append(
                        {
                            "date": evt.get("begin", "")[:10],
                            "begin_time": pd.to_datetime(evt["begin"]),
                            "peak_time": pd.to_datetime(evt["peak"])
                            if "peak" in evt
                            else None,
                            "end_time": pd.to_datetime(evt["end"])
                            if "end" in evt
                            else None,
                            "goes_class": f"{evt.get('class_type', '')}{evt.get('class_value', 0)}",
                            "peak_flux": float(evt.get("peak", 0)) * 1e-4,
                            "active_region": evt.get("active_region_num", np.nan),
                        }
                    )
                df_ev = pd.DataFrame(rows)
                all_events.append(df_ev)
                print(f"Loaded {len(df_ev)} events from {json_file.name}")
        except Exception:
            pass

    if not all_events:
        print(
            "WARNING: No SWPC flare data available. "
            "Run data acquisition or place GOES catalog in data/external/goes/"
        )
        return pd.DataFrame(
            columns=[
                "date",
                "begin_time",
                "peak_time",
                "end_time",
                "goes_class",
                "peak_flux",
                "active_region",
            ]
        )

    # Combine and deduplicate
    combined = pd.concat(all_events, ignore_index=True)
    if "peak_time" in combined.columns and combined["peak_time"].notna().any():
        combined = combined.sort_values("peak_time").drop_duplicates(
            subset=["peak_time", "goes_class"], keep="first"
        )
    return combined


def get_ground_truth_events(
    start_date: date | None = None,
    end_date: date | None = None,
) -> pd.DataFrame:
    """Get all ground-truth flare events within a date range."""
    df = load_swpc_flares()
    if df.empty:
        return df
    if start_date:
        df = df[df["date"] >= str(start_date)]
    if end_date:
        df = df[df["date"] <= str(end_date)]
    return df


def validate_nowcasting(
    detected_events: pd.DataFrame,
    truth_events: pd.DataFrame,
    tolerance_sec: int = 300,
) -> dict[str, Any]:
    """Validate a nowcast catalog against ground truth.

    Parameters
    ----------
    detected_events : pd.DataFrame
        Nowcast catalogue with 'peak_time' (float, Unix seconds or MJD) and
        'goes_class' columns.
    truth_events : pd.DataFrame
        Ground truth with 'peak_time' (datetime) and 'goes_class' columns.
    tolerance_sec : int
        Maximum temporal offset (seconds) to consider a match.

    Returns
    -------
    dict with keys: tp, fp, fn, precision, recall, f1, null_distances, ...
    """
    if detected_events.empty or truth_events.empty:
        return {
            "tp": 0,
            "fp": len(detected_events),
            "fn": len(truth_events),
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        }

    tp, fp, fn = 0, 0, 0
    matched_truth = set()

    for _, det in detected_events.iterrows():
        det_t = det.get("peak_time", 0)
        found = False
        for idx, tr in truth_events.iterrows():
            if idx in matched_truth:
                continue
            tr_t = (
                tr["peak_time"].timestamp()
                if hasattr(tr["peak_time"], "timestamp")
                else float(tr["peak_time"])
            )
            if abs(det_t - tr_t) <= tolerance_sec:
                tp += 1
                matched_truth.add(idx)
                found = True
                break
        if not found:
            fp += 1

    fn = len(truth_events) - len(matched_truth)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-10)

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "total_detected": len(detected_events),
        "total_truth": len(truth_events),
    }


def is_flare_in_window(
    truth_events: pd.DataFrame,
    t_mjd: float,
    window_sec: float = 1800.0,
) -> bool:
    """Return True if any flare peak falls in [t_mjd, t_mjd + window_sec] (in MJD-based seconds)."""
    if truth_events.empty:
        return False
    # Convert truth peak_time to MJD-based seconds for comparison
    # This requires knowing the time system used in truth_events
    for _, tr in truth_events.iterrows():
        tr_t = tr["peak_time"]
        if hasattr(tr_t, "timestamp"):
            # Convert datetime to Unix → MJD: MJD = Unix/86400 + 40587
            tr_met = tr_t.timestamp()
        elif isinstance(tr_t, (int, float)):
            tr_met = tr_t
        else:
            continue
        if 0 < tr_met - t_mjd <= window_sec:
            return True
    return False


if __name__ == "__main__":
    truth = load_swpc_flares()
    print(f"Loaded {len(truth)} ground-truth events")
    if not truth.empty:
        print(truth[["date", "goes_class", "peak_flux"]].head(10))
        print(f"Date range: {truth['date'].min()} → {truth['date'].max()}")
