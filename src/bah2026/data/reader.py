"""FITS data readers for SoLEXS and HEL1OS processed data.

Expected directory layout (after extracting tar.xz):
    data/processed/
    ├── solexs/YYYY/MM/DD/SDD2/AL1_SOLEXS_YYYYMMDD_SDD2_L1.{lc,pi,gti}
    └── hel1os/YYYY/MM/DD/lightcurve_czt{1,2}.fits, lightcurve_cdte{1,2}.fits, ...
"""

from __future__ import annotations

from datetime import date
from multiprocessing import Pool

import numpy as np
from astropy.io import fits

from bah2026.config import DATA_ROOT, CZT_BANDS, CDTE_BANDS, N_WORKERS


# ── SoLEXS ──────────────────────────────────────────────────────────────

def _solexs_dir(d: date) -> Path:
    return DATA_ROOT / "solexs" / f"{d.year:04d}" / f"{d.month:02d}" / f"{d.day:02d}" / "SDD2"


def load_solexs_lc(d: date) -> dict:
    """Load SoLEXS SDD2 light curve."""
    lc_files = list(_solexs_dir(d).glob("*_L1.lc"))
    if not lc_files:
        raise FileNotFoundError(f"No SoLEXS LC for {d}")
    with fits.open(lc_files[0]) as hdul:
        hdr = hdul["RATE"].header
        data = hdul["RATE"].data
        return {
            "time": np.asarray(data["TIME"], dtype=np.float64),
            "counts": np.asarray(data["COUNTS"], dtype=np.float64),
            "tstart": float(hdr["TSTART"]),
            "mjdrefi": int(hdr.get("MJDREFI", 40587)),
            "mjdreff": float(hdr.get("MJDREFF", 0.0)),
            "date_obs": hdr.get("DATE-OBS", str(d)),
            "tstop": float(hdr["TSTOP"]),
        }


def load_solexs_pi(d: date) -> dict:
    """Load SoLEXS SDD2 PI spectrum (86400 spectra x 340 channels)."""
    pi_files = list(_solexs_dir(d).glob("*_L1.pi"))
    if not pi_files:
        raise FileNotFoundError(f"No SoLEXS PI for {d}")
    with fits.open(pi_files[0]) as hdul:
        data = hdul["SPECTRUM"].data
        return {
            "counts": np.asarray(data["COUNTS"], dtype=np.float64),
            "channel": np.asarray(data["CHANNEL"][0], dtype=np.int16),
            "tstart": np.asarray(data["TSTART"], dtype=np.float64),
            "exposure": np.asarray(data["EXPOSURE"], dtype=np.float64),
        }


def load_solexs_gti(d: date) -> np.ndarray:
    """Load SoLEXS SDD2 GTI as (N, 2) array of (start, stop) MJD pairs."""
    gti_files = list(_solexs_dir(d).glob("*_L1.gti"))
    if not gti_files:
        return np.zeros((0, 2))
    with fits.open(gti_files[0]) as hdul:
        data = hdul[1].data
        if len(data) == 0:
            return np.zeros((0, 2))
        return np.column_stack([data["START"], data["STOP"]])


# ── HEL1OS ──────────────────────────────────────────────────────────────

def _hel1os_dir(d: date) -> Path:
    return DATA_ROOT / "hel1os" / f"{d.year:04d}" / f"{d.month:02d}" / f"{d.day:02d}"


def load_hel1os_lc(d: date, detector: str = "czt", num: int = 1) -> dict:
    """Load HEL1OS light curve."""
    lc_file = _hel1os_dir(d) / f"lightcurve_{detector}{num}.fits"
    if not lc_file.exists():
        raise FileNotFoundError(f"No HEL1OS LC for {d} ({detector}{num})")

    bands_meta = CZT_BANDS if detector == "czt" else CDTE_BANDS
    energy_ranges = list(bands_meta.values())

    result: dict = {"band_names": [], "energy_ranges": energy_ranges}

    with fits.open(lc_file) as hdul:
        all_ctr, all_err = [], []
        all_mjd = all_isot = None

        for i, band_key in enumerate(bands_meta):
            ext_idx = i + 1
            if ext_idx >= len(hdul):
                continue
            data = hdul[ext_idx].data
            result["band_names"].append(hdul[ext_idx].header["EXTNAME"])
            all_ctr.append(np.asarray(data["CTR"], dtype=np.float64))
            all_err.append(np.asarray(data["STAT_ERR"], dtype=np.float64))
            if all_mjd is None:
                all_mjd = np.asarray(data["MJD"], dtype=np.float64)
                all_isot = np.asarray(data["ISOT"], dtype="U30")

        result["mjd"] = all_mjd
        result["isot"] = all_isot
        # CdTe bands can have different row counts — truncate to min
        if all_ctr:
            min_rows = min(len(a) for a in all_ctr)
            all_ctr = [a[:min_rows] for a in all_ctr]
            all_err = [a[:min_rows] for a in all_err]
            result["mjd"] = all_mjd[:min_rows]
            result["isot"] = all_isot[:min_rows]
        result["ctr"] = np.column_stack(all_ctr) if all_ctr else np.empty((0, 0))
        result["stat_err"] = np.column_stack(all_err) if all_err else np.empty((0, 0))

    return result


def load_hel1os_spectra(d: date, detector: str = "czt", num: int = 1) -> dict:
    """Load HEL1OS energy spectra."""
    spec_file = _hel1os_dir(d) / f"hel1os_{detector}_spectra_{detector}{num}.fits"
    if not spec_file.exists():
        raise FileNotFoundError(f"No HEL1OS spectra for {d} ({detector}{num})")
    with fits.open(spec_file) as hdul:
        data = hdul["SPECTRUM"].data
        return {
            "spec_num": np.asarray(data["SPEC_NUM"], dtype=np.int32),
            "channel": np.asarray(data["CHANNEL"], dtype=np.int32),
            "counts": np.asarray(data["COUNTS"], dtype=np.float64),
            "stat_err": np.asarray(data["STAT_ERR"], dtype=np.float64),
            "tstart": np.asarray(data["TSTART"], dtype=np.float64),
            "tstop": np.asarray(data["TSTOP"], dtype=np.float64),
            "exposure": np.asarray(data["EXPOSURE"], dtype=np.float64),
            "detechans": int(hdul["SPECTRUM"].header.get("DETCHANS", 341)),
        }


# ── Day discovery ───────────────────────────────────────────────────────

def _check_solexs_day(path: Path) -> date | None:
    yd, md, dd = path.parent.parent.name, path.parent.name, path.name
    try:
        d = date(int(yd), int(md), int(dd))
    except (ValueError, OverflowError):
        return None
    sdd2 = path / "SDD2"
    if sdd2.exists() and any(sdd2.glob("*_L1.lc")):
        return d
    return None


def _check_hel1os_day(path: Path) -> date | None:
    yd, md, dd = path.parent.parent.name, path.parent.name, path.name
    try:
        d = date(int(yd), int(md), int(dd))
    except (ValueError, OverflowError):
        return None
    if any(path.glob("lightcurve_*.fits")):
        return d
    return None


def discover_solexs_days() -> list[date]:
    """Return sorted list of dates with SoLEXS SDD2 data."""
    root = DATA_ROOT / "solexs"
    if not root.exists():
        return []
    day_dirs = []
    for yd in root.iterdir():
        if not yd.is_dir():
            continue
        for md in yd.iterdir():
            if not md.is_dir():
                continue
            for dd in md.iterdir():
                if dd.is_dir():
                    day_dirs.append(dd)
    with Pool(N_WORKERS) as pool:
        results = pool.map(_check_solexs_day, day_dirs)
    return sorted(d for d in results if d is not None)


def discover_hel1os_days() -> list[date]:
    """Return sorted list of dates with HEL1OS light curve data."""
    root = DATA_ROOT / "hel1os"
    if not root.exists():
        return []
    day_dirs = []
    for yd in root.iterdir():
        if not yd.is_dir():
            continue
        for md in yd.iterdir():
            if not md.is_dir():
                continue
            for dd in md.iterdir():
                if dd.is_dir():
                    day_dirs.append(dd)
    with Pool(N_WORKERS) as pool:
        results = pool.map(_check_hel1os_day, day_dirs)
    return sorted(d for d in results if d is not None)


def discover_combined_days() -> list[date]:
    """Return sorted list of dates with BOTH SoLEXS and HEL1OS data."""
    return sorted(set(discover_solexs_days()) & set(discover_hel1os_days()))
