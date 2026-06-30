"""Multi-channel time-series sequence builder for deep-learning flare forecasting.

Builds (12, 86400) per-day channel tensors from SoLEXS (soft X-ray), HEL1OS
(hard X-ray), and GOES XRS data, then slices them into sliding windows for
CNN-LSTM / Transformer training.

Channel map
-----------
 0  SoLEXS SXR (2-22 keV)              deadtime-corrected, 1s cadence
 1  CZT1  20-40  keV                   background-subtracted, aligned to SoLEXS
 2  CZT1  40-60  keV                   "
 3  CZT1  60-80  keV                   "
 4  CZT1  80-150 keV                   "
 5  CdTe1 5-20  keV                    "
 6  CdTe1 20-30 keV                    "
 7  CdTe1 30-40 keV                    "
 8  CZT1  18-160 keV (full band)       "
 9  GOES XRS-B (0.1-0.8 nm)            interpolated to 1s cadence
10  GOES XRS-A (0.05-0.4 nm)           "
11  d(SXR)/dt                          np.diff(channel 0, prepend=channel_0[0])
"""

from __future__ import annotations

import os
from datetime import date
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
from scipy.interpolate import interp1d

from bah2026.config import (
    FEATURE_FORECAST_WINDOW_SEC,
    FEATURE_LOOKBACK_SEC,
    FEATURE_STEP_SEC,
    N_WORKERS,
    PROJECT_ROOT,
    SOLEXS_ROWS_PER_DAY,
)
from bah2026.data.corrections import (
    correct_solexs_deadtime,
    subtract_hel1os_background,
)
from bah2026.data.preprocessing import align_hel1os_to_solexs, met_to_mjd
from bah2026.data.reader import load_hel1os_lc, load_solexs_lc

os.environ.setdefault(
    "BAH2026_GOES_DIR", str(PROJECT_ROOT / "data" / "external" / "goes")
)
GOES_DIR = Path(os.environ["BAH2026_GOES_DIR"])

GOES_MJD_EPOCH = 51544.5

_CZT_BAND_TO_CH = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 8)]
_CDTE_BAND_TO_CH = [(0, 5), (1, 6), (2, 7)]


def _load_goes_day(d: date) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    pattern = f"sci_xrsf-l2-*_g16_d{d.year:04d}{d.month:02d}{d.day:02d}_*.nc"
    files = sorted(GOES_DIR.glob(pattern))
    if not files:
        return None
    try:
        from netCDF4 import Dataset

        with Dataset(str(files[0]), "r") as nc:
            t = np.asarray(nc.variables["time"][:], dtype=np.float64)
            xrsb = np.asarray(nc.variables["xrsb_flux"][:], dtype=np.float64)
            xrsa = np.asarray(nc.variables["xrsa_flux"][:], dtype=np.float64)
    except Exception:
        return None
    return GOES_MJD_EPOCH + t / 86400.0, xrsb, xrsa


def _interp_goes(
    goes_mjd: np.ndarray,
    flux: np.ndarray,
    target_mjd: np.ndarray,
) -> np.ndarray:
    valid = np.isfinite(flux) & (flux >= 0)
    if not np.any(valid):
        return np.zeros(len(target_mjd), dtype=np.float32)
    f = interp1d(
        goes_mjd[valid],
        flux[valid],
        kind="linear",
        bounds_error=False,
        fill_value=0.0,
    )
    return f(target_mjd).astype(np.float32)


def build_day_sequence(
    d: date,
    event_times,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Build sliding-window sequences for a single day.

    Parameters
    ----------
    d : date
        Calendar day to process.
    event_times : iterable of float
        Flare peak times in seconds-of-day (0..86400). A window starting at
        second ``i`` is labelled positive if any peak falls in
        ``[i, i + FEATURE_FORECAST_WINDOW_SEC)``.

    Returns
    -------
    (X, y) or None
        X : (N, 12, FEATURE_LOOKBACK_SEC) float32
        y : (N,) int8   -- None if SoLEXS data is unavailable.
    """
    try:
        sx = load_solexs_lc(d)
    except Exception:
        return None

    sx_time = np.asarray(sx["time"], dtype=np.float64)
    sx_counts = np.asarray(sx["counts"], dtype=np.float64)
    mjdrefi = int(sx["mjdrefi"])
    mjdreff = float(sx["mjdreff"])
    target_T = SOLEXS_ROWS_PER_DAY
    n = min(len(sx_time), target_T)

    channels = np.zeros((12, target_T), dtype=np.float32)
    channels[0, :n] = np.asarray(
        correct_solexs_deadtime(sx_counts)[:n], dtype=np.float32
    )

    sx_mjd = met_to_mjd(sx_time, mjdrefi, mjdreff)

    try:
        czt = load_hel1os_lc(d, "czt", 1)
        czt_ctr = np.asarray(czt["ctr"], dtype=np.float64)
        if czt_ctr.ndim == 2 and czt_ctr.shape[1] >= 5 and czt.get("mjd") is not None:
            czt_mjd = np.asarray(czt["mjd"], dtype=np.float64)
            czt_bs = subtract_hel1os_background(czt_ctr, "czt")
            aligned = align_hel1os_to_solexs(czt_mjd, czt_bs, sx_time, mjdrefi, mjdreff)
            for b, ch in _CZT_BAND_TO_CH:
                col = aligned[:, b]
                m = min(len(col), target_T)
                channels[ch, :m] = np.nan_to_num(col[:m], nan=0.0).astype(np.float32)
    except Exception:
        pass

    try:
        cdte = load_hel1os_lc(d, "cdte", 1)
        cdte_ctr = np.asarray(cdte["ctr"], dtype=np.float64)
        if (
            cdte_ctr.ndim == 2
            and cdte_ctr.shape[1] >= 3
            and cdte.get("mjd") is not None
        ):
            cdte_mjd = np.asarray(cdte["mjd"], dtype=np.float64)
            cdte_bs = subtract_hel1os_background(cdte_ctr, "cdte")
            aligned = align_hel1os_to_solexs(
                cdte_mjd, cdte_bs, sx_time, mjdrefi, mjdreff
            )
            for b, ch in _CDTE_BAND_TO_CH:
                col = aligned[:, b]
                m = min(len(col), target_T)
                channels[ch, :m] = np.nan_to_num(col[:m], nan=0.0).astype(np.float32)
    except Exception:
        pass

    g = _load_goes_day(d)
    if g is not None:
        gmjd, xrsb, xrsa = g
        channels[9, :n] = _interp_goes(gmjd, xrsb, sx_mjd)[:n]
        channels[10, :n] = _interp_goes(gmjd, xrsa, sx_mjd)[:n]

    channels[:11] = np.nan_to_num(channels[:11], nan=0.0)
    channels[:11] = np.maximum(channels[:11], 0.0)
    channels[11] = np.diff(channels[0], prepend=channels[0, 0]).astype(np.float32)

    lookback = FEATURE_LOOKBACK_SEC
    step = FEATURE_STEP_SEC
    forecast = FEATURE_FORECAST_WINDOW_SEC
    ev = np.asarray(list(event_times), dtype=np.float64) if event_times else np.empty(0)

    starts = np.arange(lookback, target_T, step)
    if starts.size == 0:
        return None
    X = np.empty((starts.size, 12, lookback), dtype=np.float32)
    y = np.zeros(starts.size, dtype=np.int8)
    for k, i in enumerate(starts):
        X[k] = channels[:, i - lookback : i]
        if ev.size and np.any((ev >= i) & (ev < i + forecast)):
            y[k] = 1
    return X, y


def _lookup_events(d: date, event_times_map) -> list:
    if d in event_times_map:
        return list(event_times_map[d])
    s = d.isoformat()
    if s in event_times_map:
        return list(event_times_map[s])
    return []


def _day_worker(args):
    d, event_times = args
    try:
        return build_day_sequence(d, event_times)
    except Exception:
        return None


def build_all_sequences(days, event_times_map, output_dir) -> None:
    """Process all days in parallel and save memory-mapped .npy sequence files.

    Produces ``output_dir/X_seq.npy`` (N_total, 12, lookback) float32 and
    ``output_dir/y_seq.npy`` (N_total,) int8, both loadable with
    ``np.load(..., mmap_mode='r')``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tasks = [(d, _lookup_events(d, event_times_map)) for d in days]
    results: list = []
    with Pool(N_WORKERS) as pool:
        for i, res in enumerate(pool.imap_unordered(_day_worker, tasks)):
            if res is not None:
                results.append(res)
            if (i + 1) % 50 == 0:
                print(
                    f"[sequence_builder] processed {i + 1}/{len(tasks)} days "
                    f"({len(results)} ok)"
                )

    if not results:
        print("[sequence_builder] no valid days produced")
        return

    x_parts = [r[0] for r in results]
    y_parts = [r[1] for r in results]
    n_total = sum(x.shape[0] for x in x_parts)
    lookback = x_parts[0].shape[2]

    x_path = output_dir / "X_seq.npy"
    y_path = output_dir / "y_seq.npy"
    X_mm = np.lib.format.open_memmap(
        x_path, mode="w+", dtype=np.float32, shape=(n_total, 12, lookback)
    )
    y_mm = np.lib.format.open_memmap(y_path, mode="w+", dtype=np.int8, shape=(n_total,))
    offset = 0
    for xd, yd in zip(x_parts, y_parts):
        m = xd.shape[0]
        X_mm[offset : offset + m] = xd
        y_mm[offset : offset + m] = yd
        offset += m
    X_mm.flush()
    y_mm.flush()
    del X_mm, y_mm

    print(
        f"[sequence_builder] saved {x_path} ({n_total}, 12, {lookback}) "
        f"and {y_path} ({n_total},)"
    )


class SequenceDataset(torch.utils.data.Dataset):
    """Memory-mapped sequence dataset for PyTorch.

    Loads X_seq.npy and y_seq.npy with mmap_mode='r' for zero-copy access.
    Supports data augmentation: time-shift, noise injection, channel dropout.
    """

    def __init__(self, x_path, y_path, indices=None, augment=False):
        self.X = np.load(x_path, mmap_mode="r")
        self.y = np.load(y_path, mmap_mode="r")
        if indices is not None:
            self.indices = np.asarray(indices)
        else:
            self.indices = np.arange(len(self.y))
        self.augment = augment

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        actual_idx = int(self.indices[idx])
        x = self.X[actual_idx].copy()
        y = self.y[actual_idx]
        if self.augment:
            x = self._augment(x)
        return torch.from_numpy(x).float(), torch.tensor(y, dtype=torch.float32)

    def _augment(self, x):
        """Apply random augmentations.

        - Time-shift: roll along time axis by +/-60 samples
        - Gaussian noise: add N(0, 0.01 * std)
        - Channel dropout: zero out 2 of 12 channels randomly
        """
        if np.random.random() < 0.5:
            shift = np.random.randint(-60, 61)
            x = np.roll(x, shift, axis=1)
        if np.random.random() < 0.5:
            noise = (
                np.random.randn(*x.shape).astype(np.float32)
                * 0.01
                * (np.std(x, axis=1, keepdims=True) + 1e-6)
            )
            x = x + noise
        if np.random.random() < 0.3:
            drop = np.random.choice(12, 2, replace=False)
            x[drop] = 0.0
        return x


def create_dataloaders(
    x_path,
    y_path,
    train_idx,
    val_idx,
    test_idx,
    batch_size: int = 256,
) -> dict:
    """Create train/val/test DataLoaders with appropriate settings."""
    train_ds = SequenceDataset(x_path, y_path, indices=train_idx, augment=True)
    val_ds = SequenceDataset(x_path, y_path, indices=val_idx, augment=False)
    test_ds = SequenceDataset(x_path, y_path, indices=test_idx, augment=False)

    return {
        "train": torch.utils.data.DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
            prefetch_factor=4,
            persistent_workers=True,
        ),
        "val": torch.utils.data.DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        ),
        "test": torch.utils.data.DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        ),
    }


def prepare_downsampled_sequences(
    x_path,
    y_path,
    output_path,
    factor: int = 10,
) -> None:
    """Downsample 1s -> `factor`s cadence (default 10s) for transformer training.

    Reshapes (N, 12, 3600) -> (N, 12, 360, factor) -> mean over last axis ->
    (N, 12, 360). Saves an .npz with keys ``X`` (float32) and ``y`` (int8) to
    ``output_path``.
    """
    X = np.load(x_path, mmap_mode="r")
    N, C, L = X.shape
    usable = (L // factor) * factor
    new_len = usable // factor
    X_ds = np.empty((N, C, new_len), dtype=np.float32)

    batch = 512
    for s in range(0, N, batch):
        chunk = np.asarray(X[s : s + batch], dtype=np.float32)
        b = chunk.shape[0]
        chunk = chunk[:, :, :usable].reshape(b, C, new_len, factor).mean(axis=3)
        X_ds[s : s + b] = chunk

    y = np.asarray(np.load(y_path, mmap_mode="r"), dtype=np.int8)
    np.savez(output_path, X=X_ds, y=y)
    print(
        f"[sequence_builder] saved downsampled -> {output_path} ({N}, {C}, {new_len})"
    )


__all__ = [
    "build_day_sequence",
    "build_all_sequences",
    "SequenceDataset",
    "create_dataloaders",
    "prepare_downsampled_sequences",
]
