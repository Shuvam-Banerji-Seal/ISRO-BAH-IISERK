"""Data preprocessing: GTI masking, background subtraction, temporal alignment."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter
from scipy.interpolate import interp1d

from bah2026.config import NOWCAST_BACKGROUND_WINDOW_SEC


def compute_gti_mask(time_mjd: np.ndarray, gti: np.ndarray) -> np.ndarray:
    """Create boolean mask selecting times within GTI intervals."""
    mask = np.zeros(len(time_mjd), dtype=bool)
    for start, stop in gti:
        mask |= (time_mjd >= start) & (time_mjd <= stop)
    return mask


def background_subtract(
    counts: np.ndarray,
    window_sec: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate and subtract background via sliding median.

    Parameters
    ----------
    counts : array
        Raw count rate.
    window_sec : int or None
        Window size (seconds). Defaults to config value.

    Returns
    -------
    background : array
    residual : array
    """
    if window_sec is None:
        window_sec = NOWCAST_BACKGROUND_WINDOW_SEC
    valid = np.where(np.isfinite(counts), counts, np.nanmedian(counts))
    bg = median_filter(valid, size=window_sec, mode="nearest")
    return bg, valid - bg


def interpolate_to_common_grid(
    mjd_src: np.ndarray,
    values_src: np.ndarray,
    mjd_grid: np.ndarray,
) -> np.ndarray:
    """Linearly interpolate from source MJD grid to target grid."""
    f = interp1d(mjd_src, values_src, kind="linear",
                 bounds_error=False, fill_value=np.nan)
    return f(mjd_grid)


def met_to_mjd(met: np.ndarray, mjdrefi: int, mjdreff: float) -> np.ndarray:
    """Convert Mission Elapsed Time (seconds from epoch) to MJD."""
    return (mjdrefi + mjdreff) + met / 86400.0


def align_hel1os_to_solexs(
    hel1os_mjd: np.ndarray,
    hel1os_ctr: np.ndarray,
    solexs_time_met: np.ndarray,
    solexs_mjdrefi: int,
    solexs_mjdreff: float,
) -> np.ndarray:
    """Interpolate HEL1OS multi-band CTR onto SoLEXS 1-second time grid.

    Returns
    -------
    aligned : array, shape (86400, nbands)
        HEL1OS CTR interpolated onto SoLEXS grid (NaN outside overlap).
    """
    solexs_mjd = met_to_mjd(solexs_time_met, solexs_mjdrefi, solexs_mjdreff)
    nbands = hel1os_ctr.shape[1] if hel1os_ctr.ndim == 2 else 1
    if hel1os_ctr.ndim == 1:
        hel1os_ctr = hel1os_ctr[:, np.newaxis]

    aligned = np.full((len(solexs_mjd), nbands), np.nan)
    for b in range(nbands):
        aligned[:, b] = interpolate_to_common_grid(hel1os_mjd, hel1os_ctr[:, b], solexs_mjd)
    return aligned
