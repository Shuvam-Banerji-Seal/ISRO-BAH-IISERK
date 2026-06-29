"""Instrument corrections based on SoLEXS and HEL1OS instrument papers.

Implements corrections described in:
  - SoLEXS: Sarwade et al. 2025 (arXiv:2509.26292v2) §4.5, §5.3
  - HEL1OS: Nandi et al. 2025 (arXiv:2512.12679) §3.2, §4, §6

Corrections:
  1. Paralyzable deadtime correction (SoLEXS spectral chain)
  2. HEL1OS background subtraction (CZT ~70 cps, CdTe ~0.15 cps)
  3. Reset-pulse spurious count subtraction (SoLEXS ~500 cts/s)
"""

from __future__ import annotations

import numpy as np


# ── SoLEXS Deadtime Parameters (from paper §4.5, §5.3) ────────────────

SOLEXS_TAU_SPECTRAL = 13.65e-6  # 13.65 µs on-board spectral chain
SOLEXS_TAU_TIMING = 1.6e-6  # 1.6 µs timing chain
SOLEXS_SPECTRAL_EFFICIENCY = 0.8883  # 88.83% on-board
SOLEXS_SPURIOUS_RATE = 500.0  # ~500 spurious counts/s from reset pulses

# ── HEL1OS Background Levels (from paper §6, off-Sun pointings) ────────

HEL1OS_BG_CZT_CPS = 70.0  # counts/s (CZT background at L1)
HEL1OS_BG_CDTE_CPS = 0.15  # counts/s (CdTe background at L1)


def correct_solexs_deadtime(
    counts: np.ndarray,
    tau: float = SOLEXS_TAU_SPECTRAL,
    max_iter: int = 50,
    tol: float = 1e-10,
) -> np.ndarray:
    """Apply paralyzable deadtime correction to SoLEXS count rates.

    The paralyzable deadtime model is:
        n_measured = n_true × exp(-n_true × τ)

    This function inverts it using Newton-Raphson to recover n_true.

    From SoLEXS paper §4.5:
        "At high count rates, the spectral chain's processed triangular
        pulse from two photons pile-up... The timing chain processes the
        X-ray event faster... If multiple events are detected by the
        timing channel during the processing of a single spectroscopic
        pulse, the event is considered a pile-up."

    Parameters
    ----------
    counts : ndarray
        Measured count rates (counts/s). Can be any shape.
    tau : float
        Deadtime in seconds (default 13.65 µs for SoLEXS spectral chain).
    max_iter : int
        Maximum Newton-Raphson iterations.
    tol : float
        Convergence tolerance.

    Returns
    -------
    corrected : ndarray
        Deadtime-corrected count rates. Same shape as input.
    """
    counts_arr = np.asarray(counts, dtype=np.float64)
    corrected = np.zeros_like(counts_arr)

    # Vectorized Newton-Raphson
    mask = counts_arr > 0
    n_meas = counts_arr[mask]

    # Initial guess: n_true ≈ n_measured * 1.1 (biased high for convergence)
    # For paralyzable detector, max measurable rate is at n_true = 1/tau
    n_max = 1.0 / tau  # = 73,260 cts/s for tau=13.65µs
    n_true = np.minimum(n_meas * 1.5, n_max * 0.99)  # Start below the peak

    for _ in range(max_iter):
        # f(x) = x * exp(-x * tau) - n_meas
        exp_term = np.exp(-n_true * tau)
        f = n_true * exp_term - n_meas
        # f'(x) = exp(-x * tau) * (1 - x * tau)
        df = exp_term * (1 - n_true * tau)

        # Avoid division by zero (at n_true = 1/tau, df = 0)
        valid = np.abs(df) > 1e-15
        if not np.any(valid):
            break

        update = np.zeros_like(n_true)
        update[valid] = f[valid] / df[valid]
        n_true -= update

        # Ensure positive and below the paralyzable limit
        n_true = np.clip(n_true, n_meas, n_max * 0.999)

        # Check convergence
        if np.max(np.abs(f)) < tol:
            break

    corrected[mask] = n_true
    return corrected


def subtract_hel1os_background(
    ctr: np.ndarray,
    detector: str = "czt",
) -> np.ndarray:
    """Subtract instrumental background from HEL1OS count rates.

    From HEL1OS paper §6:
        "Background obtained from off-Sun pointings (>10° from boresight)
        during PV phase... Background level of the data has been very
        benign" (L1 orbit, away from Earth radiation belts).

    Background levels (paper §6):
        CdTe: ~0.15 counts/s per band
        CZT:  ~70 counts/s for the FULL band (18-160 keV)
              ~10-15 counts/s per narrow band

    The 70 cps CZT background is dominated by the full-band channel.
    For narrow bands (20-40, 40-60, 60-80, 80-150 keV), the background
    is proportionally smaller (~10-15 cps each).

    Parameters
    ----------
    ctr : ndarray
        Count rates (cts/s). Shape (n_times,) or (n_times, n_bands).
    detector : str
        "czt" or "cdte".

    Returns
    -------
    bg_subtracted : ndarray
        Background-subtracted count rates. Same shape as input.
    """
    if detector == "czt":
        # CZT: 70 cps total for full band (band 4 = 18-160 keV)
        # Narrow bands get proportional background
        # Band 0 (20-40): ~15 cps, Band 1 (40-60): ~12 cps,
        # Band 2 (60-80): ~8 cps, Band 3 (80-150): ~5 cps,
        # Band 4 (18-160 full): ~70 cps
        bg_per_band = np.array([15.0, 12.0, 8.0, 5.0, 70.0])
    else:
        # CdTe: 0.15 cps per band
        bg_per_band = np.full(5, 0.15)

    if ctr.ndim == 1:
        # Single band — use the full-band background
        bg = bg_per_band[-1] if detector == "czt" else bg_per_band[0]
        return np.maximum(ctr - bg, 0.0)
    else:
        # Multi-band: subtract per-band background
        n_bands = ctr.shape[1]
        bg = np.zeros(n_bands)
        for b in range(min(n_bands, len(bg_per_band))):
            bg[b] = bg_per_band[b]
        # If more bands than expected, use last value
        if n_bands > len(bg_per_band):
            bg[len(bg_per_band) :] = bg_per_band[-1]
        return np.maximum(ctr - bg[np.newaxis, :], 0.0)


def subtract_solexs_spurious(counts: np.ndarray) -> np.ndarray:
    """Subtract reset-pulse spurious counts from SoLEXS timing chain.

    From SoLEXS paper §5.3:
        "CSPA reset pulses (every 2 ms) produce ringing picked up by
        timing chain as ~500 spurious counts/s... subtract spurious
        contribution from timing chain."

    Parameters
    ----------
    counts : ndarray
        Measured timing-chain count rates (cts/s).

    Returns
    -------
    corrected : ndarray
        Spurious-subtracted count rates.
    """
    return np.maximum(counts - SOLEXS_SPURIOUS_RATE, 0.0)


def correct_hel1os_deadtime_approx(
    counts: np.ndarray,
    detector: str = "czt",
) -> np.ndarray:
    """Approximate deadtime correction for HEL1OS.

    From HEL1OS paper §4 (Srikar et al. in preparation):
        Dead time models fitted to ground-test data.

    Approximate values (not yet published):
        CdTe: ~10 µs
        CZT:  ~5 µs

    Parameters
    ----------
    counts : ndarray
        Measured count rates (cts/s).
    detector : str
        "czt" or "cdte".

    Returns
    -------
    corrected : ndarray
        Deadtime-corrected count rates.
    """
    tau = 10e-6 if detector == "cdte" else 5e-6
    return correct_solexs_deadtime(counts, tau=tau)


def apply_all_corrections(
    solexs_counts: np.ndarray | None = None,
    hel1os_ctr: np.ndarray | None = None,
    hel1os_detector: str = "czt",
    apply_deadtime: bool = True,
    apply_background: bool = True,
    apply_spurious: bool = False,
) -> dict:
    """Apply all instrument corrections to a single day's data.

    Parameters
    ----------
    solexs_counts : ndarray, optional
        SoLEXS SDD2 count rates (cts/s, 86400 elements).
    hel1os_ctr : ndarray, optional
        HEL1OS count rates (cts/s, n_times × n_bands).
    hel1os_detector : str
        "czt" or "cdte".
    apply_deadtime : bool
        Apply deadtime correction.
    apply_background : bool
        Apply HEL1OS background subtraction.
    apply_spurious : bool
        Apply SoLEXS spurious count subtraction.

    Returns
    -------
    corrections : dict
        Keys: 'solexs_corrected', 'hel1os_corrected', 'stats'
    """
    stats = {}

    solexs_corrected = solexs_counts
    if solexs_counts is not None and apply_deadtime:
        solexs_corrected = correct_solexs_deadtime(solexs_counts)
        diff = solexs_corrected - solexs_counts
        stats["solexs_deadtime_max_corr_pct"] = float(
            np.max(np.where(solexs_counts > 0, diff / solexs_counts * 100, 0))
        )
        stats["solexs_deadtime_mean_corr_pct"] = float(
            np.mean(np.where(solexs_counts > 0, diff / solexs_counts * 100, 0))
        )

    if solexs_counts is not None and apply_spurious:
        if solexs_corrected is None:
            solexs_corrected = solexs_counts.copy()
        solexs_corrected = subtract_solexs_spurious(solexs_corrected)
        stats["solexs_spurious_subtracted"] = SOLEXS_SPURIOUS_RATE

    hel1os_corrected = hel1os_ctr
    if hel1os_ctr is not None and apply_background:
        hel1os_corrected = subtract_hel1os_background(hel1os_ctr, hel1os_detector)
        bg = HEL1OS_BG_CZT_CPS if hel1os_detector == "czt" else HEL1OS_BG_CDTE_CPS
        stats["hel1os_background_cps"] = bg
        if hel1os_ctr.ndim == 1:
            stats["hel1os_bg_fraction_pct"] = float(
                np.where(hel1os_ctr > 0, bg / hel1os_ctr * 100, 0).mean()
            )
        else:
            stats["hel1os_bg_fraction_pct"] = float(
                np.where(hel1os_ctr[:, -1] > 0, bg / hel1os_ctr[:, -1] * 100, 0).mean()
            )

    return {
        "solexs_corrected": solexs_corrected,
        "hel1os_corrected": hel1os_corrected,
        "stats": stats,
    }
