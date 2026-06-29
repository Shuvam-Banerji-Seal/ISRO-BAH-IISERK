"""RMF/ARF response convolution for proper spectral deconvolution.

The SoLEXS CALDB contains:
  - RMF ( Redistribution Matrix File): maps photon energies to detector channels
  - ARF (Ancillary Response File): effective area as a function of energy

To convert observed counts → incident photon spectrum, we need to:
  1. Apply RMF redistribution: counts = RMF @ photon_flux
  2. Apply ARF effective area: counts = ARF * RMF @ photon_flux
  3. Invert the process: photon_flux = (ARF * RMF)^(-1) @ counts

This module implements:
  - load_and_build_response(): load RMF + ARF → response matrix
  - deconvolve_spectrum(): invert the response to get photon spectrum
  - convolve_model(): forward-fold a model through the response
  - effective_area_at_energy(): interpolate ARF at given energies

References:
  - SoLEXS paper (Sarwade et al. 2025) §4: instrument response
  - HEASARC OGIP calibration memo CAL/GEN/92-002 (RMF/ARF format)
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d
from scipy.sparse import csr_matrix, lil_matrix
from scipy.optimize import nnls

from bah2026.data.calibration import (
    load_arf,
    load_rmf,
    load_channel_energies,
    EBOUNCE_FILE,
    ARF_FILE,
    RMF_FILE,
)


# ── Response matrix construction ────────────────────────────────────────


def build_response_matrix(
    n_channels: int = 340,
    n_energy_bins: int = 340,
) -> dict:
    """Build the full instrument response matrix R = ARF × RMF.

    The response maps incident photon flux (per energy) to observed
    counts (per channel):
        counts = R @ photon_flux

    Parameters
    ----------
    n_channels : int
        Number of detector channels (default 340 for SoLEXS SDD2).
    n_energy_bins : int
        Number of photon energy bins (default 340, matching channels).

    Returns
    -------
    response : dict
        Keys: 'matrix' (sparse CSR, shape n_channels × n_energy_bins),
        'arf' (effective area array), 'energies_lo', 'energies_hi',
        'channel_energies_lo', 'channel_energies_hi'.
    """
    # Load ARF
    arf_energies, arf_area = load_arf()

    # Load RMF
    fchan, nchan, matrix_data = load_rmf()

    # Load channel energies
    chan_emin, chan_emax = load_channel_energies()
    chan_centroids = (chan_emin + chan_emax) / 2.0

    # Build sparse RMF matrix
    # RMF is stored as variable-length arrays: for each energy bin i,
    # fchan[i] is the first channel, nchan[i] is the number of channels,
    # matrix[i] is the probability for each channel.
    n_energy = len(fchan)
    rmf = lil_matrix((n_channels, n_energy))

    for i in range(n_energy):
        fc = int(fchan[i])
        nc = int(nchan[i])
        if nc > 0 and fc < n_channels:
            row_slice = matrix_data[i]
            if hasattr(row_slice, "__len__"):
                vals = np.asarray(row_slice[:nc], dtype=np.float64)
            else:
                vals = np.array([float(row_slice)], dtype=np.float64)
            end = min(fc + nc, n_channels)
            rmf[fc:end, i] = vals[: end - fc]

    rmf = rmf.tocsr()

    # Interpolate ARF to RMF energy grid
    # ARF energies are the photon energy grid
    arf_interp = interp1d(
        arf_energies,
        arf_area,
        kind="linear",
        bounds_error=False,
        fill_value=0.0,
    )

    # ARF energy grid may not match RMF energy grid exactly
    # Use ARF energies as the photon energy grid
    n_photon = len(arf_energies)
    arf_vals = arf_area.copy()

    # Resize RMF if needed
    if rmf.shape[1] != n_photon:
        # Pad or truncate
        new_rmf = lil_matrix((n_channels, n_photon))
        min_cols = min(rmf.shape[1], n_photon)
        new_rmf[:n_channels, :min_cols] = rmf[:n_channels, :min_cols]
        rmf = new_rmf.tocsr()

    # Full response: R = diag(ARF) @ RMF
    # Each column of R is scaled by the ARF at that energy
    arf_diag = csr_matrix(np.diag(arf_vals))
    response = rmf @ arf_diag

    return {
        "matrix": response.tocsr(),
        "arf": arf_vals,
        "energies_lo": arf_energies,
        "energies_hi": arf_energies,  # ARF stores ENERG_LO only
        "channel_energies_lo": chan_emin,
        "channel_energies_hi": chan_emax,
        "channel_centroids": chan_centroids,
    }


# ── Forward folding ─────────────────────────────────────────────────────


def convolve_model(
    model_photon_flux: np.ndarray,
    response: dict,
) -> np.ndarray:
    """Forward-fold a model photon spectrum through the instrument response.

    Parameters
    ----------
    model_photon_flux : ndarray
        Incident photon flux per energy bin.
    response : dict
        Response from build_response_matrix().

    Returns
    -------
    counts : ndarray
        Predicted counts per channel.
    """
    R = response["matrix"]
    n_model = len(model_photon_flux)
    n_resp = R.shape[1]

    if n_model < n_resp:
        flux_padded = np.zeros(n_resp)
        flux_padded[:n_model] = model_photon_flux
    elif n_model > n_resp:
        flux_padded = model_photon_flux[:n_resp]
    else:
        flux_padded = model_photon_flux

    return R @ flux_padded


# ── Deconvolution (inverse problem) ─────────────────────────────────────


def deconvolve_spectrum(
    counts: np.ndarray,
    response: dict,
    method: str = "nnls",
    max_iter: int = 100,
) -> np.ndarray:
    """Deconvolve observed counts to get incident photon spectrum.

    Solves: counts = R @ photon_flux for photon_flux.

    Parameters
    ----------
    counts : ndarray
        Observed counts per channel.
    response : dict
        Response from build_response_matrix().
    method : str
        Deconvolution method: 'nnls' (non-negative least squares) or
        'richardson_lucy' (iterative Bayesian).
    max_iter : int
        Maximum iterations for iterative methods.

    Returns
    -------
    photon_flux : ndarray
        Deconvolved photon flux per energy bin.
    """
    R = response["matrix"].toarray()
    counts = np.asarray(counts, dtype=np.float64)
    counts = np.maximum(counts, 0.0)

    n_chan, n_energy = R.shape

    # Ensure counts matches channel count
    if len(counts) < n_chan:
        counts = np.pad(counts, (0, n_chan - len(counts)))
    elif len(counts) > n_chan:
        counts = counts[:n_chan]

    if method == "nnls":
        # Non-negative least squares
        try:
            photon_flux, _ = nnls(R, counts, maxiter=max_iter)
            return photon_flux
        except Exception:
            # Fallback: simple division
            col_sums = R.sum(axis=1)
            col_sums[col_sums == 0] = 1.0
            return counts / col_sums

    elif method == "richardson_lucy":
        # Richardson-Lucy deconvolution
        photon_flux = np.ones(n_energy) * np.mean(counts) / max(R.sum(), 1)
        R_T = R.T

        for _ in range(max_iter):
            # Forward projection
            model_counts = R @ photon_flux
            model_counts = np.maximum(model_counts, 1e-10)

            # Correction factor
            correction = R_T @ (counts / model_counts)
            correction = np.nan_to_num(correction, nan=0.0, posinf=0.0)

            # Update
            photon_flux *= correction
            photon_flux = np.nan_to_num(photon_flux, nan=0.0, posinf=0.0, neginf=0.0)

        return photon_flux

    else:
        raise ValueError(f"Unknown method: {method}")


# ── Effective area interpolation ────────────────────────────────────────


def effective_area_at_energy(
    energy_kev: float | np.ndarray,
) -> float | np.ndarray:
    """Get effective area at a given energy.

    Parameters
    ----------
    energy_kev : float or ndarray
        Photon energy (keV).

    Returns
    -------
    area : float or ndarray
        Effective area (cm²).
    """
    energies, area = load_arf()
    f = interp1d(energies, area, kind="linear", bounds_error=False, fill_value=0.0)
    return f(energy_kev)


# ── Energy flux computation ─────────────────────────────────────────────


def counts_to_energy_flux(
    counts: np.ndarray,
    response: dict | None = None,
    energy_range: tuple[float, float] = (2.0, 22.0),
) -> float:
    """Convert counts in a channel range to energy flux (erg/cm²/s).

    Uses ARF to convert counts to photon flux, then integrates
    E × photon_flux over the specified energy range.

    Parameters
    ----------
    counts : ndarray
        Observed counts per channel.
    response : dict, optional
        Pre-built response. If None, builds from CALDB.
    energy_range : tuple
        Energy range for integration (keV).

    Returns
    -------
    flux : float
        Energy flux in erg/cm²/s.
    """
    if response is None:
        response = build_response_matrix()

    # Deconvolve
    photon_flux = deconvolve_spectrum(counts, response)

    # Get energy grid
    energies = response["energies_lo"]

    # Match flux length to energy grid
    n_energy = len(energies)
    if len(photon_flux) < n_energy:
        flux_padded = np.zeros(n_energy)
        flux_padded[: len(photon_flux)] = photon_flux
        photon_flux = flux_padded
    elif len(photon_flux) > n_energy:
        photon_flux = photon_flux[:n_energy]

    # Integrate E * F(E) dE over the specified range
    mask = (energies >= energy_range[0]) & (energies <= energy_range[1])
    if mask.sum() < 2:
        return 0.0

    e_masked = energies[mask]
    f_masked = photon_flux[mask]

    # Trapezoidal integration: ∫ E * F(E) dE
    # Convert keV to erg: 1 keV = 1.602e-9 erg
    KEV_TO_ERG = 1.602e-9
    # Use np.trapezoid (numpy 2.0+) or fallback to np.trapz
    trapz_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))
    if trapz_fn is None:
        # Manual trapezoidal integration
        energy_flux = np.sum(
            0.5
            * np.diff(e_masked)
            * (e_masked[:-1] * f_masked[:-1] + e_masked[1:] * f_masked[1:])
        )
    else:
        energy_flux = trapz_fn(e_masked * f_masked, e_masked)
    energy_flux *= KEV_TO_ERG

    return float(energy_flux)


# ── Check if CALDB files exist ──────────────────────────────────────────


def has_caldb() -> bool:
    """Check if CALDB files are available."""
    return ARF_FILE.exists() and RMF_FILE.exists() and EBOUNCE_FILE.exists()
