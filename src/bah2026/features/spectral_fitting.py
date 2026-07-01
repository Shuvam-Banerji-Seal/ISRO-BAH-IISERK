"""Spectral fitting for SoLEXS PI data and HEL1OS multi-band data.

Extracts physical parameters from the energy-resolved data:
  1. Temperature T (MK) from thermal bremsstrahlung fit to SoLEXS PI spectrum
  2. Emission Measure EM (cm⁻³) from the 2-22 keV continuum
  3. Spectral index γ for HEL1OS hard X-ray bands (power-law fit)
  4. Hardness ratio evolution over time
  5. Neupert faithfulness ρ = corr(dSXR/dt, HXR) over sliding windows
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import pearsonr

from bah2026.data.calibration import load_channel_energies


# ── Thermal model ─────────────────────────────────────────────────────


def thermal_bremsstrahlung_spectrum(
    energies_kev: np.ndarray,
    temperature_mk: float,
    emission_measure: float,
) -> np.ndarray:
    """Thermal bremsstrahlung model: I(E) ∝ EM · E⁻¹ · exp(-E/kT).

    Simplified thermal continuum (no line emission). For a more accurate
    model, use a proper APEC/CHIANTI database.

    Parameters
    ----------
    energies_kev : ndarray
        Energy bin centroids (keV).
    temperature_mk : float
        Plasma temperature in MK (1 MK ≈ 0.086 keV).
    emission_measure : float
        EM = ∫ n_e² dV (cm⁻³).

    Returns
    -------
    flux : ndarray
        Model flux at each energy (arbitrary units).
    """
    kT_kev = temperature_mk * 0.086  # MK → keV
    if kT_kev <= 0:
        return np.zeros_like(energies_kev)

    # Thermal bremsstrahlung continuum
    with np.errstate(divide="ignore", invalid="ignore"):
        flux = emission_measure * energies_kev ** (-1) * np.exp(-energies_kev / kT_kev)
    flux[~np.isfinite(flux)] = 0.0
    return flux


def fit_temperature(
    counts_spectrum: np.ndarray,
    channel_energies_kev: tuple[np.ndarray, np.ndarray] | None = None,
    fit_range_kev: tuple[float, float] = (2.0, 10.0),
) -> tuple[float, float, float]:
    """Fit a 1-temperature thermal bremsstrahlung model to a SoLEXS spectrum.

    Parameters
    ----------
    counts_spectrum : ndarray, shape (340,)
        Single spectrum counts (integrated over some time bin).
    channel_energies_kev : tuple, optional
        (emin, emax) arrays for each channel.
    fit_range_kev : tuple
        Energy range for fitting (keV). Default (2, 10) avoids noisy high-energy tail.

    Returns
    -------
    temperature_mk : float
        Best-fit temperature in MK.
    emission_measure : float
        Best-fit EM (scaled).
    chi2_red : float
        Reduced chi-squared of the fit.
    """
    if channel_energies_kev is None:
        channel_energies_kev = load_channel_energies()
    emin, emax = channel_energies_kev
    centroids = (emin + emax) / 2.0

    # Select channels in fitting range
    mask = (centroids >= fit_range_kev[0]) & (centroids <= fit_range_kev[1])
    if mask.sum() < 5:
        return 0.0, 0.0, 999.0

    e_fit = centroids[mask]
    counts_fit = counts_spectrum[mask].astype(float)
    # Add small positive constant to avoid zeros
    counts_fit = np.maximum(counts_fit, 1e-10)

    # Initial guess: T≈10 MK, EM scaled to match total counts
    total_counts = np.sum(counts_fit)
    em_guess = total_counts * 1e48 / np.sum(e_fit ** (-1))

    try:
        # Fit: f(E) = EM * E^(-1) * exp(-E/(T*0.086))
        def model(e, T_mk, EM):
            return EM * e ** (-1) * np.exp(-e / (T_mk * 0.086))

        # Scale EM to avoid overflow: work in log-space
        log_em_guess = np.log10(max(em_guess, 1e40))
        log_em_guess = np.clip(log_em_guess, 40, 52)

        popt, pcov = curve_fit(
            model,
            e_fit,
            counts_fit,
            p0=[10.0, 10**log_em_guess],
            bounds=([1.0, 1e40], [100.0, 1e55]),
            max_nfev=500,
        )
        T_mk, EM = popt
        # Clip EM to reasonable range
        EM = np.clip(EM, 1e40, 1e55)
        # Reduced chi-squared
        predicted = model(e_fit, T_mk, EM)
        resid = counts_fit - predicted
        chi2_red = np.sum(resid**2 / np.maximum(counts_fit, 1)) / max(
            len(counts_fit) - 2, 1
        )
        return float(T_mk), float(EM), float(chi2_red)
    except Exception:
        return 0.0, 0.0, 999.0


# ── Hardness Ratio ────────────────────────────────────────────────────


def compute_hardness_ratio(
    counts_bands: np.ndarray,
    band_edges_kev: list[tuple[float, float]],
    reference_band: int = 1,
) -> np.ndarray:
    """Compute hardness ratio HR = hi_band / lo_band for HEL1OS bands.

    Parameters
    ----------
    counts_bands : ndarray, shape (n_times, n_bands)
        Count rates per energy band.
    band_edges_kev : list of (lo, hi)
        Energy edges for each band.
    reference_band : int
        Index of the low-energy reference band (default 1 = 20-40 keV).

    Returns
    -------
    hr : ndarray, shape (n_times, n_bands - 1)
        Hardness ratios.
    """
    lo_band = counts_bands[:, reference_band]
    hr = np.zeros_like(counts_bands)
    for b in range(counts_bands.shape[1]):
        hr[:, b] = np.where(lo_band > 0, counts_bands[:, b] / lo_band, 0.0)
    return hr


# ── Spectral Index (Power-law) ────────────────────────────────────────


def fit_spectral_index(
    band_rates: np.ndarray,
    band_centroids: np.ndarray,
) -> float:
    """Estimate photon spectral index γ from multi-band count rates.

    Assumes power-law: I(E) ∝ E^(-γ). Uses least-squares fit on
    log(rate) vs log(E). For thin-target bremsstrahlung, the electron
    index δ = γ + 1.

    Parameters
    ----------
    band_rates : ndarray, shape (n_bands,)
        Count rates in each energy band.
    band_centroids : ndarray, shape (n_bands,)
        Geometric mean energy of each band (keV).

    Returns
    -------
    gamma : float
        Photon spectral index.
    """
    mask = (band_rates > 0) & np.isfinite(band_rates)
    if mask.sum() < 3:
        return 0.0

    log_e = np.log(band_centroids[mask])
    log_r = np.log(band_rates[mask])

    try:
        # Linear fit in log-log space
        A = np.vstack([log_e, np.ones_like(log_e)]).T
        coeffs, *_ = np.linalg.lstsq(A, log_r, rcond=None)
        gamma = -coeffs[0]  # negative slope
        return float(gamma)
    except Exception:
        return 0.0


# ── Neupert Correlation ──────────────────────────────────────────────


def neupert_correlation(
    sxr_counts: np.ndarray,
    hxr_counts: np.ndarray,
    window_sec: int = 300,
    step_sec: int = 60,
) -> np.ndarray:
    """Compute sliding Neupert correlation ρ = corr(dSXR/dt, HXR).

    The Neupert effect states dF_SXR/dt ∝ F_HXR. This function computes
    the Pearson correlation over sliding windows.

    Parameters
    ----------
    sxr_counts : ndarray
        SoLEXS count rate (1 s cadence).
    hxr_counts : ndarray
        HEL1OS count rate (1 s cadence, same length as sxr).
    window_sec : int
        Window size in seconds (default 300).
    step_sec : int
        Step size in seconds (default 60).

    Returns
    -------
    rho : ndarray
        Pearson r for each window. NaN where correlation is undefined.
    """
    if len(sxr_counts) != len(hxr_counts):
        min_len = min(len(sxr_counts), len(hxr_counts))
        sxr_counts = sxr_counts[:min_len]
        hxr_counts = hxr_counts[:min_len]

    # Compute dSXR/dt
    dsxr = np.diff(sxr_counts, prepend=sxr_counts[0])
    n = len(dsxr)
    rho = np.full(n, np.nan)

    for i in range(0, n - window_sec, step_sec):
        s = dsxr[i : i + window_sec]
        h = hxr_counts[i : i + window_sec]

        # Skip if either is constant
        if np.std(s) < 1e-10 or np.std(h) < 1e-10:
            continue

        try:
            r, _ = pearsonr(s, h)
            rho[i + window_sec // 2] = r
        except Exception:
            pass

    return rho


# ── Batch processing for feature engineering ──────────────────────────


def extract_spectral_features_from_pi(
    pi_counts: np.ndarray,
    channel_energies: tuple[np.ndarray, np.ndarray] | None = None,
    time_bin: int = 300,
) -> dict[str, np.ndarray]:
    """Extract spectral features from full-day PI data.

    Parameters
    ----------
    pi_counts : ndarray, shape (86400, 340)
        Full day of PI spectra.
    channel_energies : tuple, optional
        (emin, emax) arrays.
    time_bin : int
        Binning in seconds for spectral fitting (default 300 = 5 min).

    Returns
    -------
    features : dict
        Keys: 'temperature_mk', 'emission_measure', 'chi2_red', 'spectral_index'
        Each value is an array of shape (n_bins,).
    """
    if channel_energies is None:
        channel_energies = load_channel_energies()

    n_bins = pi_counts.shape[0] // time_bin
    temps = np.zeros(n_bins)
    ems = np.zeros(n_bins)
    chi2s = np.zeros(n_bins)

    for i in range(n_bins):
        start = i * time_bin
        end = min(start + time_bin, pi_counts.shape[0])
        summed_spec = np.nansum(pi_counts[start:end, :], axis=0)
        if np.sum(summed_spec) < 10:
            temps[i] = 0.0
            ems[i] = 0.0
            chi2s[i] = 999.0
            continue
        T, EM, chi2 = fit_temperature(summed_spec, channel_energies)
        temps[i] = T
        ems[i] = EM
        chi2s[i] = chi2

    return {
        "temperature_mk": temps,
        "emission_measure": ems,
        "chi2_red": chi2s,
    }
