"""Non-thermal spectral fitting for HEL1OS hard X-ray data.

Implements the thick-target bremsstrahlung model described in the
HEL1OS paper (Nandi et al. 2025, arXiv:2512.12679) and standard
solar flare HXR spectroscopy (Brown 1971, Holman et al. 2011).

Key physics:
  - Thermal component: bremsstrahlung from hot plasma (T ~ 10-40 MK)
  - Non-thermal component: power-law electron spectrum hitting chromosphere
  - Thick-target model: I(E) ∝ E^(-γ) where γ is the photon spectral index
  - Electron spectral index δ = γ + 1 (for thin target δ = γ - 1)
  - Cut-off energy E_c: low-energy cutoff of non-thermal electrons
  - Total non-thermal electron count N_nth above E_c

Functions:
  - thick_target_spectrum(energies, gamma, ec, norm)
  - fit_non_thermal(energies, counts, fit_range)
  - separate_thermal_non_thermal(energies, counts, t_mk, em)
  - compute_electron_column(gamma, ec, flux_norm)
  - fit_combined_spectrum(energies, counts, t_mk, em)
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit
from scipy.integrate import quad


# ── Physical constants ─────────────────────────────────────────────────

K_B_KEV = 0.08617  # Boltzmann constant in keV/MK
ELECTRON_MASS_KEV = 511.0  # electron rest mass in keV
THOMSON_XSEC_CM2 = 6.65e-25  # Thomson cross-section


# ── Thick-target bremsstrahlung ────────────────────────────────────────


def thick_target_spectrum(
    energies_kev: np.ndarray,
    gamma: float,
    ec_kev: float,
    norm: float,
) -> np.ndarray:
    """Thick-target bremsstrahlung photon spectrum.

    For a power-law electron distribution F(E0) = A * E0^(-delta)
    hitting a thick target, the photon spectrum is:
        I(E) ∝ E^(-(delta-1)) * integral from E to inf of E0^(-delta+1) dE0
             = E^(-(delta-1)) * E^(-delta+2) / (delta-2)   for delta > 2
             ∝ E^(-(2*gamma - 1))  simplified

    For the photon spectral index gamma (I(E) ∝ E^(-gamma)):
        delta = gamma + 1  (thick-target relation)

    Parameters
    ----------
    energies_kev : ndarray
        Photon energies (keV).
    gamma : float
        Photon spectral index (typically 2-7 for flares).
    ec_kev : float
        Low-energy cutoff of non-thermal electrons (keV).
    norm : float
        Normalization (arbitrary units).

    Returns
    -------
    spectrum : ndarray
        Photon flux at each energy.
    """
    if gamma <= 1.0:
        return np.zeros_like(energies_kev, dtype=np.float64)

    e = np.asarray(energies_kev, dtype=np.float64)
    # Simple power-law with low-energy cutoff
    spectrum = np.where(
        e >= ec_kev,
        norm * e ** (-gamma),
        norm * ec_kev ** (-gamma) * np.exp(-(ec_kev - e) / ec_kev),
    )
    return np.maximum(spectrum, 0.0)


def thermal_bremsstrahlung(
    energies_kev: np.ndarray,
    t_mk: float,
    em: float,
) -> np.ndarray:
    """Thermal bremsstrahlung continuum (isothermal).

    I(E) ∝ EM * E^(-1) * exp(-E / (kT))

    Parameters
    ----------
    energies_kev : ndarray
        Photon energies (keV).
    t_mk : float
        Plasma temperature in MK.
    em : float
        Emission measure (scaled).

    Returns
    -------
    spectrum : ndarray
    """
    kT = t_mk * K_B_KEV
    if kT <= 0:
        return np.zeros_like(energies_kev, dtype=np.float64)
    e = np.asarray(energies_kev, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        flux = em * e ** (-1) * np.exp(-e / kT)
    return np.nan_to_num(flux, nan=0.0, posinf=0.0, neginf=0.0)


# ── Non-thermal spectral fitting ───────────────────────────────────────


def fit_non_thermal(
    energies_kev: np.ndarray,
    counts: np.ndarray,
    fit_range_kev: tuple[float, float] = (20.0, 150.0),
    ec_guess: float = 10.0,
) -> dict:
    """Fit a non-thermal power-law to the hard X-ray spectrum.

    Parameters
    ----------
    energies_kev : ndarray
        Energy bin centroids (keV).
    counts : ndarray
        Count rates per energy bin.
    fit_range_kev : tuple
        Energy range for the power-law fit (above thermal contribution).
    ec_guess : float
        Initial guess for cutoff energy (keV).

    Returns
    -------
    result : dict
        Keys: 'gamma' (photon spectral index), 'ec' (cutoff energy),
        'norm' (normalization), 'chi2_red', 'delta' (electron index),
        'n_nth' (electron column above Ec).
    """
    mask = (
        (energies_kev >= fit_range_kev[0])
        & (energies_kev <= fit_range_kev[1])
        & (counts > 0)
        & np.isfinite(counts)
    )
    if mask.sum() < 4:
        return {
            "gamma": 0.0,
            "ec": 0.0,
            "norm": 0.0,
            "chi2_red": 999.0,
            "delta": 0.0,
            "n_nth": 0.0,
        }

    e_fit = energies_kev[mask]
    c_fit = counts[mask].astype(float)
    c_fit = np.maximum(c_fit, 1e-10)

    try:

        def model(e, gamma, ec, norm):
            return thick_target_spectrum(e, gamma, ec, norm)

        popt, pcov = curve_fit(
            model,
            e_fit,
            c_fit,
            p0=[4.0, ec_guess, np.max(c_fit) * e_fit[0] ** 4],
            bounds=([1.5, 5.0, 1e-10], [10.0, 50.0, 1e10]),
            max_nfev=500,
        )
        gamma, ec, norm = popt

        # Chi-squared
        predicted = model(e_fit, *popt)
        resid = c_fit - predicted
        chi2_red = float(
            np.sum(resid**2 / np.maximum(c_fit, 1)) / max(len(c_fit) - 3, 1)
        )

        # Electron spectral index (thick-target: delta = gamma + 1)
        delta = gamma + 1.0

        # Electron column above Ec: N(>Ec) = norm / (delta-1) * Ec^(-(delta-1))
        # (simplified, in arbitrary units)
        if delta > 1.0 and ec > 0:
            n_nth = norm / (delta - 1.0) * ec ** (-(delta - 1.0))
        else:
            n_nth = 0.0

        return {
            "gamma": float(gamma),
            "ec": float(ec),
            "norm": float(norm),
            "chi2_red": chi2_red,
            "delta": float(delta),
            "n_nth": float(n_nth),
        }
    except Exception:
        return {
            "gamma": 0.0,
            "ec": 0.0,
            "norm": 0.0,
            "chi2_red": 999.0,
            "delta": 0.0,
            "n_nth": 0.0,
        }


# ── Thermal / Non-thermal separation ───────────────────────────────────


def separate_thermal_non_thermal(
    energies_kev: np.ndarray,
    counts: np.ndarray,
    t_mk: float,
    em: float,
    thermal_range_kev: tuple[float, float] = (5.0, 20.0),
    nonthermal_range_kev: tuple[float, float] = (30.0, 150.0),
) -> dict:
    """Separate thermal and non-thermal components of an HXR spectrum.

    Strategy:
      1. Fit thermal bremsstrahlung to the low-energy range (5-20 keV)
      2. Subtract thermal model from full spectrum
      3. Fit non-thermal power-law to the residual above 30 keV

    Parameters
    ----------
    energies_kev : ndarray
        Energy bin centroids (keV).
    counts : ndarray
        Count rates per energy bin.
    t_mk : float
        Initial temperature guess (MK).
    em : float
        Initial emission measure guess.
    thermal_range_kev : tuple
        Energy range for thermal fit.
    nonthermal_range_kev : tuple
        Energy range for non-thermal fit.

    Returns
    -------
    result : dict
        Keys: 't_mk', 'em', 'gamma', 'ec', 'n_nth',
        'thermal_flux', 'nonthermal_flux', 'residual',
        'thermal_fraction' (fraction of total flux that is thermal).
    """
    # Step 1: Fit thermal model
    t_fit, em_fit = t_mk, em
    try:
        thermal_mask = (
            (energies_kev >= thermal_range_kev[0])
            & (energies_kev <= thermal_range_kev[1])
            & (counts > 0)
        )
        if thermal_mask.sum() >= 4:
            e_thermal = energies_kev[thermal_mask]
            c_thermal = counts[thermal_mask].astype(float)
            c_thermal = np.maximum(c_thermal, 1e-10)

            def thermal_model(e, T, EM):
                return thermal_bremsstrahlung(e, T, EM)

            popt, _ = curve_fit(
                thermal_model,
                e_thermal,
                c_thermal,
                p0=[max(t_mk, 5.0), max(em, 1e-3)],
                bounds=([1.0, 1e-10], [100.0, 1e10]),
                max_nfev=300,
            )
            t_fit, em_fit = float(popt[0]), float(popt[1])
    except Exception:
        pass

    # Step 2: Compute thermal model and subtract
    thermal_flux = thermal_bremsstrahlung(energies_kev, t_fit, em_fit)
    residual = np.maximum(counts - thermal_flux, 0.0)

    # Step 3: Fit non-thermal to residual
    nt_result = fit_non_thermal(
        energies_kev, residual, fit_range_kev=nonthermal_range_kev
    )

    nonthermal_flux = thick_target_spectrum(
        energies_kev, nt_result["gamma"], nt_result["ec"], nt_result["norm"]
    )

    # Thermal fraction (in 5-20 keV range)
    thermal_mask = (energies_kev >= 5.0) & (energies_kev <= 20.0)
    total_in_range = np.sum(counts[thermal_mask])
    thermal_in_range = np.sum(thermal_flux[thermal_mask])
    thermal_fraction = (
        float(thermal_in_range / max(total_in_range, 1e-10))
        if total_in_range > 0
        else 0.0
    )
    # Clip to physical range [0, 1]
    thermal_fraction = float(np.clip(thermal_fraction, 0.0, 1.0))

    return {
        "t_mk": t_fit,
        "em": em_fit,
        "gamma": nt_result["gamma"],
        "ec": nt_result["ec"],
        "n_nth": nt_result["n_nth"],
        "delta": nt_result["delta"],
        "thermal_flux": thermal_flux,
        "nonthermal_flux": nonthermal_flux,
        "residual": residual,
        "thermal_fraction": thermal_fraction,
        "nonthermal_chi2": nt_result["chi2_red"],
    }


# ── Combined spectrum (SoLEXS + HEL1OS) ────────────────────────────────


def fit_combined_spectrum(
    solexs_energies: np.ndarray,
    solexs_counts: np.ndarray,
    hel1os_energies: np.ndarray,
    hel1os_counts: np.ndarray,
    t_mk_init: float = 10.0,
) -> dict:
    """Fit a combined thermal + non-thermal model to SoLEXS + HEL1OS data.

    SoLEXS covers 2-22 keV (thermal domain), HEL1OS covers 10-150 keV
    (non-thermal domain). Together they span 2-150 keV.

    Parameters
    ----------
    solexs_energies : ndarray
        SoLEXS channel centroids (keV).
    solexs_counts : ndarray
        SoLEXS counts per channel.
    hel1os_energies : ndarray
        HEL1OS band centroids (keV).
    hel1os_counts : ndarray
        HEL1OS counts per band.
    t_mk_init : float
        Initial temperature guess (MK).

    Returns
    -------
    result : dict
        Combined fit results: T, EM, gamma, ec, n_nth, thermal_fraction.
    """
    # Combine energy grids
    all_energies = np.concatenate([solexs_energies, hel1os_energies])
    all_counts = np.concatenate([solexs_counts, hel1os_counts])

    # Sort by energy
    sort_idx = np.argsort(all_energies)
    all_energies = all_energies[sort_idx]
    all_counts = all_counts[sort_idx]

    # Mask valid
    mask = (all_counts > 0) & np.isfinite(all_counts) & (all_energies > 1.0)
    if mask.sum() < 10:
        return {
            "t_mk": 0.0,
            "em": 0.0,
            "gamma": 0.0,
            "ec": 0.0,
            "n_nth": 0.0,
            "thermal_fraction": 0.0,
            "combined_range_kev": (0, 0),
        }

    e_fit = all_energies[mask]
    c_fit = all_counts[mask]

    # Fit thermal to low energy, non-thermal to high energy
    result = separate_thermal_non_thermal(e_fit, c_fit, t_mk_init, np.max(c_fit) * 10.0)

    return {
        "t_mk": result["t_mk"],
        "em": result["em"],
        "gamma": result["gamma"],
        "ec": result["ec"],
        "n_nth": result["n_nth"],
        "thermal_fraction": result["thermal_fraction"],
        "combined_range_kev": (float(e_fit[0]), float(e_fit[-1])),
    }


# ── Electron column computation ─────────────────────────────────────────


def compute_electron_column(
    gamma: float,
    ec_kev: float,
    flux_norm: float,
    delta: float | None = None,
) -> float:
    """Compute total non-thermal electron column N(>Ec).

    For a power-law electron spectrum F(E) = A * E^(-delta):
        N(>Ec) = A * integral from Ec to inf of E^(-delta) dE
               = A * Ec^(-(delta-1)) / (delta-1)   for delta > 1

    Parameters
    ----------
    gamma : float
        Photon spectral index.
    ec_kev : float
        Low-energy cutoff (keV).
    flux_norm : float
        Normalization from the power-law fit.
    delta : float, optional
        Electron spectral index (default: gamma + 1 for thick-target).

    Returns
    -------
    n_nth : float
        Total electron column above Ec (arbitrary units).
    """
    if delta is None:
        delta = gamma + 1.0
    if delta <= 1.0 or ec_kev <= 0:
        return 0.0
    return flux_norm * ec_kev ** (-(delta - 1.0)) / (delta - 1.0)
