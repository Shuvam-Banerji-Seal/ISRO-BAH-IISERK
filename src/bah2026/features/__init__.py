"""Feature engineering for solar flare forecasting.

Feature Engineering
-------------------
extract_features_window    — Extract 117 features from a time window
build_feature_matrix       — Sliding-window feature matrix over a full day
get_canonical_feature_names — Sorted list of all canonical feature names
pad_features_to_canonical  — Pad feature dict to match canonical list

Spectral Fitting
----------------
fit_temperature             — Thermal bremsstrahlung T (MK) from SoLEXS PI
fit_spectral_index          — Power-law γ from HEL1OS multi-band rates
compute_hardness_ratio      — Hardness ratio (hi/lo band) evolution
neupert_correlation         — Sliding ρ = corr(dSXR/dt, HXR)
extract_spectral_features_from_pi — Batch T/EM from PI spectra

Information Theory
------------------
transfer_entropy            — TE(HXR → SXR) causal flow
sample_entropy              — Signal complexity/regularity
mutual_information          — I(SXR; HXR) shared information
lagged_cross_correlation    — Optimal lag + max correlation

Non-thermal Fitting
-------------------
thick_target_spectrum       — Thick-target bremsstrahlung model
thermal_bremsstrahlung      — Thermal continuum model
fit_non_thermal             — Power-law fit: γ, Ec, N_nth
separate_thermal_non_thermal — Thermal + non-thermal decomposition
fit_combined_spectrum       — Combined SoLEXS + HEL1OS fit (2–150 keV)
compute_electron_column     — N(>Ec) from power-law parameters

QPP Detection
-------------
detect_qpp                  — Detect quasi-periodic pulsations
extract_qpp_features        — Sliding-window QPP features
wavelet_power               — Morlet wavelet power spectrum (FFT-based)
wavelet_power_gpu           — GPU-accelerated wavelet (PyTorch)
wavelet_power_auto          — Auto CPU/GPU selection
lomb_scargle_periodogram    — Lomb-Scargle for unevenly sampled data

Response Convolution
--------------------
build_response_matrix       — RMF × ARF instrument response
convolve_model              — Forward-fold model through response
deconvolve_spectrum         — Invert response (NNLS / Richardson-Lucy)
effective_area_at_energy    — ARF interpolation
counts_to_energy_flux       — Counts → erg/cm²/s
has_caldb                   — Check CALDB availability

Causal Network Analysis
-----------------------
granger_causality_simple        — AR-based Granger test with cross-validation
lagged_causal_correlation       — Optimal lag search for cause→effect
mediation_analysis              — Baron-Kenny mediation decomposition
build_causal_network            — Directed graph across all energy bands
extract_causal_network_features — 13 scalar features: density, cycles, lags
get_causal_feature_names        — List of causal feature names
"""

from bah2026.features.engineering import (
    extract_features_window,
    build_feature_matrix,
    get_canonical_feature_names,
    pad_features_to_canonical,
)
from bah2026.features.spectral_fitting import (
    fit_temperature,
    compute_hardness_ratio,
    fit_spectral_index,
    neupert_correlation,
    extract_spectral_features_from_pi,
)
from bah2026.features.information_theory import (
    transfer_entropy,
    sample_entropy,
    mutual_information,
    lagged_cross_correlation,
)
from bah2026.features.non_thermal import (
    thick_target_spectrum,
    thermal_bremsstrahlung,
    fit_non_thermal,
    separate_thermal_non_thermal,
    fit_combined_spectrum,
    compute_electron_column,
)
from bah2026.features.qpp import (
    detect_qpp,
    extract_qpp_features,
    wavelet_power,
    wavelet_power_gpu,
    wavelet_power_auto,
    lomb_scargle_periodogram,
)
from bah2026.features.response_convolution import (
    build_response_matrix,
    convolve_model,
    deconvolve_spectrum,
    effective_area_at_energy,
    counts_to_energy_flux,
    has_caldb,
)
from bah2026.features.causal_network import (
    granger_causality_simple,
    lagged_causal_correlation,
    mediation_analysis,
    build_causal_network,
    extract_causal_network_features,
    get_causal_feature_names,
)
from bah2026.features.advanced_features import (
    extract_all_advanced_features,
    get_advanced_feature_names,
    extract_temporal_derivatives,
    extract_multiscale_features,
    extract_goes_timeseries_features,
    extract_per_window_spectral,
    extract_wavelet_scalogram_features,
)

__all__ = [
    # Feature engineering
    "extract_features_window",
    "build_feature_matrix",
    "get_canonical_feature_names",
    "pad_features_to_canonical",
    # Spectral fitting
    "fit_temperature",
    "compute_hardness_ratio",
    "fit_spectral_index",
    "neupert_correlation",
    "extract_spectral_features_from_pi",
    # Information theory
    "transfer_entropy",
    "sample_entropy",
    "mutual_information",
    "lagged_cross_correlation",
    # Non-thermal fitting
    "thick_target_spectrum",
    "thermal_bremsstrahlung",
    "fit_non_thermal",
    "separate_thermal_non_thermal",
    "fit_combined_spectrum",
    "compute_electron_column",
    # QPP detection
    "detect_qpp",
    "extract_qpp_features",
    "wavelet_power",
    "wavelet_power_gpu",
    "wavelet_power_auto",
    "lomb_scargle_periodogram",
    # Response convolution
    "build_response_matrix",
    "convolve_model",
    "deconvolve_spectrum",
    "effective_area_at_energy",
    "counts_to_energy_flux",
    "has_caldb",
    # Causal network
    "granger_causality_simple",
    "lagged_causal_correlation",
    "mediation_analysis",
    "build_causal_network",
    "extract_causal_network_features",
    "get_causal_feature_names",
    # Advanced features (v3)
    "extract_all_advanced_features",
    "get_advanced_feature_names",
    "extract_temporal_derivatives",
    "extract_multiscale_features",
    "extract_goes_timeseries_features",
    "extract_per_window_spectral",
    "extract_wavelet_scalogram_features",
]
