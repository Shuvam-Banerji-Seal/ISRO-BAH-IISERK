"""Feature engineering for solar flare forecasting."""

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

__all__ = [
    "extract_features_window",
    "build_feature_matrix",
    "get_canonical_feature_names",
    "pad_features_to_canonical",
    "fit_temperature",
    "compute_hardness_ratio",
    "fit_spectral_index",
    "neupert_correlation",
    "extract_spectral_features_from_pi",
    "transfer_entropy",
    "sample_entropy",
    "mutual_information",
    "lagged_cross_correlation",
]
