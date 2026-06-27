"""Flare detection (nowcasting) and prediction (forecasting) models."""

from bah2026.models.nowcasting import (
    detect_flares_threshold,
    detect_flares_bayesian_blocks,
    detect_flares_wavelet,
    detect_flares_swpc,
    detect_flares_hel1os,
    coincidence_merge,
    classify_flare_goes,
    background_subtract_simple,
    background_subtract_iterative,
)
from bah2026.models.forecasting import (
    FlareForecasterLightGBM,
    FlareForecasterXGBoost,
    FlareForecasterCatBoost,
    FlareForecasterCNNLSTM,
)

__all__ = [
    "detect_flares_threshold",
    "detect_flares_bayesian_blocks",
    "detect_flares_wavelet",
    "detect_flares_swpc",
    "detect_flares_hel1os",
    "coincidence_merge",
    "classify_flare_goes",
    "background_subtract_simple",
    "background_subtract_iterative",
    "FlareForecasterLightGBM",
    "FlareForecasterXGBoost",
    "FlareForecasterCatBoost",
    "FlareForecasterCNNLSTM",
]
