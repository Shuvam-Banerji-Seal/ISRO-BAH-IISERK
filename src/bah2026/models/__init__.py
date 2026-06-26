"""Flare detection (nowcasting) and prediction (forecasting) models."""

from bah2026.models.nowcasting import (
    detect_flares_threshold,
    detect_flares_bayesian_blocks,
    detect_flares_wavelet,
    classify_flare_goes,
    background_subtract_simple,
)
from bah2026.models.forecasting import (
    FlareForecasterLightGBM,
    FlareForecasterXGBoost,
    FlareForecasterCatBoost,
    FlareForecasterCNNLSTM,
)

__all__ = [
    "detect_flares_threshold", "detect_flares_bayesian_blocks",
    "detect_flares_wavelet", "classify_flare_goes", "background_subtract_simple",
    "FlareForecasterLightGBM", "FlareForecasterXGBoost",
    "FlareForecasterCatBoost", "FlareForecasterCNNLSTM",
]
