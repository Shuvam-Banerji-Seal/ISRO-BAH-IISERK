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
from bah2026.models.cnn_lstm_v3 import (
    CNNLSTMv3,
    FocalLoss,
    FlareForecasterCNNLSTMv3,
    evaluate_model as evaluate_cnn_lstm,
    rolling_origin_cv as rolling_origin_cv_cnn,
)
from bah2026.models.transformer import (
    SpectralTemporalTransformer,
    NeupertLoss,
    FlareForecasterTransformer,
    PositionalEncoding,
    evaluate_transformer,
)
from bah2026.models.mae_pretrain import (
    MaskedAutoencoder,
    MAEEncoder,
    MAEDecoder,
    MAEPretrainer,
    prepare_pretraining_data,
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
    # v3 models
    "CNNLSTMv3",
    "FocalLoss",
    "FlareForecasterCNNLSTMv3",
    "evaluate_cnn_lstm",
    "rolling_origin_cv_cnn",
    "SpectralTemporalTransformer",
    "NeupertLoss",
    "FlareForecasterTransformer",
    "PositionalEncoding",
    "evaluate_transformer",
    "MaskedAutoencoder",
    "MAEEncoder",
    "MAEDecoder",
    "MAEPretrainer",
    "prepare_pretraining_data",
]
