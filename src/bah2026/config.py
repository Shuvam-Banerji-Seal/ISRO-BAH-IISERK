"""Central configuration: all paths, instrument constants, pipeline parameters."""

from __future__ import annotations

from pathlib import Path
import os
import json

# ── Paths ────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("BAH2026_DATA", PROJECT_ROOT / "data" / "processed"))
OUTPUT_ROOT = Path(os.environ.get("BAH2026_OUTPUT", PROJECT_ROOT / "output"))
CONFIG_FILE = PROJECT_ROOT / "bah2026_config.json"

PLOTS_DIR = OUTPUT_ROOT / "plots"
CATALOGS_DIR = OUTPUT_ROOT / "catalogs"
MODELS_DIR = OUTPUT_ROOT / "models"
HDF5_DIR = OUTPUT_ROOT / "hdf5"

PLOTS_OVERVIEW = PLOTS_DIR / "overview"
PLOTS_SPECTRAL = PLOTS_DIR / "spectral"
PLOTS_NOWCAST = PLOTS_DIR / "nowcast"
PLOTS_FORECAST = PLOTS_DIR / "forecast"
PLOTS_STATISTICS = PLOTS_DIR / "statistics"

# ── Instrument Constants ─────────────────────────────────────────────────

SOLEXS_ENERGY_KEV = (2.0, 22.0)
SOLEXS_CHANNELS = 340
SOLEXS_CADENCE_SEC = 1
SOLEXS_ROWS_PER_DAY = 86_400
# Calibration: SoLEXS→GOES is now handled by data/calibration.py
# The old SOLEXS_TO_GOES_SCALE = 1e-8 was physically wrong (fabricated constant)
# and produced a 22× error on the X6.3 flare (labeled M2.8).
# Use data.calibration.solexs_counts_to_irradiance_simple() instead.

CZT_BANDS: dict[str, tuple[float, float]] = {
    "CZT1_LC_BAND_20.00KEV_TO_40.00KEV": (20, 40),
    "CZT1_LC_BAND_40.00KEV_TO_60.00KEV": (40, 60),
    "CZT1_LC_BAND_60.00KEV_TO_80.00KEV": (60, 80),
    "CZT1_LC_BAND_80.00KEV_TO_150.00KEV": (80, 150),
    "CZT1_LC_BAND_18.00KEV_TO_160.00KEV": (18, 160),
}

CDTE_BANDS: dict[str, tuple[float, float]] = {
    "CDTE1_LC_BAND_5.00KEV_TO_20.00KEV": (5, 20),
    "CDTE1_LC_BAND_20.00KEV_TO_30.00KEV": (20, 30),
    "CDTE1_LC_BAND_30.00KEV_TO_40.00KEV": (30, 40),
    "CDTE1_LC_BAND_40.00KEV_TO_60.00KEV": (40, 60),
    "CDTE1_LC_BAND_1.80KEV_TO_90.00KEV": (1.8, 90),
}

# ── Nowcasting Parameters ────────────────────────────────────────────────

NOWCAST_THRESHOLD_SIGMA = 4.0
NOWCAST_MIN_DURATION_SEC = 240  # SWPC standard: 4-minute minimum duration
NOWCAST_BAYESIAN_BLOCKS_SIGMA = 3.5
NOWCAST_WAVELET_SIGMA = 3.5
NOWCAST_BACKGROUND_WINDOW_SEC = 600

# ── Feature Engineering Parameters ───────────────────────────────────────

FEATURE_LOOKBACK_SEC = 3600
FEATURE_STEP_SEC = 300
FEATURE_FORECAST_WINDOW_SEC = 1800
FEATURE_SPECTRAL_ENTROPY_NPERSEG = 256
FEATURE_AUTOCORR_LAGS = [5, 10, 30, 60]
FEATURE_PERCENTILES = [5, 25, 75, 95]

# ── Forecasting Parameters ───────────────────────────────────────────────

FORECAST_TRAIN_RATIO = 0.70
FORECAST_VAL_RATIO = 0.15
FORECAST_TEST_RATIO = 0.15
FORECAST_BINARY_THRESHOLD = 0.5

LGBM_N_ESTIMATORS = 1000
LGBM_LEARNING_RATE = 0.05
LGBM_MAX_DEPTH = 8
LGBM_VERBOSE = -1

XGB_N_ESTIMATORS = 1000
XGB_LEARNING_RATE = 0.05
XGB_MAX_DEPTH = 8

CATBOOST_ITERATIONS = 1000
CATBOOST_LEARNING_RATE = 0.05
CATBOOST_DEPTH = 8

CNNLSTM_INPUT_LEN = 3600
CNNLSTM_N_CHANNELS = 12
CNNLSTM_N_FEATURES = 179
CNNLSTM_LR = 1e-3
CNNLSTM_EPOCHS = 50
CNNLSTM_BATCH_SIZE = 64

# ── External Data Paths ─────────────────────────z────────────────────────────

GOES_DATA_DIR = Path(
    os.environ.get("BAH2026_GOES_DIR", str(PROJECT_ROOT / "data" / "external" / "goes"))
)
# ── Transformer / DL Parameters ─────────────────────────────────────────

TRANSFORMER_D_MODEL = 256
TRANSFORMER_NHEAD = 8
TRANSFORMER_NUM_LAYERS = 4
TRANSFORMER_LR = 5e-4
TRANSFORMER_WEIGHT_DECAY = 0.01
TRANSFORMER_EPOCHS = 100
TRANSFORMER_BATCH_SIZE = 256
TRANSFORMER_PATIENCE = 15
TRANSFORMER_FOCAL_GAMMA = 2.0
TRANSFORMER_FOCAL_ALPHA = 0.25
TRANSFORMER_LAMBDA_PHYS = 0.1
TRANSFORMER_SEQ_LEN = 360  # 1h at 10s cadence
TRANSFORMER_DOWNSAMPLE = 10  # 1s -> 10s

MAE_D_MODEL = 256
MAE_LR = 1e-3
MAE_WEIGHT_DECAY = 0.05
MAE_EPOCHS = 50
MAE_BATCH_SIZE = 512
MAE_MASK_RATIO = 0.5

# ── Download / Data Paths ────────────────────────────────────────────────

RAW_DATA_DIR = Path(os.environ.get("BAH2026_RAW", PROJECT_ROOT / "data" / "raw"))
PROCESSED_DATA_DIR = DATA_ROOT  # alias for backward compat
EXTERNAL_DATA_DIR = PROJECT_ROOT / "data" / "external"
GOES_DATA_DIR = EXTERNAL_DATA_DIR / "goes"

# SoLEXS/HEL1OS download settings
PRADAN_BASE_URL = "https://pradan1.issdc.gov.in"
PRADAN_SOLEXS_URL = f"{PRADAN_BASE_URL}/browse/Aditya-L1/SoLEXS"
PRADAN_HEL1OS_URL = f"{PRADAN_BASE_URL}/browse/Aditya-L1/HEL1OS"
SOLEXS_ZIP_DIR = RAW_DATA_DIR / "solexs"
HEL1OS_ZIP_DIR = RAW_DATA_DIR / "hel1os"
SOLEXS_PROCESSED_DIR = PROCESSED_DATA_DIR / "solexs"
HEL1OS_PROCESSED_DIR = PROCESSED_DATA_DIR / "hel1os"

# GOES download settings
GOES_BASE_URL = "https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/goes16/l2/data/xrsf-l2-flx1s"
GOES_START_DATE = "2024-02-01"
GOES_END_DATE = "2026-06-22"

# ── Parallelism ──────────────────────────────────────────────────────────

N_WORKERS = int(os.environ.get("BAH2026_WORKERS", min(os.cpu_count() or 4, 24)))
# Lazy GPU detection — defer CUDA init until model training
HAS_GPU = False
GPU_NAME = "none"
GPU_MEMORY_GB = 0.0


def detect_gpu() -> None:
    """Initialize GPU detection. Call before any GPU-accelerated code."""
    global HAS_GPU, GPU_NAME, GPU_MEMORY_GB
    if HAS_GPU:
        return
    try:
        import torch

        if torch.cuda.is_available():
            HAS_GPU = True
            GPU_NAME = torch.cuda.get_device_name(0)
            GPU_MEMORY_GB = torch.cuda.get_device_properties(0).total_memory / 1e9
    except Exception:
        pass


def has_gpu() -> bool:
    """Returns True if GPU is available. Call instead of importing HAS_GPU."""
    detect_gpu()
    return HAS_GPU


def gpu_info() -> tuple[str, float]:
    """Returns (name, memory_gb). Call instead of importing GPU_NAME/MEMORY_GB."""
    detect_gpu()
    return GPU_NAME, GPU_MEMORY_GB


USE_GPU = os.environ.get("BAH2026_GPU", "auto")

# ── Functions ────────────────────────────────────────────────────────────


def ensure_output_dirs() -> None:
    """Create all output subdirectories."""
    for d in [
        PLOTS_OVERVIEW,
        PLOTS_SPECTRAL,
        PLOTS_NOWCAST,
        PLOTS_FORECAST,
        PLOTS_STATISTICS,
        CATALOGS_DIR,
        MODELS_DIR,
        HDF5_DIR,
    ]:
        d.mkdir(parents=True, exist_ok=True)


def load_config() -> dict:
    """Load overrides from bah2026_config.json if it exists."""
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE) as f:
            return json.load(f)
    return {}


def save_default_config() -> None:
    """Write a default config file for user customization."""
    cfg = {
        "data_root": str(DATA_ROOT),
        "n_workers": N_WORKERS,
        "nowcast": {
            "threshold_sigma": NOWCAST_THRESHOLD_SIGMA,
            "min_duration_sec": NOWCAST_MIN_DURATION_SEC,
            "bayesian_blocks_sigma": NOWCAST_BAYESIAN_BLOCKS_SIGMA,
            "wavelet_sigma": NOWCAST_WAVELET_SIGMA,
            "background_window_sec": NOWCAST_BACKGROUND_WINDOW_SEC,
        },
        "features": {
            "lookback_sec": FEATURE_LOOKBACK_SEC,
            "step_sec": FEATURE_STEP_SEC,
            "forecast_window_sec": FEATURE_FORECAST_WINDOW_SEC,
        },
        "forecasting": {
            "train_ratio": FORECAST_TRAIN_RATIO,
            "val_ratio": FORECAST_VAL_RATIO,
            "lgbm_n_estimators": LGBM_N_ESTIMATORS,
            "lgbm_lr": LGBM_LEARNING_RATE,
        },
    }
    with open(CONFIG_FILE, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"Default config saved to {CONFIG_FILE}")
