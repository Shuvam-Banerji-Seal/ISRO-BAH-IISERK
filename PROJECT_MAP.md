# PROJECT_MAP — ISRO-BAH-IISERK Codebase Reference

## Top-Level Files

| File | Purpose |
|------|---------|
| `AGENTS.md` | Problem statement, dataset specs, repository structure, technical documentation |
| `README.md` | Project overview, quick start, badges |
| `pyproject.toml` | Dependencies (NumPy, SciPy, PyTorch, etc.), entry points, build config |
| `.python-version` | Python version pin |
| `.gitignore` | Ignores data/, .pt, .npy, artifacts |
| `STAGE3.md` | This file — Stage 3 deep learning pipeline guide |
| `PROJECT_MAP.md` | This file — codebase reference |

## `src/bah2026/` — Main Package

### `__init__.py`
- Package-level docstring with Stage 3 architecture overview
- Re-exports `__version__`
- Description: "CNN-LSTM (feature injection) + Transformer + MAE + Deep Ensemble"

### `config.py`
- All constants: paths, hyperparameters, energy bands, data dimensions
- **Key exports:**
  - `PROJECT_ROOT`, `DATA_ROOT`, `OUTPUT_ROOT` — directory paths
  - `HDF5_DIR`, `CATALOGS_DIR`, `MODELS_DIR`, `PLOTS_DIR` — output directories
  - `SOLEXS_CHANNELS=340`, `SOLEXS_CADENCE_SEC=1` — instrument params
  - `CZT_BANDS`, `CDTE_BANDS` — energy band definitions
  - `NOWCAST_*` — nowcast thresholds
  - `FEATURE_*` — feature extraction windows
  - `CNNLSTM_N_FEATURES=179` — feature vector dimension
  - `CNNLSTM_EPOCHS=200`, `TRANSFORMER_EPOCHS=100` — training params
  - `N_WORKERS` — CPU worker count
- Functions: `detect_gpu()`, `ensure_output_dirs()`, `has_gpu()`, `load_config()`, `save_default_config()`
- GPU constants: `GPU_MEM_GB=4`, `GPU_MAX_BATCH=256` for RTX 2050

### `main.py`
- CLI entry point (`bah2026` command)
- Subcommands: `nowcast`, `features`, `train`, `dashboard`, `pipeline`
- `cmd_nowcast()` — wrapper for phase 3 pipeline
- `cmd_features()` — wrapper for phase 4 feature extraction
- `cmd_train()` — runs `run_master_pipeline --phase 6-9`

### `data/` — Data Loading & Preprocessing

#### `__init__.py`
- Re-exports all data submodule functions

#### `reader.py`
- `load_solexs_lc(date)` → dict with time, counts, headers
- `load_solexs_pi(date)` → dict with time, energy, counts
- `load_solexs_gti(date)` → good time intervals
- `load_hel1os_lc(date, detector, unit)` → dict with MJD, count rates per band
- `load_hel1os_spectra(date, detector, unit)` → spectra BinTables
- `load_hel1os_hk(date)` → housekeeping data
- `load_hel1os_gti(date)` → per-orbit GTIs
- `discover_solexs_days()`, `discover_hel1os_days()`, `discover_combined_days()` → date lists
- All FITS loading with astropy, error-handled

#### `preprocessing.py`
- `align_hel1os_to_solexs()` — interpolates HXR data to SoLEXS time grid
- `met_to_mjd()` — mission elapsed time → Modified Julian Date
- `background_subtract()` — iterative sigma-clipping
- `interpolate_to_common_grid()` — resample to uniform cadence
- `compute_gti_mask()` — apply good time intervals

#### `corrections.py`
- `correct_solexs_deadtime(counts)` — deadtime correction for SDD2
- `subtract_hel1os_background(ctr, detector)` — background subtraction per energy band
- `subtract_solexs_spurious()` — removes electronic artefacts
- `correct_hel1os_deadtime_approx()` — approximate deadtime for HXR
- `apply_all_corrections()` — pipeline: all corrections in sequence

#### `calibration.py`
- `solexs_counts_to_irradiance_simple(counts)` → W/m² (×2.5e-8 factor)
- `classify_goes(flux_wm2)` → X/M/C/B/A class
- `load_channel_energies()`, `load_arf()`, `load_rmf()` — spectral response
- `calibrate_day(date)` — full spectral calibration with forward model

#### `ground_truth.py`
- `load_swpc_flares()` — loads SWPC GOES flare list CSV
- `validate_nowcasting(nowcast_csv, swpc_list)` — matching within ±5min
- `parse_goes_class(string)` → numeric GOES class (M5.0 → 5e-5)

#### `hdf5_builder.py`
- `build_hdf5()` — writes SoLEXS + HEL1OS data to HDF5 for fast random access
- Chunked storage (1 day per group), compression (lzf)

#### `sequence_builder.py`
- **`build_day_sequence(date, event_times)`** → `(X, y)`:
  - 12-channel tensor: SoLEXS SXR (ch0), 5 CZT bands (ch1-4,8), 3 CdTe bands (ch5-7), GOES XRS-B/A (ch9-10), dSXR/dt (ch11)
  - Sliding windows: lookback=3600s, step=300s → ~276 windows/day
  - Label: positive if flare peak in `[window_end, window_end + forecast_window)`
- **`build_all_sequences(days, event_times_map, output_dir)`** — multiprocess pool, memory-mapped .npy output
- **`SequenceDataset(x_path, y_path, indices, augment, features_path)`** — PyTorch Dataset with mmap
  - Optional feature loading: returns `(x, f, y)` when `features_path` provided
  - Augmentation: time-shift, Gaussian noise, channel dropout (2/12)
- **`create_dataloaders(x_path, y_path, features_path, train_idx, val_idx, test_idx, batch_size)`** — returns dict of train/val/test DataLoaders

#### `hel1os_concat.py`
- Standalone script: concatenates HEL1OS multi-orbit days into continuous 24h coverage
- Entry point: `bah2026-concat-hel1os`

### `features/` — Feature Engineering (179 Features)

#### `__init__.py`
- Re-exports extraction functions

#### `extract.py`
- **`extract_features(solexs_data, hel1os_data, goes_data)`** — main entry, returns 179-vector
- Sub-extractors:
  - `_time_domain()` — mean, std, skew, kurtosis, min, max, percentiles (28)
  - `_autocorr()` — lags 1–20 (20)
  - `_spectral()` — FFT dominant freq, spectral entropy, band power ratios (20)
  - `_wavelet()` — DWT Daubechies 4, scales 1–4, energy per scale (16)
  - `_cross_corr()` — SXR vs each of 5 HXR bands, max corr + lag (8)
  - `_flare_history()` — time since last flare, flare density, class (12)
  - `_physics()` — Neupert slope, EM, temperature, cooling time, nonthermal params (75)
- Multiprocessing support via `extract_all_parallel()`

#### `feature_names.py`
- Returns canonical list of 179 feature names (strings)

### `models/` — ML Models

#### `__init__.py`
- Re-exports from all submodules
- Package docstring: "CNN-LSTM v3 + Transformer + MAE + Deep Ensemble"

#### `nowcasting.py`
- `background_subtract_simple(counts, window)` — running median subtraction
- `detect_flares_threshold(counts, time, sigma)` — peak-over-threshold
- `detect_flares_bayesian_blocks(counts, time)` — Bayesian changepoint detection
- `classify_flare_goes(peak_counts)` → X/M/C/B/A class

#### `forecasting.py`
- **`FocalLoss(gamma, alpha, pos_weight)`** — focal loss for binary classification
- **`FlareForecasterCNNLSTM`** — v1 CNN-LSTM (baseline):
  - Conv1D(12→32→64) + LSTM(64→32) + Dense(32→1)
  - `fit()`, `predict_proba()`, `save()`, `load()`

#### `cnn_lstm_v3.py` — CNN-LSTM v3 with Feature Injection
- **`FocalLoss`** — shared focal loss implementation
- **`TemporalAttention(hidden_dim)`** — additive attention over time dim
- **`CNNLSTMv3(n_channels, seq_len, n_features)`** — core model:
  - 4-stage Conv1D (32→64→128→256) + BatchNorm + GELU + AdaptiveAvgPool
  - Optional `feature_proj`: Linear(179→128) + LayerNorm + GELU (when n_features > 0)
  - Forward: `conv(x)` → `permute` → `cat([conv_out, feature_proj(f)])` → `BiLSTM` → `Attention` → `Head`
  - `forward(x, features=None)` — features injected after Conv1D, before BiLSTM
- **`evaluate_model(model, loader, device)`** — full eval with TSS, AUC, F1
  - Handles `(x, y)` and `(x, f, y)` loaders
- **`FlareForecasterCNNLSTMv3(n_channels, seq_len, n_features, lr, ...)`** — training wrapper:
  - AdamW + CosineAnnealingWarmRestarts
  - Mixed precision (bfloat16), gradient clipping
  - Early stopping (patience=10)
  - `fit()` handles 2/3-element loader tuples
  - `predict_proba()`, `predict_proba_dataloader()`
  - Checkpoint save/load with full config + history
- **`rolling_origin_cv(X, y, n_folds, epochs)`** — expanding-window CV

#### `transformer.py` — Spectral-Temporal Transformer
- Rotary positional embeddings
- GEGLU feed-forward
- Neupert physics loss term
- Multi-head self-attention with causal masking
- Deep Ensemble support (5 seeds)

#### `mae_pretrain.py` — Masked Autoencoder
- Encoder: Transformer (same as above)
- Decoder: lightweight 2-layer Transformer
- Masking: random 75% patch masking
- Reconstruction loss + NMI prediction + Neupert proxy
- Output: saved encoder weights for Transformer fine-tuning

### `scripts/` — Pipeline Runners

#### `run_master_pipeline.py`
- **Single-file orchestrator** controlling all 10 phases (0–9)
- Each phase: idempotent (checks for output before running), skip-able, force-able
- Phases:
  0. Download GOES (NOAA public data)
  1. Download SoLEXS/HEL1OS (PRADAN)
  2. Decompress + preprocess
  3. Nowcasting (SXR+HXR combined)
  4. Feature extraction (179 features)
  5. Sequence building (12-ch × 3600s windows)
  6. CNN-LSTM v3 training (feature injection)
  7. MAE self-supervised pretraining
  8. Transformer training
  9. Deep Ensemble + evaluation
- `_make_loader_from_npy()` — creates TensorDataset with optional features
- Argparse: `--phase`, `--skip-*`, `--force-phase`, `--fetch-goes-date`

#### `run_v3_pipeline.py`
- Stage 3 pipeline (6 phases, renumbered):
  1. GPU feature extraction (179 features)
  2. Sequence building via `build_all_sequences`
  3. CNN-LSTM training (with feature injection)
  4. MAE pretraining
  5. Transformer training
  6. Ensemble (delegated to master pipeline)
- `--skip-*` flags for all phases

#### `run_full_pipeline.py`
- Legacy v2 pipeline: nowcast → features
- GBDT removed; delegates to master for DL training

#### `run_pipeline.py`
- Legacy step runner: nowcast → features
- GBDT removed; delegates to master for DL training

#### `verify_pipeline.py`
- Integrity checks: data split consistency, nowcast catalogue validation
- Verifies no overlapping windows between train/val/test

#### `build_goes_catalog.py`
- Downloads GOES XRS CSV from SWPC
- Parses flare list, normalises classes, writes `goes_flare_catalog.csv`
- Entry point: `bah2026-build-goes`

#### `extract_aux_files.py`
- Extracts auxiliary FITS files (spectra, event lists, housekeeping)
- Not used by main pipeline

#### `analyze_unused_data.py`
- Scans processed data, identifies unused files, estimates disk waste

### `visualization/`

#### `dashboard.py`
- Streamlit dashboard entry point
- Tabs: Overview, Nowcasting, Forecasting, Spectral Analysis
- Real-time light curve plot (SXR + HXR bands)
- Flare event markers (nowcasted + forecasted)
- Probability gauge for next 60-min window
- Runs with: `uv run streamlit run src/bah2026/visualization/dashboard.py`

#### `plots.py`
- Matplotlib plot generators: light curves, confusion matrices, reliability diagrams
- Saved to `outputs/plots/` subdirectories

### `downloads/` (in `data/`)

| Script | Purpose |
|--------|---------|
| `download_solexs.sh` | wget SoLEXS files from PRADAN |
| `download_hel1os.sh` | wget HEL1OS files from PRADAN |
| `parallel_dl.sh` | 8-worker parallel wget |
| `fast_dl.py` | Python 10-worker urllib downloader |
| `cookie_grabber.py` | Extract/refresh browser cookies for auth |
| `download_manager.sh` | Sequential download with auto-retry |
| `bulk_dl_browser.py` | Playwright browser-based downloader |
| `decompress.sh` | Unzip structured data |

## `tests/` — Unit Tests

| File | Tests |
|------|-------|
| `test_models.py` | Nowcasting: bg subtraction, threshold/Bayesian detection, GOES classification. Forecasting: CNN-LSTM v1 training. |
| `test_config.py` | All constants validated, directory structure, save/load config |
| `test_features.py` | Feature extraction dimensions, value ranges |
| `test_reader.py` | FITS loading, data integrity |
| `test_metrics.py` | TSS, HSS, skill score calculations |

## `docs/` — Documentation

| File | Content |
|------|---------|
| `PLAN.md` | 60-paper lit review, 8 novel contributions, mathematical formulations |
| `RESULTS.md` | Historical metrics: nowcast catalogue, v2 GBDT ensemble results |
| `analysis/01_data_exploration.md` | FITS exploration, band analysis |
| `analysis/02_nowcasting_pipeline.md` | Real-time detection architecture |
| `analysis/03_forecasting_pipeline.md` | Predictive model design (v2, GBDT-era) |
| `analysis/04_visualization_dashboard.md` | Streamlit dashboard design |
| `analysis/notes_solexs.md` | SoLEXS deep-dive: 747 days, gaps, quality |
| `analysis/notes_hel1os.md` | HEL1OS deep-dive: 902 days, coverage |
| `analysis/solexs_inventory.md` | Full inventory: headers, shapes, sizes |
| `analysis/hel1os_inventory.md` | Full inventory: bands, spectra, detectors |
| `analysis/combined_coverage.md` | 724 days dual-instrument overlap |

## Key Data Paths

| Path | Content |
|------|---------|
| `data/processed/solexs/YYYY/MM/DD/` | Extracted SoLEXS light curves + spectra |
| `data/processed/hel1os/YYYY/MM/DD/` | Extracted HEL1OS light curves + spectra |
| `data/external/goes/` | GOES XRS netCDF files |
| `outputs/hdf5/sequences/X_seq.npy` | 12-channel sequence windows (N, 12, 3600) |
| `outputs/catalogs/nowcast_catalogue.csv` | Combined SXR+HXR flare detections |
| `outputs/models/cnn_lstm_v3_best.pt` | Trained CNN-LSTM v3 checkpoint |
| `outputs/models/mae_encoder.pt` | MAE pretrained encoder |
| `outputs/models/transformer_best.pt` | Trained Transformer |
| `outputs/models/ensemble_metrics.json` | Deep ensemble evaluation |
