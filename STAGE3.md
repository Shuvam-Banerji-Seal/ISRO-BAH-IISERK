# Stage 3 — Deep Learning Ensemble for Solar Flare Forecasting

## Overview

Stage 3 replaces the GBDT (LightGBM/XGBoost/CatBoost) ensemble with a pure deep-learning pipeline combining three complementary architectures:

1. **CNN-LSTM v3** (Phase 6) — Conv1D feature extractor + BiLSTM + temporal attention, with **179 handcrafted features** injected after Conv1D, before BiLSTM
2. **Spectral-Temporal Transformer** (Phase 8) — attention-based model processing full 3600-step sequences, optionally initialized from MAE encoder weights
3. **MAE Self-Supervised Pretraining** (Phase 7) — Masked Autoencoder that learns representations by reconstructing masked time patches; encoder reused by Transformer

All three models produce per-window flare probabilities, combined in a **deep ensemble** (Phase 9) for calibrated uncertainty quantification via MC Dropout and multi-seed aggregation.

---

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│  Phase 0–2: Download, Preprocess (GOES + PRADAN)                   │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Phase 3: Nowcasting (combined SXR+HXR detection)                  │
│  Output: nowcast_catalogue.csv                                     │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Phase 4: Feature Extraction (179 tabular features per window)     │
│  Output: X_features.npy (N × 179), y_labels.npy (N,)              │
│  fnames: 117 time-domain + 20 freq-domain + 42 physics-based       │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Phase 5: Sequence Building (12-channel × 3600-step windows)       │
│  Output: sequences/X_seq.npy, sequences/y_seq.npy                  │
│  Strategy: rolling windows, step=300, lookback=3600s               │
└─────────────────────────────────────────────────────────────────────┘
                                    │
          ┌─────────────────────────┼─────────────────────────┐
          ▼                         ▼                         ▼
┌─────────────────────┐ ┌─────────────────────┐ ┌─────────────────────┐
│  Phase 6            │ │  Phase 7            │ │  Phase 8            │
│  CNN-LSTM v3        │ │  MAE Pretraining    │ │  Spectral-Temporal  │
│  (feature injection)│ │  (self-supervised)  │ │  Transformer        │
│  ─────────────      │ │  ─────────────      │ │  ─────────────      │
│  Conv1D(12→256)     │ │  Mask 75% patches   │ │  Rotary emb +       │
│  + concat(179→128)  │ │  Encoder → Decoder  │ │  GEGLU FFN          │
│  + BiLSTM(384→512)  │ │  + Neupert loss     │ │  + Neupert loss     │
│  + Attention → Head │ │  + NMI proxy task   │ │  + MAE init opt.    │
│  Output: probability │ │  Output: encoder.pt │ │  Output: probability │
└─────────────────────┘ └─────────────────────┘ └─────────────────────┘
          │                       │                       │
          └───────────────────────┼───────────────────────┘
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Phase 9: Deep Ensemble                                             │
│  - 5 seeds per architecture (→ 15 models)                            │
│  - MC Dropout with 50 forward passes                                 │
│  - Uncertainty: std across ensemble + MC Dropout variance            │
│  - Threshold optimisation per model via TSS on held-out val          │
│  Output: ensemble_predictions.csv, ensemble_metrics.json            │
└─────────────────────────────────────────────────────────────────────┘
```

---

## How to Run

### One-command full run (recommended for server):
```bash
uv run python -m bah2026.scripts.run_master_pipeline --phase 0-9
```

### Selective phases:
```bash
# Train only (assumes features + sequences exist)
uv run python -m bah2026.scripts.run_master_pipeline --phase 6-9

# CNN-LSTM only
uv run python -m bah2026.scripts.run_master_pipeline --phase 6

# Transformer only (with MAE init if available)
uv run python -m bah2026.scripts.run_master_pipeline --skip-cnn --skip-mae --phase 8

# Features + sequences only
uv run python -m bah2026.scripts.run_master_pipeline --phase 4-5
```

### Skip flags:
```bash
--skip-download     --skip-preprocess   --skip-nowcast
--skip-features     --skip-sequences    --skip-cnn
--skip-mae          --skip-transformer  --skip-ensemble
```

### Force re-run:
```bash
uv run python -m bah2026.scripts.run_master_pipeline \
    --force-phase 4 --force-phase 6
```

---

## Phase Details

### Phase 6 — CNN-LSTM v3 with Feature Injection

**Architecture:**
```
Input: (B, 12, 3600) time-series
  │
  ├── Conv1D block (32→64→128→256) + BatchNorm + GELU + MaxPool
  │     AdaptiveAvgPool1d(32) → (B, 256, 32) → permute → (B, 32, 256)
  │
  ├── Feature Projection: Dense(179 → 128) + LayerNorm + GELU
  │     expand: (B, 128) → (B, 32, 128)  ← tiled across all 32 timesteps
  │
  ├── Concat: (B, 32, 256) + (B, 32, 128) → (B, 32, 384)
  │
  ├── BiLSTM(384 → 256, 2 layers, dropout=0.3, bidirectional) → (B, 512)
  │
  ├── TemporalAttention(512 → pool → 512)
  │
  └── MLP Head: Linear(512→256→128→1) + GELU + Dropout(0.4→0.3) + Sigmoid
```

**Key design choices:**
- Features injected **after Conv1D, before BiLSTM** — keeps high-level CNN features and tiled features available at every LSTM timestep (Pattern B from literature)
- 179 features projected through `Dense(128)→LayerNorm→GELU`, tiled to `(B, 32, 128)` to match the 32 pooled Conv1D timesteps
- Focal Loss (γ=2.0, α=0.25) with `pos_weight=16.0` for ~6% positive rate
- CosineAnnealingWarmRestarts (T₀=10, T_mult=2) + gradient clipping (max_norm=1.0)

**Inputs:** `sequences/X_seq.npy`, `sequences/y_seq.npy`, `X_features.npy`, `y_labels.npy`

**Output:** `models/cnn_lstm_v3_best.pt` + `models/cnn_lstm_results.json`

### Phase 7 — MAE Self-Supervised Pretraining

**Architecture:**
- Encoder: Transformer with rotary positional embeddings, GEGLU FFN
- Decoder: lightweight Transformer (2 layers × 128 hidden)
- Masking: 75% of temporal patches masked
- Reconstruction target: normalized patch values
- Auxiliary heads: **NMI prediction** (flare occurrence proxy), **Neupert energy reconstruction** (physics-informed)

**Output:** `models/mae_encoder.pt` (encoder weights only)

### Phase 8 — Spectral-Temporal Transformer

**Architecture:**
- Same encoder as MAE (weights initialized from MAE when available)
- Rotary positional embeddings for time-aware attention
- Learnable class token + MLP head
- **Neupert physics loss** — penalises predictions inconsistent with Neupert effect
- Deep Ensemble (5 seeds) + MC Dropout (50 passes) for uncertainty

**Input:** `sequences/X_seq.npy`, `sequences/y_seq.npy`, optionally `models/mae_encoder.pt`

**Output:** `models/transformer_best.pt` + `models/transformer_results.json`

### Phase 9 — Deep Ensemble

- Aggregates: CNN-LSTM (3 seeds) + Transformer (5 seeds with MC Dropout)
- **Uncertainty quantification:** ensemble std + MC Dropout variance → 95% CI
- Threshold selected per model to maximise TSS on validation set
- Final prediction: mean probability across ensemble, with epistemic uncertainty
- **Metrics:** TSS, HSS2, AUC-ROC, AUC-PR, F1, Brier Score, Reliability Diagram

**Output:** `models/ensemble_predictions.npz`, `models/ensemble_metrics.json`

---

## Feature Engineering (179 Features)

Injected as a dense vector into CNN-LSTM. Categories:

| Category | Count | Examples |
|----------|-------|----------|
| Time-domain statistics | 28 | mean, std, skew, kurtosis, min, max, quantiles |
| Autocorrelation | 20 | lags 1–20 |
| Spectral features | 20 | dominant frequencies, spectral entropy, power ratios |
| Wavelet coefficients | 16 | Daubechies 4, scales 1–4 |
| Cross-correlation | 8 | SXR vs each HXR band |
| Flare history | 12 | time since last flare, flare density, class decay |
| Physics-based | 75 | Neupert slope, cooling timescale, EM, temperature, nonthermal params |

---

## Dataset Split

| Split | Ratio | Windows | Positive |
|-------|-------|---------|----------|
| Train | 70% | ~560k | ~6% |
| Val   | 15% | ~120k | ~6% |
| Test  | 15% | ~120k | ~6% |
| **Total** | **100%** | **~800k** | **~6%** |

All splits are chronological (no shuffling across time). The test set is the last 15% of data.

---

## Nowcasting Pipeline (Phase 3)

Two independent detection algorithms:

1. **Threshold-based:** sliding window (300s) + peak-over-threshold (5σ) + Bayesian Blocks refinement
2. **Wavelet-based:** CWT with Morlet wavelet, ridge detection across scales

Both applied independently to:
- SoLEXS SXR (2–22 keV, 1s cadence, channel 0)  
- HEL1OS CZT1 full band (18–160 keV, channel 8)

**Master catalogue:** union of all four detections, deduplicated by temporal overlap, validated against SWPC GOES flare list (when available). Combined SXR+HXR nowcasting is a **novel contribution** — prior challenge participants used only soft X-rays.

**Ground truth:** GOES XRS flare list from SWPC, cross-matched within ±5 minutes.

---

## Visualization & Alerting

- **Streamlit dashboard**: `uv run streamlit run src/bah2026/visualization/dashboard.py`
- Real-time light curves (SXR + HXR bands) with overlaid nowcasted/forecasted flare markers
- Probability gauge for next 60-minute forecast window
- Alert triggers: nowcast (confirmed detection) and forecast (P(flare) > threshold)

---

## Server Deployment

```bash
# Clone
cd /home/bs_ms/sbs22ms076/
git clone https://github.com/Shuvam-Banerji-Seal/ISRO-BAH-IISERK.git

# Install
cd ISRO-BAH-IISERK
uv sync

# Run
uv run python -m bah2026.scripts.run_master_pipeline --phase 0-9
```

Raw data is pre-extracted at `/DATA/SBS/` as tar.xz archives. Use `--skip-download --skip-preprocess` if already extracted.

---

## References

1. Sun et al. (2022) — CNN-LSTM with handcrafted features for flare forecasting, Pattern B injection
2. Tang et al. (2021) — Multi-modal fusion of time-series and tabular features
3. Karmakar et al. (2025) — VIT-like transformer for flare prediction
4. Liu et al. (2026) — MAE pretraining for solar time-series
5. Zheng et al. (2023) — Transfer learning GOES→new instrument

See `docs/PLAN.md` for the full 60-paper literature review and all 8 novel contributions.
