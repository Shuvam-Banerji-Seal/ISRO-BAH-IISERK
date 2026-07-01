"""BAH 2026 Challenge 15 — Solar Flare Nowcasting & Forecasting with Aditya-L1.

A Python package for detecting (nowcasting) and predicting (forecasting)
solar flares using combined soft X-ray (SoLEXS) and hard X-ray (HEL1OS)
data from ISRO's Aditya-L1 mission.

Architecture (Stage 3 — final)
-------------------------------
Nowcasting:
  SWPC onset detection on calibrated SXR + MAD threshold on HEL1OS HXR +
  temporal coincidence merge → 2,285 events catalogue.

Forecasting (three-track ensemble):
  1. CNN-LSTM v3 — 4×Conv1D → BiLSTM → Attention (on 12-ch × 3600-step
     1s data) with 179 handcrafted features injected after Conv backbone.
  2. Spectral-Temporal Transformer — dual-branch self/cross-attention on
     10s-downsampled sequences, Neupert physics-informed loss.
  3. MAE-pretrained Transformer — same arch but encoder initialised from
     Masked Autoencoder self-supervised pretraining.

  Ensemble: weighted average / stacking with deep ensemble (5 seeds)
  + MC Dropout uncertainty quantification.

Modules
-------
data        — FITS readers, calibration, corrections, preprocessing
features    — Feature engineering (179 canonical), spectral fitting,
              information theory, non-thermal fitting, QPP detection
models      — Nowcasting and forecasting (CNN-LSTM, Transformer, MAE)
visualization — Publication plots and Streamlit dashboard
scripts     — CLI pipeline runners

Entry Points
------------
bah2026                        — Full pipeline CLI (nowcast → features)
uv run python -m bah2026.scripts.run_master_pipeline  — 9-phase orchestrator
uv run python -m bah2026.scripts.run_v3_pipeline       — GPU-accelerated v3
"""

__version__ = "0.2.0"
