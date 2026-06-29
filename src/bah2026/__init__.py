"""BAH 2026 Challenge 15 — Solar Flare Nowcasting & Forecasting with Aditya-L1.

A Python package for detecting (nowcasting) and predicting (forecasting)
solar flares using combined soft X-ray (SoLEXS) and hard X-ray (HEL1OS)
data from ISRO's Aditya-L1 mission.

Modules
-------
data        — FITS readers, calibration, corrections, preprocessing
features    — Feature engineering, spectral fitting, information theory,
              non-thermal fitting, QPP detection, RMF/ARF convolution
models      — Nowcasting (SWPC onset, HXR coincidence) and forecasting
              (LightGBM, XGBoost, CatBoost, CNN-LSTM)
visualization — Publication plots and Streamlit dashboard
scripts     — CLI entry points (run_pipeline, analyze, extract, etc.)

Entry Points
------------
bah2026              — Full pipeline CLI (nowcast → features → forecast)
bah2026-run-pipeline — Pipeline runner with GPU support
bah2026-concat-hel1os — HEL1OS orbit concatenation
"""

__version__ = "0.2.0"
