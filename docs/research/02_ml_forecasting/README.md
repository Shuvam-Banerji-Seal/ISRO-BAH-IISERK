# Research Note 02: ML/DL for Solar Flare Forecasting — SOTA Review
**Part of BAH 2026 Challenge #15** | *Stored: docs/research/02_ml_forecasting/*

---

## Survey of Approaches (2015–2026)

### Classical ML on Magnetic Parameters
**Bobra & Couvidat (2015)** — SVM on 13 SHARP parameters for ≥M1 flare within 24h. TSS ≈ 0.82, HSS ≈ 0.52. Log-transformed features critical.
- *ApJ 798, 135* | DOI: 10.1088/0004-637X/798/2/135

**Bringewald (2025)** — Random Forest, KNN, XGBoost on SHARP with PCA. RF & XGB best, benefit from increased dimensionality. Combined binary + multiclass classification.
- [arXiv:2505.03385](https://arxiv.org/abs/2505.03385)

### Deep Learning on Magnetograms
**Huang et al. (2018)** — First deep CNN on line-of-sight magnetograms for flare prediction. TSS ≈ 0.52 for ≥M1.
- [arXiv:1801.10420](https://arxiv.org/abs/1801.10420) | *ApJ 856, 7*

**Pandey et al. (2023)** — Explainable DL with post hoc attention (Grad-CAM, Deep SHAP, Integrated Gradients). Full-disk magnetograms, TSS=0.51±0.05. Showed models can locate near-limb flares.
- [arXiv:2308.02682](https://arxiv.org/abs/2308.02682)

### CNN on X-ray Time Series
**Landa & Reuveni (2021)** — 1D CNN on GOES X-ray light curves (1998–2019, SC23-24). Forecasting at 1–96h horizons. **Key finding:** X-ray time series alone cannot distinguish M vs X class — spectral information needed.
- [arXiv:2101.12550](https://arxiv.org/abs/2101.12550) | *ApJS 258, 29*

### Transformer Models
**Abduallah & Wang (2024)** — SolarFlareNet: Transformer on SHARP time series for 24–72h prediction. Three parallel transformers for ≥M5, ≥M, ≥C. Operational.
- [arXiv:2405.16080](https://arxiv.org/abs/2405.16080)

**Vural et al. (2025)** — GCTAF: Global Cross-Time Attention Fusion. Learnable global tokens that summarize temporal patterns via cross-attention.
- [arXiv:2511.12955](https://arxiv.org/abs/2511.12955)

**Li et al. (2026)** — Operational forecasting using explainable LLM on SHARP time series + magnetograms.
- [arXiv:2601.22811](https://arxiv.org/abs/2601.22811)

### Ensemble Methods
**Guerra et al. (2020)** — First ensemble prediction model for major flares. Combining outputs from multiple classifiers improves skill.
- [arXiv:2008.00382](https://arxiv.org/abs/2008.00382)

**Lv et al. (2026)** — FlareCast: Deep Bayesian neural network with MLOps infrastructure. Variational inference for UQ.
- *Semantic Scholar*

### Extreme Value Theory & Optimal Prediction
**Verma et al. (2024)** — Neyman-Pearson optimal extreme event prediction for heavy-tailed time series. Applied to GOES X-ray flux. Established theoretical bounds on predictability.
- [arXiv:2407.11887](https://arxiv.org/abs/2407.11887)

### Multi-modal Fusion
**Rosales et al. (2026)** — Combining magnetograms + EUV improves over single-modality.

### Verification & Challenges
**Camporeale & Berger (2025)** — NOAA SWPC forecast verification 1998-2024. SWPC doesn't beat simple baselines. Severe calibration issues.
- [arXiv:2508.01114](https://arxiv.org/abs/2508.01114)

**Hu et al. (2025)** — Data quality defects in flare data sources. Impact on ML reproducibility.
- [arXiv:2512.13417](https://arxiv.org/abs/2512.13417)

**Leka et al. (2026)** — 4π full-heliosphere framework for limb flare prediction. Far-side helioseismology helps.
- [arXiv:2601.05209](https://arxiv.org/abs/2601.05209)

### Data Augmentation & Loss Functions
**Pandey et al. (2024)** — Embedding ordinality into binary loss for improved forecasting.
- [arXiv:2408.11768](https://arxiv.org/abs/2408.11768)

---

## Key Research Gaps (ML Perspective)

1. **No work uses spectral X-ray information** — all ML models use GOES broadband (1–8Å)
2. **Physics-informed ML is absent in flare forecasting** — no Neupert, no bremsstrahlung physics
3. **Self-supervised pretraining on X-ray data is unexplored**
4. **Uncertainty quantification is nascent** — only recent Bayesian attempts
5. **Operational systems show severe calibration issues** (Camporeale 2025)
6. **No work fuses soft + hard X-ray data** for ML forecasting
