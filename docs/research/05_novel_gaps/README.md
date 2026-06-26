# Research Note 05: Research Gaps & Novel Opportunities
**Part of BAH 2026 Challenge #15** | *Stored: docs/research/05_novel_gaps/*

---

## Identified Research Gaps

### Gap 1: No Combined Soft+Hard X-ray Flare Pipeline for Aditya-L1
- All existing work uses GOES (no spectral resolution) or single instruments
- SoLEXS (340 ch) + HEL1OS (10 bands) provide spectral information never before used for flare detection

### Gap 2: X-ray Spectral Information Not Used in Forecasting
- Every published ML model uses GOES 1–8 Å broadband total flux
- Temperature T, emission measure EM, spectral index γ — all discarded
- Landa & Reuveni (2021) explicitly showed X-ray time series alone cannot distinguish M vs X class

### Gap 3: No Physics-Informed ML for Flare Forecasting
- PINNs are well-established (Raissi et al. 2019) but never applied to X-ray flare physics
- Neupert effect provides natural physics constraint: dSXR/dt ∝ HXR
- Bremsstrahlung physics, energy conservation — all unused

### Gap 4: Precursor Detection Not Explored with Information Theory
- Transfer entropy, mutual information, Granger causality — never applied to flare precursor detection
- Cross-channel causality HXR→SXR not studied
- Hudson et al. (2020) found Hot X-ray Onsets — but not quantified with ML

### Gap 5: Uncertainty Quantification is Nascent
- Most models give point probabilities
- Only recent effort: FlareCast (Lv et al. 2026) with Bayesian DNN
- Operational NOAA forecasts show severe calibration issues (Camporeale & Berger 2025)

### Gap 6: Cross-Instrument Transfer Learning Unexplored
- GOES has 25+ years of data (1998–present) — SC23-24-25
- Aditya-L1 has only 2.5 years
- No published transfer learning between X-ray instruments

### Gap 7: Self-Supervised Learning on Unlabeled X-ray Data
- 700+ days of continuous 1-second data → excellent for pretraining
- Masked autoencoders, contrastive learning — unexplored in solar physics

### Gap 8: Spectral-Temporal Joint Modeling
- No paper treats both energy and time dimensions jointly via attention
- Landa's CNN on GOES is purely temporal (no spectral dimension)
- Transformers on time series (SolarFlareNet, GCTAF) use only temporal data

## Novel Opportunities

| # | Opportunity | Gap Addressed |
|---|-------------|---------------|
| N1 | Combined soft+hard X-ray nowcast database using SoLEXS + HEL1OS | Gap 1 |
| N2 | Neupert-constrained physics-informed loss function | Gap 3 |
| N3 | Cross-attention spectral-temporal transformer | Gap 8 |
| N4 | Transfer entropy-based precursor activity index | Gap 4 |
| N5 | Energy-dependent forecasting (T, EM, γ, HR features) | Gap 2 |
| N6 | Self-supervised MAE pretraining on quiet-Sun data | Gap 7 |
| N7 | Bayesian Deep Ensemble + conformal prediction UQ | Gap 5 |
| N8 | Cross-instrument transfer learning GOES → Aditya-L1 | Gap 6 |

## SOTA Performance Benchmarks

| Method | Task | TSS | HSS | Reference |
|--------|------|-----|-----|-----------|
| SVM (SHARP) | ≥M1, 24h | 0.82 | 0.52 | Bobra & Couvidat 2015 |
| CNN (magnetogram) | ≥M1, 24h | 0.52 | — | Huang et al. 2018 |
| CNN (full disk) | ≥M1, 24h | 0.51 | 0.38 | Pandey et al. 2023 |
| 1D CNN (GOES) | ≥M, 1h | 0.47 | 0.33 | Landa & Reuveni 2021 |
| 1D CNN (GOES) | ≥X, 1h | 0.41 | 0.28 | Landa & Reuveni 2021 |
| Transformer (SHARP) | ≥M, 24h | 0.58 | — | Abduallah & Wang 2024 |
| GCTAF (TS) | ≥M, 24h | 0.61 | — | Vural et al. 2025 |
| NOAA SWPC | ≥M, 24h | 0.35 | 0.15 | Camporeale & Berger 2025 |
| Ensemble | ≥M, 24h | 0.65 | 0.40 | Guerra et al. 2020 |

## What Makes Our Approach Unique

1. **First to use spectral X-ray data** (SoLEXS 340 ch + HEL1OS 10 bands) instead of GOES broadband
2. **First to combine soft + hard X-ray** from Aditya-L1 for any purpose
3. **First physics-informed neural network** for flare forecasting with Neupert constraint
4. **First cross-instrument transfer** from GOES (25 yr) to Aditya-L1 (2.5 yr)
5. **First uncertainty quantification** with conformal prediction for flare forecasts
6. **First self-supervised pretraining** on continuous solar X-ray data
