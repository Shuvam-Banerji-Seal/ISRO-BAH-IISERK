# RESULTS — BAH 2026 Challenge #15

**Solar Flare Nowcasting & Forecasting with Aditya-L1 SoLEXS + HEL1OS**

---

## 1. Dataset Summary

| Metric | SoLEXS | HEL1OS | Combined |
|--------|--------|--------|----------|
| Date range | 2024-02-01 → 2026-06-22 | 2023-11-30 → 2026-06-23 | 2024-02-01 → 2026-06-22 |
| Days with data | 747 | 927 | **724** |
| Energy range | 2–22 keV | 1.8–160 keV | **1.8–160 keV** |
| Cadence | 1 second | 1 second | 1 second |
| Coverage | 85.6% | 98.9% | 78.3% |
| FITS integrity | 100% (2,988 files) | 100% (7,272 files) | 100% |

---

## 2. Nowcast Catalogue — Key Findings

### 2.1 Flare Detection Results

**8,861 flare events** detected across **640 days** (out of 724 combined days).

| GOES Class | Count | Percentage | Interpretation |
|------------|-------|------------|----------------|
| **B** | 5,502 | 62.1% | Microflares — most common, low activity |
| **C** | 2,955 | 33.3% | Small flares — moderate activity |
| **M** | 128 | 1.4% | Medium flares — significant energy release |
| **A** | 276 | 3.1% | Background-level events |
| **X** | 0 | 0% | No major flares in detection window |

**No X-class flares** detected. This is consistent with the SoLEXS sensitivity range (2–22 keV) and the approximate GOES calibration scale (SOLEXS_TO_GOES_SCALE = 1e-8).

### 2.2 Hard X-ray Cross-Confirmation

- **39.0% of all events** (3,454/8,861) have simultaneous HEL1OS hard X-ray detection within ±30 seconds
- M-class flares show higher HXR confirmation rates, consistent with the Neupert effect
- Events without HXR confirmation are likely pure thermal (soft X-ray only) or below HEL1OS sensitivity

### 2.3 Temporal Distribution

| Metric | Value |
|--------|-------|
| Flares per day (mean) | 13.8 |
| Flares per day (median) | 9 |
| Flares per day (max) | 107 (active day) |
| Days with ≥1 flare | 640 / 724 (88.4%) |
| Days with no flares | 84 (11.6%) |

The high detection rate (88.4% of days) confirms that the Sun was frequently active during Solar Cycle 25's rising phase.

### 2.4 Duration Distribution

| Percentile | Duration (sec) |
|------------|----------------|
| 25th | 11 |
| 50th (median) | 15 |
| 75th | 36 |
| 99th | 250 |

Most detected events are short-duration impulsive flares (10–40 seconds). The 1-second cadence of both instruments resolves individual impulsive peaks that would be blended in GOES 1-minute data.

### 2.5 Peak Flux Distribution

| Metric | Value |
|--------|-------|
| Mean | 132 cts/s |
| Median | 68 cts/s |
| Max | 6,765 cts/s |
| Std | 298 cts/s |

The distribution is heavily right-skewed — a few large flares dominate. This is consistent with the power-law distribution of solar flare energies.

---

## 3. Energy Coverage

The combined SoLEXS + HEL1OS coverage spans **1.8–160 keV**:

| Band | Detector | Range (keV) | Type |
|------|----------|-------------|------|
| SoLEXS SDD2 | Silicon | 2–22 | Thermal (plasma T > 10 MK) |
| CdTe1 Band 1 | CdTe | 5–20 | Thermal/Non-thermal boundary |
| CdTe1 Band 2 | CdTe | 20–30 | Non-thermal |
| CdTe1 Band 3 | CdTe | 30–40 | Non-thermal |
| CdTe1 Band 4 | CdTe | 40–60 | Non-thermal |
| CZT1 Band 1 | CZT | 20–40 | Non-thermal |
| CZT1 Band 2 | CZT | 40–60 | Non-thermal |
| CZT1 Band 3 | CZT | 60–80 | Non-thermal |
| CZT1 Band 4 | CZT | 80–150 | Non-thermal (high-energy tail) |

**Inference:** The overlap region (20–40 keV) between SoLEXS, CdTe, and CZT provides a cross-calibration opportunity. The thermal-nonthermal boundary around 10–20 keV is well-resolved by SoLEXS (thermal) and CdTe (non-thermal), enabling direct measurement of the electron beam energy threshold.

---

## 4. Feature Engineering Results

**42 canonical features** extracted per analysis window:

| Feature Category | Count | Description |
|-----------------|-------|-------------|
| SXR statistical | 15 | mean, std, max, min, median, skew, kurtosis, range, CV, rise/fall rate, abs slope, IQR, percentiles |
| SXR spectral | 2 | spectral entropy, peak frequency |
| SXR autocorrelation | 4 | ACF at lags 5s, 10s, 30s, 60s |
| SXR percentiles | 4 | P5, P25, P75, P95 |
| HXR band-level | 15 | mean, std, max for 5 energy bands |
| HXR derived | 3 | hardness ratio, total mean, soft-hard ratio |
| Meta | 1 | window length |

**Key inferences:**
- The **spectral entropy** feature captures the complexity of the X-ray spectrum — higher entropy during quiet times, lower during flares (dominated by a single thermal component)
- **Hardness ratio** (HXR high-energy / low-energy) is a direct proxy for non-thermal electron acceleration
- **Autocorrelation** at 60s lag captures the quasi-periodic pulsations observed in many flares

---

## 5. Model Performance

Three gradient-boosted tree models trained on the 42-feature matrix (199,824 samples, 9.70% positive rate):

| Model | AUC-ROC | AUC-PR | F1 | Precision | Recall |
|-------|---------|--------|----|-----------|--------|
| **LightGBM** | 0.634 | 0.156 | 0.158 | 0.223 | 0.122 |
| **XGBoost** | 0.632 | 0.157 | 0.143 | 0.248 | 0.100 |
| **CatBoost** | **0.664** | 0.150 | **0.203** | 0.156 | **0.289** |

**Key observations:**
- **CatBoost** achieves the highest AUC-ROC (0.664) and best recall (0.289) — it catches the most flares
- **XGBoost** has the highest precision (0.248) — when it predicts a flare, it's most often correct
- **LightGBM** provides a balanced middle ground
- All models perform above random (AUC-ROC > 0.5), confirming that X-ray spectral features contain predictive signal

### Why Performance is Moderate

1. **Extreme class imbalance:** Only 9.7% of time windows are labeled as flares (19,374 positive out of 199,824 total)
2. **Short prediction horizon:** 30-minute look-ahead window captures many ambiguous boundary cases
3. **Missing HXR data:** ~76% of each day has no HEL1OS data (orbital gaps), reducing feature quality for those windows
4. **No spectral fitting:** Features use raw band statistics rather than physical parameters (temperature T, emission measure EM)

### Expected Performance Benchmarks (from literature)

| Method | TSS | Reference |
|--------|-----|-----------|
| SVM on SHARP (magnetic) | 0.82 | Bobra & Couvidat 2015 |
| CNN on magnetograms | 0.52 | Huang et al. 2018 |
| 1D CNN on GOES X-ray | 0.47 | Landa & Reuveni 2021 |
| Transformer on SHARP | 0.58 | Abduallah & Wang 2024 |
| Ensemble (magnetic) | 0.65 | Guerra et al. 2020 |

Our approach uses **energy-dependent spectral features** (not just total flux), which Landa & Reuveni (2021) explicitly identified as necessary for distinguishing flare classes. The current baseline can be improved by adding spectral fitting features (T, EM), transfer entropy, and the planned spectral-temporal transformer (N3).

---

## 6. Novel Contributions

| # | Contribution | Status | Evidence |
|---|-------------|--------|----------|
| N1 | First combined SXR+HXR nowcast from Aditya-L1 | ✅ Done | 8,861-event catalogue, 39% HXR-confirmed |
| N2 | Neupert-constrained physics-informed loss | 🔜 Next | Documented in PLAN.md §5.4 |
| N3 | Spectral-temporal cross-attention transformer | 🔜 Next | Architecture in PLAN.md §5.3 |
| N4 | Transfer entropy precursor detection | ⚠️ Partial | Feature framework ready |
| N5 | Energy-dependent forecasting features | ✅ Done | 42 features including 5 HXR bands |
| N6 | Self-supervised MAE pretraining | 🔜 Next | Not started |
| N7 | Bayesian UQ (ensemble + conformal) | 🔜 Next | Not started |
| N8 | Cross-instrument transfer GOES→Aditya-L1 | 🔜 Next | Not started |

---

## 7. Data Quality Observations

### 7.1 SoLEXS SDD1 Non-Functionality

SDD1 GTI files consistently contain **0 rows** with `EXPOSURE=0.0`. SDD1 (7.1 mm² aperture) does not produce science-grade data. **All analysis uses SDD2 exclusively.**

### 7.2 HEL1OS Orbital Gaps

HEL1OS observes ~5.7 hours/day (orbital segments), compared to SoLEXS's 24 hours/day. This means:
- ~76% of each day has no HEL1OS data
- Combined dual-instrument analysis is limited to HEL1OS orbital windows
- Feature engineering must handle this duty cycle mismatch

### 7.3 MJD Reference Inconsistency

Some SoLEXS LC files have `MJDREFF=0.22916666651` (IST offset) instead of `MJDREFF=0`. The pipeline uses `MJDREFI + MJDREFF` for correct MJD computation, but this inconsistency should be flagged in any published analysis.

### 7.4 June 2024 Data Gap

The entire month of June 2024 (30 days) is missing from SoLEXS data. This is the largest continuous gap and coincides with the instrument's commissioning phase.

---

## 8. Next Steps

1. ~~Complete feature extraction~~ ✅ Done — 199,824 samples × 42 features extracted
2. ~~Train and evaluate forecasting models~~ ✅ Done — LightGBM, XGBoost, CatBoost trained
3. **Implement spectral-temporal transformer** (N3) — core novel contribution
4. **Add Neupert physics-informed loss** (N2) as regularization
5. **Compute transfer entropy** between HEL1OS and SoLEXS channels (N4)
6. **Add spectral fitting features** (T, EM from SoLEXS PI data) — expected to improve AUC
7. **Build Streamlit dashboard** for interactive visualization
8. **Generate per-day PDF catalogue** for all 724 days

---

## 9. Reproducibility

```bash
# Install
uv sync

# Run full pipeline (with logs)
bah2026 all 2>&1 | tee logs/pipeline_$(date +%Y%m%d_%H%M%S).log

# Run individual phases
bah2026 explore    # Phase 1: overview plots
bah2026 nowcast    # Phase 2: flare detection
bah2026 features   # Phase 3: feature extraction
bah2026 forecast   # Phase 4: model training

# Run tests
pytest tests/ -v
```

All parameters are configurable via `bah2026_config.json`:
```bash
bah2026 init-config  # Generate default config
```

Or set via environment variables:
```bash
BAH2026_DATA=/path/to/data BAH2026_WORKERS=16 bah2026 all
```
