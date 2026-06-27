# ISRO-BAH-IISERK

**Bharatiya Antariksh Hackathon 2026 — Challenge 15**
*Forecasting and Nowcasting of Solar Flares using combined Soft and Hard X-ray data from Aditya-L1*

**Team:** IISER Kolkata | **Organizers:** ISRO × Hack2skill

---

## Overview

Automated pipeline combining **SoLEXS** (soft X-rays, 2–22 keV, 340 channels) and **HEL1OS** (hard X-rays, 1.8–160 keV, 10 bands) from Aditya-L1 to detect and predict solar flares.

**Key result:** 8,861 flare events detected across 724 days (Feb 2024 – Jun 2026).

## Quick Start

```bash
uv sync                          # install dependencies
bah2026 all                      # run full pipeline
bah2026 nowcast                  # flare detection only
bah2026 features                 # feature extraction only
bah2026 forecast                 # model training only
pytest tests/ -v                 # run 50 tests
bah2026 init-config              # generate bah2026_config.json
```

## Package Structure

```
src/bah2026/
├── config.py              # All parameters via config (no hardcoding)
├── main.py                # CLI entry point with multiprocessing
├── data/
│   ├── reader.py          # FITS loaders for SoLEXS + HEL1OS
│   ├── preprocessing.py   # Background subtraction, temporal alignment
│   └── hdf5_builder.py    # HDF5 database creation
├── features/
│   └── engineering.py     # 42 canonical features per window
├── models/
│   ├── nowcasting.py      # Threshold, Bayesian Blocks, Wavelet detection
│   └── forecasting.py     # LightGBM, XGBoost, CatBoost, CNN-LSTM
└── visualization/
    └── plots.py           # 8 publication-quality plot functions
```

## Database

| Dataset | Energy | Days | Cadence | FITS Integrity |
|---------|--------|------|---------|----------------|
| SoLEXS | 2–22 keV | 747 | 1s | 100% |
| HEL1OS | 1.8–160 keV | 927 | 1s | 100% |
| **Combined** | **1.8–160 keV** | **724** | 1s | 100% |

## Nowcast Results

| Metric | Value |
|--------|-------|
| Flares detected | 8,861 |
| Days with flares | 640 / 724 (88.4%) |
| HXR-confirmed | 3,454 (39.0%) |
| Class B | 5,502 (62.1%) |
| Class C | 2,955 (33.3%) |
| Class M | 128 (1.4%) |

## Configuration

All parameters are configurable — no hardcoded values:

```bash
bah2026 init-config              # generate config file
BAH2026_WORKERS=16 bah2026 all   # override parallelism
BAH2026_DATA=/path bah2026 all   # override data location
```

See `docs/RESULTS.md` for detailed analysis and inferences.

## Documentation

| Document | Content |
|----------|---------|
| `AGENTS.md` | Problem statement, data format, database summary |
| `docs/PLAN.md` | Research plan, mathematical framework, architecture |
| `docs/RESULTS.md` | Analysis results and inferences |
| `docs/IMPLEMENTED.md` | Implementation status vs plan |
| `docs/analysis/` | Data exploration, pipeline designs, coverage analysis |
| `docs/research/` | Literature review (5 research notes) |

## Tests

50 tests covering config, data loading, preprocessing, feature extraction, and model inference:

```bash
pytest tests/ -v
# ======================== 50 passed in 12s ========================
```

## License

Academic project for Bharatiya Antariksh Hackathon 2026.
