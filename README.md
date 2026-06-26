# ISRO-BAH-IISERK

**Bharatiya Antariksh Hackathon 2026 — Challenge 15**  
*Forecasting and Nowcasting of Solar Flares using combined Soft and Hard X-ray data from Aditya-L1*

**Team:** IISER Kolkata  
**Organizers:** ISRO × Hack2skill

---

## Overview

This repository contains the solution for Challenge 15 of the Bharatiya Antariksh Hackathon 2026. We build an automated algorithmic pipeline using time-series data from Aditya-L1's SoLEXS (soft X-ray) and HEL1OS (hard X-ray) payloads to detect and predict solar flares.

## Database Summary

| Dataset | Coverage | Days | Files | Size |
|---------|----------|------|-------|------|
| **SoLEXS** | 2–22 keV | 747 days | 2,988 FITS | 330 GB |
| **HEL1OS** | 1.8–160 keV | 902 days | 7,272 FITS | 88 GB |
| **Overlap** | Combined | **724 days** | — | — |

**SoLEXS:** 1-second cadence, full 24h/day, 340 energy channels (2–22 keV)  
**HEL1OS:** 1-second cadence, 4 detectors × 5 energy bands (1.8–160 keV)

## Project Structure

```
├── AGENTS.md                   # Full problem statement & guidance
├── README.md                   # This file
├── pyproject.toml              # Python 3.13 + dependencies
├── main.py                     # Entry point
├── data/
│   ├── downloads/              # PRADAN download scripts
│   │   ├── download_solexs.sh
│   │   ├── download_hel1os.sh
│   │   ├── parallel_dl.sh     # 8-worker parallel downloader
│   │   ├── fast_dl.py         # Python urllib downloader
│   │   ├── cookie_grabber.py  # Browser cookie extraction
│   │   └── decompress.sh      # Zip extraction to processed/
│   └── tools/
│       └── solexs_tools-1.1.tar.gz
├── docs/
│   ├── manuals/                # Instrument user manuals
│   └── analysis/               # Data analysis & pipeline plans
├── src/bah2026/                # Main package
├── notebooks/                  # Jupyter notebooks
└── tests/                      # Unit tests
```

## Quick Start

```bash
uv sync
uv run bah2026
```

## Key Documentation

- `AGENTS.md` — Problem statement, data format, database summary
- `docs/analysis/notes_solexs.md` — SoLEXS data analysis (747 days, 100% integrity)
- `docs/analysis/notes_hel1os.md` — HEL1OS data analysis (902 days, 100% integrity)
- `docs/analysis/combined_coverage.md` — Temporal overlap (724 dual-instrument days)
- `docs/analysis/01_data_exploration.md` — FITS loading, feature extraction
- `docs/analysis/02_nowcasting_pipeline.md` — Real-time flare detection
- `docs/analysis/03_forecasting_pipeline.md` — Predictive model (LightGBM + CNN-LSTM)
- `docs/analysis/04_visualization_dashboard.md` — Dashboard & alerts

## License

Academic project for Bharatiya Antariksh Hackathon 2026.
