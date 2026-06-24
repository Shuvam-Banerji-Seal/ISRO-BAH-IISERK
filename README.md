# ISRO-BAH-IISERK

**Bharatiya Antariksh Hackathon 2026 — Challenge 15**  
*Forecasting and Nowcasting of Solar Flares using combined Soft and Hard X-ray data from Aditya-L1*

**Team:** IISER Kolkata  
**Organizers:** ISRO × Hack2skill

---

## Overview

This repository contains the solution for Challenge 15 of the Bharatiya Antariksh Hackathon 2026. We build an automated algorithmic pipeline using time-series data from Aditya-L1's SoLEXS (soft X-ray) and HEL1OS (hard X-ray) payloads to detect and predict solar flares.

## Project Structure

```
├── AGENTS.md               # Full problem statement & guidance
├── main.py                 # CLI entry point
├── pyproject.toml          # Python project dependencies
├── data/
│   ├── downloads/          # PRADAN download scripts
│   └── tools/              # SoLEXS processing tools
├── docs/manuals/           # Instrument user manuals
├── notebooks/              # Exploration notebooks
├── src/
│   ├── data/               # Data loading & preprocessing
│   ├── features/           # Feature engineering
│   ├── models/             # Nowcasting & forecasting models
│   └── visualization/      # Dashboard & plotting
└── tests/
```

## Quick Start

```bash
uv sync
uv run bah2026
```

## Data Download

1. Log in to https://pradan1.issdc.gov.in
2. Copy session cookie into the download scripts
3. Run `bash data/downloads/download_solexs.sh` and `data/downloads/download_hel1os.sh`

## License

Academic project for Bharatiya Antariksh Hackathon 2026.
