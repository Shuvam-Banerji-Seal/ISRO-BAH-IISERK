# Analysis Index

## Document Map

| Doc | Purpose | Key Output |
|-----|---------|------------|
| [01_data_exploration.md](01_data_exploration.md) | Load FITS, inspect formats, identify flares | Flare detection algorithm |
| [02_nowcasting_pipeline.md](02_nowcasting_pipeline.md) | Real-time flare detection & classification | FlareNowcaster class |
| [03_forecasting_pipeline.md](03_forecasting_pipeline.md) | Predict flares N minutes ahead | FlareForecaster + CNN-LSTM |
| [04_visualization_dashboard.md](04_visualization_dashboard.md) | Dashboard + alerts | Streamlit app + plots |

---

## Data Access Quick Reference

### SoLEXS Light Curve
```python
from astropy.io import fits
with fits.open("data/processed/solexs/YYYY/MM/DD/SDD2/AL1_SOLEXS_YYYYMMDD_SDD2_L1.lc") as hdul:
    time = hdul["RATE"].data["TIME"]   # seconds
    counts = hdul["RATE"].data["COUNTS"]  # cts/s
```

### HEL1OS Light Curve
```python
with fits.open("data/processed/hel1os/YYYY/MM/DD/lightcurve_czt1.fits") as hdul:
    # HDU 5 = full band (18-160 keV)
    mjd = hdul[5].data["MJD"]
    ctr = hdul[5].data["CTR"]       # cts/s
    err = hdul[5].data["STAT_ERR"]
```

### Data Summary
| Dataset | Days | Files | Primary Product |
|---------|------|-------|----------------|
| SoLEXS | 747 | 2,990 | 1s light curves, 2-22 keV |
| HEL1OS | 927 | 9,091 | 1s light curves, 5-160 keV |
