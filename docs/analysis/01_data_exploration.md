# Data Exploration Plan

## 1. Loading SoLEXS Light Curves

### File Format
- FITS OGIP/HEASARC format with 2 HDUs
- Primary HDU (empty) + `RATE` BinTable
- 86,400 rows per file (1-second cadence, full day)
- Columns: `TIME` (float64, seconds), `COUNTS` (float64)
- Time reference: TSTART from header (mission elapsed time)

### Loading Script
```python
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_solexs_lc(year, month, day):
    """Load SoLEXS SDD2 light curve for a given date."""
    path = Path(f"data/processed/solexs/{year:04d}/{month:02d}/{day:02d}/SDD2")
    lc_file = list(path.glob("*_L1.lc"))[0]
    
    with fits.open(lc_file) as hdul:
        data = hdul["RATE"].data
        tstart = hdul["RATE"].header.get("TSTART", 0)
        time = data["TIME"]  # seconds from TSTART
        counts = data["COUNTS"]
    
    return time, counts, tstart

def load_solexs_pi(year, month, day):
    """Load SoLEXS PI spectrum with energy channels."""
    path = Path(f"data/processed/solexs/{year:04d}/{month:02d}/{day:02d}/SDD2")
    pi_file = list(path.glob("*_L1.pi"))[0]
    
    with fits.open(pi_file) as hdul:
        data = hdul["SPECTRUM"].data
        # Per-second spectra: shape (86400, 340)
        channels = data["CHANNEL"]
        counts = data["COUNTS"]  
        tstart = data["TSTART"]
        exposure = data["EXPOSURE"]
    
    return channels, counts, tstart, exposure
```

### Expected Output
```
SoLEXS Light Curve shape: (86400,)
  TIME range: 0.0 – 86399.0 seconds
  COUNTS range: 0 – 4500 cts/s (varies by solar activity)
  
SoLEXS PI Spectrum shape: (86400, 340)
  Energy channels: 0–339 (~0.059 keV/channel → 2–22 keV)
  One spectrum per second
```

---

## 2. Loading HEL1OS Light Curves

### File Format
- FITS with 6 HDUs (Primary + 5 energy-band BinTables)
- Each HDU: ~42,500 rows (1-second cadence, ~12 hr orbit)
- Columns: `MJD` (float64), `ISOT` (string), `CTR` (float64), `STAT_ERR` (float64)

### Loading Script
```python
def load_hel1os_lc(year, month, day, detector="czt", detector_num=1):
    """
    Load HEL1OS light curve.
    
    Parameters
    ----------
    detector : str
        "czt" (20-150 keV) or "cdte" (10-40 keV)
    detector_num : int
        1 or 2
    """
    path = Path(f"data/processed/hel1os/{year:04d}/{month:02d}/{day:02d}")
    lc_file = path / f"lightcurve_{detector}{detector_num}.fits"
    
    if not lc_file.exists():
        raise FileNotFoundError(f"No data for {year}-{month:02d}-{day:02d}")
    
    bands = {}
    with fits.open(lc_file) as hdul:
        for i in range(1, 6):  # HDUs 1-5 are energy bands
            extname = hdul[i].header["EXTNAME"]
            data = hdul[i].data
            bands[extname] = {
                "mjd": data["MJD"],
                "isot": data["ISOT"],
                "ctr": data["CTR"],
                "stat_err": data["STAT_ERR"]
            }
    
    return bands

def plot_hel1os_bands(bands):
    """Plot all 5 energy bands for a HEL1OS observation."""
    fig, axes = plt.subplots(5, 1, figsize=(12, 10), sharex=True)
    
    for i, (band_name, data) in enumerate(bands.items()):
        time_mjd = data["mjd"]
        # Convert MJD to hours from start
        t_hrs = (time_mjd - time_mjd[0]) * 24
        
        axes[i].plot(t_hrs, data["ctr"], 'b-', lw=0.5)
        axes[i].fill_between(t_hrs, 
                           data["ctr"] - data["stat_err"],
                           data["ctr"] + data["stat_err"],
                           alpha=0.3)
        axes[i].set_ylabel("Counts/s")
        axes[i].set_title(band_name)
    
    axes[-1].set_xlabel("Time from start (hours)")
    plt.tight_layout()
    return fig
```

### Expected Output
```
HEL1OS Bands per detector:
  CZT: 5 bands (20-40, 40-60, 60-80, 80-150, 18-160 keV)
  CdTe: 5 bands (5-20, 20-30, 30-40, 40-60, 1.8-90 keV)
  
  Each band: ~42,500 rows, 1s cadence
  CTR range (quiet Sun): 10-100 cts/s
  CTR range (flare): 100-10,000+ cts/s
```

---

## 3. Identifying Solar Flares in Light Curves

### Detection Strategy
Solar flares appear as sudden, sharp increases in count rate followed by exponential decay:

```
Counts
  ^
  |     _______
  |    /       \     ← Flare: rapid rise + gradual decay
  |   /         \
  |--/           \--←--- Quiet Sun background
  +------------------→ Time
```

### Algorithm: Threshold-Based Detection
```python
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d

def detect_flares(counts, time, sigma=5, min_duration=10):
    """
    Simple flare detection via statistical threshold.
    
    Parameters
    ----------
    counts : array
        Count rate time series
    time : array
        Time values (seconds)
    sigma : float
        Detection threshold in sigmas above background
    min_duration : int
        Minimum flare duration in seconds
    
    Returns
    -------
    list of (start_idx, peak_idx, end_idx)
    """
    # Smooth to estimate background
    background = gaussian_filter1d(counts, sigma=100)
    
    # Residual
    residual = counts - background
    
    # Noise estimate (MAD = Median Absolute Deviation)
    mad = np.median(np.abs(residual - np.median(residual)))
    threshold = sigma * 1.4826 * mad  # MAD → σ conversion
    
    # Find above-threshold regions
    above = residual > threshold
    
    # Group into flare events
    events = []
    i = 0
    while i < len(above):
        if above[i]:
            start = i
            while i < len(above) and above[i]:
                i += 1
            end = i
            
            if end - start >= min_duration:
                peak = start + np.argmax(counts[start:end])
                events.append((start, peak, end))
        else:
            i += 1
    
    return events
```

### Cross-Validation with GOES XRS
For validation, cross-reference detected flares with:
- GOES X-ray Sensor (XRS) flare catalog
- HEL1OS hard X-ray confirmation
- Known flare lists from SWPC

---

## 4. Combined SoLEXS + HEL1OS Analysis

### Energy Coverage
```
Energy (keV):  2    5   10   20   30   40   60   80   150  160
                |----|----|----|----|----|----|----|----|
SoLEXS (SDD2)   ████████████
SoLEXS (SDD1)   ████████████
HEL1OS CdTe          ████████████████████
HEL1OS CZT                    █████████████████████████
                |----|----|----|----|----|----|----|----|
```

### Fusion Strategy
```python
def align_solexs_hel1os(solexs_time, solexs_counts, 
                         hel1os_time_mjd, hel1os_ctr):
    """
    Align SoLEXS and HEL1OS light curves to common time grid.
    SoLEXS: TIME in seconds (mission elapsed)
    HEL1OS: MJD (Modified Julian Date)
    """
    # Convert SoLEXS time to MJD
    # (TSTART is in seconds, need mission epoch)
    # Then interpolate both to common 1-second grid
    
    from scipy.interpolate import interp1d
    
    # ... alignment logic ...
    
    return common_time, solexs_aligned, hel1os_aligned
```

---

## 5. Feature Engineering for Nowcasting

| Feature | Description | Source |
|---------|-------------|--------|
| Base flux | Running median over 5 min | SoLEXS |
| Rise rate | d(counts)/dt over 30s | SoLEXS |
| Hard/Soft ratio | HEL1OS/SoLEXS count ratio | Both |
| Spectral hardness | High-energy/Low-energy ratio | SoLEXS PI |
| Pre-flare quiet level | Median of preceding 10 min | Both |
| Time since last flare | Minutes since last event | Both |
| HXR timing delay | HEL1OS peak - SoLEXS peak | Both |

---

## 6. Next Steps

1. **Batch load** all 747 SoLEXS days → create time-series database
2. **Batch load** all 927 HEL1OS days → align with SoLEXS
3. **Build flare catalog** via threshold detection
4. **Validate** against GOES/XRS and known events
5. **Extract pre-flare features** for forecasting model training
