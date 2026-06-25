# Nowcasting Pipeline Plan

## Objective
Build an automated pipeline to detect (nowcast) solar flares in real-time by combining SoLEXS and HEL1OS light curves.

---

## Pipeline Architecture

```
Raw FITS → QC/GTI Filter → Time Alignment → Feature Extraction → Flare Detection → Classification → Alert
```

---

## Step 1: Data Ingestion & QC

```python
import numpy as np
from astropy.io import fits
from pathlib import Path

def load_day(year, month, day):
    """Load one day of aligned SoLEXS + HEL1OS data."""
    base = Path(f"data/processed/{year:04d}/{month:02d}/{day:02d}")
    
    solexs_lc = base / "SDD2" / f"*{year:04d}{month:02d}{day:02d}*L1.lc"
    solexs_lc = list((base / "SDD2").glob("*_L1.lc"))[0]
    
    # SoLEXS: 86400 seconds, 1s cadence
    with fits.open(solexs_lc) as hdul:
        time_s = hdul["RATE"].data["TIME"]
        counts_s = hdul["RATE"].data["COUNTS"]
        tstart_s = hdul["RATE"].header["TSTART"]
    
    # HEL1OS: multiple bands, ~42500 seconds
    lc_czt = base / "lightcurve_czt1.fits"
    lc_cdte = base / "lightcurve_cdte1.fits"
    
    if lc_czt.exists():
        with fits.open(lc_czt) as hdul:
            # HDU 5 = full band (18-160 keV)
            data_h = hdul[5].data
            mjd_h = data_h["MJD"]
            ctr_h = data_h["CTR"]
    else:
        mjd_h, ctr_h = np.array([]), np.array([])
    
    return {
        "solexs_time": time_s,
        "solexs_counts": counts_s,
        "solexs_tstart": tstart_s,
        "hel1os_mjd": mjd_h,
        "hel1os_ctr": ctr_h
    }

def apply_qti_filter(counts, gti_path):
    """Mask data outside Good Time Intervals."""
    with fits.open(gti_path) as hdul:
        gti = hdul[1].data
        mask = np.zeros(len(counts), dtype=bool)
        for start, stop in gti:
            mask |= (time >= start) & (time <= stop)
    return counts * mask
```

## Step 2: Feature Extraction

```python
def extract_features(counts, window=60):
    """
    Extract real-time features from light curve.
    
    Parameters
    ----------
    counts : array, shape (N,)
        Current count rate window
    window : int
        Look-back window in seconds
    
    Returns
    -------
    dict of feature values
    """
    from scipy.signal import savgol_filter
    
    N = len(counts)
    background = np.median(counts[:N//2])
    
    # Rise rate (slope of last 30s)
    if N >= 30:
        slope = np.polyfit(range(30), counts[-30:], 1)[0]
    else:
        slope = 0
    
    # Flux anomaly
    anomaly = counts[-1] / max(background, 1e-6)
    
    # Variance ratio (recent vs background)
    var_recent = np.var(counts[-60:])
    var_bg = np.var(counts[:60])
    var_ratio = var_recent / max(var_bg, 1e-6)
    
    return {
        "count_rate": counts[-1],
        "background": background,
        "anomaly": anomaly,
        "rise_rate": slope,
        "var_ratio": var_ratio,
        "peak_count": np.max(counts[-120:]) if N >= 120 else np.max(counts)
    }
```

## Step 3: Flare Classification

```python
class FlareNowcaster:
    """
    Real-time flare detection using sliding window analysis.
    """
    
    def __init__(self, sigma=5, hard_soft_ratio_thresh=1.5):
        self.sigma = sigma
        self.hard_soft_ratio_thresh = hard_soft_ratio_thresh
    
    def detect(self, solexs_counts, hel1os_counts, time):
        """
        Returns list of detected flare events.
        
        Returns
        -------
        list of dict:
            Each dict has keys: time_peak, flux_peak, class, 
            start, end, hard_soft_ratio, confidence
        """
        # 1. Background estimation (running 10-min median)
        from scipy.ndimage import median_filter
        bg = median_filter(solexs_counts, size=600)
        residual = solexs_counts - bg
        
        # 2. Noise estimation (MAD)
        mad = np.median(np.abs(residual))
        threshold = self.sigma * 1.4826 * mad
        
        # 3. Find events above threshold
        events = self._find_events(residual, threshold, solexs_counts)
        
        # 4. For each event, classify and measure
        results = []
        for start, peak, end in events:
            # Hard/soft X-ray ratio
            hs_ratio = self._compute_hs_ratio(
                solexs_counts[start:end],
                hel1os_counts[start:end] if len(hel1os_counts) > 0 else None
            )
            
            # Classify
            peak_flux = solexs_counts[peak]
            flare_class = self._classify(peak_flux, hs_ratio)
            
            results.append({
                "time_peak": time[peak],
                "start": time[start],
                "end": time[end],
                "flux_peak": peak_flux,
                "class": flare_class,
                "hard_soft_ratio": hs_ratio,
                "confidence": self._confidence(residual[peak])
            })
        
        return results
    
    def _classify(self, peak_flux, hs_ratio):
        """
        Classify flare by peak flux (Watts/m^2 at 0.1-0.8 nm).
        
        GOES Scale:
            A  < 1e-7
            B  < 1e-6
            C  < 1e-5
            M  < 1e-4
            X  >= 1e-4
        """
        if peak_flux < 100:
            return "A"
        elif peak_flux < 300:
            return "B"
        elif peak_flux < 600:
            return "C"
        elif peak_flux < 2000:
            return "M"
        else:
            return "X"
    
    def _find_events(self, residual, threshold, counts):
        """Find contiguous above-threshold regions."""
        events = []
        above = residual > threshold
        i = 0
        while i < len(above):
            if above[i]:
                start = i
                while i < len(above) and above[i]:
                    i += 1
                end = i
                # Require at least 10s duration
                if end - start >= 10:
                    peak = start + np.argmax(counts[start:end])
                    events.append((start, peak, end))
            else:
                i += 1
        return events
```

## Step 4: Validation Metrics

```python
def validate_nowcasting(detected, truth, tolerance=300):
    """
    Validate detected flares against ground truth.
    
    Parameters
    ----------
    detected : list of dict
        Detected flare events with time_peak
    truth : list of dict
        Known flare events from GOES/XRS catalog
    tolerance : int
        Matching tolerance in seconds
    
    Returns
    -------
    dict with TP, FP, FN, precision, recall, F1
    """
    matched = set()
    
    for d in detected:
        for t_idx, t in enumerate(truth):
            if abs(d["time_peak"] - t["time_peak"]) < tolerance:
                matched.add(t_idx)
    
    TP = len(matched)
    FP = len(detected) - TP
    FN = len(truth) - len(matched)
    
    precision = TP / max(TP + FP, 1e-6)
    recall = TP / max(TP + FN, 1e-6)
    f1 = 2 * precision * recall / max(precision + recall, 1e-6)
    
    return {
        "TP": TP, "FP": FP, "FN": FN,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }
```

## Expected Output: Nowcast Catalog

```
Column         Type        Description
-----------    --------    -----------
time_peak      float64     Peak time (seconds from start of day)
flux_peak      float64     Peak count rate (cts/s)
flare_class    str         A/B/C/M/X classification
hard_soft_ratio float64    HEL1OS/SoLEXS ratio at peak
start_time     float64     Event start time
end_time       float64     Event end time
confidence     float64     Detection confidence (0-1)
duration_sec   float64     Duration in seconds
```
