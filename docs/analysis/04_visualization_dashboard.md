# Visualization & Dashboard Plan

## Objective
Build an interactive dashboard showing real-time X-ray light curves with flare alerts for nowcasting and forecasting.

---

## Dashboard Layout

```
┌──────────────────────────────────────────────────────────┐
│  BAH 2026 — Solar Flare Dashboard                       │
│  Status: LIVE | Last Update: YYYY-MM-DD HH:MM:SS         │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  ┌──────────────────────────────────────────────────┐   │
│  │  SoLEXS Light Curve (2-22 keV)                  │   │
│  │  [counts/s]                                     │   │
│  │   ████████                                      │   │
│  │   ████  ██  ████     ← Detected flares          │   │
│  │   ██    ██    ███                                │   │
│  │   ─────────────── baseline                       │   │
│  └──────────────────────────────────────────────────┘   │
│                                                          │
│  ┌──────────────────────────────────────────────────┐   │
│  │  HEL1OS CZT (20-150 keV)                        │   │
│  │  [cts/s]                                        │   │
│  │   ██████                                        │   │
│  │   ████  ████                                    │   │
│  │   ██      ████                                  │   │
│  │   ─────────────── baseline                      │   │
│  └──────────────────────────────────────────────────┘   │
│                                                          │
│  ┌──────────────────────┐  ┌─────────────────────────┐  │
│  │ Forecast Panel        │  │ Flare Catalog           │  │
│  │ P(flares) = 0.87     │  │ Time     Class  HR      │  │
│  │ Lead time: 12 min    │  │ 14:32    M      2.3     │  │
│  │ Confidence: HIGH     │  │ 16:15    C      1.8     │  │
│  └──────────────────────┘  └─────────────────────────┘  │
└──────────────────────────────────────────────────────────┘
```

---

## Python Visualization Stack

| Component | Library | Purpose |
|-----------|---------|---------|
| Static plots | matplotlib | Publication-quality light curve plots |
| Interactive plots | plotly | Time-series with zoom, pan, hover |
| Dashboard UI | streamlit or gradio | Web-based interactive dashboard |
| Data tables | pandas | Flare catalog display |
| Alerting | Custom | Real-time threshold notifications |

---

## Plot 1: Multi-Band Light Curve Overview

```python
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from astropy.time import Time

def plot_day_overview(year, month, day, solexs_data, hel1os_data, flares):
    """
    Comprehensive view of a single day's observations.
    
    Parameters
    ----------
    solexs_data : dict with 'time', 'counts'
    hel1os_data : dict with 'mjd', 'ctr' (all 5 bands)
    flares : list of dict from nowcast catalog
    """
    fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True,
                              gridspec_kw={"height_ratios": [3, 2, 2, 2]})
    
    # --- Panel 1: SoLEXS soft X-ray ---
    time_solexs = solexs_data["time"]
    counts_solexs = solexs_data["counts"]
    
    # Convert to hours
    t_hours = time_solexs / 3600
    
    axes[0].plot(t_hours, counts_solexs, "b-", lw=0.3, alpha=0.7, label="SoLEXS 2-22 keV")
    axes[0].set_ylabel("Count Rate (cts/s)", fontsize=11)
    axes[0].set_title(f"SoLEXS + HEL1OS — {year}-{month:02d}-{day:02d}", fontsize=13)
    axes[0].legend(loc="upper right")
    
    # Overlay flare regions
    for flare in flares:
        x0 = flare["start"] / 3600
        x1 = flare["end"] / 3600
        axes[0].axvspan(x0, x1, alpha=0.3, color="red", label="_flare")
        axes[0].axvline(flare["time_peak"] / 3600, color="red", lw=1, ls="--")
    
    # --- Panel 2: HEL1OS CZT full band ---
    if hel1os_data and "mjd" in hel1os_data:
        mjd = hel1os_data["mjd"]
        t_h = (mjd - mjd[0]) * 24
        axes[1].plot(t_h, hel1os_data["ctr"], "r-", lw=0.3, alpha=0.7)
    axes[1].set_ylabel("CZT (cts/s)", fontsize=11)
    
    # --- Panel 3: Hard/Soft ratio ---
    # (computed from aligned data)
    axes[2].set_ylabel("Hard/Soft Ratio", fontsize=11)
    
    # --- Panel 4: Forecast probability ---
    # (from forecasting model)
    axes[3].set_ylabel("P(flares)", fontsize=11)
    axes[3].set_xlabel("Time from start (hours)", fontsize=12)
    axes[3].set_ylim(0, 1)
    axes[3].axhline(0.5, color="orange", ls="--", alpha=0.5)
    axes[3].axhline(0.8, color="red", ls="--", alpha=0.5)
    
    plt.tight_layout()
    return fig
```

## Plot 2: Spectral Evolution During Flare

```python
def plot_spectral_evolution(pi_data, time_grid, flare_peak_idx, window=300):
    """
    Show how the energy spectrum evolves during a flare.
    
    Parameters
    ----------
    pi_data : array, shape (86400, 340)
        PI spectra (counts per energy channel per second)
    flare_peak_idx : int
        Index of flare peak in the time grid
    window : int
        Half-width in seconds around peak
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    energy = np.arange(0, 340) * 0.059 + 2.0  # keV
    
    # Time slices
    slices = {
        "T - 10 min": flare_peak_idx - 600,
        "T - 5 min": flare_peak_idx - 300,
        "Peak": flare_peak_idx,
        "T + 5 min": flare_peak_idx + 300,
        "T + 10 min": flare_peak_idx + 600,
    }
    
    colors = ["blue", "cyan", "red", "orange", "gray"]
    
    for (label, idx), color in zip(slices.items(), colors):
        if 0 <= idx < len(pi_data):
            spectrum = pi_data[idx]
            mask = spectrum > 0
            ax1.loglog(energy[mask], spectrum[mask], "o-", color=color, 
                       ms=2, lw=1, label=label)
    
    ax1.set_xlabel("Energy (keV)", fontsize=11)
    ax1.set_ylabel("Counts", fontsize=11)
    ax1.set_title("Spectral Evolution During Flare")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Time-integrated spectrum
    start = max(0, flare_peak_idx - window)
    end = min(len(pi_data), flare_peak_idx + window)
    avg_spectrum = np.mean(pi_data[start:end], axis=0)
    mask = avg_spectrum > 0
    ax2.semilogy(energy[mask], avg_spectrum[mask], "k-o", ms=2, lw=1)
    ax2.set_xlabel("Energy (keV)", fontsize=11)
    ax2.set_ylabel("Counts (averaged)", fontsize=11)
    ax2.set_title(f"Spectrum ±{window}s around peak")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig
```

## Plot 3: Flare Classification Scatter

```python
def plot_flare_statistics(catalog_df):
    """
    Scatter plot: peak flux vs duration, colored by class.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    class_colors = {"A": "blue", "B": "cyan", "C": "green", "M": "orange", "X": "red"}
    
    for cls, color in class_colors.items():
        mask = catalog_df["flare_class"] == cls
        if mask.sum() > 0:
            ax1.scatter(
                catalog_df.loc[mask, "flux_peak"],
                catalog_df.loc[mask, "duration_sec"] / 60,  # minutes
                c=color, alpha=0.6, s=20, label=f"Class {cls}"
            )
    
    ax1.set_xlabel("Peak Count Rate (cts/s)")
    ax1.set_ylabel("Duration (minutes)")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.legend()
    ax1.set_title("Flare Duration vs Peak Flux")
    ax1.grid(True, alpha=0.3)
    
    # Class distribution
    class_counts = catalog_df["flare_class"].value_counts()
    bars = ax2.bar(class_counts.index, class_counts.values,
                   color=[class_colors[c] for c in class_counts.index])
    ax2.set_xlabel("Flare Class")
    ax2.set_ylabel("Count")
    ax2.set_title("Flare Class Distribution")
    
    plt.tight_layout()
    return fig
```

## Alert System

```python
class FlareAlertSystem:
    """Real-time flare alert with configurable thresholds."""
    
    def __init__(self, nowcaster, forecaster):
        self.nowcaster = nowcaster
        self.forecaster = forecaster
        self.alert_history = []
    
    def check(self, current_counts_solexs, current_counts_hel1os, current_time):
        """
        Run detection + forecasting on current data.
        
        Returns
        -------
        dict with:
            - nowcast: detected flare (or None)
            - forecast: predicted flare (or None)
            - alert_level: "NONE" / "WATCH" / "WARNING" / "ALERT"
        """
        # Nowcasting: any active flare?
        nowcast = self.nowcaster.detect(
            current_counts_solexs,
            current_counts_hel1os
        )
        
        # Forecasting: any flare coming?
        forecast = self.forecaster.predict(
            current_counts_solexs[-3600:],   # last hour
            current_counts_hel1os[-3600:]    # last hour
        )
        
        # Determine alert level
        alert_level = "NONE"
        if nowcast:
            alert_level = "ALERT"
        elif forecast["probability"] > 0.8:
            alert_level = "WARNING"
        elif forecast["probability"] > 0.5:
            alert_level = "WATCH"
        
        return {
            "nowcast": nowcast,
            "forecast": forecast,
            "alert_level": alert_level,
            "time": current_time
        }
```

---

## Streamlit Dashboard Template

```python
# streamlit_app.py
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Solar Flare Dashboard", layout="wide")
st.title("☀️ Solar Flare Nowcasting & Forecasting Dashboard")

# Sidebar controls
st.sidebar.header("Configuration")
year = st.sidebar.slider("Year", 2024, 2026, 2025)
month = st.sidebar.slider("Month", 1, 12, 6)
day = st.sidebar.slider("Day", 1, 31, 15)

# Load data
data = load_day(year, month, day)

# Main panels
col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("X-ray Light Curves")
    
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        subplot_titles=["SoLEXS (2-22 keV)", "HEL1OS CZT (20-150 keV)", "Forecast Probability"],
                        row_heights=[0.4, 0.3, 0.3])
    
    # SoLEXS
    fig.add_trace(go.Scatter(
        x=data["solexs_time"] / 3600,
        y=data["solexs_counts"],
        mode="lines", name="SoLEXS",
        line=dict(color="blue", width=0.5)
    ), row=1, col=1)
    
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Status")
    
    # Nowcast
    nowcast = nowcaster.detect(...)
    if nowcast:
        st.error("🔴 FLARE DETECTED")
        st.write(f"Peak flux: {nowcast[0]['flux_peak']:.0f} cts/s")
        st.write(f"Class: {nowcast[0]['flare_class']}")
    else:
        st.success("🟢 No active flare")
    
    # Forecast
    forecast = forecaster.predict(...)
    st.metric("Flare Probability", f"{forecast['probability']:.1%}")
    st.metric("Lead Time", f"{forecast['lead_time']/60:.0f} min")
    
    if forecast['confidence'] == 'HIGH':
        st.warning(f"⚠️ HIGH confidence — flare predicted in ~{forecast['lead_time']/60:.0f} min")
```

---

## Output Directory Structure

```
output/
├── plots/
│   ├── overview/
│   │   └── YYYY-MM-DD_overview.png
│   ├── spectral/
│   │   └── YYYY-MM-DD_flare_XX_keV.png
│   └── statistics/
│       ├── flare_distribution.png
│       └── flare_duration_vs_flux.png
├── catalogs/
│   ├── solexs_flares.csv
│   ├── combined_flares.csv
│   └── forecast_log.csv
└── models/
    ├── lgbm_nowcaster_v1.pkl
    ├── lgbm_forecaster_v1.pkl
    └── cnn_lstm_forecaster_v1.pt
```
