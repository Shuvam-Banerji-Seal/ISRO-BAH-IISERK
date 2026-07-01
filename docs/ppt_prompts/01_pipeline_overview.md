# Pipeline Overview — Image Generation Prompt

## Objective
Generate a professional 16:9 slide image showing the complete data pipeline from Aditya-L1 satellite data ingestion to final interpretation output. The diagram should show a vertical top-to-bottom flow with 5 major stages.

## Visual Style
- **Design aesthetic**: Clean, modern, scientific. Similar to Nature/Science journal graphical abstracts.
- **Color palette**:
  - Data Input (top): Warm amber/orange gradient (#F39C12 → #E67E22)
  - Corrections (upper middle): Cool gray (#95A5A6 → #7F8C8D)
  - GPU Batch (center): Deep blue (#3498DB → #2980B9)
  - CPU Analysis (lower center): Green (#2ECC71 → #27AE60)
  - Output Products (bottom): Crimson red (#E74C3C → #C0392B)
- **Background**: Off-white (#F8F9FA) with subtle grid pattern
- **Font**: Sans-serif (Segoe UI or Helvetica), hierarchical sizing
- **Shadows**: Soft drop shadows on all containers (y-offset 3px, blur 6px, opacity 0.15)
- **Border radius**: 12px on containers, 8px on nodes
- **Connectors**: Curved arrows, 2.5px stroke, arrowhead size 8px, color matching source container

## Layout (16:9, landscape)
```
┌─────────────────────────────────────────────────────────────┐
│  [TITLE] Solar Flare Analysis Pipeline — Aditya-L1          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────── STAGE 1: DATA INGESTION ───────────────┐  │
│  │  [SoLEXS icon]  [HEL1OS icon]  [PI icon]  [HK icon]   │  │
│  │  5 boxes in a row, orange gradient background          │  │
│  └──────────────────────┬──────────────────────────────────┘  │
│                         ▼                                    │
│  ┌─────────────── STAGE 2: CORRECTIONS ──────────────────┐  │
│  │  5 interconnected boxes, gray background               │  │
│  │  Deadtime -> Background Subtraction -> GTI -> Align    │  │
│  └──────────────────────┬──────────────────────────────────┘  │
│                         ▼                                    │
│  ┌─────────── STAGE 3: GPU BATCH (A100) ──────────────────┐  │
│  │  8 function boxes arranged in 2 rows of 4              │  │
│  │  Blue background, "133 Features in 1.1s" callout       │  │
│  │  Small GPU chip icon in corner                         │  │
│  └──────────────────────┬──────────────────────────────────┘  │
│                         ▼                                    │
│  ┌─────────── STAGE 4: CPU DAY-LEVEL ─────────────────────┐  │
│  │  10 function boxes arranged in 2 rows of 5             │  │
│  │  Green background, "46 Features in 4s" callout         │  │
│  └──────────────────────┬──────────────────────────────────┘  │
│                         ▼                                    │
│  ┌─────────── STAGE 5: OUTPUT ────────────────────────────┐  │
│  │  3 boxes: 179-Feature Vector -> CSV -> Interpretation  │  │
│  │  Red background, document icon                         │  │
│  └─────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Elements to Include

### Stage 1 — Data Ingestion (Orange, 5 boxes)
1. **SoLEXS SDD2**: Satellite icon + text "Soft X-rays 2-22 keV, 86400 samples/day"
2. **HEL1OS**: Multi-band icon + text "Hard X-rays 1.8-160 keV, 20 energy bands"
3. **PI Spectra**: Spectrum graph icon + text "86400 x 340 channels"
4. **Housekeeping**: Gauge icon + text "62 telemetry columns"
5. **GOES XRS**: Satellite dish icon + text "1-min averaged flux"

### Stage 2 — Corrections (Gray, 5 boxes in sequence)
1. **Deadtime Correction**: Formula icon + "Paralyzable, tau=13.65us"
2. **Background Subtraction**: Filter icon + "CZT: 70 cps, CdTe: 0.15 cps"
3. **Channel Calibration**: Ruler icon + "Energy response"
4. **GTI Masking**: Scissors icon + "Earth occultation removal"
5. **Temporal Alignment**: Clock icon + "Align to SoLEXS grid, linear interpolation"

### Stage 3 — GPU Batch (Blue, 8 function boxes)
Arrange in 2 rows of 4 with a "NVIDIA A100 80GB" badge:
1. **Batch Stats**: Chart icon + "mean, std, max, min, skew, kurtosis"
2. **Batch ACF**: Wave icon + "lags 5s, 10s, 30s, 60s"
3. **Spectral Entropy**: Frequency icon + "Welch PSD + FFT"
4. **Derivatives**: Slope icon + "dSXR/dt, d2SXR/dt2, dHXR/dt"
5. **Multiscale**: Pyramid icon + "5min, 15min, 30min stats"
6. **Neupert**: Correlation icon + "Sliding rho dSXR/dt vs HXR"
7. **HXR Features**: Bar chart icon + "10 bands x 3 stats"
8. **Cross-detector**: Grid icon + "CZT1/2, CdTe1/2 totals"

Include a highlighted callout box: **"133 Features computed in 1.1 seconds"**

### Stage 4 — CPU Day-Level (Green, 10 function boxes)
Arrange in 2 rows of 5:
1. **Temperature & EM**: Thermometer icon + "Thermal bremsstrahlung fit"
2. **Spectral Indices**: Gamma icon + "Power-law fit, 4 detectors"
3. **Non-thermal**: Lightning icon + "Combined spectrum fit"
4. **HK Stats**: Dashboard icon + "8 detector telemetry features"
5. **Granger Causality**: Arrow diagram icon + "HXR -> dSXR/dt"
6. **QPP Detection**: Sine wave icon + "Wavelet + Lomb-Scargle"
7. **GOES Flux**: Flux icon + "XRS-B, XRS-A, ratio"
8. **Window Spectral**: Window icon + "Per-window T, EM, gamma"
9. **Wavelet**: Spectrogram icon + "5 energy bands, cross-power"
10. **Mediation**: Path diagram icon + "Baron-Kenny analysis"

Include a highlighted callout box: **"46 Features computed in 4 seconds"**

### Stage 5 — Output (Red, 3 boxes in sequence)
1. **179-Feature Vector**: Database icon + "277 windows x 179 features"
2. **Master CSV**: Spreadsheet icon + "277 x 277 columns per day"
3. **Interpretation JSON**: Document icon + "15 analysis sections, 19 feature groups"

## Typography
- Title: 28pt bold, dark navy (#2C3E50)
- Stage headers: 16pt bold, white text on colored background
- Node titles: 14pt bold, dark text
- Node descriptions: 11pt regular, dark gray (#555)
- Callout numbers: 24pt bold, colored text
- Labels: 10pt, medium gray (#777)

## Special Effects
- Stage containers: Semi-transparent colored background (opacity 0.08) with 2px border of stage color
- Connecting arrows between stages: Wide (4px), dashed, dark gray (#666)
- GPU icon in Stage 3: Small chip/circuit board illustration
- Document icon in Stage 5: Small paper/document illustration
- Progress indicators: Small filled circles at each stage boundary
