# Feature Extraction Architecture — Image Generation Prompt

## Objective
Generate a 16:9 slide image showing how 179 features are computed from a single day of data. Three-column layout: Input sources (left), GPU batch (center), CPU day-level (right), merging to output (bottom).

## Visual Style
- **Design aesthetic**: Technical diagram style, similar to IEEE/ACM conference figures. Clean, precise, data-dense.
- **Color palette**:
  - Input column (left): Warm amber (#F39C12 → #E67E22)
  - GPU column (center): Deep teal-blue (#1ABC9C → #16A085) with subtle circuit-board pattern overlay
  - CPU column (right): Forest green (#2ECC71 → #27AE60) with subtle pattern overlay
  - Output (bottom): Crimson (#E74C3C → #C0392B)
  - Background: White (#FFFFFF)
  - Connectors: Dark gray (#7F8C8D), 2px
  - Text: Dark (#2C3E50) for headings, medium (#555555) for body
- **Font**: Consolas for technical terms, Inter/Helvetica for descriptions
- **Node style**: Rounded rectangles (radius 6px), 1.5px border, subtle shadow (y=2px, blur=4px, opacity=0.1)
- **Performance callouts**: Pill-shaped badges with white text on colored background

## Layout (16:9)
```
┌──────────────────────────────────────────────────────────────┐
│  FEATURE EXTRACTION PIPELINE — 179 Features from One Day     │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─── INPUT ───┐  ┌─── GPU BATCH (A100) ───┐  ┌─── CPU ───┐│
│  │ SoLEXS      │  │ 8 function boxes        │  │ 10 boxes  ││
│  │ HEL1OS      │  │ in 2 rows of 4          │  │ 2 rows×5  ││
│  │ PI Spectra  │  │ "133 feats in 1.1s"     │  │ "46 feats"││
│  │ HK Data     │  │ ┌──┐ ┌──┐ ┌──┐ ┌──┐   │  │ ┌──┐..    ││
│  │ GOES Flux   │  │ │S1│ │S2│ │S3│ │S4│   │  │ │C1│..    ││
│  └──────┬──────┘  │ └──┘ └──┘ └──┘ └──┘   │  │ └──┘..    ││
│         │         │ ┌──┐ ┌──┐ ┌──┐ ┌──┐   │  │ ┌──┐..    ││
│         │         │ │S5│ │S6│ │S7│ │S8│   │  │ │C6│..    ││
│         │         │ └──┘ └──┘ └──┘ └──┘   │  │ └──┘..    ││
│         └─────────┼──────────┬──────────────┼──┘           │
│                   │          │              │              │
│                   ▼          ▼              ▼              │
│              ┌─────────────────────────────────────┐       │
│              │     179-DIMENSIONAL FEATURE VECTOR  │       │
│              │   277 windows × 179 features per day│       │
│              └──────────────┬──────────────────────┘       │
│                             ▼                              │
│              ┌─────────────────────────────────────┐       │
│              │     MASTER CSV 277 × 277 columns    │       │
│              │  + Interpretation JSON (15 sections)│       │
│              └─────────────────────────────────────┘       │
└──────────────────────────────────────────────────────────────┘
```

## Elements to Include

### Title Bar
"FEATURE EXTRACTION PIPELINE — 179 Features from a Single Day of Data"
Dark text (#2C3E50), 24pt bold, with a thin accent line underneath (2px, #3498DB)

### Left Column — Input Data (Amber, 5 items stacked)
1. **SoLEXS SXR**: Satellite/wave icon + "86400 samples, 1s cadence"
2. **HEL1OS HXR**: Multi-band icon + "20 energy bands, 86400 samples"
3. **PI Spectra**: Spectrum icon + "340 channels, 86400 rows"
4. **HK Data**: Gauge icon + "62 telemetry columns"
5. **GOES Flux**: Graph icon + "XRS-B + XRS-A channels"

Each item: Small icon (16x16px) + label in 11pt. Amber dot connector to right.

### Center Column — GPU Batch (Teal, 8 function boxes)
2 rows × 4 boxes. Each box: 120x80px, teal fill, white text.
Performance badge on top: "133 Features in 1.1 seconds" — pill shape, white text on teal.

Row 1:
1. **Batch Stats**: "mean, std, max, min, skew, kurtosis, rise/fall rates"
2. **Batch ACF**: "Autocorrelation at lags 5, 10, 30, 60s"
3. **Spectral Entropy**: "Welch PSD, FFT peak frequency"
4. **Derivatives**: "dSXR/dt, d2SXR/dt2, dHXR/dt, dHR/dt"

Row 2:
5. **Multiscale**: "5min, 15min, 30min statistics + ratios"
6. **Neupert**: "Sliding rho: corr(dSXR/dt, HXR)"
7. **HXR Features**: "10 bands × 3 stats, hardness ratios"
8. **Cross-detector**: "CZT1/2 + CdTe1/2 totals, detector ratios"

### Right Column — CPU Day-Level (Green, 10 function boxes)
2 rows × 5 boxes. Each box: 100x65px, green fill, white text.
Performance badge: "46 Features in 4 seconds"

Row 1:
1. **Temperature & EM**: "Thermal bremsstrahlung, MK + cm-3"
2. **Spectral Index γ**: "Power-law, 4 detectors"
3. **Non-thermal**: "Combined spectrum, Ec, N_nth"
4. **HK Stats**: "8 telemetry features"
5. **Granger**: "HXR → dSXR/dt causality"

Row 2:
6. **QPP**: "Wavelet + Lomb-Scargle"
7. **GOES Flux**: "XRS-B, XRS-A, ratio"
8. **GOES Time Series**: "ddt, rolling std, gradient"
9. **Window Spectral**: "Per-window T, EM, gamma"
10. **Mediation + Wavelet**: "Causal pathways + scalogram"

### Bottom — Output (Red, 3 items in horizontal row)
1. **179-Feature Vector**: Large database cylinder icon + "277 windows × 179 features"
2. **Master CSV**: Spreadsheet icon + "277 columns × 277 rows per day"
3. **Interpretation**: Document icon + "15 analysis sections, 19 groups"

### Connectors
- Left column → Center: Three thick amber arrows (3px)
- Left column → Right: Three thick amber arrows (3px)  
- Center + Right → Bottom: Two teal and green arrows converging to red output
- Within columns: Thin dark gray arrows (1.5px) for sequential flow

## Typography
- Title: 24pt bold #2C3E50
- Column headers: 14pt bold white
- Function box titles: 11pt bold white
- Function descriptions: 9pt regular white or light gray
- Performance badges: 12pt bold white
- Output labels: 13pt bold #2C3E50

## Special Effects
- GPU acceleration symbol: Small lightning bolt icon in the center column header
- CPU symbol: Small processor chip icon in the right column header
- Column divider: Thin vertical dashed lines (1px, #BDC3C7) between columns
- Number emphasis on performance badges: Highlighted numbers in larger font (16pt)
