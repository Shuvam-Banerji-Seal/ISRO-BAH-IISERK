# Results & Performance — Image Generation Prompt

## Objective
Generate a 16:9 slide image showing key results in a 2×2 dashboard layout: flare detection accuracy, feature pipeline performance, model comparison, and physical validation metrics.

## Visual Style
- **Design aesthetic**: Performance dashboard style, similar to executive summary slides in scientific conferences. Clean, data-rich, with charts and callout numbers.
- **Color palette**:
  - Background: Light gray (#F0F3F5) with subtle grid
  - Quadrant 1 (Detection): Amber (#F39C12)
  - Quadrant 2 (Features): Blue (#3498DB)
  - Quadrant 3 (Models): Purple (#9B59B6)
  - Quadrant 4 (Physics): Green (#2ECC71)
  - Text: Dark navy (#2C3E50)
  - Numbers: Bold, colored by quadrant
- **Font**: Inter/Helvetica, data-dense
- **Dashboard cards**: White (#FFFFFF) background, 2px left border in quadrant color, shadow (y=3px, blur=8px)

## Layout (16:9)
```
┌──────────────────────────────────────────────────────────────┐
│  RESULTS & PERFORMANCE — Solar Flare Forecasting Pipeline    │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌───────────────┐  ┌───────────────┐                        │
│  │ FLARE         │  │ FEATURE       │                        │
│  │ DETECTION     │  │ PIPELINE      │                        │
│  │               │  │               │                        │
│  │ ● 9 flares    │  │ ● 179/179 OK  │                        │
│  │ ● 0 false pos │  │ ● 0 NaN       │                        │
│  │ ● 8 HXR conf  │  │ ● 10s/day    │                        │
│  │ ● X6.3 peak   │  │ ● 50 CSVs    │                        │
│  └───────────────┘  └───────────────┘                        │
│                                                              │
│  ┌───────────────┐  ┌───────────────┐                        │
│  │ MODEL         │  │ PHYSICAL      │                        │
│  │ PERFORMANCE   │  │ VALIDATION    │                        │
│  │               │  │               │                        │
│  │ CatBoost:     │  │ Neupert r:    │                        │
│  │ TSS 0.412     │  │ 0.877         │                        │
│  │ AUC 0.795     │  │ Temp: 7.4 MK  │                        │
│  │               │  │ Gamma: 1.66   │                        │
│  └───────────────┘  └───────────────┘                        │
│                                                              │
│  [Mini comparison bar chart at bottom]                       │
│  CatBoost ████████████ 0.412  │ XGBoost ██████████ 0.371    │
│  LightGBM █████████ 0.331    │ CNN-LSTM █████████ 0.341    │
└──────────────────────────────────────────────────────────────┘
```

## Elements to Include

### Quadrant 1 — Flare Detection (Amber, top-left)
- **Title**: "FLARE DETECTION" with target icon
- **Key metric**: "9 flares detected on 2024-05-05" (large number: 9)
- Metrics list with icons:
  - ✅ "4 X-class + 5 M-class (from SoLEXS)"
  - ✅ "8/9 confirmed by HEL1OS HXR"
  - ✅ "0 false positives (HXR gate)"
  - ✅ "X6.3 peak flare detected"
- Validation note: "GOES catalog: 2 X + 8 M + 4 C — good agreement considering calibration differences"

### Quadrant 2 — Feature Pipeline (Blue, top-right)
- **Title**: "FEATURE PIPELINE" with gear/chart icon
- **Key metric**: "179/179 Features Non-Zero" (large number: 179)
- Metrics list:
  - ✅ "100% feature coverage (179/179)"
  - ✅ "0 NaN values in output"
  - ✅ "~10 seconds per day total runtime"
  - ✅ "50+ CSVs generated across 10 months of data"
- Sub-badge: "GPU: 133 feats in 1.1s | CPU: 46 feats in 4s"

### Quadrant 3 — Model Performance (Purple, bottom-left)
- **Title**: "MODEL PERFORMANCE" with bar chart icon
- **Ranked list** (best first):
  1. 🥇 "CatBoost GPU: TSS=0.412, AUC=0.795" (gold highlight)
  2. 🥈 "XGBoost CPU: TSS=0.371, AUC=0.783" (silver)
  3. 🥉 "CNN-LSTM v3: TSS=0.341, AUC=0.741" (bronze)
  4. "LightGBM CPU: TSS=0.331, AUC=0.736"
- Note: "Training data: 158,998 windows × 179 features"

### Quadrant 4 — Physical Validation (Green, bottom-right)
- **Title**: "PHYSICAL VALIDATION" with flask/science icon
- **Key metric**: "Neupert r = 0.877" (large number)
- Metrics list:
  - 🔬 "Neupert effect confirmed (integral form)"
  - 🔬 "r=0.877 matches literature range 0.57-0.90"
  - 🔬 "Temperature: 7.4 MK (thermal bremsstrahlung)"
  - 🔬 "Spectral index γ=1.66 (electron acceleration)"
  - 🔬 "QPP: not detected (boundary-excluded, correct)"

### Bottom — TSS Comparison Bar Chart (Full width)
Horizontal bar chart showing relative TSS scores:
```
CatBoost GPU  ████████████████████░░ 0.412  ← best
XGBoost CPU   █████████████████░░░░░ 0.371
CNN-LSTM v3   ███████████████░░░░░░░ 0.341
LightGBM CPU  █████████████░░░░░░░░░ 0.331
```
Gold highlight on CatBoost bar. X-axis labeled "TSS (True Skill Score)"

## Typography
- Title: 22pt bold #2C3E50
- Dashboard card titles: 14pt bold, colored by quadrant
- Key numbers: 28pt bold, quadrant color
- Metric text: 11pt regular #555
- Bar chart labels: 10pt #666

## Special Effects
- Dashboard cards: White background with colored left border accent (4px)
- Number emphasis: Large bold numbers with slight text shadow
- Medal icons: 🥇🥈🥉 next to top 3 models
- Checkmark icons: ✅ for positive results
- Bar chart: Gradient bars from light to saturated color
- Grid background: Subtle 20px grid pattern on main background
