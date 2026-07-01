# Causal Network Analysis — Image Generation Prompt

## Objective
Generate a 16:9 slide image explaining the causal discovery framework used to understand energy flow between thermal and non-thermal emission channels during solar flares. Three-column layout showing methods, results, and physical interpretation.

## Visual Style
- **Design aesthetic**: Scientific methodology diagram, similar to causal inference papers. Networks, arrows, and flow diagrams.
- **Color palette**:
  - Background: Clean white (#FFFFFF) with light grid
  - Methods column (left): Deep blue (#2C3E50 → #3498DB)
  - Results column (center): Emerald (#2ECC71 → #27AE60)
  - Interpretation column (right): Ruby red (#E74C3C → #C0392B)
  - Causal graph nodes: Gradient circles with energy band colors
  - Arrows: Dark gray (#7F8C8D), varying widths for strength
  - Accent: Gold (#F1C40F) for key numbers
- **Font**: Inter/Helvetica, clean scientific
- **Node style**: Circles for energy bands, rounded rectangles for method boxes

## Layout (16:9)
```
┌──────────────────────────────────────────────────────────────┐
│  CAUSAL NETWORK ANALYSIS — Energy Flow in Solar Flares       │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌── METHODS ────┐    ┌── RESULTS ──────┐   ┌── INTERP ──┐ │
│  │               │    │                  │   │             │ │
│  │ ┌───────────┐ │    │ ┌──────────────┐ │   │ 6→SXR:17s  │ │
│  │ │ Granger   │─┼────┼─┤Improve: 6.6% │ │   │ Strength   │ │
│  │ │ Causality │ │    │ │Best lag: 1s  │ │   │ 0.69       │ │
│  │ └───────────┘ │    │ └──────────────┘ │   │             │ │
│  │ ┌───────────┐ │    │ ┌──────────────┐ │   │ SXR→HXR:1s │ │
│  │ │ Mediation │─┼────┼─┤Mediation:    │ │   │ Strength   │ │
│  │ │ Analysis  │ │    │ │18.1%         │ │   │ 0.54       │ │
│  │ └───────────┘ │    │ └──────────────┘ │   │             │ │
│  │ ┌───────────┐ │    │ ┌──────────────┐ │   │ 14 Feedback │ │
│  │ │ Info      │─┼────┼─┤Trans Entropy │ │   │ Loops       │ │
│  │ │ Theory    │ │    │ │0.038         │ │   │ Detected    │ │
│  │ └───────────┘ │    │ └──────────────┘ │   │             │ │
│  └───────────────┘    └──────────────────┘   │ Cycle=True  │ │
│                                              │             │ │
│  ┌───────────────── CAUSAL GRAPH ──────────┐  │             │ │
│  │      [SXR 2-22keV] ←── [CZT 20-40]     │  │             │ │
│  │         ↕                    ↕           │  │             │ │
│  │     [CZT 80-150] ←── [CZT 40-60]        │  │             │ │
│  │              ↕              ↕            │  └─────────────┘ │
│  │          [CdTe 5-20] ← [CZT 60-80]      │                  │
│  └──────────────────────────────────────────┘                  │
└──────────────────────────────────────────────────────────────┘
```

## Elements to Include

### Left Column — Methods (Blue, 3 boxes stacked)
1. **Granger Causality**: "Does HXR improve prediction of dSXR/dt beyond past SXR?"
   - Formula: AR restricted vs AR full model
   - "RidgeCV + TimeSeriesSplit cross-validation"
   - "Tested lags: 1, 5, 10, 20, 30, 60 seconds"
   - Small arrow diagram showing cause→effect

2. **Mediation Analysis (Baron-Kenny)**:
   - Path diagram: X → M → Y
   - "X: HXR 40-60 keV (cause)"
   - "M: CdTe 20-30 keV (mediator)"
   - "Y: SXR counts (outcome)"
   - "Does the intermediate energy band mediate the causal effect?"

3. **Information Theory**:
   - "Transfer Entropy: non-linear directed info flow"
   - "Mutual Information: total shared info"
   - "Sample Entropy: complexity comparison"
   - Three entropy formulas as small equations

### Center Column — Results (Green, 4 result boxes)
1. **Granger Result**: "HXR→dSXR/dt: +6.6% improvement at lag=1s"
   - Bar chart showing restricted vs full model R²
   - "Statistically significant (p < 0.01)"

2. **Mediation Result**: "18.1% of HXR→SXR effect mediated through CdTe"
   - Pie chart showing direct vs mediated proportion

3. **Network Metrics**: 
   - "Density: 0.52 (52% of connections active)"
   - "Avg in-degree: 3.6, out-degree: 3.6"
   - "14 feedback loops detected"
   - "Cyclic structure: True"

4. **Transfer Entropy**: "TE(HXR→SXR) = 0.038 bits"
   - "Non-linear coupling beyond Granger"
   - "Mutual Info: 0.083 bits"

### Right Column — Physical Interpretation (Red)
List of key findings with causal pathway visualizations:

1. **HXR → SXR (Neupert Pathway)**: 
   - "Lag: 17 seconds, Strength: 0.69"
   - "Non-thermal electrons heat plasma → thermal SXR emission"
   - Visual: Arrow from lightning icon to sun icon, labeled "17s"

2. **SXR → HXR (Feedback Pathway)**:
   - "Lag: 1 second, Strength: 0.54"
   - "Thermal conditions modulate acceleration"
   - Visual: Arrow from sun icon to lightning icon, labeled "1s"

3. **Energy Recirculation**:
   - "14 feedback loops = thermal ↔ non-thermal cycling"
   - "Energy continuously exchanged between channels"
   - Visual: Circular arrow icon

4. **Sample Entropy Contrast**:
   - "HXR: 0.319 (4× more complex than SXR: 0.074)"
   - "Consistent with stochastic acceleration mechanism"

### Bottom — Causal Graph (Full width)
A network graph showing 6 energy bands as colored circles connected by directed arrows:
- Node colors: SXR (red), CZT bands (orange→yellow), CdTe (green)
- Arrow thickness represents connection strength
- Arrow labels show lag in seconds
- Curved arrows for feedback loops
- Self-loop arrows for cycle detection

## Typography
- Title: 24pt bold dark
- Column headers: 16pt bold white
- Method/result names: 13pt bold
- Descriptions: 11pt regular
- Numbers/statistics: 14pt bold with gold highlight for key values
- Causal graph labels: 10pt

## Special Effects
- Arrow animations implied: Gradient arrow fills from transparent to solid
- Network graph: Glowing node edges, pulsing for active connections
- Energy band nodes: Circular gradients matching physical energy color (red=low, blue=high)
- Feedback loops: Circular arrow paths with dashed lines
- Subtle glow on key numbers (6.6%, 18.1%, r=0.877)
