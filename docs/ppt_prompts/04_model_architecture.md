# Model Architecture Comparison — Image Generation Prompt

## Objective
Generate a 16:9 slide image comparing four forecasting model architectures: CatBoost, XGBoost, LightGBM (GBDT family) and CNN-LSTM, Transformer (Deep Learning family). Show parameter counts, TSS scores, and architectural differences.

## Visual Style
- **Design aesthetic**: Technical comparison matrix style. Split into two conceptual families.
- **Color palette**:
  - Background: Off-white (#F8F9FA)
  - GBDT family (left): Warm amber/orange (#E67E22 → #D35400)
  - DL family (right): Deep purple (#9B59B6 → #8E44AD)
  - Input features (top bar): Steel blue (#3498DB)
  - Evaluation metrics (bottom bar): Teal (#1ABC9C)
  - Accent: Gold (#F1C40F) for best scores
  - Neutral: Light gray (#ECF0F1) for comparison baselines
- **Font**: Inter/Helvetica, technical feel
- **Node style**: Rounded rectangles with thick left border accent (4px) in family color

## Layout (16:9)
```
┌──────────────────────────────────────────────────────────────┐
│  MODEL ARCHITECTURE COMPARISON — Forecasting Solar Flares    │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌───────────────── INPUT ───────────────────────────────┐  │
│  │           179-Dimensional Feature Vector               │  │
│  │        StandardScaler → Chronological Split            │  │
│  │     70% Train | 15% Validation | 15% Test              │  │
│  └────────────────────┬───────────────────────────────────┘  │
│                       │                                      │
│         ┌─────────────┼─────────────┐                        │
│         ▼             ▼             ▼                        │
│  ┌── GBDT FAMILY ──┐  │  ┌── DL FAMILY ────────────────┐   │
│  │ ┌──────────────┐ │  │  │ ┌────────────────────────┐  │   │
│  │ │ CatBoost GPU │ │  │  │ │ CNN-LSTM v3 - 3.0M    │  │   │
│  │ │ TSS: 0.412   │ │  │  │ │ 4xConv1D→BiLSTM→Attn  │  │   │
│  │ │ AUC: 0.795   │ │  │  │ │ TSS: 0.341             │  │   │
│  │ └──────────────┘ │  │  │ └────────────────────────┘  │   │
│  │ ┌──────────────┐ │  │  │ ┌────────────────────────┐  │   │
│  │ │ XGBoost CPU  │ │  │  │ │ Transformer - 3.7M     │  │   │
│  │ │ TSS: 0.371   │ │  │  │ │ Self-Attn + Cross-Attn │  │   │
│  │ │ AUC: 0.783   │ │  │  │ │ Neupert Physics Loss   │  │   │
│  │ └──────────────┘ │  │  │ └────────────────────────┘  │   │
│  │ ┌──────────────┐ │  │  │ ┌────────────────────────┐  │   │
│  │ │ LightGBM CPU │ │  │  │ │ MAE - 5.6M (Pretrain) │  │   │
│  │ │ TSS: 0.331   │ │  │  │ │ 6-layer encoder        │  │   │
│  │ │ AUC: 0.736   │ │  │  │ │ 50% masking            │  │   │
│  │ └──────────────┘ │  │  │ └────────────────────────┘  │   │
│  └──────────────────┘  │  └─────────────────────────────┘   │
│                        │                                     │
│         ┌──────────────┼──────────────┐                      │
│         ▼              ▼              ▼                      │
│  ┌─────────── EVALUATION METRICS ────────────────────┐      │
│  │  TSS (True Skill Score) | AUC-ROC | AUC-PR | F1  │      │
│  │  Best: CatBoost TSS=0.412, AUC=0.795             │      │
│  └───────────────────────────────────────────────────┘      │
└──────────────────────────────────────────────────────────────┘
```

## Elements to Include

### Title Bar
"FORECASTING MODEL ARCHITECTURES — Comparison of GBDT and Deep Learning Approaches"
Dark text, 24pt bold. Subtitle in 12pt: "Trained on 158,998 windows × 179 features from 724 days"

### Input Features Bar (Blue)
Wide horizontal bar showing: "179-Dimensional Feature Vector" with icons for StandardScaler and a 3-way split arrow showing "70% Train | 15% Validation | 15% Test"

### Left Section — GBDT Family (Orange, 3 model boxes)

**CatBoost GPU** (Best performer — gold highlight border):
- GPU icon
- "iterations=1000, depth=8, lr=0.05"
- "task_type=GPU" 
- Large TSS score: "0.412" (24pt bold gold)
- AUC: "0.795"
- Checkmark icon for "Best Overall"

**XGBoost CPU**:
- CPU icon
- "iterations=1000, depth=8, lr=0.05"
- "n_jobs=-1"
- TSS: "0.371" | AUC: "0.783"

**LightGBM CPU**:
- CPU icon
- "iterations=1000, depth=8, lr=0.05"
- "n_jobs=-1"
- TSS: "0.331" | AUC: "0.736"

### Right Section — Deep Learning Family (Purple, 3 model boxes)

**CNN-LSTM v3** (3.0M parameters):
- Neural network icon
- Architecture: "4×Conv1D → BiLSTM → TemporalAttention → MLP"
- "FocalLoss, Mixed Precision"
- TSS: "0.341"

**Spectral-Temporal Transformer** (3.7M parameters):
- Transformer/attention icon
- Architecture: "Temporal Self-Attention + Energy Cross-Attention"
- "Neupert Physics-Informed Loss, FlashAttention"
- TSS: "N/A" (not yet trained)

**Masked Autoencoder** (5.6M parameters):
- Encoder/decoder icon
- Architecture: "6-layer encoder + 4-layer decoder"
- "50% masking, self-supervised pretraining"
- TSS: "N/A" (pretraining only)

### Evaluation Metrics Bar (Teal)
Wide horizontal bar showing key metrics:
- TSS (True Skill Score): Bar chart visualization with CatBoost highlighted
- AUC-ROC: Secondary metric
- AUC-PR: Tertiary metric
- F1 Score: Quaternary metric
- Highlight box: "Best: CatBoost TSS=0.412, AUC-ROC=0.795"

## Typography
- Title: 24pt bold dark #2C3E50
- Section headers: 16pt bold white
- Model names: 14pt bold white
- TSS scores: 22pt bold gold (#F1C40F) for best, 18pt bold white for others
- Parameters: 10pt regular light gray
- Architecture details: 9pt regular light gray

## Special Effects
- Gold glow around CatBoost box (best performer)
- Small GPU chip icon next to CatBoost
- Small CPU chip icon next to XGBoost/LightGBM
- Neural network icon next to DL models
- Performance bars: Small horizontal bar charts next to each TSS score (relative comparison)
- Dashed separator line between GBDT and DL families
