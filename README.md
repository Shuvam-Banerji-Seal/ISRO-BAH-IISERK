# ISRO-BAH-IISERK — Solar Flare Forecasting with Aditya-L1

**Bharatiya Antariksh Hackathon 2026 — Challenge #15**  
**Team:** IISER Kolkata  
**Instruments:** SoLEXS (soft X-rays, 2–22 keV) + HEL1OS (hard X-rays, 1.8–160 keV)

---

## Pipeline Overview

```mermaid
flowchart TB
    subgraph Data Ingestion
        A[SoLEXS SDD2 LC] --> B[Raw SXR Counts 86400]
        C[HEL1OS 4 detectors] --> D[Raw HXR Counts 86400x20]
        E[SoLEXS PI] --> F[PI Spectra 86400x340]
        G[HEL1OS HK] --> H[Housekeeping 62 cols]
        I[GOES XRS] --> J[GOES Flux]
    end

    subgraph Corrections
        B --> K[Deadtime corrected SXR]
        D --> L[Background subtracted HXR]
        F --> M[Calibrated PI]
        K --> N[GTI masked SXR]
        L --> O[Aligned HXR]
    end

    subgraph GPU_Batch[A100 GPU Batch]
        N --> P[SXR Windows 277x3600]
        O --> Q[HXR Windows 277x3600x20]
        P --> R[Stats + ACF + Entropy]
        P --> S[Derivatives + Multiscale]
        P --> T[Neupert Correlation]
        Q --> U[HXR Band Features]
        Q --> V[Cross-detector Stats]
    end

    subgraph CPU_Features
        M --> W[Temperature EM chi2]
        O --> X[Spectral Indices gamma]
        O --> Y[Non-thermal params]
        O --> Z[Granger + Mediation]
        O --> AA[QPP Detection]
        H --> AB[HK Statistics]
        J --> AC[GOES Flux]
        J --> AD[GOES Time Series]
        M --> AE[Window Spectral]
        O --> AF[Wavelet Scalogram]
    end

    subgraph Feature_Matrix[179 Feature Matrix]
        R --> AG
        S --> AG
        T --> AG
        U --> AG
        V --> AG
        W --> AG
        X --> AG
        Y --> AG
        Z --> AG
        AA --> AG
        AB --> AG
        AC --> AG
        AD --> AG
        AE --> AG
        AF --> AG
    end

    subgraph Output
        AG --> AH[Master CSV 277x277]
        AG --> AI[Interpretation JSON]
        AH --> AJ[CatBoost XGBoost LightGBM]
        AJ --> AK[TSS AUC Precision Recall]
    end
```

---

## Data Flow

```mermaid
flowchart LR
    subgraph Input
        A1[SoLEXS FITS] --> B1[SXR Counts]
        A1 --> C1[PI Spectra]
        D1[HEL1OS FITS] --> E1[HXR Bands]
        D1 --> F1[HK Data]
        G1[GOES NC] --> H1[Flux]
    end

    subgraph Pipeline
        B1 --> I1[generate_master_csv.py]
        C1 --> I1
        E1 --> I1
        F1 --> I1
        H1 --> I1
        I1 --> J1[CSV 277 x 277]
        I1 --> K1[Interpretation JSON]
    end

    subgraph Output
        J1 --> L1[179 GPU Features]
        J1 --> M1[80 CPU Features]
        J1 --> N1[18 Metadata]
        K1 --> O1[15 Analysis Sections]
        K1 --> P1[19 Feature Groups]
    end
```

---

## Feature Extraction Architecture

```mermaid
flowchart TD
    subgraph GPU[GPU Batch A100]
        A1[277 Windows] --> B1[GPU Tensor]
        B1 --> C1[Stats 15 feats]
        B1 --> D1[ACF 4 feats]
        B1 --> E1[Spectral Entropy 2 feats]
        B1 --> F1[Derivatives 12 feats]
        B1 --> G1[Multiscale 24 feats]
        B1 --> H1[Neupert 2 feats]
        B1 --> I1[HXR Bands 35 feats]
        B1 --> J1[Cross-detector 6 feats]
        C1 --> K1[133 GPU Feats in 1.1s]
        D1 --> K1
        E1 --> K1
        F1 --> K1
        G1 --> K1
        H1 --> K1
        I1 --> K1
        J1 --> K1
    end

    subgraph CPU[CPU Day Level]
        L1[Full Day Data] --> M1[Temperature EM chi2]
        L1 --> N1[Spectral Index 4 detectors]
        L1 --> O1[Non-thermal fit]
        L1 --> P1[HK Stats 8]
        L1 --> Q1[GOES Flux 3]
        L1 --> R1[Granger Causality]
        L1 --> S1[Mediation]
        L1 --> T1[QPP Detection]
        L1 --> U1[Info Theory 6]
        L1 --> V1[GOES Time Series 8]
        L1 --> W1[Window Spectral 8]
        L1 --> X1[Wavelet 10]
        M1 --> Y1[46 CPU Feats in 4s]
        N1 --> Y1
        O1 --> Y1
        P1 --> Y1
        Q1 --> Y1
        R1 --> Y1
        S1 --> Y1
        T1 --> Y1
        U1 --> Y1
        V1 --> Y1
        W1 --> Y1
        X1 --> Y1
    end

    K1 --> Z1[179 Total Features]
    Y1 --> Z1
    Z1 --> AA1[Master CSV 277 x 277]
    AA1 --> AB1[Interpretation JSON]
```

---

## Interpretation Pipeline

```mermaid
flowchart LR
    A[Master CSV] --> B[interpretation.py]
    B --> C[Flare Catalog]
    B --> D[Neupert Effect]
    B --> E[Cross Correlation]
    B --> F[Power Spectrum]
    B --> G[QPP Analysis]
    B --> H[Spectral Evolution]
    B --> I[Causal Network]
    B --> J[Feature Groups 19]
    C --> K[Interpretation JSON]
    D --> K
    E --> K
    F --> K
    G --> K
    H --> K
    I --> K
    J --> K
```

---

## Project Structure

```
isro-bah-iiserk/
├── AGENTS.md                    # Problem statement + implementation state
├── README.md                    # This file
├── docs/
│   ├── PLAN.md                  # Research plan + mathematical framework
│   ├── RESULTS.md               # Analysis results
│   └── analysis/                # Data exploration notes
├── src/bah2026/
│   ├── config.py                # Paths, constants, parameters
│   ├── main.py                  # CLI + pipeline orchestration
│   ├── data/
│   │   ├── reader.py            # FITS data loaders
│   │   ├── corrections.py       # Deadtime, background subtraction
│   │   ├── preprocessing.py     # Alignment, GTI masking
│   │   ├── calibration.py       # SoLEXS→GOES conversion
│   │   ├── ground_truth.py      # GOES catalog validation
│   │   └── sequence_builder.py  # DL sequence preparation
│   ├── features/
│   │   ├── engineering.py       # 179 canonical feature definitions
│   │   ├── gpu_features.py      # GPU batch functions (A100)
│   │   ├── advanced_features.py # GOES TS, wavelet, per-window spectral
│   │   ├── spectral_fitting.py  # Temperature, spectral index, Neupert
│   │   ├── non_thermal.py       # Thick-target bremsstrahlung fitting
│   │   ├── causal_network.py    # Granger causality, mediation
│   │   ├── information_theory.py # Transfer entropy, mutual info
│   │   ├── qpp.py               # QPP detection (wavelet + LS)
│   │   └── interpretation.py    # Physical interpretation pipeline
│   ├── models/
│   │   ├── nowcasting.py        # SWPC flare detection
│   │   ├── adaptive_detection.py # Adaptive threshold detection
│   │   ├── forecasting.py       # CatBoost, XGBoost, LightGBM
│   │   ├── cnn_lstm_v3.py       # 3.0M param deep learning
│   │   ├── transformer.py       # 3.7M param transformer
│   │   └── mae_pretrain.py      # 5.6M param masked autoencoder
│   ├── scripts/
│   │   └── generate_master_csv.py # Single-day analysis pipeline
│   └── visualization/
├── scripts/
│   └── run_all.py               # Batch runner for all 724 days
├── tests/                       # 120+ pytest tests
└── output/
    ├── master_csv/              # Generated CSVs + interpretations
    ├── models/                  # Trained model checkpoints
    └── hdf5/                    # Feature matrices
```

---

## Quick Start

```bash
# Single day analysis (GPU-accelerated)
.venv/bin/python3 src/bah2026/scripts/generate_master_csv.py 2024-05-05

# Output:
#   output/master_csv/master_May_5_2024.csv
#   output/master_csv/master_May_5_2024_interpretation.json

# Run tests
PYTHONPATH=src .venv/bin/python3 -m pytest tests/ -v

# Train forecasting model
PYTHONPATH=src .venv/bin/python3 -c "from bah2026.main import cmd_train; cmd_train()"
```

## Key Results

| Metric | Value |
|--------|-------|
| Detection (X-class) | X6.3 flare on 2024-05-05 |
| False positives | 0 |
| CatBoost TSS | 0.412 |
| CatBoost AUC | 0.795 |
| Neupert correlation r | 0.877 (integral form) |
| Feature coverage | 179/179 (100%) |
| Pipeline runtime/day | ~10s |
