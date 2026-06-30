# isro-pipeline

Feature engineering pipeline for solar flare nowcasting and forecasting using combined SoLEXS (soft X-ray) and HEL1OS (hard X-ray) data from Aditya-L1.

## Stages

| Stage | Description | Output |
|-------|-------------|--------|
| 0 | Master dataset assembly (cross-instrument time alignment) | `master_dataset_YYYYMMDD.npz` |
| 1 | Preprocessing: quality flags, background subtraction, GOES cross-calibration | `stage1_YYYYMMDD.npz` |
| 2 | Feature engineering (9 phases) + correlation analysis | `stage2_feature_matrix_YYYYMMDD.npz` |

## Quick Start

```bash
cd pipeline

# Run all 9 phases of Stage 2 feature engineering
python3 run_all.py

# Run a single phase
python3 run_all.py 5    # wavelet features only

# Run correlation analysis (requires stage2_feature_matrix to exist)
python3 src/stage2/correlation_analysis.py
```

## Data Setup

**Processed NPZ files** are bundled in `pipeline/data/processed/`.

**Raw FITS data** must be symlinked. Run the setup script:

```bash
./setup_data.sh /path/to/your/data/root
```

Example:
```bash
./setup_data.sh /home/alok/codes/Isro/data
```

The data root should contain:
```
<root>/
├── raw/
│   ├── solexs/YYYYMMDD/SDD2/AL1_SOLEXS_YYYYMMDD_SDD2_L1.{lc,pi,gti}
│   ├── hel1os/YYYYMMDD/events/evt.fits
│   ├── hel1os/YYYYMMDD/cdte/hel1os_cdte_spectra_cdte*.fits
│   └── caldb/solexs_tools-1.1/CALDB/{arf,response}/
└── processed/
    ├── master_dataset_YYYYMMDD.npz
    └── stage1_YYYYMMDD.npz
```

If the symlink is missing, phases that read raw data will fail with
`FileNotFoundError`. Phases 1, 2, 3, 7, 9 work from processed NPZs only.

## Output

All outputs go to `pipeline/dist/`:
- `features/` — per-phase intermediate NPZ files
- `stage2_feature_matrix_YYYYMMDD.npz` — merged feature matrix (85 features)
- `stage2_feature_matrix.csv` — CSV export
- `plots/` — correlation plots and NaN heatmap

## Dependencies

numpy, scipy, matplotlib, astropy, PyWavelets
