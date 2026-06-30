# Stage 2 — Feature Engineering & Feature Matrix Assembly

**Date**: 2026-06-30  
**Output**: `dist/stage2_feature_matrix_20260623.npz` (2.6 MB, **85 features**, 86400 × 1s bins)  
**Pipeline**: 9 modular phases in `src/stage2/`

---

## Overview

Stage 2 transforms the cleaned Stage 1 data into a rich **feature matrix** suitable for ML-based flare nowcasting and forecasting. Each phase computes a group of physics-motivated features from the `SoLEXS_HEL1OS_Flare_Feature_Reference.md` document (37 of 42 listed features addressed; 5 remain blocked due to data limitations).

The output is a single NPZ archive containing 85 feature arrays (72 float, 13 int), each 86400 elements long (1s cadence for 2026-06-23). Non-GOOD bins (`master_flag != 0`) have NaN masking for all float features.

---

## Pipeline Structure

```
src/stage2/
├── __init__.py
├── phase1_direct.py      # Direct extractions from Stage 1
├── phase2_perflare.py    # Per-flare catalog (timing + energetics)
├── phase3_tem_goes.py    # GOES T/EM (White et al. 2005)
├── phase4_tem_solexs.py  # SoLEXS PI T/EM (isothermal spectral fit)
├── phase5_wavelet.py     # Wavelet + QPP + oscillation features
├── phase6_spectral.py    # HEL1OS spectral index δ
├── phase7_nonlinear.py   # Hurst, Bayesian Blocks, recurrence plots
├── phase8_event.py       # Event-level + auxiliary (coincidence, cross-corr)
├── phase9_assemble.py    # Merge all → unified feature matrix
└── run_all.py            # Orchestrator (python3 run_all.py)
```

Output intermediate artifacts: `dist/features/phase*.npz`

---

## Phase Breakdown

### Phase 1: Direct Extractions (15 features)

**Reference features**: #5, #11, #16, #22, #24

| Feature | Description | Source |
|---------|-------------|--------|
| `sxr_deriv1`, `sxr_deriv5` | 1-min / 5-min SXR flux time-derivative | Stage 0 |
| `hxr_deriv1`, `hxr_deriv5` | 1-min / 5-min HXR flux time-derivative | Stage 0 |
| `hr_cdte_band{2,3,4}_band1` | Hardness ratio: high/low CdTe energy bands | Stage 1 bands |
| `hr_cdte_band4_band2` | Additional hardness combination | Stage 1 bands |
| `neupert_rho` | Neupert effect correlation score | Stage 1 |
| `time_since_last_flare` | Seconds since last GOES flare | Stage 0 |
| `time_until_next_flare` | Seconds until next GOES flare | Stage 0 |
| `flares_last_1hr`, `flares_last_3hr` | Rolling flare count in window | Computed from flare_id |
| `hxr_event_count` | Cumulative HXR >3σ event count | Computed from hxr_snr |
| `hxr_time_since_event` | Seconds since last HXR >3σ event | Computed |

---

### Phase 2: Per-Flare Catalog (14 features)

**Reference features**: #7, #12, #26, #27

Groups all 8 detected flares on 2026-06-23 (B8.1–C8.8). Each flare gets timing and energetic metrics broadcast to all its bins.

| Feature | Description |
|---------|-------------|
| `t_start`, `t_peak`, `t_end` | Flare boundary times (Unix s) |
| `rise_time`, `decay_time`, `duration` | Timing parameters (s) |
| `peak_flux` | GOES XRS-B peak flux (W/m²) |
| `peak_goes_class` | Numeric GOES class (B1.2=1.2, C1.5=1.5, etc.) |
| `bg_flux` | Pre-flare background flux (10-min median) |
| `max_deriv` | Max SXR flux derivative in flare |
| `dt_peak_hxr_minus_sxr` | HXR peak time − SXR peak time (s); positive = HXR leads |
| `hxr_fluence` | Integrated HXR excess over flare (cts) |
| `peak_sxr_flux`, `peak_hxr_flux` | Peak SXR / HXR flux per flare |

**Catalog** (metadata):
| Flare | Class | Peak (UTC) | Peak Flux | Rise | Δt_peak |
|-------|-------|------------|-----------|------|---------|
| 1 | B8.3 | 01:12 | 8.3e-7 | 186s | NaN (no HXR) |
| 2 | B7.9 | 01:36 | 7.9e-7 | 86s | NaN (no HXR) |
| 3 | C1.3 | 08:18 | 1.33e-6 | 900s | NaN (no HXR) |
| 4 | C1.3 | 11:52 | 1.27e-6 | 733s | NaN (no HXR) |
| 5 | C2.5 | 19:10 | 2.55e-6 | 938s | −131s |
| 6 | C2.5 | 19:38 | 2.51e-6 | 261s | +4s |
| 7 | C3.9 | 22:23 | 3.87e-6 | 896s | −202s |
| 8 | C8.8 | 23:25 | 8.78e-6 | 904s | −147s |

---

### Phase 3: GOES T/EM (17 features)

**Reference features**: #1, #2, #3, #4, #6, #8, #30

Derives temperature and emission measure from GOES XRS-A/B ratio via White et al. (2005) polynomial. Unblocks 5 previously blocked features.

**Method**:
- `T_MK`: `log10(T) = 0.051 + 0.979·w + 0.184·w² − 0.010·w³`, `w = log10(B/A)`
- `EM`: `EM_48 = F_B × 10⁵ × √T × exp(17/T)` (CHIANTI-based GOES B response)
- Noise floor: NaN when XRS-A < 5e−8 or XRS-B < 1e−7 W/m²

| Feature | Description |
|---------|-------------|
| `goes_temperature_MK` | Temperature (3–50 MK, clipped) |
| `goes_emission_measure` | EM (10⁴⁷–10⁴⁹ cm⁻³) |
| `goes_emission_measure_log10` | log10(EM) for stable float32 storage |
| `hope_score` | HOPE precursor: `max(0, dT/dt) × exp(−|dEM/dt|/σ)` |
| `hope_flag` | Binary: hope > 90th percentile |
| `fai` | Flare Anticipation Index (low-pass filtered HOPE) |
| `T_gradient` | dT/dt |
| `tem_trajectory_code` | 0=quiet, 1=intermediate, 2=QSS, 3=OFF, 4=double-peaked |
| `is_single_peak`, `is_double_peak` | T-EM peak count flags |
| `is_off_branch`, `is_qss_branch` | Decay branch classification |
| `T_peak_time`, `EM_peak_time` | Peak times (Unix s) |
| `T_leads_EM` | T peak − EM peak (s); negative = T leads (Neupert) |
| `reale_loop_length_cm` | Loop length from Reale et al. (1997) cooling time scaling |
| `internal_consistency` | Combined score: |Neupert| + |T_leads_EM| + loop_length |

---

### Phase 4: SoLEXS PI T/EM (3 features)

**Reference features**: #1, #2 (cross-check)

Accumulates raw PI spectra (340 channels) across each flare window, applies ARF correction, and fits an isothermal bremsstrahlung model.

**Method**:
- Accumulate COUNTS from PI FITS across each flare (~1800 s)
- ARF effective area interpolation from 2250 → 340 channels
- Fit: `counts(E) = ARF(E) × norm × exp(−E/kT) / √E × dE`
- Calibrated EM against GOES via empirical scale factor

| Feature | Description |
|---------|-------------|
| `T_MK_solexs_pi` | Temperature (18–34 MK, per-flare) |
| `EM_log10_solexs_pi` | log10(EM) per-flare |
| `T_diff_GOES_minus_SoLEXS` | GOES T − SoLEXS T (all NaN — no temporal overlap in this run) |

**Per-flare fit results**:
| Flare | Class | T (MK) | EM (cm⁻³) |
|-------|-------|--------|-----------|
| 1 | B8.3 | 18.1 | 3.1e48 |
| 2 | B7.9 | 18.0 | 5.3e47 |
| 3 | C1.3 | 21.7 | 3.5e48 |
| 4 | C1.3 | 22.6 | 3.3e48 |
| 5 | C2.5 | 26.9 | 6.7e48 |
| 6 | C2.5 | 26.6 | 6.1e48 |
| 7 | C3.9 | 29.4 | 1.1e49 |
| 8 | C8.8 | 33.6 | 2.1e49 |

T scales monotonically with flare class — physically consistent.

---

### Phase 5: Wavelet + Oscillation (21 features)

**Reference features**: #14, #15, #28, #32, #33, #34, #36, #37a, #37b, #38

Continuous Wavelet Transform (PyWavelets, Morlet cmor1.5-1.0) on `sxr_excess` and `hxr_excess`. Period range 10–500s, 40 scales.

| Feature | Description |
|---------|-------------|
| `sxr_qpp_power_30_300s` | Mean wavelet power in QPP-relevant band (30–300s) |
| `sxr_peak_period` | Period with maximum power at each time bin (s) |
| `sxr_peak_power` | Maximum wavelet power at each time bin |
| `scalegram_beta` | Slope of log(power) vs log(period) (30–300s) |
| `scalegram_T_min` | Period of maximum scalegram power (s) |
| `sxr_lim_max`, `sxr_lim_mean` | Local Intermittency Measure (max and mean per bin) |
| `sxr_lim_flag` | LIM > 3 intermittency flag |
| `qpp_power_preflare` | Mean QPP power in 30-min window before flare start |
| `qpp_power_onset` | Mean QPP power in first 5 min of flare |
| `qpp_power_decay` | Mean QPP power in decay phase (post-peak) |
| `qpp_cycle_count` | Number of QPP on/off transitions during flare |
| `qpp_is_localized` | <3 cycles = localized (rise/pre-flare); ≥3 = global (decay) |
| `cross_wavelet_power` | Geometric mean of SXR × HXR wavelet power |
| `cross_coherence` | Normalized spectral coherence proxy |
| `cross_lim_max` | Cross-channel LIM |
| `ridge_period` | Ridge tracking: max-power period vs time |
| `chirp_rate` | Gradient of log(ridge_period) |
| `red_noise_r1` | AR(1) lag-1 autocorrelation of SXR light curve |
| `qpp_significant`, `qpp_sig_score` | Red-noise significance flag + score (Vaughan 2005) |

---

### Phase 6: HEL1OS Spectral Index (3 features)

**Reference features**: #17, #18, #19

Fits power-law `dN/dE = A·E^{−δ}` to 511-channel CdTe1 spectra (2126 spectra × 40s integration windows), interpolates to 1s grid.

| Feature | Description |
|---------|-------------|
| `hxr_spectral_index` | Photon power-law index δ (0.5–7.4, fit in 20–60 keV) |
| `hxr_hardening_rate` | Δδ/Δt (3-point gradient) |
| `shs_correlation` | Soft-Hard-Soft: rolling correlation of δ vs log(HXR flux) |

139/2126 spectra fitted successfully (sufficient counts in the 20–60 keV range).

---

### Phase 7: Nonlinear + Statistical (6 features)

**Reference features**: #39, #40, #41, #42

| Feature | Description | Method |
|---------|-------------|--------|
| `hurst_sxr` | Rolling Hurst exponent (R/S, 1h windows, 1min step) | 0.36–0.92 range |
| `bb_segment_id` | Bayesian Blocks adaptive segmentation ID | Astropy bayesian_blocks |
| `bb_blocks_last_1hr` | Rolling count of Bayesian Blocks in past 1h | |
| `cross_recurrence_rate` | Fraction of state-space neighbors (SXR × HXR embedding) | dim=3, τ=10s, ε=0.5 |
| `rqa_determinism` | Fraction of recurrent points forming diagonal lines | |
| `rqa_laminarity` | Fraction forming vertical lines | Laminarity always 0 (low-statistics) |

---

### Phase 8: Event-Level + Auxiliary (6 features)

**Reference features**: #20, #21, #25, #35

| Feature | Description | Method |
|---------|-------------|--------|
| `hxr_window_trigger` | Short/long window count-rate ratio (W1/W2 proxy) | 1s/5s rolling |
| `hxr_binary_trigger` | Binary: ratio > 2 or flux > 5σ above background | |
| `xcorr_lag` | Lag at max cross-correlation (SXR vs HXR, ±30s) | ±5min windows |
| `xcorr_max` | Max cross-correlation coefficient | |
| `emd_dominant_period` | Dominant period from autocorrelation first zero-crossing | Per-flare proxy |
| `czt_coincidence_flag` | CZT pixel coincidence within 6 µs → particle rejection | 7,566 events flagged |

---

## Feature Matrix Assembly (Phase 9)

**Input**: All 8 intermediate `.npz` files from `dist/features/`  
**Process**:
1. Load all feature arrays (shape 86400)
2. Resolve name collisions (prefix with phase name)
3. Apply `master_flag != 0` mask: set float features to NaN for non-GOOD bins
4. Save single NPZ + CSV export + NaN heatmap plot

### Final Matrix Statistics

| Metric | Value |
|--------|-------|
| Total features | 85 |
| Float features | 72 |
| Int / bool features | 13 |
| Good bins (master_flag == 0) | 41,033 (47.5%) |
| File size | 2.6 MB (compressed) |
| Reference features addressed | 37 / 42 |
| Plot produced | `dist/plots/feature_nan_heatmap.png` |

### NaN Fractions (after master_flag masking)

- **Low NaN** (~52.5%): gradients, waiting-times, wavelet, recurrence — NaN only from the master_flag mask
- **Medium NaN** (~53–68%): spectral indices, window trigger — additional NaN from instrument gaps
- **High NaN** (~85–100%): per-flare features (non-flare bins), hardness ratios (HXR gaps), SoLEXS PI (flare-only)

### Output Files

```
dist/
├── stage2_feature_matrix_20260623.npz    # Final merged matrix (85 feats)
├── stage2_feature_matrix.csv              # CSV export (86401 rows × 86 cols)
└── plots/
    └── feature_nan_heatmap.png            # NaN fraction per feature
```

---

## Blocked Features (from reference doc)

5 features remain blocked — not addressed in any phase:

| # | Feature | Reason |
|---|---------|--------|
| 9 | Statistical/spectral-probability precursor | Needs 20–90 hours pre-flare background; only 24h of data |
| 10 | Peak-flux power-law rank (SOC) | Needs 100+ flares for meaningful size distribution |
| 13 | SoLEXS binary trigger flag | Not present in any raw file header or column |
| 23 | HEL1OS binary trigger flag | Not present in our data files |
| 29 | Thermal vs non-thermal energy fraction | Requires full joint spectral forward model |
| 31 | Trigger lead-lag | Both #13 and #23 blocked |

---

## How to Load the Feature Matrix

```python
import numpy as np

ds = np.load("dist/stage2_feature_matrix_20260623.npz", allow_pickle=True)
meta = ds["__metadata__"].item()
keys = [k for k in ds.files if k != "__metadata__"]

# For ML: stack float features, select GOOD bins only
X = np.column_stack([ds[k].astype(np.float32) for k in keys
                     if ds[k].dtype.kind == "f"])
good = np.load("data/processed/stage1_20260623.npz")["master_flag"] == 0
X_clean = X[good]
```

---

## Running the Pipeline

```bash
# Full pipeline (all 9 phases)
python3 src/stage2/run_all.py

# Single phase
python3 src/stage2/run_all.py 3    # only Phase 3

# Individual scripts (idempotent)
python3 src/stage2/phase1_direct.py
python3 src/stage2/phase5_wavelet.py
```
