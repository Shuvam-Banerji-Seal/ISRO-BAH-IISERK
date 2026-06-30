# Feature Importance: Category-Wise Ranking

**Source**: Stage 2 correlation analysis — Spearman ρ on GOOD bins (master_flag==0, 41,033 of 86,400)

**Targets**:
| Target | Purpose | Balance (0/1) |
|--------|---------|---------------|
| `y_in_flare` | Nowcast: is a flare currently in progress? | 35,212 / 5,821 |
| `y_flare_30m` | Forecast: will a flare start within 30 min? | 35,780 / 5,253 |
| `y_flare_1h` | Forecast: will a flare start within 1 hour? | 30,647 / 10,386 |
| `y_deep_quiet` | Deep quiet: >30 min from any flare | 15,715 / 25,318 |
| `y_flare_class` | Flare class (0=quiet, 1=B, 2=C+) | 0 / 10,240 / 30,793 |

**Ranking method**: Within each category, features sorted by |ρ| for **y_in_flare** (nowcast). Features with all-NaN correlations (flare-internal) are listed separately.

---

## 1. Nonlinear / Statistical — Phase 7

6 features from recurrence quantification analysis (RQA), Hurst exponent, and Bayesian Blocks. **Strongest forecast precursors.**

| # | Feature | Nowcast ρ | FC30m ρ | FC1h ρ | DeepQ ρ | Class ρ | Validity |
|---|---------|-----------|---------|--------|---------|---------|----------|
| 1 | `rqa_determinism` | **−0.505** | **−0.480** | **−0.608** | +0.614 | −0.719 | 39,583 |
| 2 | `cross_recurrence_rate` | **−0.525** | **−0.457** | **−0.564** | +0.628 | −0.716 | 39,583 |
| 3 | `hurst_sxr` | **+0.468** | **+0.434** | **+0.456** | −0.594 | +0.658 | 39,583 |
| 4 | `bb_blocks_last_1hr` | +nan | +nan | +nan | +nan | +nan | 41,033 |
| 5 | `bb_segment_id` | +nan | +nan | +nan | +nan | +nan | 41,033 |
| 6 | `rqa_laminarity` | +nan | +nan | +nan | +nan | +nan | 39,583 |

**Interpretation:**
- RQA determinism drops (system becomes more chaotic) before flares — **best single forecast feature** (ρ=−0.48)
- SXR-HXR cross-recurrence drops before flares — the two channels decouple as instability builds
- Hurst exponent rises (more persistent/predictable) before flares
- Bayesian Blocks and laminarity have zero variance in GOOD bins (constant-valued) — remove from ML

---

## 2. Wavelet / QPP — Phase 5

21 features from continuous wavelet transform (CWT, Morlet, 10–500s), cross-wavelet, QPP cycle analysis, and LIM. **Dominant nowcast category.**

| # | Feature | Nowcast ρ | FC30m ρ | FC1h ρ | DeepQ ρ | Class ρ | Validity |
|---|---------|-----------|---------|--------|---------|---------|----------|
| 1 | `qpp_cycle_count` | **+0.809** | −0.126 | +0.099 | −0.418 | +0.190 | 41,033 |
| 2 | `qpp_is_localized` | **−0.598** | +0.093 | −0.214 | +0.308 | −0.140 | 41,033 |
| 3 | `cross_coherence` | **+0.524** | +0.248 | +0.256 | −0.599 | +0.478 | 41,033 |
| 4 | `sxr_peak_power` | **+0.522** | +0.340 | +0.418 | −0.688 | +0.637 | 41,033 |
| 5 | `cross_lim_max` | **+0.520** | +0.270 | +0.315 | −0.608 | +0.536 | 41,033 |
| 6 | `cross_wavelet_power` | **+0.519** | +0.323 | +0.361 | −0.688 | +0.631 | 41,033 |
| 7 | `qpp_sig_score` | **+0.507** | +0.341 | +0.429 | −0.700 | +0.653 | 41,033 |
| 8 | `sxr_qpp_power_30_300s` | **+0.505** | +0.303 | +0.400 | −0.693 | +0.584 | 41,033 |
| 9 | `sxr_lim_mean` | **+0.499** | +0.309 | +0.394 | −0.636 | +0.616 | 41,033 |
| 10 | `sxr_lim_max` | **+0.472** | +0.283 | +0.364 | −0.570 | +0.570 | 41,033 |
| 11 | `sxr_lim_flag` | **+0.448** | +0.181 | +0.245 | −0.505 | +0.253 | 41,033 |
| 12 | `ridge_period` | +0.108 | +0.173 | +0.071 | −0.167 | +0.187 | 41,033 |
| 13 | `sxr_peak_period` | +0.108 | +0.173 | +0.071 | −0.167 | +0.187 | 41,033 |
| 14 | `chirp_rate` | −0.001 | −0.001 | −0.001 | +0.001 | −0.001 | 41,033 |
| 15 | `qpp_power_decay` | +nan | +nan | +0.497 | +nan | +nan | 5,821 |
| 16 | `qpp_power_onset` | +nan | +nan | −0.147 | +nan | +nan | 5,821 |
| 17 | `qpp_power_preflare` | +nan | +nan | −0.122 | +nan | +nan | 5,821 |
| 18 | `qpp_significant` | +nan | +nan | +nan | +nan | +nan | 41,033 |
| 19 | `red_noise_r1` | +nan | +nan | +nan | +nan | +nan | 41,033 |
| 20 | `scalegram_T_min` | +nan | +nan | +nan | +nan | +nan | 41,033 |
| 21 | `scalegram_beta` | +nan | +nan | +nan | +nan | +nan | 41,033 |

**Interpretation:**
- QPP cycle count (ρ=+0.81) is excellent for nowcasting but is a byproduct of flare duration
- Cross-coherence (ρ=+0.52), wavelet power (ρ=+0.52), and LIM (ρ=+0.50) are genuine physics features
- QPP power features are stronger for 1h forecast than 30min — QPP precursors may build over longer timescales
- Chirp rate and scalegram metrics are zero-variance — remove from ML

---

## 3. GOES Temperature / Emission Measure — Phase 3

17 features from White et al. (2005) GOES XRS-A/B ratio method, T-EM trajectory classification, Reale loop scaling.

| # | Feature | Nowcast ρ | FC30m ρ | FC1h ρ | DeepQ ρ | Class ρ | Validity |
|---|---------|-----------|---------|--------|---------|---------|----------|
| 1 | `is_double_peak` | **+1.000** | −0.156 | −0.003 | −0.516 | +0.234 | 41,033 |
| 2 | `tem_trajectory_code` | **+1.000** | −0.156 | −0.003 | −0.516 | +0.234 | 41,033 |
| 3 | `goes_temperature_MK` | **−0.573** | +0.160 | +0.048 | +0.292 | +0.054 | 15,868 |
| 4 | `goes_emission_measure_log10` | **+0.446** | +0.045 | +0.099 | −0.630 | +0.052 | 15,868 |
| 5 | `internal_consistency` | +0.365 | +0.108 | +0.029 | −0.357 | +0.385 | 40,747 |
| 6 | `fai` | +0.169 | +0.036 | +0.097 | −0.175 | +0.089 | 41,033 |
| 7 | `hope_score` | +0.013 | −0.002 | +0.006 | −0.012 | +0.001 | 15,868 |
| 8 | `T_gradient` | −0.004 | +0.002 | −0.001 | −0.002 | +nan | 14,605 |
| 9 | `T_leads_EM` | +nan | +nan | **+0.766** | +nan | +nan | 5,821 |
| 10 | `T_peak_time` | +nan | +nan | +0.497 | +nan | +nan | 5,821 |
| 11 | `EM_peak_time` | +nan | +nan | +0.497 | +nan | +nan | 5,821 |
| 12 | `goes_emission_measure` | +nan | +nan | +nan | +nan | +nan | 15,868 |
| 13 | `hope_flag` | +nan | +nan | +nan | +nan | +nan | 41,033 |
| 14 | `is_off_branch` | +nan | +nan | +nan | +nan | +nan | 41,033 |
| 15 | `is_qss_branch` | +nan | +nan | +nan | +nan | +nan | 41,033 |
| 16 | `is_single_peak` | +nan | +nan | +nan | +nan | +nan | 41,033 |
| 17 | `reale_loop_length_cm` | +nan | +nan | −0.766 | +nan | +nan | 5,821 |

**Interpretation:**
- `is_double_peak` and `tem_trajectory_code` are perfect proxies (ρ=1.0) — they encode the T-EM trajectory which is always in "flare" mode when a flare is active. **Drop from ML** — they are labels, not predictors.
- `goes_temperature_MK` (ρ=−0.57): temperature dips before/at flare onset — could be a precursor
- `goes_emission_measure_log10` (ρ=+0.45): EM rises during flares
- Per-flare features (T_peak, EM_peak, T_leads_EM) only have 5,821 valid bins — useful only for 1h forecast
- HOPE / FAI scores are weak — they were designed for larger flares (M/X class), not C-class

---

## 4. Direct Extractions — Phase 1

15 features from derivatives, hardness ratios, waiting-times, and event counts.

| # | Feature | Nowcast ρ | FC30m ρ | FC1h ρ | DeepQ ρ | Class ρ | Validity |
|---|---------|-----------|---------|--------|---------|---------|----------|
| 1 | `flares_last_1hr` | **+0.701** | +0.048 | +0.068 | −0.737 | +0.120 | 41,033 |
| 2 | `hxr_time_since_event` | **−0.454** | −0.203 | −0.234 | +0.473 | −0.451 | 41,033 |
| 3 | `time_until_next_flare` | **−0.453** | **−0.464** | **−0.616** | +0.505 | −0.724 | 39,495 |
| 4 | `hxr_event_count` | +0.346 | **+0.374** | **+0.522** | −0.529 | +0.717 | 41,033 |
| 5 | `flares_last_3hr` | +0.346 | −0.097 | −0.065 | −0.410 | −0.057 | 41,033 |
| 6 | `time_since_last_flare` | −0.301 | +0.076 | +0.120 | +0.436 | +0.132 | 41,033 |
| 7 | `neupert_rho` | +0.195 | +0.032 | −0.108 | −0.208 | +0.318 | 40,747 |
| 8 | `hxr_deriv5` | −0.161 | +0.099 | +0.065 | +0.057 | −0.040 | 40,897 |
| 9 | `hr_cdte_band4_band1` | −0.080 | −0.029 | −0.050 | +0.081 | −0.050 | 7,227 |
| 10 | `hr_cdte_band4_band2` | +0.055 | −0.054 | −0.003 | −0.002 | +0.015 | 1,274 |
| 11 | `hxr_deriv1` | −0.040 | +0.048 | +0.040 | +0.008 | +0.001 | 41,017 |
| 12 | `hr_cdte_band2_band1` | +0.022 | +0.012 | −0.005 | −0.040 | +0.001 | 7,214 |
| 13 | `sxr_deriv1` | −0.014 | +0.064 | +0.089 | +0.021 | −0.003 | 41,017 |
| 14 | `sxr_deriv5` | −0.001 | +0.081 | +0.102 | +0.052 | −0.014 | 40,897 |
| 15 | `hr_cdte_band3_band1` | +0.001 | +0.003 | −0.003 | +0.004 | +0.005 | 7,212 |

**Interpretation:**
- `flares_last_1hr` (ρ=+0.70) is a simple but strong nowcast feature — flares cluster in time
- `time_until_next_flare` (ρ=−0.46 for FC30m, −0.62 for FC1h) is a **trivial feature** (it directly encodes the forecast target) — **remove from ML** to avoid data leakage
- `hxr_event_count` (ρ=+0.37 for FC30m) is a genuine forecast precursor — HXR activity builds before flares
- `neupert_rho` (ρ=+0.20) is weak — the Neupert effect is subtle for C-class flares
- Hardness ratios have low validity (1,274–7,227 bins) due to HXR data gaps — unreliable

---

## 5. Event-Level / Auxiliary — Phase 8

6 features from window triggers, cross-correlation lag, EMD, and CZT coincidence.

| # | Feature | Nowcast ρ | FC30m ρ | FC1h ρ | DeepQ ρ | Class ρ | Validity |
|---|---------|-----------|---------|--------|---------|---------|----------|
| 1 | `xcorr_max` | +0.137 | +0.084 | +0.006 | −0.242 | +0.210 | 40,747 |
| 2 | `xcorr_lag` | −0.041 | −0.054 | −0.027 | −0.034 | +0.089 | 40,747 |
| 3 | `hxr_binary_trigger` | +0.026 | +0.018 | +0.020 | −0.031 | +0.028 | 41,033 |
| 4 | `czt_coincidence_flag` | +0.017 | +0.008 | +0.007 | −0.021 | +0.000 | 41,033 |
| 5 | `hxr_window_trigger` | +0.000 | −0.000 | −0.002 | −0.000 | −0.009 | 27,383 |
| 6 | `emd_dominant_period` | +nan | +nan | −0.222 | +nan | +nan | 5,821 |

**Interpretation:**
- Cross-correlation features are weak for this dataset — only C-class flares, where SXR-HXR coupling is modest
- CZT coincidence and window triggers are near-zero — designed for particle events, not flare detection
- EMD dominant period only valid during flares — weak for 1h forecast

---

## 6. HEL1OS Spectral Index — Phase 6

3 features from power-law fitting of CdTe1 spectra (20–60 keV).

| # | Feature | Nowcast ρ | FC30m ρ | FC1h ρ | DeepQ ρ | Class ρ | Validity |
|---|---------|-----------|---------|--------|---------|---------|----------|
| 1 | `hxr_spectral_index` | +0.077 | −0.012 | +0.019 | −0.040 | +0.018 | 40,576 |
| 2 | `hxr_hardening_rate` | +0.071 | −0.011 | +0.018 | −0.038 | +0.017 | 40,576 |
| 3 | `shs_correlation` | +nan | +nan | +nan | +nan | +nan | 37 |

**Interpretation:**
- Very weak correlations — spectral fitting only succeeded on 139/2126 spectra due to low counts
- SHS correlation is essentially all-NaN (37 valid bins)

---

## 7. SoLEXS PI T/EM — Phase 4

3 features from per-flare isothermal spectral fit of SoLEXS PI data.

| # | Feature | Nowcast ρ | FC30m ρ | FC1h ρ | DeepQ ρ | Class ρ | Validity |
|---|---------|-----------|---------|--------|---------|---------|----------|
| 1 | `T_MK_solexs_pi` | +nan | +nan | +0.497 | +nan | +nan | 5,821 |
| 2 | `EM_log10_solexs_pi` | +nan | +nan | +0.497 | +nan | +nan | 5,821 |
| 3 | `T_diff_GOES_minus_SoLEXS` | +nan | +nan | +nan | +nan | +nan | 0 |

**Interpretation:**
- Flare-only features (5,821 bins) — no value for nowcast (always NaN outside flares)
- Moderate ρ for 1h forecast but only 8 data points (per-flare constants broadcast to all bins)
- `T_diff_GOES_minus_SoLEXS` is all-NaN — GOES and SoLEXS PI don't cover overlapping times with both valid

---

## 8. Per-Flare Catalog — Phase 2

14 features from the 8-flare catalog. All are flare-internal (NaN outside flares).

| # | Feature | Nowcast ρ | FC30m ρ | FC1h ρ | DeepQ ρ | Class ρ | Validity |
|---|---------|-----------|---------|--------|---------|---------|----------|
| 1 | `bg_flux` | +nan | +nan | **−0.766** | +nan | +nan | 5,821 |
| 2 | `decay_time` | +nan | +nan | **+0.766** | +nan | +nan | 5,821 |
| 3 | `dt_peak_hxr_minus_sxr` | +nan | +nan | **−0.766** | +nan | +nan | 5,821 |
| 4 | `duration` | +nan | +nan | +0.645 | +nan | +nan | 5,821 |
| 5 | `peak_goes_class` | +nan | +nan | +0.549 | +nan | +nan | 5,821 |
| 6 | `hxr_fluence` | +nan | +nan | +0.497 | +nan | +nan | 5,821 |
| 7 | `max_deriv` | +nan | +nan | +0.497 | +nan | +nan | 5,821 |
| 8 | `peak_flux` | +nan | +nan | +0.497 | +nan | +nan | 5,821 |
| 9 | `peak_sxr_flux` | +nan | +nan | +0.497 | +nan | +nan | 5,821 |
| 10 | `t_end` | +nan | +nan | +0.497 | +nan | +nan | 5,821 |
| 11 | `t_peak` | +nan | +nan | +0.497 | +nan | +nan | 5,821 |
| 12 | `t_start` | +nan | +nan | +0.497 | +nan | +nan | 5,821 |
| 13 | `rise_time` | +nan | +nan | −0.147 | +nan | +nan | 5,821 |
| 14 | `peak_hxr_flux` | +nan | +nan | −0.122 | +nan | +nan | 5,821 |

**Interpretation:**
- These are per-flare constants broadcast to all flare bins — no variance within a flare
- ρ values for FC1h are inflated by the binary in-flare/out-of-flare structure (only 8 data points)
- `decay_time` (ρ=+0.77) and `dt_peak_hxr_minus_sxr` (ρ=−0.77) are trivially correlated with duration
- **Use with caution** — they create perfect separation for nowcast but are not available in real-time until the flare ends

---

## Summary: Best Features for ML

### For Nowcasting (flares in progress)
| Rank | Feature | Category | ρ | Note |
|------|---------|----------|---|------|
| 1 | `flares_last_1hr` | Direct | +0.70 | Simple, powerful — flare clustering |
| 2 | `cross_coherence` | Wavelet | +0.52 | SXR-HXR coherence during flares |
| 3 | `sxr_peak_power` | Wavelet | +0.52 | Wavelet power in SXR |
| 4 | `cross_wavelet_power` | Wavelet | +0.52 | Cross-power |
| 5 | `qpp_sig_score` | Wavelet | +0.51 | QPP significance |
| 6 | `sxr_lim_mean` | Wavelet | +0.50 | Intermittency |
| 7 | `rqa_determinism` | Nonlinear | −0.50 | Chaos drops during flares |
| 8 | `hurst_sxr` | Nonlinear | +0.47 | Persistence rises |
| 9 | `hxr_event_count` | Direct | +0.35 | HXR activity |

### For Forecasting (30 min ahead)
| Rank | Feature | Category | ρ | Note |
|------|---------|----------|---|------|
| 1 | `rqa_determinism` | Nonlinear | −0.48 | **Best precursor** — system becomes chaotic |
| 2 | `cross_recurrence_rate` | Nonlinear | −0.46 | SXR-HXR decouple before flare |
| 3 | `hurst_sxr` | Nonlinear | +0.43 | Persistence builds |
| 4 | `hxr_event_count` | Direct | +0.37 | HXR foreshock activity |
| 5 | `qpp_sig_score` | Wavelet | +0.34 | QPP builds before flare |
| 6 | `sxr_peak_power` | Wavelet | +0.34 | Pre-flare wavelet power |
| 7 | `cross_wavelet_power` | Wavelet | +0.32 | Pre-flare cross-power |
| 8 | `sxr_lim_mean` | Wavelet | +0.31 | Intermittency builds |
| 9 | `sxr_qpp_power_30_300s` | Wavelet | +0.30 | QPP band power |

### Features to Remove from ML
| Feature | Reason |
|---------|--------|
| `is_double_peak`, `tem_trajectory_code` | ρ=1.0 — they are flare labels, not predictors |
| `time_until_next_flare`, `time_since_last_flare` | Directly encode the target — leakage |
| `flares_last_1hr`, `flares_last_3hr` | Correlated with flare clustering — partial leakage |
| `bb_segment_id`, `bb_blocks_last_1hr`, `rqa_laminarity` | All-NaN or constant |
| `chirp_rate`, `scalegram_beta`, `scalegram_T_min` | All-NaN or constant |
| `red_noise_r1`, `qpp_significant` | All-NaN |
| `goes_emission_measure` (float) | All-NaN — use log10 version |
| `hope_flag`, `is_off_branch`, `is_qss_branch`, `is_single_peak` | All-NaN |
| `T_diff_GOES_minus_SoLEXS` | All-NaN — no temporal overlap |
| `shs_correlation` | Only 37 valid bins |
| All per-flare features (Phase 2) | Not available in real-time for prediction |

### Clean Feature Count for ML
After removing leakage + zero-variance + all-NaN: **~18 float features** viable for modeling.

**Top-3 for nowcast model:** `sxr_peak_power`, `cross_coherence`, `hurst_sxr`
**Top-3 for forecast model:** `rqa_determinism`, `cross_recurrence_rate`, `hurst_sxr`
