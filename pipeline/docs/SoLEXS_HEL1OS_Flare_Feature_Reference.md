# Feature Engineering Reference for Solar Flare Forecasting/Nowcasting
## SoLEXS (Soft X-ray) + HEL1OS (Hard X-ray) Time Series — Aditya-L1

This document consolidates every physics-motivated feature discussed, organized by data source, with the supporting literature for each. Use it as the master reference for the feature engineering pipeline.

---

## I. SoLEXS-only (Soft X-ray / Thermal) Features

| # | Feature | Definition / How to Compute | Citation |
|---|---------|------------------------------|----------|
| 1 | Isothermal temperature (T) | Derived from ratio R = XRS-A(short)/XRS-B(long) via polynomial fit | White, Thomas & Schwartz (2005); Woods et al. (2024), *arXiv:2509.05234* |
| 2 | Emission measure (EM) | Derived alongside T from the same flux-ratio fit | White et al. (2005); sunkit-instruments documentation |
| 3 | HOPE signature ("horizontal branch") | Initial near-constant-T, rising-EM segment in the T–EM diagram before the impulsive phase | Hudson et al., "Anticipating Solar Flares," *Solar Physics* (2025), DOI:10.1007/s11207-024-02418-4 |
| 4 | Flare Anticipation Index (FAI) | Precursor alert score built from the HOPE T–EM branch; gives few-minute lead time | Hudson et al. (2025), *Solar Physics* |
| 5 | Flux time-derivative | d(XRS-A)/dt, d(XRS-B)/dt — onset trigger and Neupert-effect input | Woods et al., *arXiv:2509.05234* |
| 6 | T–EM trajectory shape | Single- vs. double-peaked evolution; which decay branch (OFF vs. QSS) the flare follows | Limb flare T/EM case study, *arXiv:2406.13339*; Kołomański, *A&A* (2011) |
| 7 | Rise time, decay/e-folding time, duration | Standard light-curve timing parameters | Ryan et al., "What Determines the X-Ray Intensity and Duration of a Solar Flare?" *ApJ* (2019) |
| 8 | Reale-law loop length | L = τ_LC·√T / (α·F(ζ)), where ζ is the slope of the decay-phase EM–T diagram | Reale et al. (1997); validated in *arXiv:1610.09811*; reviewed in *arXiv:0705.3254* |
| 9 | Statistical/spectral-probability precursor | Evolving shape of the probability density of small-scale SXR background fluctuations, 20–90 hr before a major flare | "On statistical precursors of solar flares," *ScienceDirect* (2025) |
| 10 | Peak-flux power-law rank (SOC) | Where a flare's peak flux sits on the known SOC power-law size distribution | Aschwanden, "Self-organized Criticality in Solar and Stellar Flares," *ApJ* 880, 105 (2019) |
| 11 | Waiting-time / recency features | Time since last flare, rolling flare count, locally fitted waiting-time power-law slope | Aschwanden, "25 Years of SOC: Numerical Detection Methods," *arXiv:1506.08142* |
| 12 | Tier-0 schema features | Start time, peak time, end time, preflare background flux, detection threshold, peak flux, max flux-derivative | Aschwanden & Freeland, "Automated Solar Flare Statistics in Soft X-rays over 37 Years of GOES," *ApJ* 754:112 (2012) |
| 13 | SoLEXS binary trigger flag | Instrument's own onboard flare flag (usable as feature or label cross-check) | "The Aditya-L1 mission of ISRO," *arXiv:2212.13046* |
| 14 | Pre-flare QPP wavelet power | Long-period (minutes) oscillation power before onset; precursor signal | "Preflare very long-periodic pulsations...," *A&A* (2020) |
| 15 | Onset QPP/FFT band power | Sudden rise in power at specific frequency bands at flare start; nowcasting trigger | Observations of preflare QPP in coronal loop/microwave, *arXiv:2003.09567* |
| 32 | Decay-phase QPP / sausage-mode power | Globally distributed wavelet power, >3 coherent cycles, located in the decay phase | "Detection and Interpretation of Long-Lived X-Ray QPP in the X-Class Solar Flare on 2013 May 14," *arXiv:1706.03689*; Srivastava, "Flares and QPPs in Some Stellar Coronae" (talk) |
| 33 | QPP localization class (cycle count) | <3 cycles = rise/pre-flare localized power; >3 cycles = decay-phase global power | Srivastava (talk, derived from Doyle et al. 2018, *MNRAS*); "Localizing QPPs...," *arXiv:2408.05463* |
| 36 | Wavelet scalegram: N(T), slope β, T_min | Multiresolution scalegram converted to a timescale distribution function; T_min correlates with loop size | Aschwanden, "Wavelet Analysis of Solar Flare Hard X-Rays," *ApJ* 505, 941 (1998) — *(also applicable to SoLEXS)* |
| 38 | Red-noise-corrected QPP significance | Excess power above a broken-power-law (red+white noise) confidence envelope, instead of raw FFT/wavelet power | Vaughan, "A simple test for periodic signals in red noise" (2005); Pugh et al., *A&A* (2017) |
| 41 | Rolling Hurst exponent (pre-flare window) | R/S analysis on quiet-Sun SoLEXS background; persistence vs. randomness before onset | "Complex Network for Solar Active Regions," *arXiv:1707.02371*; "Statistical properties of solar Hα flare activity" (2017) |
| 42 | Bayesian Blocks segmentation | Adaptive Poisson change-point segmentation of the light curve; block count, change-point times, rate-jump magnitude | Scargle (1998, 2013); applied in stellar/Sgr A* X-ray flare detection, e.g. *arXiv:astro-ph/0605096* |

---

## II. HEL1OS-only (Hard X-ray / Non-thermal) Features

| # | Feature | Definition / How to Compute | Citation |
|---|---------|------------------------------|----------|
| 16 | Hardness ratio | Flux in a high-energy HEL1OS bin ÷ flux in a low-energy bin, rolling in time | Fletcher (2002), "Spectral and Spatial Variations of Flare HXR Footpoints" |
| 17 | Photon/electron power-law spectral index | Fitted per short time window (e.g. OSPEX thick-target fit) | "On the Relationship between Continuum Enhancement and HXR Emission...," *arXiv:astro-ph/0412171* |
| 18 | Spectral-index hardening rate | Δ(spectral index)/Δt within a single flare | "Hard X-ray Emitting Energetic Electrons and Photospheric Electric Currents," *A&A* (2015) |
| 19 | Soft-Hard-Soft (SHS) pattern | Correlation between spectral hardness and flux intensity on ~10 s timescales | Fletcher (2002) |
| 20 | HEL1OS windowed-trigger feature | Continuous-valued version of the mission's own W1/W2 (short window vs. long window) count-rate comparison, with Count_Th/Flux_Th removed | Varma et al. (2023), cited in "SUIT On-board Intelligence for Flare Observations" |
| 21 | Cosmic-ray/particle coincidence flag | Sub-6 µs coincidence across CZT detector modules → reject as particle hit, not photon | "HEL1OS – A Hard X-ray Spectrometer on Board Aditya-L1," *arXiv:2512.12679* |
| 22 | HXR-specific waiting-time statistics | Recency/clustering features computed from historical HXR flare timing (not SXR) | Boffetta et al. (1999), cited in *arXiv:1506.08142*; Aschwanden & McTiernan (2010), *ApJ* 717, 683 |
| 23 | HEL1OS-derived binary trigger flag | Instrument-side flare flag (fires earlier than SoLEXS flag in flight examples) | "Test and Calibration of SUIT on board Aditya-L1," *arXiv:2503.23476* |
| 34 | Instantaneous frequency / chirp rate | Wavelet-ridge tracking of period vs. time; slope = QPP acceleration/deceleration | Definition of "quasi-period" as a varying instantaneous period: "Detecting Fast-Variation Pulsations in Solar HXR and Radio Emissions," *arXiv:2506.14433* |
| 37 | Local Intermittency Measure (LIM) | Squared cross-wavelet power between two simultaneous channels; LIM² > 3 flags local intermittency, distinguishing avalanche vs. cascade energy release | Dinkelaker & MacKinnon, "Wavelets, Intermittency and Solar Flare Hard X-rays 1," *Solar Physics* 282, 471 (2013) — *(cross-wavelet version is also a combined feature, see #37 in section III)* |

---

## III. Combined SoLEXS + HEL1OS (Joint Physics) Features

| # | Feature | Definition / How to Compute | Citation |
|---|---------|------------------------------|----------|
| 24 | Neupert score (R_N) | Pearson correlation between d(SXR)/dt and HXR flux, per flare | "Observational Evidence Linking Loop Length and Thermal–Nonthermal Peak Timing in Solar Flares," *ApJ* (2026) |
| 25 | Normalized cross-correlation P_zy[m] with lag | Running cross-correlation between SoLEXS and HEL1OS arrays; lag-at-max-correlation as an extra feature | Original proposal; supported by "Hard X-Ray Emission from Partially Occulted Solar Flares," *arXiv:1612.02856* (correlation/lag histograms) |
| 26 | SXR–HXR peak time delay (Δt_peak) | HXR peak time − SXR peak time; correlates (R≈0.88) with loop length | *ApJ* (2026), loop-length/timing paper above |
| 27 | SXR peak flux vs. HXR fluence correlation | Proxy for electron-beam-driven evaporation vs. additional heating agent | Veronig et al., "The Neupert Effect in Solar Flares and Implications for Coronal Heating," *arXiv:astro-ph/0208089* |
| 28 | Cross-instrument QPP coherence | Whether SoLEXS and HEL1OS show wavelet power at the same period | "Quasi-periodic Pulsations before and during a Solar Flare in AR 12242," *ApJ* (2019) |
| 29 | Thermal vs. non-thermal energy fraction | Simultaneous spectral fitting across both instruments' bands | "HEL1OS – A Hard X-ray Spectrometer on Board Aditya-L1," *arXiv:2512.12679* |
| 30 | Internal consistency score | Agreement between R_N, Δt_peak, and Reale-law loop length — a data-quality/validation meta-feature | Derived (combines features #8, #24, #26 — no single source) |
| 31 | Trigger lead-lag | Time difference between SoLEXS flag firing and HEL1OS-derived flag firing | "Test and Calibration of SUIT on board Aditya-L1," *arXiv:2503.23476* |
| 35 | EMD-derived periods (cross-check) | Empirical Mode Decomposition applied independently to both channels' light curves, periods compared against wavelet-derived periods | Doyle, J.G. et al. (2018), *MNRAS* — "Flares on Magnetically Active DMe Stars" |
| 37 (joint) | Local Intermittency Measure between SoLEXS and HEL1OS | Cross-wavelet power spectrum between the two channels (rather than within one), LIM² > 3 flags intermittent joint energy release | Dinkelaker & MacKinnon, *Solar Physics* 282, 471 (2013) |
| 39 | Cross/Joint Recurrence Plot (CRP/JRP) phase-shift feature | Nonlinear recurrence-based synchronization between SoLEXS and HEL1OS — captures multi-lag, multi-scale coupling that Pearson correlation misses | "Statistical Properties of Solar Hα Flare Activity," *J. Space Weather Space Clim.* (2017) |
| 40 | RQA measures: determinism, laminarity | Recurrence-quantification-analysis statistics computed jointly on the two channels; flags dynamical-state transitions | "Recurrence Quantification Analysis of Two Solar Cycle Indices," *J. Space Weather Space Clim.* (2017) |

---

## IV. Practical / Implementation Notes

- **Cadence caveat:** SoLEXS is documented at 1-second temporal resolution (Solar Low Energy X-ray Spectrometer on board Aditya-L1: Ground Calibration paper, *arXiv:2509.26292*). This resolves QPPs with periods of a few seconds and longer but **not** sub-second QPPs reported in some HXR events — confirm HEL1OS's actual cadence/event-mode availability before relying on features #34, #36, or #42 for fast pulsations.
- **Feature #38 (red-noise significance test) should gate features #14, #15, #28, #32, #33** — i.e., don't feed raw QPP power into the model; feed the excess power above the red-noise confidence envelope, or a binary "statistically significant" flag. This directly reduces false-alarm rate from spurious periodicity in red-noise-dominated flare backgrounds.
- **Feature #42 (Bayesian Blocks)** requires unbinned photon arrival times for full benefit; if only binned light curves are available, it still works but loses some of its edge over threshold-based segmentation.
- **Features #21 (coincidence rejection) and #25 (cross-correlation)** together form a two-stage anomaly filter: #21 rejects likely particle hits at the event level (if event-mode data is available), #25 rejects single-channel spikes at the light-curve level.
- **Mission-native baseline:** Features #13, #20, #23, #31 reproduce or extend the actual onboard SUIT/SoLEXS/HEL1OS trigger-fusion algorithm (Varma et al., 2023) — treat this group as your tier-0 baseline that any ML model should beat.

---

*Compiled from the full conversation's literature search. 42 features total: 17 SoLEXS-only, 9 HEL1OS-only, 10 combined (some features span groups, e.g. #37, #38).*
