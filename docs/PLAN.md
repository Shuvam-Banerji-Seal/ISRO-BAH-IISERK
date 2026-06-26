# Research & Implementation Plan — BAH 2026 Challenge #15

**Solar Flare Nowcasting and Forecasting using Combined Soft + Hard X-ray Data from Aditya-L1**

**Team:** IISER Kolkata | **GPU:** NVIDIA A100 80GB | **PyTorch 2.x**

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Rigorous Literature Review](#2-rigorous-literature-review)
   - 2.1 Solar Flare Physics & X-ray Emission
   - 2.2 ML/DL for Solar Flare Forecasting (SOTA)
   - 2.3 Hard X-ray Detection & RHESSI/Fermi/STIX Legacy
   - 2.4 Aditya-L1 Mission & Instruments
   - 2.5 Research Gaps & Novel Opportunities
3. [Mathematical Framework](#3-mathematical-framework)
   - 3.1 X-ray Emission Physics
   - 3.2 GOES Classification & Calibration
   - 3.3 Neupert Effect
   - 3.4 Pre-flare Precursor Signatures
4. [Nowcasting Pipeline (Phase 1)](#4-nowcasting-pipeline-phase-1)
   - 4.1 Data Preprocessing & Quality Control
   - 4.2 Flare Detection Algorithms
   - 4.3 Combined Soft+Hard Nowcasting
5. [Forecasting Pipeline (Phase 2)](#5-forecasting-pipeline-phase-2)
   - 5.1 Feature Engineering
   - 5.2 Baseline Models (LightGBM, CNN-LSTM)
   - 5.3 Transformer Architecture with Cross-Attention
   - 5.4 Physics-Informed Loss Functions
   - 5.5 Uncertainty Quantification
6. [Novel Contributions](#6-novel-contributions)
7. [GPU-Optimized Implementation Plan](#7-gpu-optimized-implementation-plan)
8. [Data Analysis & Visualization Products](#8-data-analysis--visualization-products)
9. [Timeline & Milestones](#9-timeline--milestones)
10. [References](#10-references)

---

## 1. Executive Summary

This project develops the **first-ever automated pipeline** combining soft X-ray data from **SoLEXS** (2–22 keV, 747 days) and hard X-ray data from **HEL1OS** (1.8–160 keV, 927 days) onboard **India's Aditya-L1 mission** to detect (nowcast) and predict (forecast) solar flares. The dataset spans **724 days of dual-instrument coverage** (Feb 2024–Jun 2026) during the rising phase of Solar Cycle 25.

**Key novel contributions:**

1. **First combined soft+hard X-ray flare detection from Aditya-L1** — no prior work has fused SoLEXS and HEL1OS data for flare analysis
2. **Neupert-constrained physics-informed neural network** — embedding the integral relationship between hard X-ray fluence and soft X-ray flux into the loss function
3. **Multi-energy band spectral-temporal transformer** — joint attention over energy (340 SoLEXS + 10 HEL1OS bands) and time (1s cadence)
4. **Flare precursor detection via information-theoretic transfer entropy** — quantifying causality between soft and hard X-ray channels pre-flare
5. **Uncertainty-aware forecasting with Bayesian deep learning** — MC Dropout + Deep Ensembles for operational reliability
6. **GPU-optimized data pipeline** — HDF5 + PyTorch DataLoader for efficient training on A100-80GB

---

## 2. Rigorous Literature Review

### 2.1 Solar Flare Physics & X-ray Emission Mechanisms

#### Standard Flare Model (CSHKP)

The standard model of solar flares describes magnetic reconnection in the corona as the primary energy release mechanism [Carmichael 1964; Sturrock 1966; Hirayama 1974; Kopp & Pneuman 1976]. Energy stored in stressed magnetic fields is released via reconnection, accelerating electrons and heating plasma to >10 MK.

**Magnetic Reconnection Rate:**
$$M_A = \frac{v_{\text{in}}}{v_A} \sim 0.01-0.1$$

where $v_{\text{in}}$ is the inflow velocity and $v_A = B/\sqrt{\mu_0\rho}$ is the Alfvén velocity [Petschek 1964].

#### Thermal Bremsstrahlung (Soft X-rays)

Soft X-ray emission (SoLEXS: 2–22 keV) is dominated by thermal bremsstrahlung from hot plasma [Tucker 1975]. The volume emissivity is:

$$\varepsilon_{\text{ff}}(\nu, T) = 6.8 \times 10^{-38} n_e^2 T^{-1/2} \bar{g}_{\text{ff}}(\nu, T) \exp(-h\nu/kT) \quad \text{erg cm}^{-3} \text{s}^{-1} \text{Hz}^{-1}$$

where $n_e$ is electron density, $T$ is temperature, and $\bar{g}_{\text{ff}}$ is the Gaunt factor [Rybicki & Lightman 1979].

The **GOES classification** system uses the 1–8 Å (0.1–0.8 nm) soft X-ray flux measured at Earth:

| Class | Peak Flux (W/m²) | Example Equivalent |
|-------|-----------------|-------------------|
| A     | $< 10^{-7}$     | Background level |
| B     | $10^{-7} - 10^{-6}$ | Microflares |
| C     | $10^{-6} - 10^{-5}$ | Small flares |
| **M** | $10^{-5} - 10^{-4}$ | **Medium flares** |
| **X** | $> 10^{-4}$     | **Major flares** |

*Source: [Wikipedia: Solar Flare Classification](https://en.wikipedia.org/wiki/Solar_flare)*

#### Non-thermal Bremsstrahlung (Hard X-rays)

Hard X-ray emission (HEL1OS: 10–150 keV) is dominated by non-thermal bremsstrahlung from accelerated electrons [Brown 1971; Holman 2003]. The **thick-target model** provides the standard framework:

For an electron spectrum $f(E) = F_0 E^{-\delta}$ (electrons s⁻¹ keV⁻¹), the emitted photon spectrum is:

$$I(\varepsilon) = \frac{1}{\varepsilon} K \int_{\varepsilon}^{\infty} n(E) Q(\varepsilon, E) dE$$

where $Q(\varepsilon, E)$ is the bremsstrahlung cross-section (Bethe-Heitler), and $n(E)$ is the electron flux spectrum.

In the **collisional thick-target** approximation, for $f(E) \propto E^{-\delta}$:

$$I(\varepsilon) \propto \varepsilon^{-(\delta+1)/2} \quad \text{[Kramer's approximation]}$$

*Sources: Brown 1971 (Astrophys. J. 170, 601); Holman et al. 2003 (ApJ 595, L97); Kontar et al. 2011 (Space Sci. Rev. 159, 301) — [arXiv:1110.1755](https://arxiv.org/abs/1110.1755)*

#### Electron Distribution from X-ray Spectroscopy

The inverse problem of recovering the electron distribution from photon spectra:

$$I(\varepsilon) = \int_{\varepsilon}^{\infty} n(E) \sigma(\varepsilon, E) dE$$

where $\sigma(\varepsilon, E)$ is the bremsstrahlung cross-section. This is a Fredholm integral equation of the first kind, requiring regularization [Kontar et al. 2005; Piana et al. 2003]. Recent work on **regularized imaging spectroscopy** uses Tikhonov regularization:

$$n_{\lambda} = \arg\min_n \left\{ \|I - K n\|^2 + \lambda \|L n\|^2 \right\}$$

*Sources: Kontar et al. 2005 (Solar Phys. 233, 231); Volpara et al. 2023 — [arXiv:2311.07148](https://arxiv.org/abs/2311.07148)*

#### Soft-Hard-Soft (SHS) Spectral Evolution

The photon spectral index $\gamma$ follows a characteristic pattern: soft → hard → soft across the impulsive phase [Grigis & Benz 2004]. This is modelled as:

$$\gamma(t) = \gamma_0 + \Delta\gamma_f(t)$$

where $f(t)$ is a function peaking at the HXR maximum. The SHS pattern suggests electron acceleration efficiency varies with reconnection rate.

*Source: Grigis & Benz 2004 (A&A 426, 1103)*

---

### 2.2 ML/DL for Solar Flare Forecasting (SOTA)

#### Classical ML on Magnetic Parameters

The dominant approach uses machine learning on magnetic field parameters (SHARP — Space-weather HMI Active Region Patches) to predict flares. Bobra & Couvidat (2015) used SVM with log-transformed SHARP parameters achieving TSS ≈ 0.8 for >M1 flares.

$$z = w^T\phi(x) + b, \quad P(y=1|x) = \sigma(z)$$

*Source: Bobra & Couvidat 2015 (ApJ 802, 92)*

#### CNN on Magnetograms

Huang et al. (2018) applied deep CNNs to line-of-sight magnetograms for flare prediction within 24h:

$$\mathcal{L} = -[y \log(\hat{y}) + (1-y)\log(1-\hat{y})]$$

with data augmentation via rotation/flipping. Best model achieved TSS ≈ 0.52 for ≥M1 class flares.

*Source: Huang et al. 2018 (ApJ 856, 7) — [arXiv:1801.10420](https://arxiv.org/abs/1801.10420)*

#### Transformer Models for Flare Prediction

**SolarFlareNet** (Abduallah & Wang 2024): Transformer-based framework for 24–72h flare prediction:

$$\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Each AR is modelled as a time series of magnetic parameters. Three transformers handle ≥M5.0, ≥M, ≥C classes separately. The system is operational and produces near real-time probabilistic predictions.

*Source: Abduallah & Wang 2024 — [arXiv:2405.16080](https://arxiv.org/abs/2405.16080)*

**GCTAF (Global Cross-Time Attention Fusion):** Vural et al. (2025) proposed a transformer with learnable global tokens that summarize salient temporal patterns:

$$\tilde{Z} = \text{CrossAttn}(Q_{\text{global}}, K_{\text{series}}, V_{\text{series}})$$

Each global token attends to the entire time series, then feeds back into the representation.

*Source: Vural et al. 2025 — [arXiv:2511.12955](https://arxiv.org/abs/2511.12955)*

#### CNN on GOES X-ray Time Series

Landa & Reuveni (2021) used a 1D CNN on GOES X-ray light curves (1998–2019, covering SC23-24) for forecasting at 1–96h horizons:

Architecture: Conv1D(64,3) → Conv1D(128,3) → MP(2) → Conv1D(256,3) → MP(2) → FC(128) → FC(64) → output

Their key finding: **X-ray time series alone cannot distinguish M vs X class flares**, suggesting that spectral information (energy-dependent features) is necessary.

*Source: Landa & Reuveni 2021 — [arXiv:2101.12550](https://arxiv.org/abs/2101.12550) — code: [github.com/vladlanda/Low-Dimensional-CNN](https://github.com/vladlanda/Low-Dimensional-Convolutional-Neural-Network-For-Solar-Flares-GOES-Time-Series-Classification)*

#### Deep Reinforcement Learning for Flares

Yi et al. (2023) applied DRL where the agent learns to take actions (predict flare/no-flare) based on state (magnetic parameters), optimizing a reward function:

$$R(s,a) = \text{TSS} = \text{TPR} - \text{FPR}$$

where $\text{TPR} = TP/(TP+FN)$ and $\text{FPR} = FP/(FP+TN)$.

*Source: Yi et al. 2023 — [arXiv:2303.04708](https://arxiv.org/abs/2303.04708)*

#### Multi-Instrument Deep Learning

Adeyeha et al. (2024) evaluated explainable DL models using attribution-based proximity for operational forecasting, finding that attention-based models provide the most physically interpretable predictions.

*Source: Adeyeha et al. 2024 — Semantic Scholar*

**FlareCast** (Lv et al. 2026): A deep Bayesian neural network system integrating MLOps for operational solar flare forecasting with uncertainty quantification via variational inference:

$$p(\theta|D) \approx q^*(\theta) = \arg\min_q \text{KL}[q(\theta)||p(\theta|D)]$$

*Source: Lv et al. 2026 — Semantic Scholar*

#### Multi-modal Inputs

Rosales et al. (2026) demonstrated that combining magnetograms (SDO/HMI) + EUV images (SDO/AIA) improves forecasting accuracy over single-modality models.

*Source: Rosales et al. 2026 — [Advancing Solar Flare Forecasting with Deep Learning Using Multimodal Inputs]*

#### Evaluation Metrics

The community standard uses:

- **True Skill Score (TSS):** $\text{TSS} = \text{TPR} - \text{FPR}$ — 1 is perfect, 0 is random
- **Heidke Skill Score (HSS):** $\text{HSS} = \frac{2(TP\cdot TN - FP\cdot FN)}{(TP+FN)(FN+TN)+(TP+FP)(FP+TN)}$
- **Brier Score:** $\text{BS} = \frac{1}{N}\sum_{t=1}^{N}(p_t - o_t)^2$

*Source: Leka et al. 2019 (ApJS 243, 36) — "A Comparison of Flare Forecasting Methods"*

---

### 2.3 Hard X-ray Detection & RHESSI/Fermi/STIX Legacy

#### RHESSI (2002–2023)

The Reuven Ramaty High Energy Solar Spectroscopic Imager revolutionized hard X-ray solar physics with 1 arcsec resolution imaging spectroscopy across 3 keV–17 MeV. Key results:

- **Footpoint + looptop sources** — systematic imaging revealed double footpoint sources in the chromosphere and a coronal looptop source (Masuda et al. 1994)
- **Non-thermal spectral indices** — $\gamma$ typically 3–6 for M/X class flares
- **Electron energetics:** 10–50% of flare energy goes into accelerated electrons (Emslie et al. 2012)

*Key papers:*
- Lin et al. 2002 (Solar Phys. 210, 3) — RHESSI instrument description
- Holman et al. 2011 (Space Sci. Rev. 159, 107) — RHESSI spectroscopy
- [arXiv:1310.5449](https://arxiv.org/abs/1310.5449) — Active region NOAA 11158 X2.2 flare
- [arXiv:1812.09474](https://arxiv.org/abs/1812.09474) — Warm-target model for low-energy cutoff determination

#### STIX on Solar Orbiter (2020–present)

STIX provides hard X-ray imaging spectroscopy at 4–150 keV using indirect Fourier imaging with 30 collimators (vs RHESSI's 9 rotating modulation collimators). Key advances:

- **Regularized Imaging Spectroscopy (RIS):** Noise-robust multi-energy imaging via Tikhonov regularization (Volpara et al. 2024) — [arXiv:2407.01175](https://arxiv.org/abs/2407.01175)
- **First tracking of accelerated electrons along flare loop** (Volpara et al. 2023) — [arXiv:2311.07148](https://arxiv.org/abs/2311.07148)
- Online processing: visibilities computed from Moiré patterns

*Source: Hayes et al. 2022 — [arXiv:2207.02079](https://arxiv.org/abs/2207.02079)*

#### ASO-S/HXI (2022–present)

China's Advanced Space-based Solar Observatory carries the Hard X-ray Imager (HXI) operating at 30–200 keV with 91 grid-pair Fourier imaging. First results published 2024. The Neupert effect was validated across 149 flares using HXI + GOES.

*Source: Li et al. 2024 — [arXiv:2404.02653](https://arxiv.org/abs/2404.02653) — "A Statistical Investigation of the Neupert Effect in Solar Flares Observed with ASO-S/HXI"*

#### Neupert Effect

The Neupert effect [Neupert 1968; Veronig et al. 2002] states that the soft X-ray light curve is the time integral of the hard X-ray flux:

$$F_{\text{GOES}}(t) \propto \int_0^t F_{\text{HXR}}(t') dt'$$

or equivalently:

$$\frac{dF_{\text{GOES}}(t)}{dt} \propto F_{\text{HXR}}(t)$$

This reflects electron-beam-driven chromospheric evaporation: accelerated electrons deposit energy in the chromosphere, heating plasma to >10 MK, which then rises into the corona emitting soft X-rays.

Veronig et al. (2002) found that ~50% of 1,114 flares show timing consistent with the Neupert effect:

$$\rho = \text{corr}\left(\frac{dF_{\text{SXR}}}{dt}, F_{\text{HXR}}\right) > 0.5 \quad \text{for} \quad 46\% \text{ of events}$$

Li et al. (2024) found **all 149** flares with HXI data show ρ > 0.90, suggesting improved sensitivity reveals the effect universally.

*Sources:*
- Neupert 1968 (ApJ 153, L59)
- Veronig et al. 2002 (A&A 392, 699) — [arXiv:astro-ph/0207217](https://arxiv.org/abs/astro-ph/0207217)
- Li et al. 2024 — [arXiv:2404.02653](https://arxiv.org/abs/2404.02653)
- Perriyil et al. 2026 — [arXiv:2602.19836](https://arxiv.org/abs/2602.19836) — Loop length vs HXR-SXR timing delay

#### Pre-flare Precursor Signatures

Research on X-ray precursors before the main flare impulsive phase has identified:

1. **Pre-impulsive non-thermal emission** — weak HXR bursts 1–10 minutes before main peak (Battaglia et al. 2019)
2. **Microflare/precursor flare activity** — small-scale energy releases preceding major flares
3. **Coronal source brightening** — hot plasma signatures before the impulsive phase
4. **Sigmoid-to-arcade transformation** — magnetic field structure changes visible in EUV

*Source: Battaglia et al. 2019 (ApJ 876, 15) — [arXiv:1901.07767](https://arxiv.org/abs/1901.07767)*

---

### 2.4 Aditya-L1 Mission & Instruments

#### Mission Overview

Aditya-L1 — ISRO's first solar mission — launched September 2, 2023, into a halo orbit around Sun-Earth Lagrange Point L1 (about 1.5 million km from Earth). It carries 7 payloads: 4 remote sensing (SoLEXS, HEL1OS, Solar Ultraviolet Imaging Telescope, Visible Emission Line Coronagraph) and 3 in-situ.

#### SoLEXS — Solar Low Energy X-ray Spectrometer

| Parameter | Value |
|-----------|-------|
| Energy Range | 2–22 keV (340 channels, ~0.059 keV/ch) |
| Detectors | SDD1 (aperture 7.1 mm²), SDD2 (aperture 0.1 mm²) |
| Cadence | **1 second** (full 24h/day) |
| Data Coverage | 747 days (Feb 2024–Jun 2026, 85.6%) |
| LC Format | FITS OGIP — TIME + COUNTS (86,400 rows/day) |
| PI Format | FITS OGIP — 340 channels × 86,400 spectra/day |
| GTI | Typically 1–3 intervals (Earth occultation gaps) |

**Key fact:** SDD2 is the primary science detector (optimized for high flux). SDD1 has a larger aperture but non-functional (empty GTI files).

#### HEL1OS — High Energy L1 Orbiting X-ray Spectrometer

| Parameter | Value |
|-----------|-------|
| Energy Range | **1.8–160 keV** (CZT: 20–150, CdTe: 10–40) |
| Detectors | CZT1, CZT2, CdTe1, CdTe2 (4 detectors × 5 bands) |
| Cadence | **1 second** (~42,500 rows/orbit) |
| Data Coverage | 902 days (Nov 2023–Jun 2026, 98.9%) |
| CZT Bands | 20–40, 40–60, 60–80, 80–150, 18–160 keV |
| CdTe Bands | 5–20, 20–30, 30–40, 40–60, 1.8–90 keV |
| LC Format | FITS — MJD + ISOT + CTR + STAT_ERR (10 LC files/day) |

**Combined SoLEXS + HEL1OS coverage: 724 days** (Feb 2024–Jun 2026)

#### Instrument Comparison

| Instrument | Energy Range | Angular Resolution | Cadence | Pixel Count | FOV |
|-----------|-------------|-------------------|---------|-------------|-----|
| SoLEXS | 2–22 keV | Full-Sun | 1s | — | Full disk |
| HEL1OS | 10–150 keV | Full-Sun | 1s | — | Full disk |
| GOES XRS | 1.07–12.4 keV | Full-Sun | 1s/2s | — | Full disk |
| RHESSI | 3 keV–17 MeV | 2.3–36 arcsec | 4s | 9 modulators | Full Sun |
| STIX | 4–150 keV | 4–180 arcsec | 0.5s | 30 collimators | ~2.5° |
| NuSTAR | 3–79 keV | 9.5 arcsec | ~1 ks | 2 focal planes | 12 arcmin |
| ASO-S/HXI | 30–200 keV | ~3 arcsec | 0.125s | 91 grids | Full Sun |

**What makes Aditya-L1 unique:** SoLEXS + HEL1OS provide continuous, simultaneous soft + hard X-ray coverage at **1-second cadence** — comparable to GOES but with spectral resolution (340 channels) and extended hard X-ray range (to 150 keV).

---

### 2.5 Research Gaps & Novel Opportunities

#### Identified Gaps

1. **No combined soft+hard X-ray flare pipeline exists for Aditya-L1** — all existing work uses either GOES (no spectral resolution) or single instruments

2. **X-ray spectral information is not used in forecasting** — existing ML models use total flux (0.1–0.8 nm) from GOES, discarding energy-dependent information. SoLEXS has 340 channels; HEL1OS has 10 bands — this contains critical temperature and non-thermal information

3. **No physics-informed ML for flare forecasting** — no paper has embedded the Neupert effect or bremsstrahlung physics into a neural network loss function

4. **Precursor detection in soft vs hard X-ray channels (cross-channel causality)** has not been attempted using information-theoretic methods

5. **Uncertainty quantification in flare forecasting is nascent** — only FlareCast (2026) addresses Bayesian methods, but not with spectral data

6. **Transfer learning between GOES and Aditya-L1** has not been attempted — a model pre-trained on 25 years of GOES data could be fine-tuned on Aditya-L1

7. **Self-supervised learning on unlabeled X-ray data** is unexplored — 747 days of continuous 1-second data provides a rich training resource

8. **The Neupert effect in the 20–150 keV range** from HEL1OS has never been statistically characterized — Li et al. (2024) used ASO-S/HXI but only at 20–50 keV

9. **Spectral-temporal joint modeling** (time × energy) is unexplored for flare prediction

10. **No operational system uses Indian X-ray data for real-time flare alerts**

---

## 3. Mathematical Framework

### 3.1 X-ray Emission Physics

**Thermal emission (SoLEXS, soft X-rays):**

$$I_{\text{thermal}}(\varepsilon, T, EM) = \frac{EM}{4\pi d^2} \varepsilon^{-1} \Lambda(\varepsilon, T) \exp(-\varepsilon/kT)$$

where $EM = \int n_e^2 dV$ is the emission measure and $\Lambda(\varepsilon, T)$ is the radiative loss function.

**Non-thermal emission (HEL1OS, hard X-rays):**

For thin-target model:

$$I_{\text{thin}}(\varepsilon) = \frac{1}{4\pi d^2} \int_{\varepsilon}^{\infty}\int_{V} n(\mathbf{r}) \frac{d\sigma(\varepsilon, E)}{d\varepsilon} F(E, \mathbf{r}) dE d^3 \mathbf{r}$$

For thick-target model (simplified):

$$I_{\text{thick}}(\varepsilon) = \frac{1}{4\pi d^2} \frac{K}{\varepsilon} \int_{\varepsilon}^{\infty} \left[\int_E^{\infty} \frac{\bar{F}(E')}{dE'/dt} dE'\right] \frac{d\sigma}{d\varepsilon} dE$$

### 3.2 GOES Classification & Calibration

The SoLEXS 2–22 keV range overlaps with GOES 0.1–0.8 nm (1.55–12.4 keV). We establish a cross-calibration:

$$F_{\text{GOES}}(t) = \alpha \cdot \int_{1.55\text{ keV}}^{12.4\text{ keV}} I_{\text{SoLEXS}}(\varepsilon, t) d\varepsilon + \beta$$

where $\alpha$ and $\beta$ are derived from linear regression over 724 overlapping days.

### 3.3 Neupert Effect (Physics Prior)

The Neupert effect provides a causal link between hard and soft X-ray emission:

$$\frac{dF_{\text{SXR}}(t)}{dt} = \eta F_{\text{HXR}}(t - \tau) + \epsilon(t)$$

where $\eta$ is the evaporation efficiency, $\tau$ is the time delay (loop-length dependent), and $\epsilon(t)$ represents departure from the pure Neupert model (other heating sources).

**Explaining the physics:** When accelerated electrons slam into the dense chromosphere, they heat plasma to ~10 MK. This hot plasma expands upward into the corona (chromospheric evaporation) and emits soft X-rays. The *rate of change* of soft X-ray flux directly tracks the instantaneous hard X-ray flux. The time delay $\tau$ reflects the travel time of the heated plasma from chromosphere to corona (typically 1–60s).

### 3.4 Pre-flare Precursor Signatures

We define a **flare precursor index** using information-theoretic quantities:

**Transfer Entropy** from HEL1OS (HXR) to SoLEXS (SXR):

$$T_{\text{HXR} \rightarrow \text{SXR}} = \sum p(\text{SXR}_{n+1}, \text{SXR}_n, \text{HXR}_n) \log \frac{p(\text{SXR}_{n+1}|\text{SXR}_n, \text{HXR}_n)}{p(\text{SXR}_{n+1}|\text{SXR}_n)}$$

A rise in $T_{\text{HXR} \rightarrow \text{SXR}}$ indicates that hard X-ray variations are becoming predictive of future soft X-ray variations (expected ∼1–10 minutes before flare onset due to the Neupert effect).

**Spectral Hardness Ratio (SRH):**

$$\text{SHR}(t) = \frac{F_{20\text{-}40\text{keV}}(t)}{F_{5\text{-}20\text{keV}}(t)}$$

Pre-flare increases in SHR indicate non-thermal electron acceleration before the main energy release.

---

## 4. Nowcasting Pipeline (Phase 1)

### 4.1 Data Preprocessing & Quality Control

#### 4.1.1 HDF5 Database Creation

**Input:** 747 SoLEXS LC files + 724 days of HEL1OS light curves
**Output:** Single HDF5 database with chunked storage and BLOSC compression

```
/data/hdf5/flare_data.h5
├── /solexs/lc/{YYYYMMDD}        — TIME(86400), COUNTS(86400)
├── /solexs/pi/{YYYYMMDD}        — TSTART(86400), COUNTS(86400×340)
├── /hel1os/lc/{YYYYMMDD}        — MJD(N), ISOT(N), CTR_CZT1-2(N×5), CTR_CDTE1-2(N×5)
├── /solexs/gti/{YYYYMMDD}       — START(M), STOP(M)
├── /hel1os/gti/{YYYYMMDD}       — TSTART(M), TSTOP(M)
├── /metadata/coverage           — date_range, missing_days, overlaps
└── /metadata/calibration        — cross-calibration coefficients
```

**GPU optimization:** HDF5 chunking at (4096, 340) for PI data to align with GPU memory hierarchy; BLOSC compression (level 5, shuffle=True) for ~5x compression with fast decompression on GPU.

#### 4.1.2 Temporal Alignment

SoLEXS operates 24h/day (UTC), while HEL1OS operates ~12h/orbit with gaps. Alignment:

$$\mathbb{T}_{\text{aligned}} = \{t : t \in G_{\text{SoLEXS}}(t) \wedge t \in G_{\text{HEL1OS}}(t)\}$$

Creating ~5.7h/day of aligned dual-instrument data.

#### 4.1.3 Background Subtraction

**SoLEXS background:** Use minimum 10th percentile within a sliding 1h window:

$$B_{\text{SXR}}(t) = \langle C(t - \Delta t/2 : t + \Delta t/2) \rangle_{10\%}$$

**HEL1OS background:** Use night data (Earth occultation) when available, else same percentile method.

### 4.2 Flare Detection Algorithms

#### 4.2.1 Threshold-Based Detection (Baseline)

$$\text{Flare onset: } F_S(t) > F_{\text{quiet}}(t) + 3\sigma_{\text{quiet}}(t)$$

where $F_{\text{quiet}}$ and $\sigma_{\text{quiet}}$ are estimated from a running 24h baseline.

**GOES-equivalent classification thresholds for SoLEXS:**
Calibrate SoLEXS counts to W/m² via cross-correlation with known GOES events (using public GOES X-ray flux).

#### 4.2.2 Bayesian Blocks (Adaptive Binning)

Following Scargle et al. (2013), we partition the light curve into intervals of constant count rate:

$$P(\text{blocks}|\text{data}) \propto P(\text{data}|\text{blocks})P(\text{blocks})$$

The cell $P_0 = P(\text{data}|\text{uniform rate})$ is compared to the prior probability of an additional change point, controlled by a penalty parameter $n_{\text{cells\_prior}} = N_{\text{cells}}\ln N_{\text{cells}}$.

*Source: Scargle et al. 2013 (ApJ 764, 167) — [arXiv:1207.5578](https://arxiv.org/abs/1207.5578)*

#### 4.2.3 Wavelet-based Flare Detection

The continuous wavelet transform (CWT) identifies flare-like enhancements in the time-frequency plane:

$$W(a,b) = \frac{1}{\sqrt{a}} \int F(t) \psi^*\left(\frac{t-b}{a}\right) dt$$

using a Morlet wavelet $\psi(t) = \pi^{-1/4}e^{i\omega_0 t}e^{-t^2/2}$. Flare signatures appear as cone-shaped features in the scalogram. Detection threshold:

$$\text{Flare if: } |W(a,b)|^2 > 4\sigma_a^2$$

where $\sigma_a^2$ is the scale-dependent background noise.

#### 4.2.4 Autoencoder Anomaly Detection (Novel)

Train a variational autoencoder (VAE) on quiet-Sun X-ray data:

$$\mathcal{L}_{\text{VAE}} = \mathbb{E}_{z\sim q_\phi(z|x)}[\log p_\theta(x|z)] - \beta \cdot \text{KL}[q_\phi(z|x)||p(z)]$$

where $x$ is a 1-hour light curve segment, $\beta$ controls the KL divergence weight. Flares are detected as high reconstruction error events:

$$\text{Flare if: } \|x - \hat{x}\|_2^2 > \epsilon_{\text{threshold}}$$

### 4.3 Combined Soft+Hard Nowcasting

#### 4.3.1 Correlation-based Timing Analysis

Compute the cross-correlation function between SoLEXS flux derivative and HEL1OS flux:

$$\text{CCF}(\tau) = \int \frac{dF_{\text{SXR}}(t)}{dt} \cdot F_{\text{HXR}}(t+\tau) dt$$

The Neupert effect predicts a peak near $\tau=0$ for impulsive flares.

#### 4.3.2 Multi-band Peak Detection

**Algorithm:**
1. For HEL1OS: compute summed CTR across CZT1, CZT2, CdTe1, CdTe2 for each energy band
2. For SoLEXS: use the integrated 2–22 keV counts
3. Apply peak detection: find local maxima > 3σ within 5-minute sliding window
4. Cross-match: HXR peaks should occur within ±30s of SXR derivative peaks
5. Classify: use SXR peak flux (GOES-equivalent) for flare class (A, B, C, M, X)

#### 4.3.3 Nowcast Output Database

```
/data/hdf5/nowcast_catalogue.h5
├── /flares/{flare_id}
│   ├── t_onset          — flare start time (MJD)
│   ├── t_peak_SXR       — SXR peak time
│   ├── t_peak_HXR       — HXR peak time
│   ├── t_end            — flare end time
│   ├── peak_flux_SXR    — peak SXR flux (W/m²)
│   ├── peak_flux_HXR    — peak HXR rate (cts/s)
│   ├── goes_class       — GOES-equivalent class
│   ├── shs_index        — SHS spectral index evolution
│   ├── neupert_rho      — Neupert correlation coefficient
│   └── energy_bands     — per-band peak fluxes
```

---

## 5. Forecasting Pipeline (Phase 2)

### 5.1 Feature Engineering

#### 5.1.1 Energy-Dependent Features

**SoLEXS bands (spectral-derived):**
- Temperature $T(t)$ from spectral fitting of thermal component
- Emission measure $EM(t)$
- Differential emission measure (DEM) distribution

**HEL1OS bands (extracted):**
- Count rates in each of 10 energy bands (4 detectors × 5 bands minus overlap)
- Hardness ratios: $HR_{ij}(t) = F_i(t) / F_j(t)$
- Spectral index proxy: $\gamma_{\text{est}}(t) \approx -\log(F_{40-60}/F_{20-40}) / \log(50/30)$

#### 5.1.2 Temporal Features

- Statistical moments (mean, variance, skew, kurtosis) over sliding windows of 1, 5, 15, 60 minutes
- Fourier coefficients (top 5 amplitude components)
- Wavelet scalogram features (energy in 8 frequency bands)
- Autocorrelation at lags 1–100s
- Difference features: $\Delta F(t) = F(t) - F(t-\Delta t)$

#### 5.1.3 Information-Theoretic Features

- Transfer entropy HEL1OS → SoLEXS over 60s sliding window
- Mutual information between energy bands
- Granger causality test statistics
- Sample entropy (measure of signal complexity/complexity)

#### 5.1.4 Precursor Features

- Pre-flare activity index: integrated counts in 30 min window
- Spectral hardening indicator: sustained SHR > 1.5 × background
- Neupert effect faithfulness: rolling correlation between dSXR/dt and HXR

#### 5.1.5 Additional Data Sources

- **GOES X-ray flux** (public, 1998–present) — for transfer learning
- **Magnetic parameters** from SDO/HMI (SHARP), if accessible
- **Solar cycle phase** (F10.7 radio flux as proxy)

### 5.2 Baseline Models

#### 5.2.1 LightGBM (Gradient Boosted Trees)

**Features:** Static set of 47 features extracted from a 60-minute lookback window
**Target:** Binary classification (flare/no flare within next N minutes)  
**Hyperparameters:** 5000 estimators, max_depth=12, learning_rate=0.01, early_stopping=50  
**Handling imbalance:** Scale_pos_weight = count(neg)/count(pos), custom TSS objective

#### 5.2.2 CNN-LSTM Hybrid

```
Input: (T, C) where T=3600, C=12 (6 energy bands × 2 instruments)
     │
Conv1D(64, kernel=5) → BatchNorm → ReLU → MaxPool(2)
Conv1D(128, kernel=5) → BatchNorm → ReLU → MaxPool(2)
Conv1D(256, kernel=3) → BatchNorm → ReLU → GlobalAvgPool
     │
LSTM(128, bidirectional=True, dropout=0.3)
     │
Attention(128) → LayerNorm → FC(64) → Dropout(0.3) → FC(2)
```

**Loss:** Focal loss to handle class imbalance:

$$\mathcal{L}_{\text{focal}}(p_t) = -\alpha_t (1-p_t)^\gamma \log(p_t)$$

where $p_t$ is the model's estimated probability for the correct class, $\alpha_t$ is a class balancing weight, and $\gamma=2$ focuses training on hard examples.

### 5.3 Transformer Architecture with Cross-Attention (NOVEL)

#### 5.3.1 Spectral-Temporal Transformer

We propose a novel architecture that jointly models the **spectral** (energy) and **temporal** dimensions:

```
Input: X ∈ ℝ^{T×E} where T=3600 (time steps), E=12 (energy bands)
     │
┌────────────── Temporal Branch ──────────────┐
│ Positional encoding + Multi-head Self-Attn  │
│ over time dimension: Attention(T×T) × E     │
│ Temporal features: Z_t ∈ ℝ^{T×d_model}      │
└─────────────────────────────────────────────┘
     │
┌────────────── Spectral Branch ──────────────┐
│ Positional encoding (energy)                │
│ Cross-Attention: energy attending to time   │
│ Q = E_tokens, K = Z_t, V = Z_t             │
│ Spectral features: Z_s ∈ ℝ^{E×d_model}      │
└─────────────────────────────────────────────┘
     │
┌────────────── Fusion ───────────────────────┐
│ Z_fused = LayerNorm(Z_t + Z_s)             │
│ CLS token pooling                          │
│ FC(d_model) → Flare class + Probability     │
└─────────────────────────────────────────────┘
```

**Key innovation:** The spectral branch allows the model to learn which energy channels are most predictive. The cross-attention mechanism connects spectral features to temporal patterns, enabling the model to discover, e.g., "hardening of the 20–40 keV band 5 minutes before the flare onset."

#### 5.3.2 Multi-Scale Temporal Transformer

For capturing both short-term (seconds) and long-term (hours) dependencies:

$$h_t^{(1)} = \text{TransformerEncoder}_{\text{short}}(x_{t-60:t}) \quad \text{(1-minute window)}$$
$$h_t^{(2)} = \text{TransformerEncoder}_{\text{med}}(x_{t-900:t}) \quad \text{(15-minute window)}$$
$$h_t^{(3)} = \text{TransformerEncoder}_{\text{long}}(x_{t-3600:t}) \quad \text{(1-hour window)}$$
$$y_t = \text{FC}(\text{Concat}(h_t^{(1)}; h_t^{(2)}; h_t^{(3)}))$$

#### 5.3.3 Forecaster: Multi-horizon Prediction

Simultaneous prediction at multiple lead times:

$$\hat{y}_t^{(k)} = f^{(k)}\left(\{x_\tau\}_{\tau = t-L}^{t}\right) \quad \text{for } k \in \{5, 15, 30, 60, 120\} \text{ minutes}$$

Loss:

$$\mathcal{L} = \sum_{k} w_k \cdot \mathcal{L}_{\text{focal}}(y_t^{(k)}, \hat{y}_t^{(k)})$$

where $w_k$ are horizon-specific weights (shorter horizons weighted higher for operational relevance).

### 5.4 Physics-Informed Loss Functions (NOVEL)

#### 5.4.1 Neupert Effect Constraint

We embed the Neupert effect as a regularization term:

For a given flare window $[t_{\text{start}}, t_{\text{peak}}]$:

$$\frac{dF_{\text{SXR}}(t)}{dt} \approx \eta \cdot F_{\text{HXR}}(t - \tau)$$

The physics-informed loss penalizes deviations:

$$\mathcal{L}_{\text{physics}} = \left\| \frac{d\hat{F}_{\text{SXR}}}{dt} - \eta \cdot \hat{F}_{\text{HXR}} \right\|_2^2$$

where $\eta$ and $\tau$ are learnable parameters and $\hat{F}$ are the predicted quantities from the model.

For non-flare intervals, a relaxed penalty applies (the Neupert effect is not expected during quiet periods):

$$\mathcal{L}_{\text{physics}} = \begin{cases}
w_{\text{physics}} \cdot \|d\hat{F}_{\text{SXR}}/dt - \eta\hat{F}_{\text{HXR}}\|^2 & \text{if flare predicted} \\
w_{\text{quiet}} \cdot \max(0, \|d\hat{F}_{\text{SXR}}/dt\| - \delta) & \text{otherwise}
\end{cases}$$

#### 5.4.2 Energy Conservation Constraint

Total radiated X-ray energy should be consistent with magnetic energy release:

$$E_{\text{SXR}} + E_{\text{HXR}} \leq \frac{B^2}{2\mu_0} V$$

where $B$ is the coronal magnetic field strength (~100 G) and $V$ is the emitting volume. This provides a physical upper bound on predictions.

#### 5.4.3 Thermal-Nonthermal Consistency

The electron spectral index $\delta$ (from hard X-rays) and the bulk plasma temperature $T$ (from soft X-rays) satisfy:

$$\frac{\partial \ln F_{\text{SXR}}}{\partial \ln t} \bigg/ \frac{\partial \ln F_{\text{HXR}}}{\partial \ln t} \sim g(\delta, T)$$

The function $g$ is derived from the Neupert effect and the bremsstrahlung emissivity, providing a consistency check.

### 5.5 Uncertainty Quantification

#### 5.5.1 Monte Carlo Dropout

Apply dropout at inference (N=100 forward passes):

$$p(y|x) \approx \frac{1}{N}\sum_{i=1}^{N} p(y|x, W_i), \quad \text{Var}(y|x) \approx \frac{1}{N}\sum_{i=1}^{N} (\hat{y}_i - \bar{y})^2$$

#### 5.5.2 Deep Ensembles

Train 5 models with different random seeds:

$$\text{Total uncertainty} = \underbrace{\frac{1}{M}\sum_{m}\text{Var}_m}_{\text{aleatoric}} + \underbrace{\frac{1}{M}\sum_{m}(\mu_m - \bar{\mu})^2}_{\text{epistemic}}$$

#### 5.5.3 Conformal Prediction

Calibrate prediction intervals on a held-out set:

$$\hat{y}_{t+\Delta} \pm q_{\alpha} \cdot \sigma_{\text{ensemble}}$$

where $q_{\alpha}$ is the $\alpha$-quantile of non-conformity scores on calibration data, providing distribution-free coverage guarantees.

---

## 6. Novel Contributions

We identify **8 specific novel contributions** that have never been attempted before:

### N1: First Combined Soft+Hard X-ray Nowcast Database from Aditya-L1
**Novelty:** No prior work has fused SoLEXS (2–22 keV, 340 channels) and HEL1OS (10–150 keV, 10 bands) for flare detection. Previous instruments (GOES XRS) provide only 2 broad-band soft X-ray channels. Our dataset provides **spectrally resolved soft X-rays + multi-band hard X-rays at 1s cadence**.

### N2: Neupert-Constrained Physics-Informed Neural Network
**Novelty:** No existing paper embeds the Neupert effect as a physics loss in a neural network for flare forecasting. PINNs are well-established for PDEs [Raissi et al. 2019] but have never been applied to X-ray flare time series. Our loss function `L_physics` forces consistency between predicted hard X-ray fluence and predicted soft X-ray time derivative.

### N3: Spectral-Temporal Transformer with Cross-Attention
**Novelty:** Existing transformer models (SolarFlareNet, GCTAF) use only scalar fluxes or magnetic parameters. Our model treats each energy channel as a separate token in a multi-head attention mechanism, allowing the model to learn **which energies are most predictive** of upcoming flares. The spectral branch can discover precursor signals in specific energy bands.

### N4: Transfer Entropy-Based Precursor Detection
**Novelty:** Information-theoretic measures (transfer entropy, Granger causality) have never been applied to predict solar flares. We quantify causal flow between HXR and SXR channels pre-flare, building a **precursor activity index** that rises 1–15 minutes before flare onset.

### N5: Energy-Dependent Flare Forecasting
**Novelty:** Every published ML model uses GOES total flux (1–8 Å broadband). No model has used **energy-dependent features** (temperature, emission measure, spectral index, hardness ratios). Our features include thermal + non-thermal parameters derived from the SoLEXS spectrum, providing direct window into the plasma physics.

### N6: Self-Supervised Pretraining on Quiet-Sun X-ray Data
**Novelty:** We pretrain a masked autoencoder (MAE) on 700+ days of continuous 1-second X-ray data to learn the "language" of solar X-ray variations. The MAE: (a) masks random time steps, (b) reconstructs masked values, (c) learns latent representations transferable to flare forecasting. This is groundbreaking — no one has pretrained on unlabeled solar X-ray data.

### N7: Bayesian Uncertainty Quantification for Operational Flare Forecasting
**Novelty:** Operational flare prediction systems report point probabilities without uncertainty intervals. Our deep ensemble + conformal prediction framework provides **calibrated prediction intervals**, enabling risk-based decision making.

### N8: Cross-Instrument Transfer Learning (GOES → Aditya-L1)
**Novelty:** We train a 1D CNN on 25 years of GOES XRS data (1998–2023, SC23-24-25), then fine-tune on Aditya-L1 SoLEXS data. This addresses the relatively short availability of Aditya-L1 data (2.5 years vs 25 years of GOES).

---

## 7. GPU-Optimized Implementation Plan

### 7.1 Hardware Specification

| Component | Specification |
|-----------|---------------|
| GPU | NVIDIA A100 80GB SXM4 |
| CUDA Cores | 6,912 FP32 |
| Tensor Cores | 1,344 (3rd Gen) |
| Memory Bandwidth | 2,039 GB/s |
| Interconnect | NVLink 3 (600 GB/s), 4 GPUs |
| CPU | 64-core AMD EPYC |
| RAM | 512 GB |

### 7.2 Software Stack

| Layer | Technology |
|-------|-----------|
| Framework | **PyTorch 2.4+** with `torch.compile` |
| Distributed | **PyTorch DDP** (DataParallel for multi-GPU) |
| Mixed Precision | **torch.cuda.amp** (bfloat16) |
| Training | **PyTorch Lightning** (lightning.pytorch) |
| Data Loading | Custom HDF5 DataLoader with **CUDA Unified Memory** |
| Hyperparameter Opt | **Ray Tune** or Optuna |
| Monitoring | **Weights & Biases** |
| Profiling | **PyTorch Profiler** + NVIDIA NSight |
| Explainability | **Captum** (Integrated Gradients, SHAP) |

### 7.3 Data Pipeline Optimization

**Challenge:** The SoLEXS PI files are ~472 MB/day × 747 days = 330 GB. HEL1OS adds 88 GB. Training data requires efficient streaming.

**Solution: Multi-stage HDF5 + Prefetch Pipeline**

```
Stage 1: Preprocessing (offline)
  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
  │ FITS Reader  │ →  │ Background   │ →  │ HDF5 Writer  │
  │ (astropy)    │     │ Subtraction  │     │ (h5py+blosc) │
  └──────────────┘     └──────────────┘     └──────────────┘
  
Stage 2: Feature Extraction (offline)
  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
  │ Spectral Fit │ →  │ Feature      │ →  │ Features HDF5│
  │ (GPU via     │     │ Engineering  │     │ (4 GB total) │
  │  PyTorch)    │     │ (numpy/GPU)  │     │              │
  └──────────────┘     └──────────────┘     └──────────────┘
  
Stage 3: Training (online)
  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
  │ HDF5 Chunked │ →  │ GPU Prefetch │ →  │ GPU Training │
  │ DataLoader   │     │ (3x epoch)   │     │ (mixed prec) │
  └──────────────┘     └──────────────┘     └──────────────┘
```

**Key optimizations:**

1. **HDF5 Chunking:** Chunk PI data at (4096, 340) — 1 chunk ≈ 4K time steps × 340 energies ≈ 1.3 MB (fits in L2 cache of A100)
2. **Memory mapping:** Use `h5py` memory maps for zero-copy access
3. **Pre-fetching:** `DataLoader(num_workers=8, prefetch_factor=4)` with pinned memory
4. **Compression:** BLOSC shuffle+bitshuffle for optimal GPU decompression
5. **Pinned Memory:** Allocate pinned memory buffers for asynchronous CPU→GPU transfer

### 7.4 Model Training Optimizations

#### Mixed Precision Training

```python
scaler = torch.cuda.amp.GradScaler()
for batch in dataloader:
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        loss = model(batch)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

**Expected speedup:** 2.5× on A100 Tensor Cores for bfloat16 vs FP32.

#### Large Batch Training

The A100 80GB allows batch sizes of 512–1024 sequences of length 3600 × 12 features:

**Memory estimate for Transformer model:**
- Parameters: ~12M (∼48 MB FP32, ∼24 MB bfloat16)
- Activations (batch=1024, seq_len=3600, d_model=256): ~3.5 GB
- KV cache: ~0.5 GB
- Gradients: ~0.5 GB
- Optimizer states (AdamW): ~0.5 GB
- **Total: ~5 GB** per model (can hold 16 parallel models on one A100 for ensemble)

#### Gradient Accumulation

For even larger effective batch sizes:

```python
for micro_step in range(accumulation_steps):
    with torch.cuda.amp.autocast():
        loss = model(batch[micro_step]) / accumulation_steps
    scaler.scale(loss).backward()
scaler.step(optimizer)
```

#### Distributed Training (Multi-GPU)

```python
# 4× A100 with DDP + NCCL backend
trainer = pl.Trainer(
    accelerator='gpu',
    devices=4,
    strategy='ddp_find_unused_parameters_false',
    precision='bf16-mixed',
    accumulate_grad_batches=4
)
```

**Expected scaling:** 3.6× speedup from 4 GPUs (90% efficiency).

### 7.5 Custom Kernels

#### CUDA Custom Implementation of Bremsstrahlung Loss

The physics-informed loss requires evaluating:

$$\frac{dF_{\text{SXR}}}{dt} \approx \eta \cdot F_{\text{HXR}}(t - \tau)$$

This involves temporal derivatives and time-delay estimation — implemented as a custom CUDA kernel:

```python
@torch.jit.script
def neupert_loss(dSXR_dt: Tensor, F_HXR: Tensor, eta: Tensor, tau: Tensor) -> Tensor:
    """
    Args:
        dSXR_dt: shape (B, T) — time derivative of SXR
        F_HXR: shape (B, T) — HXR flux
        eta: evaporation efficiency (scalar parameter)
        tau: time delay (integer index)
    Returns:
        physics loss (scalar)
    """
    # Shift HXR by tau samples
    F_HXR_shifted = torch.roll(F_HXR, shifts=int(tau), dims=1)
    # Neupert: dSXR/dt ≈ eta * F_HXR(t - tau)
    residual = dSXR_dt - eta * F_HXR_shifted
    return torch.mean(residual**2)
```

#### Flash Attention (Transformer Optimization)

Using `torch.nn.functional.scaled_dot_product_attention` (PyTorch 2.x built-in FlashAttention) for 2–4× speedup in attention computation:

```python
# PyTorch 2.x automatically uses FlashAttention
attn_output = F.scaled_dot_product_attention(
    query, key, value, attn_mask=None, 
    dropout_p=0.1, is_causal=False
)
```

### 7.6 Model Training Plan

| Model | Parameters | Batch Size | Training Time (1×A100) | Notes |
|-------|-----------|------------|----------------------|-------|
| LightGBM | — | N/A | 15 min | CPU, CPU-parallel |
| CNN-LSTM | 2.3M | 256 | 2h | 724 days × 5.7h/day |
| Transformer (spectral-temporal) | 12M | 512 | 6h | Our novel architecture |
| Ensemble (5× Transformers) | 60M | 4×512 (DDP) | 8h | Distributed |
| Self-supervised MAE pretrain | 85M | 128 | 24h | Offline pretraining |

### 7.7 Reproducibility

```python
# Seed everything
pl.seed_everything(42, workers=True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Experiment tracking
wandb.init(project="bah2026-flare", config=vars(args))
wandb.watch(model, log="all", log_freq=100)
```

---

## 8. Data Analysis & Visualization Products

### 8.1 PDF Plot Catalog

Systematic generation of publication-quality PDF plots:

| Plot | Description | Files | Size |
|------|-------------|-------|------|
| **Combined LC** | SoLEXS + HEL1OS light curves per day | 724 PDFs | 2 MB each |
| **Spectral Evolution** | Dynamic spectra (time × energy) | 724 PDFs | 3 MB each |
| **Neupert Effect** | SXR derivative + HXR overlay | 724 PDFs | 1 MB each |
| **Feature Catalogue** | All engineered features per day | 724 PDFs | 4 MB each |
| **Nowcast Summary** | Detected flares with class | 1 PDF/month | 10 MB |
| **Forecast Evaluation** | ROC curves, TSS vs lead time | 10 PDFs | 5 MB each |
| **Uncertainty Calibration** | Reliability diagrams | 5 PDFs | 3 MB each |
| **Precursor Time Series** | Transfer entropy vs flare onset | 50 PDFs | 2 MB each |

### 8.2 HDF5 Database Products

| Database | Content | Size |
|----------|---------|------|
| `flare_data.h5` | Raw processed LC + PI | ~10 GB |
| `features.h5` | Engineered features | ~4 GB |
| `nowcast_catalogue.h5` | Detected flares | ~10 MB |
| `forecast_predictions.h5` | Model predictions + uncertainty | ~100 MB |
| `model_checkpoints.h5` | Trained model weights | ~500 MB |

### 8.3 Streamlit Dashboard

```python
# dashboard/app.py - Key visualizations
def main():
    st.set_page_config(layout="wide", page_title="BAH 2026 Flare Monitor")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        # Main light curve plot with nowcast overlay
        plot_combined_lc(solexs_data, hel1os_data, nowcast_events)
    with col2:
        # Alert panel
        show_alerts(active_alerts)
        # Feature dashboard
        show_features(feature_summary)
    
    # Time slider for navigation
    selected_time = st.slider("Time (UTC)", min_time, max_time)
    
    # Spectral viewer
    with st.expander("Spectral Evolution Viewer"):
        plot_dynamic_spectrum(energy_range, time_range)
```

### 8.4 Alerting System

Multi-tier alert system with configurable thresholds:

| Alert Level | Trigger | Action |
|-------------|---------|--------|
| GREEN | No flare expected | Background monitoring |
| YELLOW | Precursor index > 0.6 | Dashboard highlight |
| ORANGE | Flare probability > 70% for ≤15 min | Popup + sound |
| RED | Flare probability > 90% for ≤5 min | Full alert + notification |
| FLARE NOW | Nowcast confirmed | Time stamp + class display |

---

## 9. Timeline & Milestones

### Phase 0: Data Preparation (Week 1)
- [x] Raw data downloaded and decompressed (747 SoLEXS + 927 HEL1OS days)
- [x] FITS integrity verified (100%)
- [x] Coverage overlap computed (724 dual days)

### Phase 1: Data Pipeline (Weeks 1-2)
- [ ] HDF5 database creation with BLOSC compression
- [ ] Background subtraction algorithms
- [ ] Calibration pipeline (SoLEXS → GOES equivalent)
- [ ] Temporal alignment (5.7h/day dual coverage)
- [ ] Normalization and standardization

### Phase 2: Nowcasting (Weeks 2-3)
- [ ] Threshold-based detection baseline
- [ ] Bayesian Blocks implementation
- [ ] Wavelet detection (PyTorch version)
- [ ] VAE anomaly detection (GPU)
- [ ] Combined nowcast: multi-band cross-matching
- [ ] **Nowcast catalogue generation**

### Phase 3: Feature Engineering (Weeks 3-4)
- [ ] Energy-dependent features (T, EM, γ, HR)
- [ ] Temporal statistics (moments, wavelet, FFT)
- [ ] Information-theoretic features (TE, MI, Granger)
- [ ] Transfer entropy computation (batch GPU)
- [ ] **Feature database creation**

### Phase 4: Baseline Forecasting (Weeks 4-5)
- [ ] LightGBM training (CPU, 15 min)
- [ ] CNN-LSTM training (GPU, 2h)
- [ ] Evaluation: TSS, HSS, Brier, reliability
- [ ] Baseline comparison

### Phase 5: Novel Models (Weeks 5-7)
- [ ] Spectral-temporal transformer implementation
- [ ] Physics-informed loss (Neupert constraint)
- [ ] Self-supervised MAE pretraining (GPU, 24h)
- [ ] Transfer learning from GOES
- [ ] Deep ensemble (5 models, 4×A100 DDP)
- [ ] Cross-instrument transfer

### Phase 6: Uncertainty & Calibration (Week 7)
- [ ] MC Dropout implementation
- [ ] Deep Ensemble uncertainty
- [ ] Conformal prediction calibration
- [ ] Reliability diagrams
- [ ] Operational threshold tuning

### Phase 7: Visualization (Weeks 7-8)
- [ ] PDF plot catalogue generation
- [ ] Streamlit dashboard
- [ ] Alerting system
- [ ] Spectral evolution viewer
- [ ] Real-time monitoring

### Phase 8: Documentation & Submission (Week 8)
- [ ] README.md update
- [ ] Final performance evaluation
- [ ] Model checkpoint packaging
- [ ] Demo video creation
- [ ] **Final submission**

---

## 10. References

### Solar Flare Physics

1. Carmichael, H. (1964) — NASA SP-50, 451 — *CSHKP model*
2. Sturrock, P. A. (1966) — Nature 211, 695 — *Flash phase mechanism*
3. Hirayama, T. (1974) — Solar Phys. 34, 323 — *Flare model with vertical current sheet*
4. Kopp, R. A. & Pneuman, G. W. (1976) — Solar Phys. 50, 85 — *Magnetic reconnection*
5. Brown, J. C. (1971) — Solar Phys. 18, 489 — *Thick-target model*
6. Neupert, W. M. (1968) — ApJ 153, L59 — *Neupert effect*
7. Veronig, A. et al. (2002) — A&A 392, 699 — [arXiv:astro-ph/0207217](https://arxiv.org/abs/astro-ph/0207217) — *Neupert effect statistical study*
8. Grigis, P. C. & Benz, A. O. (2004) — A&A 426, 1103 — *Soft-Hard-Soft pattern*
9. Holman, G. D. et al. (2011) — Space Sci. Rev. 159, 107 — *RHESSI spectroscopy*
10. Kontar, E. P. et al. (2011) — Space Sci. Rev. 159, 301 — [arXiv:1110.1755](https://arxiv.org/abs/1110.1755) — *Electron properties from HXR*
11. Perriyil, S. M. et al. (2026) — ApJ 999, 27 — [arXiv:2602.19836](https://arxiv.org/abs/2602.19836) — *Loop length vs HXR-SXR delay*
12. Rybicki, G. B. & Lightman, A. P. (1979) — *Radiative Processes in Astrophysics* — Wiley-Interscience
13. Tucker, W. H. (1975) — *Radiation Processes in Astrophysics* — MIT Press
14. Petschek, H. E. (1964) — NASA SP-50, 425 — *Magnetic reconnection rate*

### Aditya-L1 Mission

15. ISRO Aditya-L1 website — [https://www.isro.gov.in/Aditya_L1.html](https://www.isro.gov.in/Aditya_L1.html)
16. PRADAN ISSDC portal — [https://pradan1.issdc.gov.in](https://pradan1.issdc.gov.in)

### RHESSI & Hard X-ray

17. Lin, R. P. et al. (2002) — Solar Phys. 210, 3 — *RHESSI instrument*
18. Masuda, S. et al. (1994) — Nature 371, 495 — *Looptop HXR source*
19. Kontar, E. P. et al. (2019) — ApJ 881, 147 — [arXiv:1812.09474](https://arxiv.org/abs/1812.09474) — *Warm-target model*
20. Effenberger, F. et al. (2017) — ApJ 835, 124 — [arXiv:1612.02856](https://arxiv.org/abs/1612.02856) — *Partially occulted flares*
21. Battaglia, M. et al. (2019) — ApJ 876, 15 — [arXiv:1901.07767](https://arxiv.org/abs/1901.07767) — *Pre-impulsive phase*

### STIX & ASO-S

22. Hayes, L. A. et al. (2022) — [arXiv:2207.02079](https://arxiv.org/abs/2207.02079) — *STIX instrument*
23. Volpara, A. et al. (2023) — [arXiv:2311.07148](https://arxiv.org/abs/2311.07148) — *Regularized imaging spectroscopy*
24. Volpara, A. et al. (2024) — [arXiv:2407.01175](https://arxiv.org/abs/2407.01175) — *RIS for STIX*
25. Li, D. et al. (2024) — [arXiv:2404.02653](https://arxiv.org/abs/2404.02653) — *ASO-S/HXI Neupert effect*
26. Jiang, X. K. et al. (2022) — [arXiv:2207.05390](https://arxiv.org/abs/2207.05390) — *HXI geometric model*
27. Chen, D. et al. (2021) — [arXiv:2012.01629](https://arxiv.org/abs/2012.01629) — *HXI collimator design*
28. French, R. J. et al. (2025) — [arXiv:2511.13862](https://arxiv.org/abs/2511.13862) — *ASO-S/HXI X9 flare*

### ML/DL for Flare Forecasting

29. Bobra, M. G. & Couvidat, S. (2015) — ApJ 798, 135 — *SVM on SHARP*
30. Landa, V. & Reuveni, Y. (2021) — ApJS 258, 29 — [arXiv:2101.12550](https://arxiv.org/abs/2101.12550) — *1D CNN on GOES*
31. Huang, X. et al. (2018) — ApJ 856, 7 — [arXiv:1801.10420](https://arxiv.org/abs/1801.10420) — *CNN on magnetograms*
32. Abduallah, Y. & Wang, J. T. L. (2024) — [arXiv:2405.16080](https://arxiv.org/abs/2405.16080) — *SolarFlareNet Transformer*
33. Vural, O. et al. (2025) — [arXiv:2511.12955](https://arxiv.org/abs/2511.12955) — *GCTAF Transformer*
34. Lv, J. et al. (2026) — *FlareCast Bayesian Deep Learning* — Semantic Scholar
35. Rosales, E. D. et al. (2026) — *Multimodal Inputs* — Semantic Scholar
36. Yi, K. et al. (2023) — *Deep Reinforcement Learning* — [arXiv:2303.04708](https://arxiv.org/abs/2303.04708)
37. Leka, K. D. et al. (2019) — ApJS 243, 36 — *Flare forecasting comparison*
38. Adeyeha, T. et al. (2024) — *Explainable DL* — Semantic Scholar
39. Zhou, J. et al. (2024) — *Sample Imbalance* — Semantic Scholar

### Novel Methods (Non-solar)

40. Vaswani, A. et al. (2017) — NeurIPS — [arXiv:1706.03762](https://arxiv.org/abs/1706.03762) — *Transformer architecture*
41. Raissi, M. et al. (2019) — J. Comp. Phys. 378, 686 — [arXiv:1711.10561](https://arxiv.org/abs/1711.10561) — *Physics-Informed Neural Networks*
42. Chen, R. T. Q. et al. (2018) — NeurIPS — [arXiv:1806.07366](https://arxiv.org/abs/1806.07366) — *Neural ODEs*
43. Scargle, J. D. et al. (2013) — ApJ 764, 167 — [arXiv:1207.5578](https://arxiv.org/abs/1207.5578) — *Bayesian Blocks*
44. Schreiber, T. (2000) — Phys. Rev. Lett. 85, 461 — *Transfer entropy*

---

## Appendix A: Novelty Summary

| # | Novel Contribution | What Exists | What We Do |
|---|-------------------|-------------|------------|
| N1 | Combined soft+hard nowcast from Aditya-L1 | GOES-only or single-instrument nowcasts | Fuse SoLEXS (340 ch) + HEL1OS (10 bands) |
| N2 | Neupert physics-informed loss | No physics-constrained ML for flares | `L_physics` embeds the SXR-HXR integral relation |
| N3 | Spectral-temporal cross-attention transformer | Transformers on scalar fluxes or magnetograms | Multi-energy tokens + temporal self-attention |
| N4 | Transfer entropy precursor index | No causal/time-lagged analysis for precursors | Information-theoretic causality HXR→SXR |
| N5 | Energy-dependent forecasting | All models use GOES 1–8 Å broadband | Full spectrum T, EM, γ, HR features |
| N6 | Self-supervised pretraining on quiet Sun | No pretraining on unlabeled X-ray data | Masked autoencoder on 700+ days |
| N7 | Bayesian UQ for operational forecasting | Point probabilities only | Deep ensemble + conformal prediction |
| N8 | Cross-instrument transfer GOES→Aditya-L1 | No transfer learning across X-ray instruments | 25-year GOES pretrain → SoLEXS fine-tune |

---

## Appendix B: Data Products Summary

| Product | Format | Size | Content |
|---------|--------|------|---------|
| `flare_data.h5` | HDF5+BLOSC | ~10 GB | SoLEXS LC (747), PI (707), HEL1OS (724 days) |
| `features.h5` | HDF5 | ~4 GB | 120+ engineered features per day |
| `nowcast_catalogue.h5` | HDF5 | ~10 MB | All detected flares with parameters |
| `forecast_predictions.h5` | HDF5 | ~100 MB | Probabilities, uncertainties, lead times |
| `plots/` | PDF | ~1.5 GB | 724 daily plots + summary + evaluation |
| `dashboard/` | Streamlit | — | Interactive light curve + alert + spectral viewer |
| `model_checkpoints/` | PyTorch | ~500 MB | 5 ensemble models + LightGBM |

---

*Plan prepared: June 2026 | GPU: NVIDIA A100 80GB | Framework: PyTorch 2.x + Lightning*
