# Research Note 04: Hard X-ray Detection & RHESSI/STIX Legacy
**Part of BAH 2026 Challenge #15** | *Stored: docs/research/04_hard_xray/*

---

## Key Instruments for Hard X-ray Solar Physics

### RHESSI (2002–2023)
- Energy range: 3 keV–17 MeV
- Imaging: 9 rotating modulation collimators → 2.3–36 arcsec resolution
- Key contributions:
  - Systematic footpoint + looptop imaging spectroscopy
  - Non-thermal spectral indices γ = 3–6 (M/X flares)
  - Electron energetics: 10–50% of flare energy in accelerated electrons
  - Warm-target model: low-energy cutoff determination (Kontar et al. 2019)

*Key papers:*
- Lin et al. 2002 — *Solar Phys. 210, 3* — Instrument
- Holman et al. 2011 — *Space Sci. Rev. 159, 107* — Spectroscopy review
- Kontar et al. 2019 — [arXiv:1812.09474](https://arxiv.org/abs/1812.09474) — Warm-target model
- Effenberger et al. 2017 — [arXiv:1612.02856](https://arxiv.org/abs/1612.02856) — Partially occulted flares

### STIX on Solar Orbiter (2020–present)
- Energy: 4–150 keV
- Imaging: 30 collimators (indirect Fourier imaging)
- Key advances:
  - Regularized Imaging Spectroscopy (RIS): Tikhonov regularization across energy channels
  - First electron flux spectral imaging along flare loop (Volpara et al. 2023)
  - Online visibility computation from Moiré patterns

*Key papers:*
- Hayes et al. 2022 — [arXiv:2207.02079](https://arxiv.org/abs/2207.02079) — STIX instrument
- Volpara et al. 2023 — [arXiv:2311.07148](https://arxiv.org/abs/2311.07148) — RIS for STIX
- Volpara et al. 2024 — [arXiv:2407.01175](https://arxiv.org/abs/2407.01175) — Improved RIS method

### ASO-S/HXI (2022–present)
- Energy: 30–200 keV
- Imaging: 91 grid pairs
- Key result: Neupert effect validated across 149 flares (Li et al. 2024)
- First Chinese hard X-ray solar imager

*Key papers:*
- Li et al. 2024 — [arXiv:2404.02653](https://arxiv.org/abs/2404.02653) — HXI Neupert effect
- Jiang et al. 2022 — [arXiv:2207.05390](https://arxiv.org/abs/2207.05390) — HXI geometric model

### NuSTAR (2012–present)
- Energy: 3–79 keV
- Unique: Focusing optics (~9.5 arcsec resolution), 12 arcmin FOV
- Key result: Non-thermal emission down to <7 keV in microflares (Glesener et al. 2020)
- Not a solar-dedicated instrument

*Key papers:*
- Glesener et al. 2020 — [arXiv:2003.12864](https://arxiv.org/abs/2003.12864) — NuSTAR microflare

## Electron Acceleration & Transport Physics

**Kong et al. (2022)** — Numerical modeling connecting looptop and footpoint HXR sources via Fokker-Planck + MHD. Direct simulation of electron acceleration, transport, and bremsstrahlung emission.
- [arXiv:2211.15333](https://arxiv.org/abs/2211.15333)

**Kontar et al. (2023)** — Efficiency of electron acceleration during impulsive phase. Found acceleration efficiency close to 100% of available energy in some events.
- [arXiv:2304.01088](https://arxiv.org/abs/2304.01088)

## Flare Onset & Precursors

**Hudson et al. (2020)** — "Hot X-ray Onsets": Enhanced isothermal plasma at 10–15 MK detected up to tens of seconds before impulsive phase (before non-thermal HXR). Challenges standard flare heating models.
- [arXiv:2007.05310](https://arxiv.org/abs/2007.05310)

**Sharykin et al. (2025)** — Multiwavelength precursors before X4.9 limb flare: current sheet formation, filament eruption, tether-cutting reconnection scenario.
- [arXiv:2504.15097](https://arxiv.org/abs/2504.15097)

## Relevance to HEL1OS

- HEL1OS is **non-imaging** (Sun-as-a-star spectrometer) — like GOES but with spectral resolution
- No prior work has used HEL1OS for flare physics because it's so new (launched 2023)
- Key operational advantage: **continuous 1-second cadence** unlike STIX/RHESSI's observing constraints
- Combined with SoLEXS, provides the first **full energy coverage from 2–150 keV** at 1s cadence
