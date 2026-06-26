# Research Note 01: Solar Flare Physics & X-ray Emission Mechanisms
**Part of BAH 2026 Challenge #15** | *Stored: docs/research/01_physics/*

---

## Standard Flare Model (CSHKP)

The standard model describes magnetic reconnection in the corona:
- **Energy storage:** Magnetic energy builds up in stressed coronal loops
- **Trigger:** Instability in the current sheet triggers reconnection
- **Energy release:** Magnetic energy → kinetic energy of accelerated particles + thermal plasma
- **Emission:** SXR from hot plasma (thermal bremsstrahlung), HXR from accelerated electrons (non-thermal bremsstrahlung)

**Key equation — Reconnection rate:**
$$M_A = v_{\text{in}}/v_A \sim 0.01\text{--}0.1$$

### References
- Carmichael 1964 (NASA SP-50)
- Sturrock 1966 (Nature 211, 695)
- Hirayama 1974 (Solar Phys. 34, 323)
- Kopp & Pneuman 1976 (Solar Phys. 50, 85)

## Thermal Bremsstrahlung (Soft X-rays)

Electrons in hot plasma emit X-rays via free-free transitions:

$$\varepsilon_{\text{ff}}(\nu, T) = 6.8 \times 10^{-38} n_e^2 T^{-1/2} \bar{g}_{\text{ff}} \exp(-h\nu/kT)$$

For SoLEXS (2–22 keV), this corresponds to plasma temperatures T > 10 MK.

### References
- Rybicki & Lightman 1979 — *Radiative Processes in Astrophysics*
- Tucker 1975 — *Radiation Processes in Astrophysics*

## Non-thermal Bremsstrahlung (Hard X-rays)

Accelerated electrons with power-law spectrum $f(E) = F_0 E^{-\delta}$ produce:

**Thick-target:** $I(\varepsilon) \propto \varepsilon^{-(\delta-1)/2}$ (electrons lose all energy in target)

**Thin-target:** $I(\varepsilon) \propto \varepsilon^{-\delta+1}$ (optically thin coronal source)

### References
- Brown 1971 — *Solar Phys. 18, 489* — Original thick-target model
- Holman et al. 2011 — *Space Sci. Rev. 159, 107* — RHESSI spectroscopy
- Kontar et al. 2011 — [arXiv:1110.1755](https://arxiv.org/abs/1110.1755) — Electron properties from HXR

## GOES X-ray Classification

| Class | Peak Flux (W/m²) | Frequency (SC25) |
|-------|-----------------|------------------|
| A | <10⁻⁷ | Several/hour |
| B | 10⁻⁷–10⁻⁶ | ~10/day |
| C | 10⁻⁶–10⁻⁵ | ~5/day |
| M | 10⁻⁵–10⁻⁴ | ~2/day |
| X | >10⁻⁴ | ~1/week |

## Neupert Effect

The empirical correlation: F_SXR(t) ∝ ∫ F_HXR(t') dt'

**Physical interpretation:** Non-thermal electrons (HXR) heat chromospheric plasma via collisions → heated plasma (10+ MK) rises into corona → emits SXR.

### References
- Neupert 1968 — *ApJ 153, L59*
- Veronig et al. 2002 — [arXiv:astro-ph/0207217](https://arxiv.org/abs/astro-ph/0207217) — Statistical study of 1,114 flares
- Li et al. 2024 — [arXiv:2404.02653](https://arxiv.org/abs/2404.02653) — ASO-S/HXI validation (ρ > 0.90 for all 149 flares)
- Perriyil et al. 2026 — [arXiv:2602.19836](https://arxiv.org/abs/2602.19836) — Loop length vs HXR-SXR delay

## Electron Acceleration

Stochastic acceleration by turbulence: Fokker-Planck equation

$$\frac{\partial f}{\partial t} = \frac{1}{p^2}\frac{\partial}{\partial p}\left(D_{pp} p^2 \frac{\partial f}{\partial p}\right) - \frac{f}{\tau_{\text{esc}}} + Q(p,t)$$

### References
- Petrosian & Chen 2010 — [arXiv:1002.2673](https://arxiv.org/abs/1002.2673) — Stochastic acceleration model
- Kong et al. 2022 — [arXiv:2211.15333](https://arxiv.org/abs/2211.15333) — Numerical modeling
- Kontar et al. 2023 — [arXiv:2304.01088](https://arxiv.org/abs/2304.01088) — Acceleration efficiency

## Flare Onset & Precursors

**Hot Onset:** Enhanced 10–15 MK plasma before non-thermal HXR (Hudson et al. 2020)
- Implies non-collisional heating mechanism
- Relevant for precursor detection in SoLEXS data

### References
- Hudson et al. 2020 — [arXiv:2007.05310](https://arxiv.org/abs/2007.05310) — Hot X-ray onsets
- Battaglia et al. 2019 — [arXiv:1901.07767](https://arxiv.org/abs/1901.07767) — Pre-impulsive phase
- Sharykin et al. 2025 — [arXiv:2504.15097](https://arxiv.org/abs/2504.15097) — Multiwavelength precursors
