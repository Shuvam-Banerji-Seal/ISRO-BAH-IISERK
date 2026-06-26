# Research Note 03: Aditya-L1 Mission & Instruments
**Part of BAH 2026 Challenge #15** | *Stored: docs/research/03_aditya_l1/*

---

## Mission Overview

- **Launch:** September 2, 2023, PSLV-C57
- **Orbit:** Halo orbit around Sun-Earth L1 (~1.5M km from Earth)
- **7 payloads:** 4 remote sensing (VELC, SUIT, SoLEXS, HEL1OS) + 3 in-situ (ASPEX, PAPA, Magnetometer)
- **Primary goal:** Study solar corona, chromosphere, and dynamics of solar flares/CMEs

## SoLEXS — Solar Low Energy X-ray Spectrometer

| Property | Value |
|----------|-------|
| Energy Range | **2–22 keV** (340 channels, ~0.059 keV/ch) |
| Detectors | SDD1 (7.1 mm², non-functional) + SDD2 (0.1 mm², primary) |
| Timing | **1-second cadence**, 24h/day |
| Data Product | FITS OGIP: LC (86,400 rows/day, 1.39 MB) + PI (86,400 spectra/day × 340 ch, ~472 MB) |
| Coverage | 747 days (Feb 2024–Jun 2026) |
| Integrity | **100%** — 2,988 FITS files, zero corruption |

## HEL1OS — High Energy L1 Orbiting X-ray Spectrometer

| Property | Value |
|----------|-------|
| Energy Range | **10–150 keV** (CZT: 20–150, CdTe: 10–40) — combined range **1.8–160 keV** |
| Detectors | CZT1, CZT2, CdTe1, CdTe2 — 5 bands each |
| Timing | **1-second cadence**, ~12h/orbit |
| Data Products | LC (FITS, 10 files/day, 11 MB each) + spectra (20s accumulation) + event files |
| Coverage | 927 days (Nov 2023–Jun 2026, 98.9%) |
| First Light | Oct 29, 2023 — recorded impulsive phase of flares, consistent with GOES |

## Combined SoLEXS + HEL1OS

- **Dual coverage:** 724 days (Feb 2024–Jun 2026, 78.3% of all data days)
- **Longest streak:** 57 days (Jul 3–Aug 28, 2024)
- **Energy range:** 1.8–160 keV
- **Cadence:** 1-second (both instruments)

## Publications Using Aditya-L1 Data

**ISRO Announcements:**
- HEL1OS first light: [isro.gov.in](https://www.isro.gov.in/HEL1OS_captures_glimpseof_solarflares.html) (Nov 7, 2023)
- SoLEXS operational since early 2024

**No published papers yet using SoLEXS + HEL1OS combined data** — this project is the first to use both simultaneously for flare analysis.

## Key References

- ISRO Aditya-L1 page: [isro.gov.in/Aditya_L1.html](https://www.isro.gov.in/Aditya_L1.html)
- PRADAN data portal: [pradan1.issdc.gov.in](https://pradan1.issdc.gov.in)
- User manuals: `docs/manuals/SoLEXS_UserManual.pdf`, `docs/manuals/HEL1OS_UserManual.pdf`
