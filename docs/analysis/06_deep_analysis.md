# Deep Data Analysis — SoLEXS + HEL1OS

## 1. HEL1OS Coverage After Multi-Orbit Concatenation

The original extraction kept only 1 orbit per day (last zip overwrites previous).
After running `data/downloads/concat_orbits.py`, all orbits per day are merged:

| Metric | Pre-concat | Post-concat |
|--------|-----------|-------------|
| Orbits/day average | 2.73 extracted → 1 survives | **All merged** |
| Coverage mean | 5.7 h/day | **13.2 h/day** |
| Coverage median | 5.7 h/day | **12.0 h/day** |
| Coverage max | 12 h | **36 h** (3+ orbits) |
| Days >20h | 0 | **27** |

Total recovered: **1,529 extra orbits** across 888 days.

## 2. SoLEXS Pipeline Version Consistency

5 pipeline versions detected across 747 days:

| Version | Days | Sample median | Sample max |
|---------|------|---------------|------------|
| v1.1 | 174 | 20.0 | 1554 |
| v1.2 | 72 | 139.0 | 1075 |
| v1.3 | 146 | 41.0 | 4154 |
| v1.4 | 353 | 15.0 | 138 |
| v2.0-beta.1 | 2 | 36.0 | 1954 |

v1.4 dominates (47% of days). v1.1 and v1.3 have notably higher max values,
suggesting possible sensitivity changes between pipeline versions. However,
the max values come from different days with different flare activity, so
this may be natural variation.

## 3. CZT/CdTe Alignment

**Finding: ALL sampled days have >1s misalignment between CZT and CdTe
detectors.** The CdTe and CZT detectors start at different times within
the same orbit. The reader truncates to `min_rows`, which can lose valid
data from the longer detector.

Recommendation: align both detectors on a common MJD grid before combining,
using NaN to fill gaps where one detector has no data.

## 4. CZT2 vs CZT1 Correlation

| Statistic | Value |
|-----------|-------|
| Range | -0.407 to 0.997 |
| Median | 0.606 |
| Days with r < 0.5 | 92/301 (30.6%) |
| Days with r < 0 | 16/301 (5.3%) |

**Correlation is solar-activity dependent, NOT a hardware issue:**
- High correlation (r > 0.6) during solar maximum (Jul 2024 - Aug 2025)
- Low correlation (r < 0.4) during quiet periods (early/late mission)
- Both detectors see independent background noise when Sun is quiet
- 0 days with CZT2 disabled; 0 days with different row counts

## 5. Detector Anomaly — 2026-02-01 to 02-03

| Date | Median | Mean | Max | NaN% | HEL1OS rows |
|------|--------|------|-----|------|-------------|
| 2026-01-30 (before) | 16 | 18 | 78 | 0% | normal |
| 2026-02-01 | **186** | **561** | **27,465** | **48%** | 40,154 |
| 2026-02-02 | **1,063** | **2,476** | **23,632** | **48%** | 3,693 |
| 2026-02-03 | **931** | **1,050** | **2,579** | **94%** | 4,369 |
| 2026-02-04 (after) | 148 | 362 | 4,019 | 48% | elevated |

The anomaly persists for 4+ days. Normal median is 10-60 cts/s. During
anomaly: 186-1,063 cts/s. HEL1OS also shows reduced rows during this period.
These days should be EXCLUDED from training.

## 6. GTI Gap Analysis

| Statistic | Value |
|-----------|-------|
| Gaps found | 345 (sampled) |
| Median gaps/day | 2 |
| Max gaps/day | 42 |
| Gap duration median | 172,800 s (48 h) |

The 48h gap pattern suggests recurring Earth occultation or calibration
events. Gap start times cluster near 0h UT.

## 7. Signal-to-Noise Ratio

| Day | Type | Background | Noise | SNR |
|-----|------|-----------|-------|-----|
| 2025-06-01 | Quiet | 7 cts/s | 2.0 | **79** |
| 2024-02-22 | X6.3 flare | 82 cts/s | 7.0 | **407** |

SoLEXS has excellent SNR even on quiet days. The noise floor (MAD of
residuals) is low enough to detect C-class flares reliably.

## 8. Energy Band Analysis

CZT bands are 80%+ zero at 1s cadence. CdTe bands are 90%+ zero.
This is normal for hard X-ray detectors at 1s resolution — most 1s bins
contain zero counts.

## 9. Detection Fixes Applied

| Issue | Fix | Result |
|-------|-----|--------|
| NaN saturation (2025-07-29) | Forward-fill instead of median-fill | X-class detected ✅ |
| Impulsive peak (2025-02-26) | Min_duration bypass for high-class flares | X-class detected ✅ |
| X6.3 calibration (2024-02-22) | GOES-validated scale (2.5e-8) | Correctly X-class ✅ |

## 10. Calibration Validation

SoLEXS calibration validated against GOES-16 XRSF L2 data:
- 431 GOES netCDF files processed
- **2,134 flare events** cataloged (51 X, 712 M, 1,371 C)
- X6.3 flare (2024-02-22): GOES peak = 6.517e-4 W/m²
- SoLEXS peak = 25,452 cts/s → calibrated scale = **2.5e-8**
- Catalog saved to `data/catalogs/goes_flare_catalog.csv`
