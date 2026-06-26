# SoLEXS Processed Data Analysis

> Auto-generated analysis of Aditya-L1 SoLEXS Level-1 processed data.
> Analysis date: 2026-06-26

---

## 1. Date Range

| Metric | Value |
|--------|-------|
| Earliest date | **2024-02-01** |
| Latest date | **2026-06-22** |
| Span | 873 calendar days (~2.39 years) |

SoLEXS data covers the full operational period of Aditya-L1 from early 2024 through mid-2026.

---

## 2. File Counts

| Category | Count |
|----------|-------|
| Unique days with data | **747** |
| Total FITS files | **2,990** |
| LC files (SDD2) | 747 |
| PI files (SDD2) | 747 |
| GTI files (SDD2) | 747 |
| GTI files (SDD1) | 747 |
| HK files (SDD2) | 2 |

Each processed day contains exactly 4 standard FITS files (SDD2 LC, PI, GTI + SDD1 GTI). Two days (2025-01-19, 2025-02-02) additionally contain HK (housekeeping) files.

Total disk usage: **329.67 GB** (processed), **4.02 GB** (raw zips).

---

## 3. File Types by Detector

### SDD1 (Silicon Drift Detector 1 — aperture 7.1 mm²)
| Extension | Count | Description |
|-----------|-------|-------------|
| `.gti` | 747 | Good Time Intervals |

**SDD1 produces only GTI files** — no light curves or spectra. The GTI data has **0 rows** for every SDD1 file (empty GTI table), meaning SDD1 is not producing usable time-resolved data.

### SDD2 (Silicon Drift Detector 2 — aperture 0.1 mm², high-flux)
| Extension | Count | Description |
|-----------|-------|-------------|
| `.lc` | 747 | Light curve (1-sec count rate) |
| `.pi` | 747 | PI spectrum (340 energy channels × 86,400 spectra/day) |
| `.gti` | 747 | Good Time Intervals (1–221 rows per day) |
| `.hk` | 2 | Housekeeping telemetry (only on 2 days) |

**SDD2 is the primary science detector.** All flare analysis should use SDD2 data.

---

## 4. FITS Headers — Complete Reference

### 4.1 Light Curve (`.lc`) — HDU 0 (PRIMARY)

| Keyword | Example Value | Description |
|---------|--------------|-------------|
| `SIMPLE` | `True` | Standard FITS |
| `BITPIX` | `8` | 8-bit (no compression) |
| `NAXIS` | `0` | No data in primary |
| `EXTEND` | `True` | Extensions present |
| `MISSION` | `ADITYA-L1` | Mission name |
| `TELESCOP` | `AL1` | Telescope ID |
| `INSTRUME` | `SoLEXS` | Instrument name |
| `ORIGIN` | `SoLEXSPOC` | Data origin (SoLEXS Payload Operations Centre) |
| `CREATOR` | `solexs_pipeline-1.4` | Processing pipeline version |
| `FILENAME` | `AL1_SOLEXS_YYYYMMDD_SDD2_L1.lc` | Filename |
| `CONTENT` | `LIGHT CURVE` | Content type |
| `DATE` | `2026-05-05` | File creation date (YYYY-MM-DD) |
| `OBS_DATE` | `20240201` | Observation date (YYYYMMDD) |
| `OBS_ID` | `UNP_9999_999999` | Observation ID (early data) / `N00_0000_000271` (later) |
| `DATASUM` | `0` | Data checksum |

### 4.2 Light Curve (`.lc`) — HDU 1 (RATE BinTable)

| Keyword | Example Value | Description |
|---------|--------------|-------------|
| `XTENSION` | `BINTABLE` | Binary table extension |
| `EXTNAME` | `RATE` | Extension name |
| `HDUCLASS` | `OGIP` | OGIP standard |
| `HDUVERS` | `1.1.0` | OGIP version |
| `HDUCLAS1` | `LIGHTCURVE` | OGIP class 1 |
| `HDUCLAS2` | `TOTAL` | Total counts (not background-subtracted) |
| `HDUCLAS3` | `COUNTS` | Count-rate data |
| `FILTER` | `SDD2` | Detector filter |
| `TTYPE1` | `TIME` | Column 1 name |
| `TFORM1` | `D` | Column 1 type (float64) |
| `TTYPE2` | `COUNTS` | Column 2 name |
| `TFORM2` | `D` | Column 2 type (float64) |
| `TSTART` | `1706745600.0` | Mission elapsed time start (s) |
| `TSTOP` | `1706831999.0` | Mission elapsed time stop (s) |
| `TIMEDEL` | `1` | Time resolution (1 second) |
| `TIMZERO` | `0` | Time zero offset |
| `MJDREFI` | `40587` | MJD reference integer part |
| `MJDREFF` | `0` or `0.22916666651` | MJD reference fractional part |
| `TIMESYS` | `UTC` | Time system |
| `TIMEREF` | `LOCAL` | Time reference frame |
| `TIMEUNIT` | `s` | Time unit (seconds) |
| `DATE-OBS` | `2024-02-01 00:00:00` | Observation start (UTC) |
| `DATE-END` | `2024-02-01 23:59:59` | Observation end (UTC) |
| `NUMBAND` | `4` | Number of energy bands |
| `DATASUM` | varies | Column data checksum |

**Data shape:** (86,400,) — one count per second for 24 hours.
**Columns:** `TIME` (float64, MET seconds), `COUNTS` (float64, count rate).

### 4.3 PI Spectrum (`.pi`) — HDU 1 (SPECTRUM BinTable)

| Keyword | Example Value | Description |
|---------|--------------|-------------|
| `EXTNAME` | `SPECTRUM` | Extension name |
| `HDUCLASS` | `OGIP` | OGIP standard |
| `HDUCLAS1` | `SPECTRUM` | OGIP class 1 |
| `HDUCLAS2` | `TOTAL` | Total counts |
| `HDUCLAS3` | `COUNTS` | Count-rate data |
| `HDUCLAS4` | `TYPE:II` | Type II PHA file (time-resolved) |
| `CHANTYPE` | `PI` | Pulse Invariant channels |
| `POISSERR` | `False` | No Poisson errors |
| `DETCHANS` | `340` | Number of energy channels (2–22 keV, ~0.059 keV/channel) |
| `CORRSCAL` | `1.0` | Correction scaling |
| `AREASCAL` | `1.0` | Area scaling |
| `TTYPE1` | `TSTART` | Start time of spectrum |
| `TFORM1` | `D` | float64 |
| `TUNIT1` | `s` | Seconds |
| `TTYPE2` | `TELAPSE` | Elapsed time |
| `TFORM2` | `D` | float64 |
| `TTYPE3` | `SPEC_NUM` | Spectrum sequence number |
| `TFORM3` | `J` | int32 |
| `TTYPE4` | `CHANNEL` | Energy channel array |
| `TFORM4` | `340K` | Array of 340 int16 values |
| `TTYPE5` | `COUNTS` | Counts per channel |
| `TFORM5` | `340D` | Array of 340 float64 values |
| `TTYPE6` | `EXPOSURE` | Exposure time |
| `TFORM6` | `D` | float64 |
| `TUNIT6` | `s` | Seconds |

**Data shape:** (86,400,) — one spectrum per second.
**Columns:** `TSTART` (float64), `TELAPSE` (float64), `SPEC_NUM` (int32), `CHANNEL` (int16, shape 340), `COUNTS` (float64, shape 340), `EXPOSURE` (float64).

### 4.4 GTI (`.gti`) — SDD1

| Keyword | Example Value | Description |
|---------|--------------|-------------|
| `CONTENT` | `GOOD TIME INTERVAL` | Content type |
| `TSTART` | *(empty)* | Not populated for SDD1 |
| `TSTOP` | *(empty)* | Not populated for SDD1 |
| `EXPOSURE` | `0.0` | Zero exposure (SDD1 has no data) |
| `TTYPE1` | `START` | Start column |
| `TFORM1` | `I` | int16 |
| `TTYPE2` | `STOP` | Stop column |
| `TFORM2` | `I` | int16 |

**Data shape:** (0,) — **always empty** for SDD1. SDD1 is not producing science data.

### 4.5 GTI (`.gti`) — SDD2

| Keyword | Example Value | Description |
|---------|--------------|-------------|
| `TSTART` | `2024-02-01T00:00:01+00:00` | GTI start (ISO-8601) |
| `TSTOP` | `2024-02-01T23:59:59+00:00` | GTI stop (ISO-8601) |
| `EXPOSURE` | `86394.0` – `86398.0` | Usable exposure (seconds, out of 86,400) |
| `TTYPE1` | `START` | Start column |
| `TFORM1` | `D` | float64 (MJD) |
| `TTYPE2` | `STOP` | Stop column |
| `TFORM2` | `D` | float64 (MJD) |

**Data shape:** varies — 1 to 221 rows (see §6).
**Columns:** `START` (float64, MJD), `STOP` (float64, MJD).

### 4.6 Housekeeping (`.hk`) — Only 2 days

| Keyword | Value |
|---------|-------|
| `SIMPLE` | `True` |
| `BITPIX` | `8` |
| `NAXIS` | `0` |
| `EXTEND` | `True` |
| `DATASUM` | `0` |

**HDU 1 (HK BinTable) columns:**
`SDD_TEMP`, `ELECTRONIC_BOX_TEMPERATURE`, `COOLER_CURRENT`, `BACK_CONTACT`, `SUN_ANGLE`, `HV_ENABLE`, `RESET_ENABLE`, `FLARE_TRIGGER`, `FAST_COUNTS_LOW`, `FAST_COUNTS_MED`, `FAST_COUNTS_HIGH`, `FAST_COUNTS`, `SLOW_COUNTS`

**Data shape:** (86,400,) — one housekeeping sample per second.

---

## 5. Data Integrity

| Metric | Result |
|--------|--------|
| Total FITS files checked | 2,990 |
| Successfully opened | **2,990 (100%)** |
| Corrupted/failed | **0** |

**All FITS files open successfully with `astropy.io.fits` and pass verification.** No corruption detected.

---

## 6. Date Gaps

Out of 873 expected calendar days (2024-02-01 to 2026-06-22), **126 days are missing** (85.8% coverage).

### Major Gaps

| Period | Missing Days | Likely Cause |
|--------|-------------|--------------|
| 2024-06-01 to 2024-06-30 | 30 days (entire month) | Extended instrument shutdown or data gap |
| 2025-02-03 to 2025-02-07 | 5 days | |
| 2025-07-20 to 2025-07-27 | 8 days | |
| 2025-09-26 to 2025-09-29 | 4 days | |
| 2024-12-22 to 2024-12-27 | 3 days | Holiday period |

### Complete Missing Dates

<details>
<summary>Click to expand all 126 missing dates</summary>

```
2024/03/27  2024/03/28  2024/03/30
2024/06/01  2024/06/02  2024/06/03  2024/06/04  2024/06/05
2024/06/06  2024/06/07  2024/06/08  2024/06/09  2024/06/10
2024/06/11  2024/06/12  2024/06/13  2024/06/14  2024/06/15
2024/06/16  2024/06/17  2024/06/18  2024/06/19  2024/06/20
2024/06/21  2024/06/22  2024/06/23  2024/06/24  2024/06/25
2024/06/26  2024/06/27  2024/06/28  2024/06/29  2024/06/30
2024/07/02  2024/08/29  2024/09/09  2024/09/24  2024/10/12
2024/11/06  2024/12/22  2024/12/23  2024/12/26  2024/12/27
2025/01/12  2025/01/13  2025/01/21  2025/01/27  2025/01/29
2025/01/30  2025/02/03  2025/02/04  2025/02/05  2025/02/06
2025/02/07  2025/03/02  2025/03/05  2025/03/14  2025/03/16
2025/03/21  2025/04/06  2025/04/07  2025/04/16  2025/04/28
2025/04/29  2025/05/01  2025/05/02  2025/05/05  2025/05/06
2025/05/12  2025/05/15  2025/05/20  2025/05/21  2025/05/25
2025/06/07  2025/06/10  2025/06/11  2025/06/22  2025/06/24
2025/07/02  2025/07/03  2025/07/05  2025/07/06  2025/07/11
2025/07/12  2025/07/13  2025/07/17  2025/07/20  2025/07/21
2025/07/22  2025/07/23  2025/07/24  2025/07/25  2025/07/26
2025/07/27  2025/08/05  2025/08/14  2025/08/15  2025/08/23
2025/08/26  2025/08/31  2025/09/14  2025/09/21  2025/09/22
2025/09/24  2025/09/26  2025/09/27  2025/09/28  2025/09/29
2025/10/16  2025/10/24  2025/10/29  2025/10/30  2025/11/11
2025/11/17  2025/12/28  2025/12/31  2026/02/13  2026/02/17
2026/02/20  2026/04/16  2026/04/17  2026/04/23  2026/04/29
2026/05/22  2026/06/04  2026/06/09
```

</details>

---

## 7. File Sizes

| File Type | Count | Min | Median | Max | Mean |
|-----------|-------|-----|--------|-----|------|
| LC (`.lc`) | 747 | 1.33 MB | 1.33 MB | 1.33 MB | 1.33 MB |
| PI (`.pi`) | 747 | 450.56 MB | 450.56 MB | 450.56 MB | 450.56 MB |
| GTI SDD1 (`.gti`) | 747 | 5,760 B | 5,760 B | 5,760 B | 5,760 B |
| GTI SDD2 (`.gti`) | 747 | 8,640 B | 8,640 B | 8,640 B | 8,640 B |
| HK (`.hk`) | 2 | 8.99 MB | 8.99 MB | 8.99 MB | 8.99 MB |

**All LC and PI files are identical in size** — this is expected because:
- LC: 86,400 rows × 16 bytes/row (2 float64 cols) = 1,382,400 bytes of data + headers = ~1.33 MB
- PI: 86,400 rows × 5,468 bytes/row (includes 340-element arrays) = ~450.56 MB

No outliers detected (IQR method).

---

## 8. Naming Conventions

### Directory Structure
```
data/processed/solexs/
└── YYYY/
    └── MM/
        └── DD/
            ├── SDD1/
            │   └── AL1_SOLEXS_YYYYMMDD_SDD1_L1.gti
            └── SDD2/
                ├── AL1_SOLEXS_YYYYMMDD_SDD2_L1.lc
                ├── AL1_SOLEXS_YYYYMMDD_SDD2_L1.pi
                ├── AL1_SOLEXS_YYYYMMDD_SDD2_L1.gti
                └── AL1_SOLEXS_YYYYMMDD_SDD2_L1.hk  (rare)
```

### Filename Pattern
```
AL1_SOLEXS_YYYYMMDD_SDD{1,2}_L1.{lc,pi,gti,hk}
```

| Component | Values | Description |
|-----------|--------|-------------|
| `AL1` | Fixed | Aditya-L1 mission |
| `SOLEXS` | Fixed | Instrument |
| `YYYYMMDD` | 20240201–20260622 | Observation date |
| `SDD{1,2}` | `SDD1` or `SDD2` | Detector |
| `L1` | Fixed | Level-1 processing |
| `{ext}` | `lc`, `pi`, `gti`, `hk` | Data product |

---

## 9. Raw vs Processed

| Metric | Raw Zips | Processed Days |
|--------|----------|---------------|
| Total | 750 zips | 747 days |
| Unique dates | 747 | 747 |
| Match | **100%** | **100%** |

### Raw Zip Naming
```
AL1_SLX_L1_YYYYMMDD_v{1.0,1.1}.zip
```

### Version Distribution
| Version | Count | Notes |
|---------|-------|-------|
| `v1.0` | 657 | Original processing |
| `v1.1` | 93 | Re-processed data |

### Duplicate Versions (same date, both v1.0 and v1.1)
| Date | Versions |
|------|----------|
| 2024-10-01 | v1.0, v1.1 |
| 2024-10-25 | v1.0, v1.1 |
| 2024-12-12 | v1.0, v1.1 |

Three dates have both v1.0 and v1.1 raw zips. The decompression step selected one version (appears to prefer v1.0 when both exist).

### Pipeline Version in FITS Headers
| Pipeline | Count (sampled) |
|----------|----------------|
| `solexs_pipeline-1.1` | ~20% of early-2024 data |
| `solexs_pipeline-1.4` | ~80% of data (current) |

The pipeline version evolved over time. The `CREATOR` keyword tracks this.

---

## 10. Anomalies and Observations

### 10.1 SDD1 GTI Always Empty
Every SDD1 GTI file has **0 rows** and `EXPOSURE=0.0`. SDD1 (7.1 mm² aperture) appears to not produce science-grade data in Level-1 processing. **All science analysis should use SDD2.**

### 10.2 MJDREFF Inconsistency
- Most LC files: `MJDREFF = 0`
- Some LC files (e.g., 2024-09-15): `MJDREFF = 0.22916666651` (corresponding to 05:30:00 UT, the IST offset for India)

This inconsistency in the MJD reference fractional part suggests some data was processed with a different time reference epoch. When computing absolute times from `TIME` column, use `MJDREFI + MJDREFF` to get the full MJD reference.

### 10.3 GTI Row Count Variation
SDD2 GTI files have between **1 and 221 rows**, indicating variable numbers of data gaps throughout the day:

| GTI Rows | File Count | Interpretation |
|----------|------------|----------------|
| 1 | 35 | Single contiguous interval (clean day) |
| 2–5 | 517 | Typical (few occultation gaps) |
| 6–30 | 174 | Moderate gaps |
| 31–100 | 5 | Heavy fragmentation |
| 100+ | 4 | Extreme fragmentation (up to 221 intervals) |

Higher GTI row counts indicate frequent Earth occultation gaps or data quality issues. Days with 200+ GTI rows should be treated with caution.

### 10.4 EXPOSURE Range
SDD2 GTI exposure ranges from **39,581 s** (~11 hours) to **86,398 s** (~24 hours), with most days having >85,000 s exposure. Low-exposure days correspond to high GTI row counts (frequent gaps).

### 10.5 Large Gap: June 2024
The entire month of June 2024 (30 days) is missing from both raw and processed data. This likely corresponds to an instrument shutdown, commissioning phase, or data downlink issue.

### 10.6 Housekeeping Files Rare
Only **2 out of 747 days** have `.hk` files (2025-01-19 and 2025-02-02). Housekeeping data was either not included in most raw downloads or was excluded during processing.

### 10.7 Date-Consistent Processing
The `DATE` keyword in FITS headers reflects when the file was *processed*, not when the observation occurred. Processing dates range from 2025-01-16 to 2026-05-05, indicating batch reprocessing of older data.

### 10.8 OBS_ID Evolution
- Early data (2024-02-01): `OBS_ID = UNP_9999_999999` (unplanned/calibration)
- Later data: `OBS_ID = N00_0000_XXXXXX` (sequential observation IDs)

---

## Summary

| Property | Value |
|----------|-------|
| Date range | 2024-02-01 to 2026-06-22 |
| Days with data | 747 / 873 expected (85.8%) |
| Primary detector | SDD2 (LC + PI + GTI) |
| Time resolution | 1 second |
| Energy range | 2–22 keV (340 channels, ~0.059 keV/channel) |
| LC file size | 1.33 MB/day (fixed) |
| PI file size | 450.56 MB/day (fixed) |
| Total processed data | ~330 GB |
| Data integrity | 100% (0 corrupted files) |
| Pipeline version | v1.1 (early) → v1.4 (current) |
