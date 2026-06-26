# HEL1OS Processed Data — Comprehensive Analysis

**Generated:** 2026-06-26  
**Data root:** `/store/shuvam/ISRO-BAH-IISERK/data/processed/hel1os/`

---

## 1. Date Range

| Metric | Value |
|--------|-------|
| Earliest date | 2023-11-30 |
| Latest date | 2026-06-23 |
| Calendar span | 936 days |
| Days with data | 902 (excluding 25 empty dirs) |
| Missing days | 10 |

HEL1OS data begins ~1 month after launch (Aditya-L1 launched 2023-09-02, entered L1 orbit ~2023-11-30).

---

## 2. File Count

| Metric | Count |
|--------|-------|
| Processed day directories (YYYY/MM/DD) | 927 |
| Days with actual data files | 902 |
| Empty day directories (no files) | 25 |
| Total FITS files (top-level) | 7,216 |
| Total .txt files (top-level) | 1,804 |
| Total files (top-level) | 9,020 |
| Files per day (standard) | 10 (8 FITS + 2 TXT) |

---

## 3. File Types Per Day

Every day with data contains exactly **10 files** in a flat directory:

| File | Type | Description |
|------|------|-------------|
| `lightcurve_czt1.fits` | FITS | CZT1 hard X-ray light curve (5 energy bands) |
| `lightcurve_czt2.fits` | FITS | CZT2 hard X-ray light curve (5 energy bands) |
| `lightcurve_cdte1.fits` | FITS | CdTe1 soft/hard X-ray light curve (5 energy bands) |
| `lightcurve_cdte2.fits` | FITS | CdTe2 soft/hard X-ray light curve (5 energy bands) |
| `hel1os_czt_spectra_czt1.fits` | FITS | CZT1 energy spectra (341 channels, 20s accumulation) |
| `hel1os_czt_spectra_czt2.fits` | FITS | CZT2 energy spectra (341 channels, 20s accumulation) |
| `hel1os_cdte_spectra_cdte1.fits` | FITS | CdTe1 energy spectra (511 channels, 20s accumulation) |
| `hel1os_cdte_spectra_cdte2.fits` | FITS | CdTe2 energy spectra (511 channels, 20s accumulation) |
| `czt1dispix.txt` | TXT | CZT1 detector pixel configuration (4 integers) |
| `czt2dispix.txt` | TXT | CZT2 detector pixel configuration (0 or 4 bytes) |

**File type distribution (across 902 days):**

| Filename | Occurrence |
|----------|-----------|
| lightcurve_czt1.fits | 902 |
| lightcurve_czt2.fits | 902 |
| lightcurve_cdte1.fits | 902 |
| lightcurve_cdte2.fits | 902 |
| hel1os_czt_spectra_czt1.fits | 902 |
| hel1os_czt_spectra_czt2.fits | 902 |
| hel1os_cdte_spectra_cdte1.fits | 902 |
| hel1os_cdte_spectra_cdte2.fits | 902 |
| czt1dispix.txt | 902 |
| czt2dispix.txt | 902 |

---

## 4. FITS Headers — Full Keyword Dump

### 4.1 Lightcurve Primary Header (shared across all LC files)

```
SIMPLE       = True
BITPIX       = 8
NAXIS        = 0
EXTEND       = True
MISSION      = 'Aditya-L1'
INSTRUME     = 'HEL1OS'
TELESCOP     = 'Aditya-L1'
CREATOR      = 'HEL1OS-L1-PIPELINE'
POC          = 'HEL1OS-POC'
ENTITY       = 'SAG, URSC, ISRO'
MJDSTART     = <float>          (MJD of observation start)
MJDSTOP      = <float>          (MJD of observation stop)
ISOSTART     = '<ISO-8601>'     (UTC start time)
ISOSTOP      = '<ISO-8601>'     (UTC stop time)
SUNRA        = <float>          (Sun RA, degrees)
SUNDEC       = <float>          (Sun Dec, degrees)
SUNYAW       = <float>          (Sun yaw, degrees)
SUNROLL      = <float>          (Sun roll, degrees)
SUNPITCH     = <float>          (Sun pitch, degrees)
BORESRA      = <float>          (Boresight RA, degrees)
BORESDEC     = <float>          (Boresight Dec, degrees)
L1VER        = '0.1'            (L1 pipeline version)
L1REL        = '20230815'       (L1 pipeline release date)
CHECKSUM     = '<str>'
DATASUM      = '<str>'
HISTORY      = Creation Time: <ISO-8601>
```

**Sample values (2024-06-15 lightcurve_czt1.fits):**
```
MJDSTART = 60476.659194617
MJDSTOP  = 60476.99996493
ISOSTART = '2024-06-15T15:49:14.415'
ISOSTOP  = '2024-06-15T23:59:56.970'
SUNRA    = 84.545
SUNDEC   = 23.373
SUNYAW   = 0.176
SUNROLL  = 89.944
SUNPITCH = 89.833
BORESRA  = 84.729
BORESDEC = 23.324
L1VER    = '0.1'
L1REL    = '20230815'
```

### 4.2 Lightcurve Band Header (BinTableHDU, per energy band)

```
XTENSION   = 'BINTABLE'
BITPIX     = 8
NAXIS      = 2
NAXIS1     = 54                (row byte width)
NAXIS2     = <int>             (number of time bins)
PCOUNT     = 0
GCOUNT     = 1
TFIELDS    = 4
EXTNAME    = '<DETNAM>_LC_BAND_<ELOW>KEV_TO_<EHIGH>KEV'
DETNAM     = 'CZT1' | 'CZT2' | 'CdTe1' | 'CdTe2'
TTYPE1     = 'MJD'            TFORM1 = 'D'     TUNIT1 = 'MJD'
TTYPE2     = 'ISOT'           TFORM2 = '30A'   TUNIT2 = 'UT'
TTYPE3     = 'CTR'            TFORM3 = 'D'     TUNIT3 = 'cts/sec'
TTYPE4     = 'STAT_ERR'       TFORM4 = 'D'     TUNIT4 = 'cts/sec'
TSTART     = <float>          (MJD)
TSTOP      = <float>          (MJD)
STARETIM   = <float>          (total exposure seconds)
TIMEBIN    = 1.0              (seconds)
ELOW       = <float>          (keV)
EHIGH      = <float>          (keV)
DATAMIN    = <float>          (minimum count rate)
DATAMAX    = <float>          (maximum count rate)
CHECKSUM   = '<str>'
DATASUM    = '<str>'
HISTORY    = Creation Time: <ISO-8601>
```

**Data record dtype:** `[('MJD', '>f8'), ('ISOT', 'S30'), ('CTR', '>f8'), ('STAT_ERR', '>f8')]`

### 4.3 Spectra Primary Header (shared across all spectra files)

Same keywords as lightcurve primary header (MISSION, INSTRUME, TELESCOP, CREATOR, POC, ENTITY, MJDSTART/STOP, ISOSTART/STOP, SUN*, BORES*, L1VER, L1REL, CHECKSUM, DATASUM, HISTORY).

### 4.4 Spectra BinTableHDU Header (SPECTRUM extension)

```
XTENSION   = 'BINTABLE'
BITPIX     = 8
NAXIS      = 2
NAXIS1     = <int>             (row byte width: 6858 for CZT, 10258 for CdTe)
NAXIS2     = <int>             (number of spectra, ~1400-2200)
PCOUNT     = 0
GCOUNT     = 1
TFIELDS    = 8
EXTNAME    = 'SPECTRUM'
DETNAM     = 'CZT1' | 'CZT2' | 'CdTe1' | 'CdTe2'
CHANTYPE   = 'PHA'
TSTART     = <float>           (MJD)
TSTOP      = <float>           (MJD)
TTYPE1     = 'SPEC_NUM'        TFORM1 = 'I'
TTYPE2     = 'CHANNEL'         TFORM2 = '341J' (CZT) or '511J' (CdTe)   TDIM2 = '(341)' or '(511)'
TTYPE3     = 'COUNTS'          TFORM3 = '341D' (CZT) or '511D' (CdTe)   TDIM3 = '(341)' or '(511)'
                                TUNIT3 = 'cts'
TTYPE4     = 'STAT_ERR'        TFORM4 = '341D' or '511D'                  TDIM4 = '(341)' or '(511)'
TTYPE5     = 'ROWID'           TFORM5 = '12A'
TTYPE6     = 'TSTART'          TFORM6 = 'D'     TUNIT6 = 's'
TTYPE7     = 'TSTOP'           TFORM7 = 'D'     TUNIT7 = 's'
TTYPE8     = 'EXPOSURE'        TFORM8 = 'D'     TUNIT8 = 's'
RESPFILE   = 'none'
ANCRFILE   = 'none'
HDUCLASS   = 'OGIP'
HDUCLAS1   = 'SPECTRUM'
HDUCLAS2   = 'TOTAL'
HDUCLAS3   = 'COUNT'
HDUCLAS4   = 'TYPE:II'
TELESCOP   = 'Aditya-L1'
INSTRUME   = 'HEL1OS'
BACKFILE   = 'none'
CORRFILE   = 'none'
CORRSCAL   = 0.0
AREASCAL   = <float>           (0.92578125 for CZT, 1.0 for CdTe)
BACKSCAL   = 1.0
HDUVERS    = '1.2.1'
HDUVERS1   = '1.2.1'
DETCHANS   = 341 (CZT) | 511 (CdTe)
TLMIN2     = 0
TLMAX2     = 340 (CZT) | 510 (CdTe)
DATE_OBS   = '<YYYY-MM-DD>'
TIME_OBS   = '<HH:MM:SS.mmm>'
DATE_END   = '<YYYY-MM-DD>'
TIME_END   = '<HH:MM:SS.mmm>'
CHECKSUM   = '<str>'
DATASUM    = '<str>'
HISTORY    = Creation Time: <ISO-8601>
```

**Spectra record dtype:** `[('SPEC_NUM', '>i2'), ('CHANNEL', '>i4', (N,)), ('COUNTS', '>f8', (N,)), ('STAT_ERR', '>f8', (N,)), ('ROWID', 'S12'), ('TSTART', '>f8'), ('TSTOP', '>f8'), ('EXPOSURE', '>f8')]`

---

## 5. Data Integrity

**All 7,216 top-level FITS files passed validation** (astropy.io.fits open + HDU count check).

No corrupted files detected in the entire processed dataset.

---

## 6. Date Gaps

**10 missing days** out of 937 calendar days:

| Missing Date | Notes |
|-------------|-------|
| 2023-12-24 | Single day gap |
| 2024-06-30 | Single day gap |
| 2024-09-09 | Single day gap |
| 2024-10-12 | Single day gap |
| 2025-04-16 | Single day gap |
| 2025-10-16 | Single day gap |
| 2026-02-17 | Single day gap |
| 2026-04-16 | Single day gap |
| 2026-05-22 | Single day gap |
| 2026-06-04 | Single day gap |

All gaps are single isolated days — no multi-day outages. Likely due to data downlink/scheduling or temporary instrument downtime.

**Note:** There are additionally **25 empty day directories** (directory exists but contains no files). These span 2024-05-23 to 2026-01-19 and are likely extraction failures or days where data was not available at processing time. These are distinct from the 10 "missing" days (no directory at all).

---

## 7. File Sizes

| File Type | Min (MB) | Median (MB) | Max (MB) | Mean (MB) |
|-----------|----------|-------------|----------|-----------|
| lightcurve_czt1.fits | 0.195 | 11.140 | 15.000 | 9.572 |
| lightcurve_czt2.fits | 0.195 | 11.140 | 15.000 | 9.566 |
| lightcurve_cdte1.fits | 0.184 | 11.115 | 14.972 | 9.561 |
| lightcurve_cdte2.fits | 0.179 | 11.129 | 14.988 | 9.570 |
| hel1os_czt_spectra_czt1.fits | 0.211 | 14.123 | 21.415 | 12.299 |
| hel1os_czt_spectra_czt2.fits | 0.211 | 14.117 | 19.028 | 12.120 |
| hel1os_cdte_spectra_cdte1.fits | 0.313 | 21.110 | 28.438 | 18.147 |
| hel1os_cdte_spectra_cdte2.fits | 0.313 | 21.110 | 28.457 | 18.149 |
| czt1dispix.txt | 0.000 | 0.000 | 0.000 | 0.000 |
| czt2dispix.txt | 0.000 | 0.000 | 0.000 | 0.000 |

**Total processed data volume:** 87.19 GB  
**Raw zip files total:** ~110 GB (2,537 zips)

**Size variation** reflects varying observation durations per orbit (Earth occultation gaps, instrumental dead time).

---

## 8. Multiple Orbits Per Day

Each day directory contains a **single flat set of 10 files**. Multiple HEL1OS orbits per day (typically 2, ~12 hr each) are **merged into one continuous dataset** during extraction. This is evidenced by:

- Observation start times varying: some start at 00:00 UT, others at ~12:00 UT, others mid-day (e.g., `2023-12-05` starts at `16:19:00`)
- `STARETIM` (total exposure) varying from ~27,600s (~7.7 hr) to ~43,200s (~12 hr)
- `NAXIS2` (row count) varying from ~27,600 to ~43,200 rows per band

The processing pipeline merges all orbits from a given UTC day into single light curve and spectra files.

---

## 9. Naming Conventions

### Standard filenames (10 per day, 902 days):
```
lightcurve_czt1.fits
lightcurve_czt2.fits
lightcurve_cdte1.fits
lightcurve_cdte2.fits
hel1os_czt_spectra_czt1.fits
hel1os_czt_spectra_czt2.fits
hel1os_cdte_spectra_cdte1.fits
hel1os_cdte_spectra_cdte2.fits
czt1dispix.txt
czt2dispix.txt
```

### Raw zip naming convention:
```
HLS_YYYYMMDD_HHMMSS_XXXXXsec_lev1_V111.zip
│          │       │            │
│          │       │            └── Pipeline version (V111)
│          │       └── Orbit duration in seconds
│          └── Start time (HHMMSS UT)
└── Date
```

### Nested anomalous directory pattern (7 days):
```
YYYY/MM/DD/YYYY/MM/DD/HLS_.../czt/  (lightcurves + spectra)
                                      /cdte/ (lightcurves + spectra)
                                      /aux/cztdis/ (dispix files)
```

---

## 10. Raw vs Processed

| Metric | Count |
|--------|-------|
| Raw zip files in `data/raw/hel1os/` | 2,537 |
| Processed day directories | 927 |
| Days with actual data | 902 |
| Empty day directories | 25 |
| Ratio (raw zips / processed days) | **2.74** |

The 2.74:1 ratio confirms multiple orbits per day are merged during extraction. Additionally, 2 calibration zips exist:
- `CAL_epoch20231001_CZTResponseReader.zip` (423 KB)
- `CAL_epoch20231001_CdTeResponseReader.zip` (382 KB)

These are not observation data but calibration reference files.

**25 raw zips could not be extracted** (or produced empty directories), corresponding to the 25 empty day dirs.

---

## 11. Energy Bands (from FITS headers)

### CZT Detectors (Hard X-rays: 18–160 keV)

| Detector | Band | EXTNAME | ELOW (keV) | EHIGH (keV) |
|----------|------|---------|------------|-------------|
| CZT1 | 1 | `CZT1_LC_BAND_20.00KEV_TO_40.00KEV` | 20.0 | 40.0 |
| CZT1 | 2 | `CZT1_LC_BAND_40.00KEV_TO_60.00KEV` | 40.0 | 60.0 |
| CZT1 | 3 | `CZT1_LC_BAND_60.00KEV_TO_80.00KEV` | 60.0 | 80.0 |
| CZT1 | 4 | `CZT1_LC_BAND_80.00KEV_TO_150.00KEV` | 80.0 | 150.0 |
| CZT1 | Full | `CZT1_LC_BAND_18.00KEV_TO_160.00KEV` | 18.0 | 160.0 |
| CZT2 | 1 | `CZT2_LC_BAND_20.00KEV_TO_40.00KEV` | 20.0 | 40.0 |
| CZT2 | 2 | `CZT2_LC_BAND_40.00KEV_TO_60.00KEV` | 40.0 | 60.0 |
| CZT2 | 3 | `CZT2_LC_BAND_60.00KEV_TO_80.00KEV` | 60.0 | 80.0 |
| CZT2 | 4 | `CZT2_LC_BAND_80.00KEV_TO_150.00KEV` | 80.0 | 150.0 |
| CZT2 | Full | `CZT2_LC_BAND_18.00KEV_TO_160.00KEV` | 18.0 | 160.0 |

### CdTe Detectors (Soft+Hard X-rays: 1.8–90 keV)

| Detector | Band | EXTNAME | ELOW (keV) | EHIGH (keV) |
|----------|------|---------|------------|-------------|
| CdTe1 | 1 | `CDTE1_LC_BAND_5.00KEV_TO_20.00KEV` | 5.0 | 20.0 |
| CdTe1 | 2 | `CDTE1_LC_BAND_20.00KEV_TO_30.00KEV` | 20.0 | 30.0 |
| CdTe1 | 3 | `CDTE1_LC_BAND_30.00KEV_TO_40.00KEV` | 30.0 | 40.0 |
| CdTe1 | 4 | `CDTE1_LC_BAND_40.00KEV_TO_60.00KEV` | 40.0 | 60.0 |
| CdTe1 | Full | `CDTE1_LC_BAND_1.80KEV_TO_90.00KEV` | 1.8 | 90.0 |
| CdTe2 | 1 | `CDTE2_LC_BAND_5.00KEV_TO_20.00KEV` | 5.0 | 20.0 |
| CdTe2 | 2 | `CDTE2_LC_BAND_20.00KEV_TO_30.00KEV` | 20.0 | 30.0 |
| CdTe2 | 3 | `CDTE2_LC_BAND_30.00KEV_TO_40.00KEV` | 30.0 | 40.0 |
| CdTe2 | 4 | `CDTE2_LC_BAND_40.00KEV_TO_60.00KEV` | 40.0 | 60.0 |
| CdTe2 | Full | `CDTE2_LC_BAND_1.80KEV_TO_90.00KEV` | 1.8 | 90.0 |

### Spectra Channels

| Detector | Channels | Channel Range | Energy Range |
|----------|----------|---------------|--------------|
| CZT1 | 341 | 0–340 | ~20–160 keV |
| CZT2 | 341 | 0–340 | ~20–160 keV |
| CdTe1 | 511 | 0–510 | ~1.8–90 keV |
| CdTe2 | 511 | 0–510 | ~1.8–90 keV |

### Overlapping Energy Coverage

The CZT and CdTe detectors overlap in the **20–60 keV** range, providing cross-calibration opportunity. The CdTe detectors extend down to **1.8 keV** (soft X-rays), while CZT extends up to **160 keV** (hard X-rays). Combined coverage: **1.8–160 keV**.

---

## 12. Anomalies

### 12.1 Empty Day Directories (25 days)

These directories exist but contain **zero files**:

| Date Range | Count |
|-----------|-------|
| 2024-05-23 to 2024-05-24 | 2 days |
| 2024-10-09 | 1 day |
| 2025-11-11 to 2025-11-12 | 2 days |
| 2025-12-06 to 2025-12-30 (sparse) | 16 days |
| 2026-01-01 to 2026-01-02 | 2 days |
| 2026-01-19 | 1 day |
| **Total** | **25 days** |

The 2025-12 cluster (16 days) suggests a sustained processing issue or data unavailability during that period.

### 12.2 Nested Extraction Artifacts (7 days)

These days have a correct top-level file set **plus** a nested directory tree duplicating the data:

| Date | Nested Path |
|------|------------|
| 2024-03-21 | `2024/03/21/HLS_.../czt/`, `cdte/`, `aux/cztdis/` |
| 2024-07-26 | Same pattern |
| 2024-08-29 | Same pattern |
| 2024-10-27 | Same pattern |
| 2025-01-30 | Same pattern |
| 2025-07-29 | Same pattern |
| 2025-08-14 | Same pattern |

These are **extraction artifacts** where the zip contained the full directory hierarchy (`HLS_.../czt/`, `HLS_.../cdte/`, etc.) and the extraction script did not strip the inner path. The top-level files are the correct ones to use.

### 12.3 Calibration Files in Raw Directory

Two non-observation zips exist in the raw directory:
- `CAL_epoch20231001_CZTResponseReader.zip` (423 KB)
- `CAL_epoch20231001_CdTeResponseReader.zip` (382 KB)

These are detector response calibration data, not science observations.

### 12.4 Empty dispix Files

`czt2dispix.txt` is **0 bytes** (empty) in many days. `czt1dispix.txt` contains 4 integers (e.g., `0\n15\n111\n240`). These are detector pixel mask configuration files and are not needed for flare analysis.

### 12.5 Data Record Characteristics

- **Light curve cadence:** 1 second (TIMEBIN=1.0)
- **Light curve columns:** MJD (float64), ISOT (string UTC), CTR (float64, cts/sec), STAT_ERR (float64, cts/sec)
- **Spectra accumulation:** 20 seconds per spectrum
- **Spectra columns:** SPEC_NUM, CHANNEL (array), COUNTS (array), STAT_ERR (array), ROWID, TSTART, TSTOP, EXPOSURE
- **STAT_ERR formula:** Appears to be Poisson sqrt(N) — for count rate, `STAT_ERR = sqrt(CTR)`

### 12.6 Observation Timing Variation

Observation start times are not uniform across days:

| Pattern | Example | Duration | Notes |
|---------|---------|----------|-------|
| Full day (00:00–24:00) | 2024-12-15 | ~43,181s | Two merged orbits |
| Partial (15:49–24:00) | 2024-06-15 | ~29,433s | Late start, possible S/C event |
| Partial (16:19–24:00) | 2023-12-05 | ~27,612s | Early mission, shorter orbit |
| Full day (12:00–24:00) | 2026-01-15 | ~43,172s | Half-day start |

The variation in row counts (27,600–43,200) directly reflects the observation duration within each UTC day.

---

## Summary Table

| Property | Value |
|----------|-------|
| Date range | 2023-11-30 to 2026-06-23 |
| Days with data | 902 / 937 calendar days |
| Total FITS files | 7,216 |
| File integrity | 100% (7,216/7,216 passed) |
| Files per day | 10 (8 FITS + 2 TXT) |
| Total data volume | 87.19 GB |
| Light curve cadence | 1 second |
| Spectra accumulation | 20 seconds |
| CZT energy range | 18–160 keV (5 bands) |
| CdTe energy range | 1.8–90 keV (5 bands) |
| Combined energy range | 1.8–160 keV |
| Missing days | 10 (isolated single days) |
| Empty day dirs | 25 (extraction failures) |
| Anomalous nested dirs | 7 (extraction artifacts) |
| Raw zips | 2,537 (~110 GB) |
| Raw:Processed ratio | 2.74:1 (multiple orbits merged) |
