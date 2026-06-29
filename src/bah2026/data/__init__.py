"""Data I/O, calibration, corrections, and preprocessing for Aditya-L1 SoLEXS + HEL1OS.

Readers
-------
load_solexs_lc       — SoLEXS SDD2 light curve (1s cadence, 86400 rows)
load_solexs_pi       — SoLEXS SDD2 PI spectrum (86400 × 340 channels)
load_solexs_gti      — SoLEXS SDD2 good time intervals
load_hel1os_lc       — HEL1OS light curve (CZT/CdTe, 5 energy bands)
load_hel1os_spectra  — HEL1OS energy spectra (CZT/CdTe, 341/511 channels)
load_hel1os_hk       — HEL1OS housekeeping (62 columns: temps, HV, pile-up)
load_hel1os_gti      — HEL1OS GTI per detector
load_hel1os_all_gti  — HEL1OS GTI for all 4 detectors

Calibration
-----------
solexs_counts_to_irradiance_simple — Quick-look SoLEXS → GOES W/m²
classify_goes                      — GOES class from irradiance (A/B/C/M/X)
load_channel_energies              — SoLEXS CALDB channel → energy mapping
load_arf                           — Ancillary Response File (effective area)
load_rmf                           — Redistribution Matrix File
calibrate_day                      — Full response-based calibration

Corrections
-----------
correct_solexs_deadtime     — Paralyzable deadtime (τ=13.65µs)
subtract_hel1os_background  — CZT/CdTe background subtraction
subtract_solexs_spurious    — Reset-pulse spurious count removal
correct_hel1os_deadtime_approx — Approximate HEL1OS deadtime
apply_all_corrections       — Combined correction wrapper

Preprocessing
-------------
compute_gti_mask            — Boolean mask from GTI intervals
background_subtract         — Sliding median background
background_subtract_iterative — Outlier-rejecting background
forward_fill_nan            — Forward-fill NaN (for saturated flares)
interpolate_to_common_grid  — Linear interpolation to target MJD grid
met_to_mjd                  — Mission Elapsed Time → MJD
align_hel1os_to_solexs      — Interpolate HEL1OS onto SoLEXS grid

Discovery
---------
discover_solexs_days   — All dates with SoLEXS data
discover_hel1os_days   — All dates with HEL1OS data
discover_combined_days — Dates with BOTH instruments

Ground Truth
------------
load_swpc_flares    — SWPC flare catalogue
validate_nowcasting — Compare nowcast vs ground truth
parse_goes_class    — Parse GOES class string → numeric
"""

from bah2026.data.reader import (
    load_solexs_lc,
    load_solexs_pi,
    load_solexs_gti,
    load_hel1os_lc,
    load_hel1os_spectra,
    load_hel1os_hk,
    load_hel1os_gti,
    load_hel1os_all_gti,
    discover_solexs_days,
    discover_hel1os_days,
    discover_combined_days,
    is_anomaly_day,
)
from bah2026.data.preprocessing import (
    background_subtract,
    interpolate_to_common_grid,
    compute_gti_mask,
    met_to_mjd,
    align_hel1os_to_solexs,
    background_subtract_iterative,
    forward_fill_nan,
)
from bah2026.data.hdf5_builder import build_hdf5
from bah2026.data.calibration import (
    solexs_counts_to_irradiance_simple,
    classify_goes,
    load_channel_energies,
    load_arf,
    load_rmf,
    calibrate_day,
)
from bah2026.data.corrections import (
    correct_solexs_deadtime,
    subtract_hel1os_background,
    subtract_solexs_spurious,
    correct_hel1os_deadtime_approx,
    apply_all_corrections,
)
from bah2026.data.ground_truth import (
    load_swpc_flares,
    validate_nowcasting,
    parse_goes_class,
)

__all__ = [
    # Readers
    "load_solexs_lc",
    "load_solexs_pi",
    "load_solexs_gti",
    "load_hel1os_lc",
    "load_hel1os_spectra",
    "load_hel1os_hk",
    "load_hel1os_gti",
    "load_hel1os_all_gti",
    "discover_solexs_days",
    "discover_hel1os_days",
    "discover_combined_days",
    "is_anomaly_day",
    # Preprocessing
    "background_subtract",
    "interpolate_to_common_grid",
    "compute_gti_mask",
    "met_to_mjd",
    "align_hel1os_to_solexs",
    "background_subtract_iterative",
    "forward_fill_nan",
    # HDF5
    "build_hdf5",
    # Calibration
    "solexs_counts_to_irradiance_simple",
    "classify_goes",
    "load_channel_energies",
    "load_arf",
    "load_rmf",
    "calibrate_day",
    # Corrections
    "correct_solexs_deadtime",
    "subtract_hel1os_background",
    "subtract_solexs_spurious",
    "correct_hel1os_deadtime_approx",
    "apply_all_corrections",
    # Ground truth
    "load_swpc_flares",
    "validate_nowcasting",
    "parse_goes_class",
]
