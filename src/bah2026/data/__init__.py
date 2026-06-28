"""Data I/O, preprocessing, and HDF5 database for Aditya-L1 SoLEXS + HEL1OS."""

from bah2026.data.reader import (
    load_solexs_lc,
    load_solexs_pi,
    load_solexs_gti,
    load_hel1os_lc,
    load_hel1os_spectra,
    discover_solexs_days,
    discover_hel1os_days,
    discover_combined_days,
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
)
from bah2026.data.ground_truth import (
    load_swpc_flares,
    validate_nowcasting,
    parse_goes_class,
)

__all__ = [
    "load_solexs_lc",
    "load_solexs_pi",
    "load_solexs_gti",
    "load_hel1os_lc",
    "load_hel1os_spectra",
    "discover_solexs_days",
    "discover_hel1os_days",
    "discover_combined_days",
    "background_subtract",
    "interpolate_to_common_grid",
    "compute_gti_mask",
    "met_to_mjd",
    "align_hel1os_to_solexs",
    "background_subtract_iterative",
    "build_hdf5",
    "solexs_counts_to_irradiance_simple",
    "classify_goes",
    "load_channel_energies",
    "load_swpc_flares",
    "validate_nowcasting",
    "parse_goes_class",
]
