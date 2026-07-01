"""Physical interpretation pipeline for solar flare analysis.

Produces structured interpretations (JSON) from feature values:
  1. Per-group feature interpretation (all 179 features in 19 groups)
  2. Physical phenomenon analysis (Neupert, QPP, spectral, causal)
  3. Data quality & processing summary
"""

from __future__ import annotations

import json
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr
from scipy.signal import find_peaks

from bah2026.config import GOES_DATA_DIR

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ═════════════════════════════════════════════════════════════════════════════
# FEATURE-GROUP INTERPRETATION KNOWLEDGE BASE
# ═════════════════════════════════════════════════════════════════════════════
# Each feature group: name, canonical keys, physical meaning, expected ranges,
# flare vs quiet behavior, physical process.
# ═════════════════════════════════════════════════════════════════════════════

FEATURE_GROUPS = [
    {
        "group": "SXR_basic_statistics",
        "n_features": 15,
        "features": [
            "sxr_mean",
            "sxr_std",
            "sxr_max",
            "sxr_min",
            "sxr_median",
            "sxr_skew",
            "sxr_kurtosis",
            "sxr_range",
            "sxr_cv",
            "sxr_rise_rate",
            "sxr_fall_rate",
            "sxr_abs_slope",
            "sxr_iqr",
            "sxr_spec_entropy",
            "sxr_peak_freq",
        ],
        "physical_process": "Thermal plasma emission from hot (2-22 keV) coronal plasma. Count rate tracks the total emission measure EM = integral n_e^2 dV.",
        "flare_behavior": "Mean, max, std, range INCREASE by 10-100x during flares. Skew becomes positive (bursty rise, gradual decay). Kurtosis increases (heavy tails from impulsive peaks). Rise_rate >> fall_rate (fast rise, slow decay).",
        "quiet_behavior": "Low mean (50-100 cps), low std, near-symmetric distribution (skew~0). Quasi-periodic due to p-mode oscillations.",
        "interpreting_values": {
            "sxr_mean": "Mean SXR count rate. Proxy for background + gradual flare emission. 50-200 cps = quiet, >1000 cps = flaring.",
            "sxr_std": "Intra-window variability. High during flares (impulsive spikes). Low during quiet periods.",
            "sxr_max": "Peak count rate in window. <200 cps = quiet; 200-1000 = C-class; 1000-10000 = M-class; >10000 = X-class.",
            "sxr_skew": "Positive skew = fast rise / slow decay (flare signature). Near-zero = symmetric noise (quiet).",
            "sxr_kurtosis": "High kurtosis = impulsive spikes (flare). Low kurtosis = Gaussian noise (quiet).",
            "sxr_rise_rate": "Mean positive gradient. High during flare onset (impulsive phase).",
            "sxr_fall_rate": "Mean negative gradient (absolute). High during decay phase. Usually |rise| > |fall|.",
            "sxr_cv": "Coefficient of variation = std/mean. <0.5 = steady emission; >1.0 = highly variable (flaring).",
            "sxr_spec_entropy": "Spectral flatness of the Fourier power spectrum. Low = narrowband (quiet Sun oscillations). High = broadband (flare turbulence).",
            "sxr_peak_freq": "Frequency of peak power in Fourier spectrum. <0.01 Hz = flare envelope; 0.001-0.005 Hz = p-mode oscillations.",
        },
    },
    {
        "group": "HXR_band_statistics",
        "n_features": 30,
        "features": [
            "hxr_b0_max",
            "hxr_b0_mean",
            "hxr_b0_std",
            "hxr_b1_max",
            "hxr_b1_mean",
            "hxr_b1_std",
            "hxr_b2_max",
            "hxr_b2_mean",
            "hxr_b2_std",
            "hxr_b3_max",
            "hxr_b3_mean",
            "hxr_b3_std",
            "hxr_b4_max",
            "hxr_b4_mean",
            "hxr_b4_std",
            "hxr_b5_max",
            "hxr_b5_mean",
            "hxr_b5_std",
            "hxr_b6_max",
            "hxr_b6_mean",
            "hxr_b6_std",
            "hxr_b7_max",
            "hxr_b7_mean",
            "hxr_b7_std",
            "hxr_b8_max",
            "hxr_b8_mean",
            "hxr_b8_std",
            "hxr_b9_max",
            "hxr_b9_mean",
            "hxr_b9_std",
        ],
        "physical_process": "Non-thermal bremsstrahlung from accelerated electrons. Bands: b0-b4 = CZT (20-40, 40-60, 60-80, 80-150, 18-160 keV), b5-b9 = CdTe (5-20, 20-30, 30-40, 40-60, 1.8-90 keV).",
        "flare_behavior": "All bands increase during flares. Higher bands (b2-b4, b7-b9) show larger relative increase = spectral hardening. b0 (20-40 keV) usually brightest.",
        "quiet_behavior": "Near-zero for all bands (background-subtracted). Some low-level counts from cosmic rays and instrumental background.",
        "interpreting_values": {
            "hxr_b4_mean": "CZT full band (18-160 keV) mean. <5 cps = quiet; 5-50 cps = C-class; 50-500 cps = M-class; >500 cps = X-class.",
            "hxr_b0_to_b3_ratio": "Hardness proxy: higher ratio of high/lo bands = harder spectrum.",
        },
    },
    {
        "group": "derivative_features",
        "n_features": 12,
        "features": [
            "dsxr_dt_mean",
            "dsxr_dt_std",
            "dsxr_dt_max",
            "dsxr_dt_min",
            "d2sxr_dt2_mean",
            "d2sxr_dt2_std",
            "dhxr_dt_mean",
            "dhxr_dt_std",
            "dhxr_dt_max",
            "dhr_dt_mean",
            "dhr_dt_max",
            "dsxr_dhxr_ratio_mean",
        ],
        "physical_process": "Time derivatives: dSXR/dt tracks energy injection rate (Neupert effect). dHXR/dt tracks acceleration variability. dHR/dt tracks spectral evolution rate.",
        "flare_behavior": "dsxr_dt_max >> 0 at flare onset (impulsive rise). Negative during decay. d2sxr_dt2 positive during acceleration, negative during deceleration. dsxr_dhxr_ratio_mean ~ 1 when Neupert holds.",
        "quiet_behavior": "All derivatives near-zero. Small fluctuations from photon counting noise.",
        "interpreting_values": {
            "dsxr_dt_max": "Peak rate of change of SXR. High values = explosive energy release. >100 cps/s = major flare.",
            "dsxr_dhxr_ratio_mean": "Neupert efficiency: ratio of mean dSXR/dt to mean HXR. ~1 when Neupert holds perfectly.",
        },
    },
    {
        "group": "multiscale_temporal",
        "n_features": 24,
        "features": [
            "sxr_mean_5m",
            "sxr_std_5m",
            "sxr_max_5m",
            "hxr_mean_5m",
            "hxr_std_5m",
            "hxr_max_5m",
            "sxr_mean_15m",
            "sxr_std_15m",
            "sxr_max_15m",
            "hxr_mean_15m",
            "hxr_std_15m",
            "hxr_max_15m",
            "sxr_mean_30m",
            "sxr_std_30m",
            "sxr_max_30m",
            "hxr_mean_30m",
            "hxr_std_30m",
            "hxr_max_30m",
            "sxr_5m_to_60m_ratio",
            "hxr_5m_to_60m_ratio",
            "sxr_acceleration_trend",
            "hxr_acceleration_trend",
            "sxr_15m_slope",
            "hxr_15m_slope",
        ],
        "physical_process": "Multi-timescale statistics capture flare evolution. 5min = impulsive phase; 15min = gradual phase; 30min = decay. Ratios track acceleration.",
        "flare_behavior": "5m >> 60m during flare onset (fast rise). 15m/30m capture gradual phase. sxr_5m_to_60m_ratio > 2 = flaring. sxr_acceleration_trend > 0 = still rising.",
        "quiet_behavior": "All scales similar. Ratios ~1. Slopes near-zero.",
    },
    {
        "group": "autocorrelation",
        "n_features": 4,
        "features": ["sxr_acf_5s", "sxr_acf_10s", "sxr_acf_30s", "sxr_acf_60s"],
        "physical_process": "Autocorrelation measures memory/persistence of SXR emission. High ACF at short lags = smooth, long-duration events. Low ACF = bursty/noisy.",
        "flare_behavior": "High at all lags during flares (sustained emission). ACF decays slowly (long correlation time ~minutes).",
        "quiet_behavior": "Moderate at short lags (p-mode oscillations ~5min). Decays faster.",
    },
    {
        "group": "neupert_correlation",
        "n_features": 2,
        "features": ["neupert_rho_mean", "neupert_rho_std"],
        "physical_process": "The Neupert effect: d(thermal energy)/dt = HXR flux. Sliding Pearson r between dSXR/dt and HXR over 300s windows. High r = Neupert holds.",
        "flare_behavior": "rho_mean > 0.3 during flares = strong Neupert. rho_std increases when Neupert varies (e.g., multiple energy release episodes).",
        "quiet_behavior": "rho_mean ~ 0 (no correlation). rho_std ~ 0 (uniformly uncorrelated).",
    },
    {
        "group": "hardness_evolution",
        "n_features": 3,
        "features": [
            "hardness_ratio_slope",
            "hardness_ratio_mean",
            "hardness_ratio_std",
        ],
        "physical_process": "Hardness ratio (HR = HXR_high / HXR_low) tracks electron spectral index. Harder = more high-energy electrons. SHS pattern: soft->hard->soft.",
        "flare_behavior": "HR_mean increases (hardening). HR_slope > 0 = hardening (impulsive phase), < 0 = softening (decay). HR_std captures variability.",
        "quiet_behavior": "HR_mean ~ constant. HR_slope ~ 0. HR_std low.",
    },
    {
        "group": "temperature_emission",
        "n_features": 3,
        "features": ["sxr_temperature_mk", "sxr_emission_measure", "sxr_chi2_red"],
        "physical_process": "From thermal bremsstrahlung fit to SoLEXS PI spectrum. T = isothermal plasma temperature (MK). EM = emission measure = n_e^2 V. chi2_red = fit quality.",
        "flare_behavior": "T rises from ~2 MK (quiet) to 10-30 MK (flare). EM increases by 1-3 orders of magnitude (plasma fills loops). chi2_red high during flares (multi-thermal, not isothermal).",
        "quiet_behavior": "T ~ 1.5-3 MK (coronal background). EM ~ 1e47-1e49. chi2_red ~ 1 (good isothermal fit).",
    },
    {
        "group": "spectral_indices",
        "n_features": 4,
        "features": [
            "hxr_spectral_index_gamma",
            "hxr_gamma_czt2",
            "hxr_gamma_cdte1",
            "hxr_gamma_cdte2",
        ],
        "physical_process": "Photon spectral index gamma from power-law fit: I(E) ~ E^(-gamma). gamma = 2-7 for non-thermal emission. Electron index delta = gamma + 1 (thick target).",
        "flare_behavior": "gamma decreases (hardens) during impulsive phase. gamma ~ 2-4 for X-class flares. gamma ~ 4-7 for M/C class. Gamma varies between detectors (different energy ranges).",
        "quiet_behavior": "gamma undefined or very steep (>7) due to low counts.",
    },
    {
        "group": "nonthermal_parameters",
        "n_features": 4,
        "features": [
            "nonthermal_gamma",
            "nonthermal_ec",
            "nonthermal_n_nth",
            "thermal_fraction",
        ],
        "physical_process": "From combined SoLEXS+HEL1OS spectral fit. gamma = non-thermal index. Ec = low-energy cutoff (keV). n_nth = total non-thermal electrons above Ec. thermal_fraction = fraction of thermal emission.",
        "flare_behavior": "Thermal_fraction < 1 during flares (non-thermal component present). gamma = 3-6, Ec = 10-30 keV. n_nth increases with flare strength.",
        "quiet_behavior": "thermal_fraction ~ 1 (no non-thermal). Other parameters ~ 0 (fit fails, no non-thermal signal).",
    },
    {
        "group": "goes_flux",
        "n_features": 3,
        "features": ["goes_xrsb_flux", "goes_xrsa_flux", "goes_xrsa_xrsb_ratio"],
        "physical_process": "GOES XRS: 1-minute averaged full-Sun flux. XRS-B (1-8A) = hot plasma, XRS-A (0.5-4A) = very hot plasma. Ratio = temperature proxy.",
        "flare_behavior": "XRS-B > 1e-6 = C-class, > 1e-5 = M-class, > 1e-4 = X-class. XRS-A/XRS-B ratio increases with temperature (higher during flares).",
        "quiet_behavior": "XRS-B ~ 2-5e-8. XRS-A ~ 2-5e-9. Ratio ~ 0.1.",
    },
    {
        "group": "information_theory",
        "n_features": 6,
        "features": [
            "transfer_entropy_hxr_to_sxr",
            "mutual_information_sxr_hxr",
            "sample_entropy_sxr",
            "sample_entropy_hxr",
            "lagged_cross_corr",
            "lagged_cross_corr_lag",
        ],
        "physical_process": "Information flow between SXR and HXR. Transfer entropy = directed info flow (HXR->SXR). Mutual info = total shared info. Sample entropy = complexity/predictability.",
        "flare_behavior": "Transfer entropy increases (HXR drives SXR = Neupert). Mutual info increases (strong coupling). Sample entropy of HXR > SXR (HXR more complex/stochastic). Lagged cross-corr peaks at negative lag (HXR leads).",
        "quiet_behavior": "Low transfer entropy (no directed flow). Low mutual info. Sample entropy reflects counting noise.",
    },
    {
        "group": "qpp_detection",
        "n_features": 4,
        "features": ["qpp_detected", "qpp_period", "qpp_amplitude", "qpp_significance"],
        "physical_process": "Quasi-Periodic Pulsations: oscillatory signals during flares. Periods 10-300s. Origins: MHD oscillations, oscillatory reconnection, sausage/kink modes.",
        "flare_behavior": "Detected ~50% of X-class flares. Period typically 30-120s. Amplitude 5-30% modulation. Significance > 0.3 = candidate.",
        "quiet_behavior": "Not detected. Significance near-zero.",
    },
    {
        "group": "housekeeping",
        "n_features": 8,
        "features": [
            "hk_czt1temp",
            "hk_czt2temp",
            "hk_cdte1temp",
            "hk_cdte2temp",
            "hk_czthvmon",
            "hk_cdtehvmon",
            "hk_czt1satctr",
            "hk_cdte1pilectr",
        ],
        "physical_process": "Detector housekeeping: temperatures, bias voltages, saturation/pile-up counters. Monitor instrument health and data quality.",
        "flare_behavior": "Temperatures may rise slightly during flares (heating). Saturation counters increase during very bright flares (deadtime effects).",
        "quiet_behavior": "Stable temperatures (~15-25C for CZT, -30 to -40C for CdTe). HV stable (~600-950V). Low saturation/pile-up.",
    },
    {
        "group": "causal_network",
        "n_features": 13,
        "features": [
            "causal_network_density",
            "avg_in_degree",
            "avg_out_degree",
            "avg_centrality",
            "n_feedback_loops",
            "cycle_detected",
            "hxr_to_sxr_lag",
            "hxr_to_sxr_strength",
            "sxr_to_hxr_lag",
            "sxr_to_hxr_strength",
            "neupert_granger_improvement",
            "neupert_best_lag",
            "max_mediation_proportion",
        ],
        "physical_process": "Causal discovery across energy bands. Granger causality: does HXR improve SXR prediction? Mediation: is the HXR->SXR path mediated by intermediate energies?",
        "flare_behavior": "Density increases (more connections active). HXR->SXR lag ~1-60s (Neupert delay). Strength > 0.5. Feedback loops = energy recirculation. Cycle_detected = True (thermal <-> non-thermal coupling).",
        "quiet_behavior": "Low density, few connections. No cycles. Granger improvement ~ 0%. Mediation ~ 0.",
    },
    {
        "group": "correction_statistics",
        "n_features": 2,
        "features": ["deadtime_max_pct", "bg_fraction_pct"],
        "physical_process": "Instrument corrections: deadtime (paralyzable, tau=13.65us) and HXR background subtraction (CZT=70 cps, CdTe=0.15 cps).",
        "flare_behavior": "Deadtime increases during flares (higher count rate). >10% for X-class flares. Background fraction decreases (signal >> background).",
        "quiet_behavior": "Deadtime ~ 1-3%. Background fraction ~ 30-50% of HXR counts.",
    },
    {
        "group": "advanced_goes_timeseries",
        "n_features": 8,
        "features": [
            "goes_xrsb_ddt_max",
            "goes_xrsb_rolling_std_300s",
            "goes_xrsb_rolling_std_1800s",
            "goes_xrsa_rolling_mean_300s",
            "goes_class_current",
            "goes_xrsb_gradient_1h",
            "goes_flare_history_24h",
            "goes_xrsb_prev_peak_ratio",
        ],
        "physical_process": "GOES time-series derivatives and statistics. ddt_max = fastest flux increase. Rolling std = variability on 5/30min scales. Gradient = long-term trend.",
        "flare_behavior": "ddt_max high during flare onset. Rolling std high during impulsive phase. Class_current = log10(peak_flux). Gradient positive during rise. Prev_peak_ratio < 1 if current flare > previous.",
        "quiet_behavior": "ddt_max ~ 0. Rolling std low. Gradient ~ 0.",
    },
    {
        "group": "per_window_spectral",
        "n_features": 8,
        "features": [
            "sxr_temp_window",
            "sxr_em_window",
            "sxr_gamma_window",
            "hxr_gamma_window_czt1",
            "hxr_gamma_window_cdte1",
            "shs_index",
            "spectral_hardening_rate",
            "nonthermal_fraction_window",
        ],
        "physical_process": "Window-level spectral fits. Same as temperature/spectral groups but computed per 1-hour window. Tracks spectral evolution during flare.",
        "flare_behavior": "SXR temperature rises to 10-30 MK. CZT gamma hardens (decreases). SHS > 0 = hardening. Nonthermal fraction increases.",
        "quiet_behavior": "T ~ 2 MK. Gamma undefined. SHS ~ 0.",
    },
    {
        "group": "wavelet_scalogram",
        "n_features": 10,
        "features": [
            "wavelet_energy_10_30s",
            "wavelet_energy_30_60s",
            "wavelet_energy_60_120s",
            "wavelet_energy_120_300s",
            "wavelet_energy_300_600s",
            "wavelet_peak_period",
            "wavelet_peak_significance",
            "wavelet_spectral_entropy",
            "wavelet_hxr_energy_30_120s",
            "wavelet_cross_power_sxr_hxr",
        ],
        "physical_process": "Morlet wavelet power in period bands. Energy in different bands = power at different timescales. Peak period = dominant oscillation. Cross-power = shared SXR-HXR oscillations.",
        "flare_behavior": "All bands increase (broadband power). Energy shifts to longer periods during gradual phase. Cross-power high if SXR and HXR oscillate together (QPP).",
        "quiet_behavior": "Low energy in all bands. Peak period undefined. Cross-power near-zero.",
    },
]


# ═════════════════════════════════════════════════════════════════════════════
# INTERPRETATION FUNCTIONS
# ═════════════════════════════════════════════════════════════════════════════


def analyze_feature_group(
    group: dict,
    feature_values: dict[str, float],
    is_flare_window: bool = False,
    n_windows: int = 277,
) -> dict:
    """Analyze a feature group and produce an interpretation.

    Parameters
    ----------
    group : dict
        Feature group definition from FEATURE_GROUPS.
    feature_values : dict
        Feature name -> scalar value mapping.
    is_flare_window : bool
        Whether this window is during a flare.
    n_windows : int
        Total windows for context.

    Returns
    -------
    dict with keys: group, n_available, n_total, status, summary, details.
    """
    available = {k: v for k, v in feature_values.items() if k in group["features"]}
    n_avail = len(available)
    n_total = group["n_features"]

    if n_avail == 0:
        return {
            "group": group["group"],
            "n_available": 0,
            "n_total": n_total,
            "status": "no_data",
            "summary": "No features from this group available.",
            "details": [],
        }

    # Count non-zero features
    n_nonzero = sum(1 for v in available.values() if abs(v) > 1e-10)
    frac_nonzero = n_nonzero / max(n_avail, 1)

    # Determine activity level
    if frac_nonzero > 0.8:
        activity = "active"
    elif frac_nonzero > 0.3:
        activity = "partial"
    else:
        activity = "inactive"

    # Build summary
    if is_flare_window:
        summary = group["flare_behavior"]
    else:
        summary = group["quiet_behavior"]

    # Key-value interpretations
    details = []
    for feat_name, value in sorted(available.items()):
        if feat_name in group.get("interpreting_values", {}):
            base = group["interpreting_values"][feat_name]
            details.append(f"{feat_name} = {value:.4f}: {base}")
        else:
            details.append(f"{feat_name} = {float(value):.4f}")

    return {
        "group": group["group"],
        "n_available": n_avail,
        "n_total": n_total,
        "n_nonzero": n_nonzero,
        "activity": activity,
        "physical_process": group["physical_process"],
        "summary": summary,
        "details": details[:5],  # Top 5 most informative
    }


def interpret_feature_vector(
    feature_values: dict[str, float],
    is_flare_window: bool = False,
    n_windows: int = 277,
) -> list[dict]:
    """Interpret all features by group.

    Parameters
    ----------
    feature_values : dict
        Mapping of feature name to scalar value.
        Feature names may have 'gpu_' or 'cpu_' prefixes (from CSV columns).
    is_flare_window : bool
        Whether this window contains a flare.
    n_windows : int
        Total windows.

    Returns
    -------
    list of group interpretations.
    """
    # Normalize: strip gpu_/cpu_ prefix for matching against bare names
    # in FEATURE_GROUPS, but keep original values
    normalized: dict[str, float] = {}
    for k, v in feature_values.items():
        base = k
        for prefix in ["gpu_", "cpu_"]:
            if k.startswith(prefix):
                base = k[len(prefix) :]
                break
        normalized[base] = v

    interpretations = []
    for group in FEATURE_GROUPS:
        # Pass normalized values so feature names match the knowledge base
        interp = analyze_feature_group(group, normalized, is_flare_window, n_windows)
        interpretations.append(interp)
    return interpretations


def _goes_class_label(flux: float) -> str:
    if flux >= 1e-4:
        return f"X{flux / 1e-4:.1f}"
    elif flux >= 1e-5:
        return f"M{flux / 1e-5:.1f}"
    elif flux >= 1e-6:
        return f"C{flux / 1e-6:.1f}"
    return "B"


def _fmt_goes(flux: float) -> str:
    if flux >= 1e-4:
        return f"X{flux / 1e-4:.2f}"
    elif flux >= 1e-5:
        return f"M{flux / 1e-5:.2f}"
    elif flux >= 1e-6:
        return f"C{flux / 1e-6:.2f}"
    return f"B{flux / 1e-7:.2f}"


def _utc_array(time_s, mjdrefi, mjdreff):
    mjd = mjdrefi + mjdreff + time_s / 86400.0
    return np.array([datetime(1858, 11, 17) + timedelta(days=float(m)) for m in mjd])


# ═════════════════════════════════════════════════════════════════════════════
# PHYSICAL PHENOMENON ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════


def build_physical_interpretation(
    target_date,
    counts: np.ndarray,
    hxr_fb: np.ndarray,
    hxr_bands: np.ndarray,
    hxr_cdte1: np.ndarray,
    time_s: np.ndarray,
    sxr_headers: dict,
    gti: np.ndarray,
    goes_xrsb_arr: np.ndarray | None,
    goes_xrsa_arr: np.ndarray | None,
    flares: list,
    qpp_result: dict,
    gc_result: dict,
    feature_values: dict[str, float] | None = None,
    df_columns: int = 0,
    n_gpu_features: int = 0,
    n_cpu_features: int = 0,
    n_nonzero: int = 0,
    runtime_sec: float = 0.0,
) -> dict:
    """Build comprehensive physical interpretation.

    Returns structured dict with:
      - data_quality
      - flare_catalog
      - goes_comparison
      - neupert_effect
      - cross_correlation
      - power_spectrum
      - qpp_analysis
      - spectral_evolution
      - causal_network
      - feature_interpretations (per-group)
      - processing
    """
    from bah2026.data.calibration import solexs_counts_to_irradiance_simple
    from bah2026.features.spectral_fitting import neupert_correlation
    from bah2026.features.information_theory import lagged_cross_correlation
    from bah2026.features.qpp import lomb_scargle_periodogram, wavelet_power_auto

    valid = np.isfinite(counts) & np.isfinite(hxr_fb)
    counts_v = counts[valid]
    hxr_v = hxr_fb[valid]
    hxr_b = (
        hxr_bands[valid] if (hxr_bands.ndim == 2 and hxr_bands.shape[1] >= 2) else None
    )
    utc_v = _utc_array(time_s, sxr_headers["mjdrefi"], sxr_headers["mjdreff"])[valid]

    report: dict = {
        "date": str(target_date),
        "instrument": "Aditya-L1 SoLEXS SDD2 + HEL1OS CZT1/CdTe1",
        "pipeline": "generate_master_csv.py v3",
    }

    # ── 1. Data quality ──────────────────────────────────────────────
    hxr_finite = int(np.sum(np.isfinite(hxr_fb)))
    gti_cov = 0.0
    if gti is not None and len(gti) > 0 and len(time_s) > 1:
        ts = time_s[-1] - time_s[0]
        gt = float(np.sum(gti[:, 1] - gti[:, 0]))
        gti_cov = round(100.0 * gt / max(ts, 1e-6), 1)
    report["data_quality"] = {
        "solexs_coverage_sec": len(counts),
        "hel1os_finite_fraction": round(hxr_finite / max(len(hxr_fb), 1), 3),
        "gti_coverage_pct": gti_cov,
        "corrections": "Deadtime (paralyzable, tau=13.65us) + HXR background subtraction",
        "solexs_to_goes_calibration": "Energy-channel response via load_channel_energies()",
    }

    # ── 2. Flare catalog ─────────────────────────────────────────────
    flist = []
    for i, f in enumerate(flares):
        utc_all = _utc_array(time_s, sxr_headers["mjdrefi"], sxr_headers["mjdreff"])
        pk = utc_all[min(f["peak_idx"], len(utc_all) - 1)]
        st = utc_all[min(f["start_idx"], len(utc_all) - 1)]
        en = utc_all[min(f["end_idx"], len(utc_all) - 1)]
        gflux = float(solexs_counts_to_irradiance_simple(np.array([f["peak_flux"]]))[0])
        flist.append(
            {
                "flare_no": i + 1,
                "goes_class_estimated": _fmt_goes(gflux),
                "combined_class": f.get("combined_class", "?"),
                "peak_utc": pk.strftime("%Y-%m-%d %H:%M:%S"),
                "duration_sec": round(float(f["duration_sec"]), 1),
                "goes_equivalent_flux_Wm2": gflux,
                "hxr_confirmed": bool(f.get("hxr_confirmed", False)),
            }
        )
    report["flare_catalog"] = {
        "total": len(flares),
        "hxr_confirmed": sum(1 for f in flares if f.get("hxr_confirmed", False)),
        "flares": flist,
    }

    # ── 3. GOES comparison ───────────────────────────────────────────
    if goes_xrsb_arr is not None:
        peak_flux = float(np.nanmax(goes_xrsb_arr))
        report["goes_comparison"] = {
            "goes_peak_class": _goes_class_label(peak_flux),
            "goes_peak_flux_Wm2": peak_flux,
        }

    # ── 4. Neupert effect ────────────────────────────────────────────
    best_f = None
    best_hxr_val = 0
    for f in flares:
        v = f.get("hxr_peak_czt", 0)
        if v and np.isfinite(v) and float(v) > best_hxr_val:
            best_hxr_val = float(v)
            best_f = f
    if best_f is not None and best_f.get("hxr_peak_czt", 0) > 0:
        pad = 300
        s = max(0, best_f["start_idx"] - pad)
        e = min(len(counts), best_f["end_idx"] + pad)
        sxr_seg = counts[s:e]
        hxr_seg = hxr_fb[s:e]
        m = np.isfinite(sxr_seg) & np.isfinite(hxr_seg)
        if m.sum() > 50:
            sxr_v = sxr_seg[m]
            hxr_v_seg = hxr_seg[m]
            dsxr_v = np.diff(sxr_v, prepend=sxr_v[0])
            r_n, p_n = pearsonr(dsxr_v, hxr_v_seg)
            rs = neupert_correlation(sxr_v, hxr_v_seg, window_sec=120, step_sec=30)
            rv = rs[np.isfinite(rs)]
            report["neupert_effect"] = {
                "pearson_r": round(float(r_n), 4),
                "p_value": f"{p_n:.2e}",
                "sliding_rho_mean": round(float(np.nanmean(rv)), 4),
                "sliding_rho_max": round(float(np.nanmax(rv)), 4),
            }

    # ── 5. Cross-correlation ─────────────────────────────────────────
    xx_r, xx_lag = lagged_cross_correlation(
        hxr_v.astype(np.float32), counts_v.astype(np.float32), max_lag=200
    )
    report["cross_correlation"] = {
        "hxr_vs_sxr_r": round(float(xx_r), 4),
        "lag_seconds": int(xx_lag),
        "hxr_leads_sxr_by_seconds": abs(int(xx_lag)),
    }

    # ── 6. Power spectrum ────────────────────────────────────────────
    if best_f is not None and best_f.get("hxr_peak_czt", 0) > 0:
        pad = 300
        s = max(0, best_f["start_idx"] - pad)
        e = min(len(hxr_fb), best_f["end_idx"] + pad)
        hs = hxr_fb[s:e]
        mv = np.isfinite(hs)
        if mv.sum() > 100:
            hls = hs[mv] - np.nanmean(hs[mv])
            fr = np.linspace(1.0 / 300, 1.0 / 10, 500)
            lf, lp = lomb_scargle_periodogram(np.arange(len(hls)), hls, fr)
            top3 = np.argsort(lp)[-3:][::-1]
            report["power_spectrum"] = {
                "top_periods": [
                    {
                        "period_s": round(float(1.0 / lf[i]), 1),
                        "power": round(float(lp[i]), 4),
                    }
                    for i in top3
                ]
            }

    # ── 7. QPP ───────────────────────────────────────────────────────
    wv_pks = []
    if best_f is not None and best_f.get("hxr_peak_czt", 0) > 0:
        pad = 300
        s = max(0, best_f["start_idx"] - pad)
        e = min(len(hxr_fb), best_f["end_idx"] + pad)
        hs = hxr_fb[s:e]
        mv = np.isfinite(hs)
        if mv.sum() > 100:
            pw, sc, pr = wavelet_power_auto(hs[mv], dt=1.0, s_min=10 / 4, s_max=300)
            gws = np.mean(pw, axis=1) if pw.size > 0 else np.array([])
            if len(gws) > 0:
                wp, _ = find_peaks(gws, height=np.max(gws) * 0.3)
                wv_pks = [round(float(pr[p]), 1) for p in wp]
    report["qpp_analysis"] = {
        "detected": bool(qpp_result.get("detected", False)),
        "period_s": round(float(qpp_result.get("period", 0)), 1),
        "significance": round(float(qpp_result.get("significance", 0)), 4),
        "ls_periods": [round(float(p), 1) for p in qpp_result.get("ls_periods", [])],
        "wavelet_gws_periods": wv_pks,
    }

    # ── 8. Spectral evolution ────────────────────────────────────────
    hr_full = 0.0
    hr_fmax = 0.0
    sh_full = 0.0
    if hxr_b is not None and hxr_b.shape[1] >= 2:
        hr = np.where(
            hxr_b[:, 0] > 0, hxr_b[:, 1] / np.maximum(hxr_b[:, 0], 1e-10), 0.0
        )
        hr_full = float(np.nanmean(hr))
        if best_f is not None and best_f.get("hxr_peak_czt", 0) > 0:
            pad = 300
            s = max(0, best_f["start_idx"] - pad)
            e = min(len(hxr_bands), best_f["end_idx"] + pad)
            hb = hxr_bands[s:e]
            mvb = np.isfinite(hb[:, 0]) & np.isfinite(hb[:, 1])
            if mvb.sum() > 10:
                hrf = np.where(
                    hb[mvb, 0] > 0, hb[mvb, 1] / np.maximum(hb[mvb, 0], 1e-10), 0.0
                )
                hr_fmax = float(np.nanmax(hrf))
    sh = np.where(hxr_v > 0, counts_v / hxr_v, 0.0)
    sh_full = float(np.nanmean(sh))
    report["spectral_evolution"] = {
        "hardness_ratio_mean": round(hr_full, 4),
        "hardness_ratio_flare_max": round(hr_fmax, 4),
        "sxr_over_hxr_ratio_mean": round(sh_full, 1),
    }

    # ── 9. Causal network ────────────────────────────────────────────
    report["causal_network"] = {
        "granger_hxr_to_dsxr_improvement_pct": round(
            float(gc_result.get("improvement", 0) * 100), 2
        ),
        "granger_best_lag_s": int(gc_result.get("best_lag", 0)),
    }

    # ── 10. Feature-group interpretations ─────────────────────────────
    if feature_values:
        report["feature_interpretations"] = interpret_feature_vector(
            feature_values,
            is_flare_window=(flares is not None and len(flares) > 0),
            n_windows=max(1, (len(counts) - 3600) // 300 + 1),
        )

    # ── 11. Processing ────────────────────────────────────────────────
    report["processing"] = {
        "runtime_sec": round(runtime_sec, 1),
        "n_windows": max(1, (len(counts) - 3600) // 300 + 1),
        "n_columns": df_columns,
        "n_gpu_features": n_gpu_features,
        "n_cpu_features": n_cpu_features,
        "feature_coverage_pct": round(100.0 * n_nonzero / max(n_gpu_features, 1), 1),
    }

    return report


def save_interpretation(report: dict, csv_path: str | Path) -> Path:
    """Save interpretation JSON alongside the master CSV."""
    csv_path = Path(csv_path)
    int_path = csv_path.with_name(
        csv_path.stem.replace("master_", "").replace(".csv", "")
        + "_interpretation.json"
    )
    # Put in same directory as CSV
    int_path = csv_path.parent / (csv_path.stem + "_interpretation.json")
    with open(int_path, "w") as fp:
        json.dump(report, fp, indent=2)
    return int_path
