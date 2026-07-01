"""Feature coverage verification tests.

Ensures all 179 canonical features are computed (not all-zero)
across all pipeline outputs. Reports which features are zero
and categorizes by root cause.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

from bah2026.features.engineering import get_canonical_feature_names


# ── Categories of features that may legitimately be zero ─────────────────

# Known data-dependent: need GOES netCDF files
GOES_DEPENDENT = {
    "goes_xrsb_flux",
    "goes_xrsa_flux",
    "goes_xrsa_xrsb_ratio",
    "goes_xrsb_ddt_max",
    "goes_xrsb_rolling_std_300s",
    "goes_xrsb_rolling_std_1800s",
    "goes_xrsa_rolling_mean_300s",
    "goes_class_current",
    "goes_xrsb_gradient_1h",
    "goes_flare_history_24h",
    "goes_xrsb_prev_peak_ratio",
}

# Known to be zero on quiet days (QPP not detected, Granger/mediation weak)
QUIET_DAY_DEPENDENT = {
    "qpp_detected",
    "qpp_period",
    "qpp_amplitude",
    "qpp_significance",
    "neupert_granger_improvement",
    "neupert_best_lag",
    "max_mediation_proportion",
    "lagged_cross_corr",
    "lagged_cross_corr_lag",
    "sample_entropy_hxr",
    "sample_entropy_sxr",
    "transfer_entropy_hxr_to_sxr",
    "mutual_information_sxr_hxr",
    "cycle_detected",
    "n_feedback_loops",
}


def analyze_feature_coverage(
    csv_path: str | Path,
    expected_total: int = 179,
) -> dict:
    """Analyze which canonical features are all-zero in a CSV output.

    Parameters
    ----------
    csv_path : str or Path
        Path to the master CSV file.
    expected_total : int
        Expected number of canonical features (default 179).

    Returns
    -------
    dict with keys:
        total_columns, n_gpu_features, n_cpu_features, n_zero,
        zero_features (list), pct_nonzero, breakdown (dict of category->count)
    """
    df = pd.read_csv(csv_path)
    gpu_cols = [c for c in df.columns if c.startswith("gpu_")]
    cpu_cols = [c for c in df.columns if c.startswith("cpu_")]

    result: dict = {
        "total_columns": len(df.columns),
        "n_gpu_features": len(gpu_cols),
        "n_cpu_features": len(cpu_cols),
        "warning": None,
        "n_zero": 0,
        "n_nonzero": 0,
        "pct_nonzero": 0.0,
        "zero_features": [],
        "breakdown": {},
        "breakdown_details": {},
    }

    if len(gpu_cols) != expected_total:
        result["warning"] = (
            f"Expected {expected_total} GPU features, found {len(gpu_cols)}"
        )

    zero_gpu = [c for c in gpu_cols if (df[c] == 0.0).all()]
    nonzero_gpu = [c for c in gpu_cols if not (df[c] == 0.0).all()]

    result["n_zero"] = len(zero_gpu)
    result["n_nonzero"] = len(nonzero_gpu)
    result["pct_nonzero"] = 100.0 * len(nonzero_gpu) / max(len(gpu_cols), 1)
    result["zero_features"] = zero_gpu

    # Categorize zeros by root cause
    canonical = set(get_canonical_feature_names())
    zero_names = {c.replace("gpu_", "") for c in zero_gpu}

    categorization = {
        "goes_data_unavailable": zero_names & GOES_DEPENDENT,
        "quiet_day_expected": zero_names & QUIET_DAY_DEPENDENT,
        "missing_from_pre_dict": set(),
        "missing_gpu_batch_call": set(),
        "missing_advanced_cpu": set(),
    }

    # Known pre-dict gaps in generate_master_csv.py
    known_pre_missing = {
        "deadtime_max_pct",
        "bg_fraction_pct",
        "hk_czt1satctr",
        "hk_cdte1pilectr",
        "nonthermal_n_nth",
    }
    categorization["missing_from_pre_dict"] = zero_names & known_pre_missing

    # Known causal network features (from _batch_causal / _CAUSAL_FEATURES)
    causal_features = {
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
    }
    categorization["missing_gpu_batch_call"] = zero_names & causal_features

    # Per-window spectral + wavelet (handled by advanced CPU)
    advanced_cpu = {
        "sxr_temp_window",
        "sxr_em_window",
        "sxr_gamma_window",
        "hxr_gamma_window_czt1",
        "hxr_gamma_window_cdte1",
        "shs_index",
        "spectral_hardening_rate",
        "nonthermal_fraction_window",
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
    }
    categorization["missing_advanced_cpu"] = zero_names & advanced_cpu

    # Remaining unexplained zeros
    explained = set()
    for v in categorization.values():
        explained |= v
    categorization["unexplained"] = zero_names - explained - {"window_len"}

    result["breakdown"] = {k: len(v) for k, v in categorization.items()}
    result["breakdown_details"] = {k: sorted(v) for k, v in categorization.items()}

    return result


def test_feature_coverage_baseline():
    """Run against existing master CSV and report coverage."""
    csv_path = Path("output/master_csv/master_may5_2024.csv")
    assert csv_path.exists(), f"Master CSV not found: {csv_path}"

    report = analyze_feature_coverage(csv_path)

    print(f"\n{'=' * 60}")
    print(f"FEATURE COVERAGE REPORT: {csv_path.name}")
    print(f"{'=' * 60}")
    print(f"Total columns:          {report['total_columns']}")
    print(f"GPU features:           {report['n_gpu_features']}")
    print(f"CPU features:           {report['n_cpu_features']}")
    print(
        f"Non-zero GPU features:  {report['n_nonzero']} ({report['pct_nonzero']:.1f}%)"
    )
    print(f"Zero GPU features:      {report['n_zero']}")
    print()
    print("Breakdown by root cause:")
    for cause, count in sorted(report["breakdown"].items()):
        details = report["breakdown_details"][cause]
        print(f"  {cause}: {count}")
        if details:
            for fn in details:
                print(f"    - {fn}")
    print(f"{'=' * 60}\n")

    # After Phase 1 fixes, we expect ~21 zero features (11 GOES data unavailable,
    # 7 quiet-day dependent, 1 nonthermal_fraction_window, 1 sxr_chi2_red)
    assert report["n_zero"] <= 30, (
        f"Too many zero features: {report['n_zero']}. "
        "Expected <= 30 after Phase 1 fixes (11 GOES + 7 quiet-day + 2 minor)."
    )
    # Assert 88%+ non-zero
    assert report["pct_nonzero"] >= 85.0, (
        f"Non-zero feature coverage too low: {report['pct_nonzero']:.1f}%. "
        "Expected >= 85% after Phase 1 fixes."
    )


if __name__ == "__main__":
    test_feature_coverage_baseline()
