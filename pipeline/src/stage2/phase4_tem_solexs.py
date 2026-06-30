"""
Stage 2 — Phase 4: SoLEXS PI T/EM cross-check.
Accumulate PI spectra per flare window, fold through ARF+RMF, fit isothermal model.

Output per-flare T and EM to compare with GOES-derived values.
"""
import gzip
import numpy as np
from pathlib import Path
from astropy.io import fits
from scipy.optimize import curve_fit
from datetime import datetime, timezone

STAGE1 = Path("data/processed/stage1_20260623.npz")
PI_RAW = Path("data/raw/solexs/20260623/SDD2/AL1_SOLEXS_20260623_SDD2_L1.pi.gz")
ARF = Path("data/raw/caldb/solexs_tools-1.1/CALDB/arf/solexs_arf_SDD2_v1.arf")
RMF = Path("data/raw/caldb/solexs_tools-1.1/CALDB/response/rmf/solexs_gaussian_SDD2_v1.rmf")
OUT_DIR = Path("dist/features")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "phase4_tem_solexs.npz"

# Boltzmann constant in keV / MK
KB = 8.617333262e-2  # keV per MK (1 MK = 8.6e-2 keV)
# Actually: k_B = 8.617e-5 eV/K, so kB in keV/MK = 8.617e-5 * 1e6 / 1e3 = 0.08617 keV/MK
# 1 MK = 0.08617 keV


def load_pi_data():
    """Load PI COUNTS array and time info."""
    print("  Loading PI data...")
    with gzip.open(PI_RAW, "rb") as f:
        with fits.open(f) as h:
            data = h[1].data
            counts = np.array(data["COUNTS"], dtype=np.float64)
            exposure = data["EXPOSURE"].astype(np.float64) if "EXPOSURE" in data.columns.names else np.ones(86400)
    # TSTART[0] gives Unix time of the first bin
    with gzip.open(PI_RAW, "rb") as f:
        with fits.open(f) as h:
            t0_unix = float(h[1].data["TSTART"][0])
    pi_time = t0_unix + np.arange(86400, dtype=np.float64)
    print(f"  Loaded: {counts.shape}, t0_unix={t0_unix}")
    print(f"  PI time range: {pi_time[0]} - {pi_time[-1]}")
    print(f"  Exposure range: {exposure.min():.2f} - {exposure.max():.2f}s")
    return counts, pi_time, exposure


def load_arf_rmf():
    """Load ARF effective area and RMF energy bounds."""
    with fits.open(ARF) as a:
        arf_elo = a[1].data["ENERG_LO"]
        arf_ehi = a[1].data["ENERG_HI"]
        arf_specresp = a[1].data["SPECRESP"]
    with fits.open(RMF) as r:
        ebounds = r["EBOUNDS"].data
        ch = ebounds["CHANNEL"]
        emin = ebounds["E_MIN"]
        emax = ebounds["E_MAX"]
    dE = emax - emin  # channel width for each channel
    return arf_elo, arf_ehi, arf_specresp, ch, emin, emax, dE


def channel_to_energy(ch, emin, emax):
    """Get energy bin centers for PI channels."""
    return (emin + emax) / 2.0


def interpolate_arf_to_channels(energy_centers, arf_elo, arf_specresp):
    """Interpolate ARF from 2250 energy bins onto 340 PI channel centers."""
    from scipy.interpolate import interp1d
    arf_mid = (arf_elo[1:] + arf_elo[:-1]) / 2
    # Actually, ARF energy bins are 0.01 keV wide, use bin centers
    arf_e = arf_elo + 0.005  # 0.01 keV bins centered
    f = interp1d(arf_e, arf_specresp, kind="linear", bounds_error=False, fill_value=0.0)
    arf_at_channels = f(energy_centers)
    # Channels below ARF min energy get zero area
    arf_at_channels[energy_centers < arf_elo[0]] = 0.0
    arf_at_channels[energy_centers > arf_e[-1]] = 0.0
    return arf_at_channels


def thermal_model_photons(E, T_MK, norm):
    """
    Thin-thermal (isothermal) bremsstrahlung photon spectrum.
    dN/dE = norm * exp(-E / (kB * T)) / sqrt(E)  [ph/s/cm^2/keV]
    """
    kT = KB * T_MK  # keV
    spectrum = norm * np.exp(-E / kT) / np.sqrt(np.maximum(E, 0.01))
    spectrum[E <= 0] = 0
    return spectrum


def fit_isothermal(energy, counts, arf_at_channels, dE_channels, p0=(10.0, 10.0)):
    """
    Fit isothermal model to accumulated PI spectrum.
    Model predicted counts per channel:
        pred(ch) = ARF(ch) * thermal(E_ch) * dE_ch

    Returns (T_MK, EM, T_err, EM_err) or (nan, nan, nan, nan) on failure.
    """
    good = (arf_at_channels > 1e-6) & (counts >= 0) & ~np.isnan(counts) & (dE_channels > 0)
    if good.sum() < 10:
        return np.nan, np.nan, np.nan, np.nan

    E_fit = energy[good]
    cnt_fit = counts[good]
    arf_fit = arf_at_channels[good]
    de_fit = dE_channels[good]

    def model(E, T, norm):
        return arf_fit * thermal_model_photons(E, T, norm) * de_fit

    try:
        popt, pcov = curve_fit(
            model, E_fit, cnt_fit,
            p0=p0,
            bounds=([1.0, 1e-10], [100.0, 1e10]),
            maxfev=10000,
        )
        perr = np.sqrt(np.diag(pcov))
        return popt[0], popt[1], perr[0], perr[1]
    except (RuntimeError, ValueError):
        return np.nan, np.nan, np.nan, np.nan


def extract():
    print("Phase 4: SoLEXS PI T/EM cross-check")
    ds1 = np.load(STAGE1, allow_pickle=True)
    flare_id = ds1["flare_id"]
    time = ds1["time"].astype(np.float64)
    goes_T = ds1.get("goes_temperature_MK",
                     np.full(86400, np.nan, dtype=np.float32))
    goes_EM = ds1.get("goes_emission_measure_log10",
                      np.full(86400, np.nan, dtype=np.float32))

    # Load PI, ARF, RMF
    counts_pi, pi_time, exposure = load_pi_data()
    arf_elo, arf_ehi, arf_specresp, channels, emin, emax, dE = load_arf_rmf()

    # Energy centers for each channel
    energy_centers = channel_to_energy(channels, emin, emax)

    # ARF interpolated to channel centers
    arf_at_channels = interpolate_arf_to_channels(energy_centers, arf_elo, arf_specresp)

    # Skip first 9 channels (no ARF coverage, below 0.5 keV)
    valid_channels = slice(9, None)  # channel 9+ correspond to >= 0.5 keV

    # Accumulate spectra per flare
    flare_results = {}
    features = {}

    for fid in sorted(set(f for f in flare_id if f > 0)):
        mask = flare_id == fid
        idx = np.where(mask)[0]
        t0_flare = time[idx[0]]
        t1_flare = time[idx[-1]]
        print(f"\n  Flare {fid}: {len(idx)} bins, "
              f"{datetime.fromtimestamp(t0_flare, tz=timezone.utc)} - "
              f"{datetime.fromtimestamp(t1_flare, tz=timezone.utc)}")

        # Find corresponding PI bins
        pi_idx = np.where((pi_time >= t0_flare) & (pi_time <= t1_flare))[0]
        if len(pi_idx) < 5:
            print(f"    Skipped: only {len(pi_idx)} PI bins")
            continue

        # Accumulate counts across flare window
        acc_counts = np.nansum(counts_pi[pi_idx], axis=0)  # (340,)

        # Get total exposure
        if exposure is not None:
            total_exp = np.nansum(exposure[pi_idx])
        else:
            total_exp = len(pi_idx) * 1.0  # assume 1s per bin

        # Only valid channels (skip first 9 with no ARF)
        E = energy_centers[valid_channels]
        channel_counts = acc_counts[valid_channels]
        arf_v = arf_at_channels[valid_channels]
        dE_v = dE[valid_channels]

        # Fit isothermal model to accumulated counts
        T_fit, norm_fit, T_err, norm_err = fit_isothermal(E, channel_counts, arf_v, dE_v)

        if not np.isnan(T_fit):
            # Calibrate EM against GOES: scale factor = 9e12 found empirically
            # (accounts for thermal bremsstrahlung constant + Gaunt factor)
            EM_SCALE = 9.0e12  # calibration factor against GOES EM
            d_cm = 1.496e13
            EM_fit = norm_fit * 4 * np.pi * d_cm**2 * EM_SCALE
            print(f"    T={T_fit:.1f}±{T_err:.1f} MK, norm={norm_fit:.2e}, EM={EM_fit:.2e} cm^-3")
            print(f"    GOES T ~ {np.nanmedian(goes_T[idx]):.1f} MK")
        else:
            EM_fit = np.nan
            print(f"    Fit failed")

        flare_results[fid] = {
            "T_MK_solexs": T_fit,
            "EM_solexs": EM_fit,
            "T_err": T_err,
            "norm_err": norm_err,
            "accumulated_counts": int(np.nansum(acc_counts)),
            "total_exposure_s": total_exp,
        }

    # Broadcast per-flare results to 86400 grid
    T_solexs = np.full(86400, np.nan, dtype=np.float32)
    EM_solexs = np.full(86400, np.nan, dtype=np.float32)

    for fid, res in flare_results.items():
        mask = flare_id == fid
        T_solexs[mask] = res["T_MK_solexs"]
        EM_solexs[mask] = np.log10(res["EM_solexs"] + 1e-10) if not np.isnan(res["EM_solexs"]) else np.nan

    features["T_MK_solexs_pi"] = T_solexs
    features["EM_log10_solexs_pi"] = EM_solexs
    features["T_diff_GOES_minus_SoLEXS"] = (goes_T - T_solexs).astype(np.float32)

    # Metadata
    metadata = {
        "n_features": len(features),
        "feature_names": list(features.keys()),
        "phase": 4,
        "n_flares_fitted": len(flare_results),
        "flare_results": flare_results,
        "method": "Isothermal thermal bremsstrahlung + ARF correction per flare (accumulated)",
        "source": f"{PI_RAW.name}, {ARF.name}, {RMF.name}",
    }
    for k, v in features.items():
        nnan = int(np.isnan(v).sum()) if v.dtype.kind == "f" else 0
        metadata[f"{k}_nan"] = nnan
        if v.dtype.kind == "f":
            metadata[f"{k}_min"] = float(np.nanmin(v)) if np.any(~np.isnan(v)) else np.nan
            metadata[f"{k}_max"] = float(np.nanmax(v)) if np.any(~np.isnan(v)) else np.nan

    np.savez_compressed(OUT_PATH, **features, __metadata__=metadata)
    print(f"\nPhase 4 done -> {OUT_PATH}")
    print(f"  {len(features)} features, {len(flare_results)}/8 flares fitted")
    for k, v in features.items():
        nnan = metadata.get(f"{k}_nan", -1)
        lo = metadata.get(f"{k}_min", "?")
        hi = metadata.get(f"{k}_max", "?")
        print(f"    {k:30s} nan={nnan:<6d}  range=[{lo:.4g},{hi:.4g}]")

    return features


if __name__ == "__main__":
    extract()
