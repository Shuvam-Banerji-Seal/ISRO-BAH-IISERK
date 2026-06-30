"""
Stage 2 — Phase 6: HEL1OS spectral index (#17, #18, #19).
Fit power-law to HEL1OS CdTe spectra, interpolate to 1s grid.
"""
import numpy as np
from pathlib import Path
from astropy.io import fits
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d

STAGE1 = Path("data/processed/stage1_20260623.npz")
CDTE1 = Path("data/raw/hel1os/20260623/cdte/hel1os_cdte_spectra_cdte1.fits")
CDTE2 = Path("data/raw/hel1os/20260623/cdte/hel1os_cdte_spectra_cdte2.fits")
OUT_DIR = Path("dist/features")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "phase6_spectral_index.npz"


def power_law(E, A, delta):
    """Simple power-law: dN/dE = A * E^(-delta)"""
    return A * E ** (-delta)


def extract():
    ds1 = np.load(STAGE1, allow_pickle=True)
    t_unix = ds1["time"].astype(np.float64)
    n = 86400

    # Load CdTe1 spectra
    print("  Loading CdTe1 spectra...")
    with fits.open(CDTE1) as h:
        spec = h[1].data
        mjdstart = h[0].header.get("MJDSTART", 61214.50726711752)
        n_spec = len(spec)
        ct_data = spec["COUNTS"].astype(np.float64)
        t_start = spec["TSTART"].astype(np.float64)
        t_stop = spec["TSTOP"].astype(np.float64)

    print(f"  MJDSTART = {mjdstart:.8f}, n_spec = {n_spec}")

    # Channel-energy mapping for CdTe
    ch_energy = 1.8 + np.arange(511, dtype=np.float64) * (90.0 - 1.8) / 511.0

    # Convert spectra timestamps to Unix
    mjdref = 40587
    t_mid_unix = (mjdstart + (t_start + t_stop) / 2.0 / 86400.0 - mjdref) * 86400.0

    # Fit power-law per spectrum in 20-60 keV range (nonthermal)
    en_range = (ch_energy >= 20) & (ch_energy <= 60)

    delta_arr = np.full(n_spec, np.nan, dtype=np.float32)
    A_arr = np.full(n_spec, np.nan, dtype=np.float32)
    fit_good = np.zeros(n_spec, dtype=bool)

    for i in range(n_spec):
        counts = np.asarray(ct_data[i], dtype=np.float64)
        if np.all(np.isnan(counts)) or np.nansum(counts) < 5:
            continue
        E_fit = ch_energy[en_range]
        ct_fit = counts[en_range]
        good = (ct_fit > 0) & ~np.isnan(ct_fit)
        if good.sum() < 5:
            continue
        try:
            popt, _ = curve_fit(
                power_law, E_fit[good], ct_fit[good],
                p0=(100, 3.0),
                bounds=([1e-10, 0.5], [1e10, 10.0]),
                maxfev=2000,
            )
            delta_arr[i] = popt[1]
            A_arr[i] = popt[0]
            fit_good[i] = True
        except (RuntimeError, ValueError):
            pass

    n_good = fit_good.sum()
    print(f"  Fitted {n_good}/{n_spec} spectra")

    # Interpolate to 1s grid using spectrum mid-times in Unix
    delta_1s = np.full(n, np.nan, dtype=np.float32)
    if n_good > 3:
        good_i = np.where(~np.isnan(delta_arr))[0]
        # Check that times are in range
        print(f"  t_mid_unix range: {t_mid_unix[good_i[0]]:.0f} - {t_mid_unix[good_i[-1]]:.0f}")
        print(f"  target t_unix range: {t_unix[0]:.0f} - {t_unix[-1]:.0f}")
        if t_mid_unix[good_i[0]] > t_unix[0] - 100 and t_mid_unix[good_i[-1]] < t_unix[-1] + 100:
            f_delta = interp1d(
                t_mid_unix[good_i], delta_arr[good_i],
                kind="linear", bounds_error=False, fill_value=np.nan
            )
            delta_1s = f_delta(t_unix).astype(np.float32)
        else:
            print("  Time ranges don't overlap — interpolation skipped")

    features = {
        "hxr_spectral_index": delta_1s,
    }
    # #18 Hardening rate (3-point gradient)
    g = ~np.isnan(delta_1s)
    dh = np.full(n, np.nan, dtype=np.float32)
    if g.sum() > 10:
        dh[g] = np.gradient(delta_1s[g])
    features["hxr_hardening_rate"] = dh

    # #19 SHS: rolling correlation of delta vs log(flux)
    hxr_flux = ds1["hxr_flux"].astype(np.float64)
    shs = np.full(n, np.nan, dtype=np.float32)
    window = 30  # 30s
    half = window // 2
    for i in range(half, n - half):
        if g[i] and ~np.isnan(hxr_flux[i]) and hxr_flux[i] > 0:
            seg_d = delta_1s[i - half:i + half + 1]
            seg_f = hxr_flux[i - half:i + half + 1]
            good2 = ~(np.isnan(seg_d) | np.isnan(seg_f) | (seg_f <= 0))
            if good2.sum() > 5:
                cmat = np.corrcoef(seg_d[good2], np.log(seg_f[good2]))
                if cmat.ndim == 2 and cmat.shape == (2, 2):
                    shs[i] = cmat[0, 1]
    features["shs_correlation"] = shs

    metadata = {
        "n_features": len(features),
        "feature_names": list(features.keys()),
        "phase": 6,
        "n_spectra": n_spec,
        "n_fitted": n_good,
        "delta_range": (float(np.nanmin(delta_arr)), float(np.nanmax(delta_arr))),
        "source": f"{CDTE1.name}, {CDTE2.name}",
    }
    for k, v in features.items():
        nnan = int(np.isnan(v).sum()) if v.dtype.kind == "f" else 0
        metadata[f"{k}_nan"] = nnan
        if v.dtype.kind == "f":
            vv = v[~np.isnan(v)]
            metadata[f"{k}_min"] = float(vv.min()) if len(vv) > 0 else np.nan
            metadata[f"{k}_max"] = float(vv.max()) if len(vv) > 0 else np.nan

    np.savez_compressed(OUT_PATH, **features, __metadata__=metadata)
    print(f"Phase 6 done -> {OUT_PATH}")
    print(f"  {len(features)} features")
    for k, v in features.items():
        nnan = metadata.get(f"{k}_nan", -1)
        lo = metadata.get(f"{k}_min", "?")
        hi = metadata.get(f"{k}_max", "?")
        print(f"    {k:30s} nan={nnan:<6d}  range=[{lo:.4g},{hi:.4g}]")

    return features


if __name__ == "__main__":
    extract()
