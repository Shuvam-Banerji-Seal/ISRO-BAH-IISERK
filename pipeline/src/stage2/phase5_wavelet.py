"""
Stage 2 — Phase 5: Wavelet + oscillation features.
CWT, QPP power, coherence, ridges, scalegram, red-noise significance.
"""
import numpy as np
from pathlib import Path
from scipy import signal
from datetime import datetime, timezone

STAGE1 = Path("data/processed/stage1_20260623.npz")
PHASE2 = Path("dist/features/phase2_perflare.npz")
OUT_DIR = Path("dist/features")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "phase5_wavelet.npz"

# Wavelet parameters
DT = 1.0  # 1s cadence
MIN_PERIOD = 10  # seconds
MAX_PERIOD = 500  # seconds
N_SCALES = 40


def morlet_wavelet(freq, s, w=6):
    """Morlet wavelet in frequency domain for CWT."""
    from scipy.signal._wavelets import morlet
    # Use scipy's cwt with morlet2
    pass


def compute_cwt_power(x, dt=DT, min_period=MIN_PERIOD, max_period=MAX_PERIOD, n_scales=N_SCALES):
    """Compute CWT power spectrum using PyWavelets. Returns (periods, power)."""
    import pywt
    n = len(x)
    if n < 20:
        return None, None

    x_filled = np.nan_to_num(x, nan=0.0)

    # Scales as periods in seconds
    periods = np.geomspace(min_period, max_period, n_scales)
    # Convert periods to scales for pywt (scale relates to period for Morlet)
    sampling_period = dt
    scales = periods / sampling_period

    # CWT using pywt
    coeffs, _ = pywt.cwt(x_filled, scales, "cmor1.5-1.0", sampling_period=sampling_period)
    power = np.abs(coeffs) ** 2  # (n_scales, n)

    return periods, power


def compute_red_noise_baseline(x, dt=DT):
    """Fit AR(1) process to get red-noise power spectrum for significance testing."""
    from scipy.optimize import minimize
    n = len(x)
    # Remove NaN
    xc = x[~np.isnan(x)]
    if len(xc) < 50:
        return None
    # Estimate lag-1 autocorrelation
    xc = xc - np.mean(xc)
    r1 = np.correlate(xc[:-1], xc[1:], mode="valid")[0] / np.correlate(xc, xc, mode="valid")[0]
    r1 = np.clip(r1, 0, 0.99)
    # Red noise power spectrum
    freqs = np.fft.rfftfreq(len(x), d=dt)[1:]  # skip DC
    # AR(1) power: P(f) = P0 / (1 + r1^2 - 2*r1*cos(2*pi*f*dt))
    p_red = 1.0 / (1 + r1**2 - 2 * r1 * np.cos(2 * np.pi * freqs * dt))
    return freqs, p_red, r1


def compute_wavelet_features():
    """Main computation entry point."""
    ds1 = np.load(STAGE1, allow_pickle=True)
    p2 = np.load(PHASE2, allow_pickle=True)

    t = ds1["time"].astype(np.float64)
    sxr = ds1["sxr_excess"].astype(np.float64)
    hxr = ds1["hxr_excess"].astype(np.float64)
    master_flag = ds1["master_flag"]
    flare_id = ds1["flare_id"]

    # Clean arrays for CWT: NaN where flag != 0
    sxr_clean = sxr.copy()
    hxr_clean = hxr.copy()
    sxr_clean[master_flag != 0] = np.nan
    hxr_clean[master_flag != 0] = np.nan

    n = 86400
    features = {}

    print("  Computing CWT for SXR...")
    sxr_periods, sxr_power = compute_cwt_power(sxr_clean)
    if sxr_power is None:
        print("  SXR CWT failed — not enough valid data")
        return features

    print(f"  SXR CWT: {len(sxr_periods)} scales × {sxr_power.shape[1]} time bins")

    print("  Computing CWT for HXR...")
    hxr_periods, hxr_power = compute_cwt_power(hxr_clean)
    if hxr_power is not None:
        print(f"  HXR CWT: {len(hxr_periods)} scales × {hxr_power.shape[1]} time bins")

    # ── Per-time-bin wavelet summary features ─────────────────
    # Mean power in 30-300s band (QPP-relevant)
    band_mask = (sxr_periods >= 30) & (sxr_periods <= 300)
    if band_mask.sum() > 0:
        features["sxr_qpp_power_30_300s"] = np.nanmean(sxr_power[band_mask], axis=0).astype(np.float32)

    # Peak period at each time bin (period with max power)
    peak_idx = np.nanargmax(sxr_power, axis=0)
    features["sxr_peak_period"] = sxr_periods[peak_idx].astype(np.float32)
    features["sxr_peak_period"][np.all(np.isnan(sxr_power), axis=0)] = np.nan
    # Peak power
    features["sxr_peak_power"] = np.nanmax(sxr_power, axis=0).astype(np.float32)

    # ── #36 Scalegram ─────────────────────────────────────────
    # Timescale distribution: N(T) = sum of power at each scale
    scalegram = np.nansum(sxr_power, axis=1)  # (n_scales,)
    # Fit slope β: log(N) ~ β * log(T)  (over 30-300s)
    sg_mask = (sxr_periods >= 30) & (sxr_periods <= 300)
    if sg_mask.sum() > 3:
        logT = np.log10(sxr_periods[sg_mask])
        logN = np.log10(np.maximum(scalegram[sg_mask], 1e-30))
        beta_slope = np.polyfit(logT, logN, 1)[0]
    else:
        beta_slope = np.nan
    # T_min = period at which first significant peak occurs (scale before power drops)
    # Simplified: period of max scalegram
    T_min = sxr_periods[np.nanargmax(scalegram)]

    features["scalegram_beta"] = np.full(n, beta_slope, dtype=np.float32)
    features["scalegram_T_min"] = np.full(n, T_min, dtype=np.float32)

    # ── #37a LIM (Local Intermittency Measure) ────────────────
    # LIM² = local wavelet power / mean power at each scale
    mean_power_per_scale = np.nanmean(sxr_power, axis=1, keepdims=True) + 1e-30
    lim_sxr = sxr_power / mean_power_per_scale
    features["sxr_lim_max"] = np.nanmax(lim_sxr, axis=0).astype(np.float32)
    features["sxr_lim_mean"] = np.nanmean(lim_sxr, axis=0).astype(np.float32)
    # LIM > 3 flag (significant intermittency)
    features["sxr_lim_flag"] = (np.nanmax(lim_sxr, axis=0) > 3).astype(np.int8)

    # ── #14, 15, 32: Per-flare windowed QPP power ────────────
    pre_power = np.full(n, np.nan, dtype=np.float32)
    onset_power = np.full(n, np.nan, dtype=np.float32)
    decay_power = np.full(n, np.nan, dtype=np.float32)
    qpp_cycles = np.zeros(n, dtype=np.int8)

    if "t_start" in p2 and "t_peak" in p2 and "t_end" in p2:
        p2_meta = p2["__metadata__"].item() if "__metadata__" in p2 else {}
        t_start_arr = p2["t_start"].astype(np.float64)
        t_peak_arr = p2["t_peak"].astype(np.float64)
        t_end_arr = p2["t_end"].astype(np.float64)

        for fid in sorted(set(f for f in flare_id if f > 0)):
            mask = flare_id == fid
            idx = np.where(mask)[0]
            if len(idx) < 30:
                continue

            tr = t[idx]
            ts = t_start_arr[idx[0]]
            tp = t_peak_arr[idx[0]]
            te = t_end_arr[idx[0]]
            if np.isnan(ts) or np.isnan(tp) or np.isnan(te):
                continue

            # Pre-flare: 30 min before start
            pre_win = (t >= ts - 1800) & (t < ts)
            # Onset: first 5 min of flare
            onset_win = (t >= ts) & (t < ts + 300)
            # Decay: after peak to end
            decay_win = (t >= tp) & (t < te)

            qpp_mask = (sxr_periods >= 30) & (sxr_periods <= 300)

            for win_name, win_idx, out_arr in [
                ("pre-flare", pre_win, pre_power),
                ("onset", onset_win, onset_power),
                ("decay", decay_win, decay_power),
            ]:
                w = np.where(win_idx)[0]
                if len(w) < 10:
                    continue
                power_vals = sxr_power[:, w]
                if qpp_mask.sum() > 0:
                    mean_p = np.nanmean(power_vals[qpp_mask])
                    out_arr[mask] = mean_p

            # QPP cycle count (proxy for #33)
            # Count times where peak power at QPP periods toggles
            power_broadband = sxr_power[:, idx]
            peak_per = sxr_periods[np.nanargmax(power_broadband, axis=0)]
            in_qpp = (peak_per >= 30) & (peak_per <= 300)
            transitions = np.sum(np.diff(in_qpp.astype(int)) != 0)
            qpp_cycles[mask] = min(transitions // 2, 20)

    features["qpp_power_preflare"] = pre_power
    features["qpp_power_onset"] = onset_power
    features["qpp_power_decay"] = decay_power
    features["qpp_cycle_count"] = qpp_cycles
    features["qpp_is_localized"] = (qpp_cycles < 3).astype(np.int8)  # #33: <3 = local, >=3 = global

    # ── #28 Cross-instrument coherence ────────────────────────
    if hxr_power is not None:
        cross_power = sxr_power * hxr_power  # rough proxy for cross-spectrum magnitude
        cross_power = np.sqrt(cross_power)  # geometric mean = common power
        features["cross_wavelet_power"] = np.nanmean(cross_power, axis=0).astype(np.float32)
        # Coherence proxy: where both have power at similar periods
        sxr_norm = sxr_power / (np.nanmean(sxr_power, axis=1, keepdims=True) + 1e-30)
        hxr_norm = hxr_power / (np.nanmean(hxr_power, axis=1, keepdims=True) + 1e-30)
        coh_proxy = np.nanmean(np.minimum(sxr_norm, hxr_norm), axis=0)
        features["cross_coherence"] = coh_proxy.astype(np.float32)
        # #37b Cross-channel LIM
        cross_lim = cross_power / (np.nanmean(cross_power, axis=1, keepdims=True) + 1e-30)
        features["cross_lim_max"] = np.nanmax(cross_lim, axis=0).astype(np.float32)

    # ── #34 Instantaneous frequency proxy ─────────────────────
    # Ridge tracking: follow the max-power period over time
    ridge_period = sxr_periods[np.nanargmax(sxr_power, axis=0)]
    ridge_period = np.where(np.all(np.isnan(sxr_power), axis=0), np.nan, ridge_period)
    features["ridge_period"] = ridge_period.astype(np.float32)
    # Chirp rate = gradient of log ridge period
    log_rp = np.log(np.maximum(np.nan_to_num(ridge_period, nan=100), 10))
    features["chirp_rate"] = np.gradient(log_rp).astype(np.float32)

    # ── #38 Red-noise significance flag ───────────────────────
    print("  Computing red-noise baseline...")
    rn_result = compute_red_noise_baseline(sxr_clean)
    if rn_result is not None:
        freqs_rn, p_red, r1 = rn_result
        # For each wavelet scale, compute significance relative to red noise
        # Simplified: flag any time bin where power exceeds 95% confidence envelope
        # Convert wavelet periods to approximate frequencies
        wave_freqs = 1.0 / sxr_periods
        red_power_interp = np.interp(wave_freqs, freqs_rn, p_red)
        # Normalize: power / red_noise_baseline per scale
        significance = np.mean(sxr_power, axis=1) / (red_power_interp + 1e-30)
        # Global significance > 3 = "significant QPP" flag
        global_sig = np.nanmean(sxr_power, axis=0) / np.nanmean(red_power_interp + 1e-30)
        features["red_noise_r1"] = np.full(n, r1, dtype=np.float32)
        features["qpp_significant"] = (global_sig > 3.0).astype(np.int8)
        features["qpp_sig_score"] = global_sig.astype(np.float32)

    # ── Metadata ──────────────────────────────────────────────
    metadata = {}
    for k, v in features.items():
        nnan = int(np.isnan(v).sum()) if v.dtype.kind == "f" else 0
        metadata[f"{k}_nan"] = nnan
        if v.dtype.kind == "f":
            vv = v[~np.isnan(v)]
            metadata[f"{k}_min"] = float(vv.min()) if len(vv) > 0 else np.nan
            metadata[f"{k}_max"] = float(vv.max()) if len(vv) > 0 else np.nan
        elif v.dtype.kind in ("i", "b"):
            metadata[f"{k}_min"] = int(v.min())
            metadata[f"{k}_max"] = int(v.max())

    metadata["n_features"] = len(features)
    metadata["feature_names"] = list(features.keys())
    metadata["phase"] = 5
    metadata["wavelet"] = "Morlet w=6"
    metadata["period_range"] = f"{MIN_PERIOD}-{MAX_PERIOD}s"
    metadata["source"] = f"{STAGE1.name}, {PHASE2.name}"

    np.savez_compressed(OUT_PATH, **features, __metadata__=metadata)
    print(f"Phase 5 done -> {OUT_PATH}")
    print(f"  {len(features)} features saved")
    for k, v in features.items():
        nnan = metadata.get(f"{k}_nan", -1)
        lo = metadata.get(f"{k}_min", "?")
        hi = metadata.get(f"{k}_max", "?")
        print(f"    {k:30s} nan={nnan:<6d}  range=[{lo:.4g},{hi:.4g}]")

    return features


if __name__ == "__main__":
    compute_wavelet_features()
