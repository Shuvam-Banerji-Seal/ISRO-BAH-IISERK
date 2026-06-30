"""
Stage 2 — Phase 3: GOES T/EM derivation.
Unblocks features #1 (T), #2 (EM), #3 (HOPE), #4 (FAI), #6 (T-EM trajectory), #8 (Reale loop length), #30 (internal consistency).

Uses White, Thomas & Schwartz (2005) polynomial method:
    T from XRS-B / XRS-A ratio
    EM from XRS-B flux + temperature response
"""
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from scipy.ndimage import uniform_filter1d

STAGE0 = Path("data/processed/master_dataset_20260623.npz")
STAGE1 = Path("data/processed/stage1_20260623.npz")
OUT_DIR = Path("dist/features")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "phase3_tem_goes.npz"


def compute_tem(flux_a, flux_b):
    """
    Compute T (MK) and EM (cm^-3) from GOES XRS-A (0.5-4A) and XRS-B (1-8A).

    Uses White et al. (2005) polynomial for T:
        log10(T) = 0.051 + 0.979·w + 0.184·w² - 0.010·w³,  w = log10(B/A)

    Emission measure from GOES B-channel response (CHIANTI-based):
        EM_48 = F_B · 1e5 · T^0.5 · exp(17 / T)   [T in MK, F_B in W/m²]
    """
    fa = np.where(flux_a > 2e-8, flux_a, 2e-8)
    fb = np.where(flux_b > 5e-8, flux_b, 5e-8)

    R = fb / fa
    R = np.clip(R, 1.05, 35.0)

    w = np.log10(R)
    log10_T = 0.051 + 0.979 * w + 0.184 * w**2 - 0.010 * w**3
    T_MK = 10**log10_T
    T_MK = np.clip(T_MK, 3.0, 50.0)

    cf = np.exp(17.0 / T_MK)
    em_48 = fb * 1e5 * np.sqrt(T_MK) * cf
    em_48 = np.clip(em_48, 0.001, 1e4)

    EM = em_48 * 1e48

    # NaN out quiet periods
    quiet = (flux_a < 5e-8) | (flux_b < 1e-7)
    T_MK = np.where(quiet, np.nan, T_MK)
    EM = np.where(quiet, np.nan, EM)

    return T_MK.astype(np.float32), EM.astype(np.float64)


def compute_hope_score(T, EM, window=60):
    """
    HOPE precursor score: T rises while EM stays flat (pre-evaporation heating).
    Normalized within each flare window.

    Score = max(0, dT/dt) * exp(-|dEM/dt| / sigma_EM)
    where sigma_EM is the rolling robust scatter of dEM/dt.
    """
    T_filled = np.nan_to_num(T, nan=np.nanmedian(T[~np.isnan(T)]) if np.any(~np.isnan(T)) else 3.0)
    EM_filled = np.nan_to_num(EM, nan=0.0)

    T_smooth = uniform_filter1d(T_filled, size=window, mode="nearest")
    EM_smooth = uniform_filter1d(EM_filled, size=window, mode="nearest")

    dT = np.gradient(T_smooth)
    dEM = np.gradient(EM_smooth)

    sigma_EM = np.nanpercentile(np.abs(dEM), 75) + 1e-10

    hope = np.maximum(0, dT) * np.exp(-np.abs(dEM) / sigma_EM)
    hope = np.clip(hope / (np.nanpercentile(hope, 95) + 1e-10), 0, 1)

    hope = np.where(np.isnan(T) | np.isnan(EM), np.nan, hope)
    return hope.astype(np.float32)


def compute_tem_trajectory_shape(T, EM, flare_id):
    """Classify each flare's T-EM trajectory. Returns per-bin integer code."""
    n = len(T)
    code = np.zeros(n, dtype=np.int8)

    for fid in sorted(set(f for f in flare_id if f > 0)):
        mask = flare_id == fid
        idx = np.where(mask)[0]
        if len(idx) < 10:
            continue

        t_local = T[idx].copy()
        em_local = EM[idx].copy()
        good = ~(np.isnan(t_local) | np.isnan(em_local))
        if good.sum() < 10:
            continue

        t_g = t_local[good]
        em_g = em_local[good]

        T_peaks = np.sum((t_g[1:-1] > t_g[:-2]) & (t_g[1:-1] > t_g[2:]))
        EM_peaks = np.sum((em_g[1:-1] > em_g[:-2]) & (em_g[1:-1] > em_g[2:]))

        if T_peaks >= 2 or EM_peaks >= 2:
            code[idx] = 4
        else:
            peak_idx = np.nanargmax(t_local)
            decay = slice(peak_idx, None)
            if peak_idx < len(t_local) - 3:
                logT_d = np.log(t_g[t_g > 0])
                logEM_d = np.log(em_g[em_g > 0])
                if len(logT_d) > 5 and len(logEM_d) > 5:
                    zeta = np.polyfit(logT_d[-min(20, len(logT_d)):], logEM_d[-min(20, len(logEM_d)):], 1)[0]
                    if zeta > 1.5:
                        code[idx] = 3
                    elif zeta < 0.5:
                        code[idx] = 2
                    else:
                        code[idx] = 1
                else:
                    code[idx] = 0
            else:
                code[idx] = 0
    return code


def compute_loops(T, EM, tau_decay, flare_id):
    """
    Reale-law loop length (cm) = tau_LC * sqrt(T) / (alpha * F(zeta))
    """
    n = len(T)
    L = np.full(n, np.nan, dtype=np.float64)

    for fid in sorted(set(f for f in flare_id if f > 0)):
        mask = flare_id == fid
        idx = np.where(mask)[0]
        if len(idx) < 10:
            continue

        t_l = T[idx].copy()
        e_l = EM[idx].copy()
        g = ~(np.isnan(t_l) | np.isnan(e_l))
        if g.sum() < 10:
            continue
        t_g = t_l[g]
        e_g = e_l[g]

        peak_loc = np.argmax(t_g)
        if peak_loc >= len(t_g) - 3:
            continue

        t_decay = t_g[peak_loc:]
        e_decay = e_g[peak_loc:]
        if len(t_decay) < 5:
            continue

        x = np.arange(len(t_decay))
        y = np.log(t_decay)
        slope = np.polyfit(x, y, 1)[0]
        tau_lc = -1.0 / slope if slope < 0 else 500.0

        logT_d = np.log(t_decay)
        logEM_d = np.log(e_decay)
        if len(logT_d) < 5:
            continue
        zeta = np.polyfit(logT_d, logEM_d, 1)[0]
        zeta = np.clip(zeta, 0.1, 3.0)

        if zeta < 0.35:
            Fz = 0.63 * zeta / 0.35
        elif zeta < 0.7:
            Fz = 0.63 + (zeta - 0.35) * (1.0 - 0.63) / 0.35
        elif zeta < 1.6:
            Fz = 1.0 + (zeta - 0.7) * (1.6 - 1.0) / 0.9
        else:
            Fz = 1.6

        alpha = 3.7e-4
        L_loop = tau_lc * np.sqrt(np.nanmean(t_decay * 1e6)) / (alpha * Fz)
        L[idx] = L_loop

    return L.astype(np.float32)


def extract():
    ds0 = np.load(STAGE0, allow_pickle=True)
    ds1 = np.load(STAGE1, allow_pickle=True)

    t = ds0["time"].astype(np.float64)
    goes_a = ds0["goes_flux_a"].astype(np.float64)
    goes_b = ds0["goes_flux"].astype(np.float64)
    flare_id = ds1["flare_id"]
    master_flag = ds1["master_flag"]

    # ── #1 Temperature, #2 Emission Measure ──────────────────
    T_MK, EM = compute_tem(goes_a, goes_b)

    # Log-scale EM for storage (float32 friendly); store linear as float64
    EM_log = np.log10(EM + 1e-30).astype(np.float32)
    features = {
        "goes_temperature_MK": T_MK,
        "goes_emission_measure_log10": EM_log,
        "goes_emission_measure": EM.astype(np.float64),
    }

    # ── #3 HOPE signature ─────────────────────────────────────
    hope = compute_hope_score(T_MK, EM_log)
    features["hope_score"] = hope
    hope_threshold = np.nanpercentile(hope[hope > 0], 90) if np.any(hope > 0) else 0.5
    features["hope_flag"] = (hope > hope_threshold).astype(np.int8)

    # ── #4 Flare Anticipation Index ───────────────────────────
    # Continuous version of HOPE strength, low-pass filtered
    from scipy.ndimage import uniform_filter1d
    fai = uniform_filter1d(np.nan_to_num(hope, nan=0), size=61, mode="constant")  # ~1 min
    features["fai"] = fai.astype(np.float32)

    # ── #5 T gradient ─────────────────────────────────────────
    features["T_gradient"] = np.gradient(np.nan_to_num(T_MK, nan=np.nan)).astype(np.float32)

    # ── #6 T-EM trajectory shape ──────────────────────────────
    traj_code = compute_tem_trajectory_shape(T_MK, EM, flare_id)
    features["tem_trajectory_code"] = traj_code
    features["is_single_peak"] = ((traj_code > 0) & (traj_code < 4)).astype(np.int8)
    features["is_double_peak"] = (traj_code == 4).astype(np.int8)
    features["is_off_branch"] = (traj_code == 3).astype(np.int8)
    features["is_qss_branch"] = (traj_code == 2).astype(np.int8)

    # ── #7 T peak lead relative to EM peak ────────────────────
    T_peak_t = np.full(86400, np.nan, dtype=np.float32)
    EM_peak_t = np.full(86400, np.nan, dtype=np.float32)
    for fid in sorted(set(f for f in flare_id if f > 0)):
        mask = flare_id == fid
        idx = np.where(mask)[0]
        if len(idx) >= 5:
            T_peak_t[idx] = t[idx[np.nanargmax(T_MK[idx])]]
            EM_peak_t[idx] = t[idx[np.nanargmax(EM[idx])]]
    features["T_peak_time"] = T_peak_t
    features["EM_peak_time"] = EM_peak_t
    features["T_leads_EM"] = (EM_peak_t - T_peak_t).astype(np.float32)

    # ── #8 Reale-law loop length ──────────────────────────────
    tau_decay = ds1.get("duration", np.full(86400, 500.0, dtype=np.float32))
    loop_L = compute_loops(T_MK, EM, tau_decay, flare_id)
    features["reale_loop_length_cm"] = loop_L

    # ── #30 Internal consistency score ────────────────────────
    # Combines R_N, T_leads_EM, Reale loop length (dt_peak added in Phase 2)
    neupert = ds1["neupert_rho"].astype(np.float32)
    t_leads = features["T_leads_EM"]

    def robust_normalize(x):
        lo, hi = np.nanpercentile(x[~np.isnan(x)], 5), np.nanpercentile(x[~np.isnan(x)], 95) if np.any(~np.isnan(x)) else (0, 1)
        if np.isnan(lo) or lo == hi or np.isnan(hi):
            return np.zeros_like(x)
        return np.clip((np.nan_to_num(x) - lo) / (hi - lo), 0, 1)

    rn = robust_normalize(np.abs(neupert))
    tl = 1.0 - robust_normalize(np.abs(t_leads))
    rl = robust_normalize(np.nan_to_num(loop_L, nan=0))
    consistency = (rn + tl + rl) / 3.0
    consistency = np.where(np.isnan(neupert) & np.isnan(t_leads) & np.isnan(loop_L), np.nan, consistency)
    features["internal_consistency"] = consistency.astype(np.float32)

    # ── Metadata ──────────────────────────────────────────────
    metadata = {}
    for k, v in features.items():
        nnan = int(np.isnan(v).sum()) if v.dtype.kind == "f" else 0
        metadata[f"{k}_nan"] = nnan
        metadata[f"{k}_min"] = float(np.nanmin(v)) if v.dtype.kind == "f" else int(v.min())
        metadata[f"{k}_max"] = float(np.nanmax(v)) if v.dtype.kind == "f" else int(v.max())

    metadata["n_features"] = len(features)
    metadata["feature_names"] = list(features.keys())
    metadata["phase"] = 3
    metadata["source"] = f"{STAGE0.name}, {STAGE1.name}"
    metadata["tem_method"] = "White et al. (2005) polynomial + GOES response"
    metadata["tem_notes"] = "XRS-A floored to 1e-8, XRS-B floored to 1e-9"

    np.savez_compressed(OUT_PATH, **features, __metadata__=metadata)
    print(f"Phase 3 done -> {OUT_PATH}")
    print(f"  {len(features)} features saved")
    for k, v in features.items():
        nnan = metadata.get(f"{k}_nan", -1)
        lo = metadata.get(f"{k}_min", "?")
        hi = metadata.get(f"{k}_max", "?")
        print(f"    {k:30s} nan={nnan:<6d}  range=[{lo:.4g},{hi:.4g}]")

    return features


if __name__ == "__main__":
    extract()
