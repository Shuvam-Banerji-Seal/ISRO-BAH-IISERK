#!/usr/bin/env python3
"""GPU extraction — parallel CPU loading, GPU compute, v3 features + day-level precomputed."""

import sys, os, time, warnings, json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
os.environ.setdefault(
    "BAH2026_DATA", os.path.join(os.path.dirname(__file__), "../../../data/processed")
)
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, torch
from multiprocessing import Pool, cpu_count
from bah2026.data import discover_combined_days
from bah2026.config import CATALOGS_DIR, HDF5_DIR

HDF5_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_LOG = HDF5_DIR / "processed_days.txt"
LOOKBACK, STEP = 3600, 300
N_WORKERS = min(cpu_count(), 24)
CHUNK = 200


def _worker_load(args):
    d, event_times = args
    from bah2026.data.reader import (
        load_solexs_lc,
        load_solexs_pi,
        load_solexs_gti,
        load_hel1os_lc,
        load_hel1os_hk,
        load_hel1os_spectra,
    )
    from bah2026.data.preprocessing import align_hel1os_to_solexs
    from bah2026.data.corrections import (
        correct_solexs_deadtime,
        subtract_hel1os_background,
    )

    pre = {}
    try:
        sxr = load_solexs_lc(d)
        counts = correct_solexs_deadtime(
            np.where(
                np.isfinite(sxr["counts"]), sxr["counts"], np.nanmedian(sxr["counts"])
            )
        ).astype(np.float32)
        time_s = sxr["time"]
        hxr4 = np.zeros((len(counts), 20), dtype=np.float32)
        for idx, (det, num) in enumerate(
            [("czt", 1), ("czt", 2), ("cdte", 1), ("cdte", 2)]
        ):
            try:
                hx = load_hel1os_lc(d, detector=det, num=num)
                if hx["ctr"].size > 0:
                    ctr = subtract_hel1os_background(hx["ctr"], det)
                    a = align_hel1os_to_solexs(
                        hx["mjd"], ctr, time_s, sxr["mjdrefi"], sxr["mjdreff"]
                    )
                    ml = min(len(counts), a.shape[0])
                    hxr4[:ml, idx * 5 : (idx + 1) * 5] = a[:ml, :5].astype(np.float32)
            except Exception:
                pass
        pi_win = None
        pi_sum = None
        try:
            pi = load_solexs_pi(d)
            if pi["counts"].size > 0:
                pr = pi["counts"].astype(np.float32)
                pi_sum = np.nansum(pr, axis=0) if pr.ndim == 2 else pr
                n_w = (len(pr) - LOOKBACK) // STEP + 1
                if n_w > 0:
                    pi_win = np.zeros((n_w, 340), dtype=np.float32)
                    for wi in range(n_w):
                        pi_win[wi] = np.nansum(
                            pr[wi * STEP : wi * STEP + LOOKBACK], axis=0
                        )
        except Exception:
            pass

        # ── Day-level features ────────────────────────────────────
        # Deadtime percentage from SoLEXS GTI
        try:
            gti = load_solexs_gti(d)
            if gti.size > 0:
                total_span = time_s[-1] - time_s[0] if len(time_s) > 1 else 1.0
                good_time = np.sum(gti[:, 1] - gti[:, 0])
                pre["deadtime_max_pct"] = float(
                    max(
                        0.0,
                        min(100.0, (1.0 - good_time / max(total_span, 1e-6)) * 100.0),
                    )
                )
            else:
                pre["deadtime_max_pct"] = 0.0
        except Exception:
            pre["deadtime_max_pct"] = 0.0

        # Background fraction from HXR full-band (band 4 = CZT 18-160 keV)
        try:
            hxr_full = hxr4[:, 4]
            valid_hxr = hxr_full[np.isfinite(hxr_full)]
            if len(valid_hxr) > 0:
                median_val = np.median(valid_hxr)
                bg_threshold = max(median_val * 1.1, 0.1)
                pre["bg_fraction_pct"] = float(
                    np.mean(valid_hxr < bg_threshold) * 100.0
                )
            else:
                pre["bg_fraction_pct"] = 0.0
        except Exception:
            pre["bg_fraction_pct"] = 0.0

        # HK features from HEL1OS housekeeping
        for k in [
            "hk_czt1temp",
            "hk_czt2temp",
            "hk_cdte1temp",
            "hk_cdte2temp",
            "hk_czthvmon",
            "hk_cdtehvmon",
            "hk_czt1satctr",
            "hk_cdte1pilectr",
        ]:
            pre[k] = 0.0
        try:
            hk = load_hel1os_hk(d)
            for hk_key, pre_key in [
                ("czt1temp", "hk_czt1temp"),
                ("czt2temp", "hk_czt2temp"),
                ("cdte1temp", "hk_cdte1temp"),
                ("cdte2temp", "hk_cdte2temp"),
                ("czthvmon", "hk_czthvmon"),
                ("cdtehvmon", "hk_cdtehvmon"),
            ]:
                if hk_key in hk and len(hk[hk_key]) > 0:
                    vals = hk[hk_key][np.isfinite(hk[hk_key])]
                    pre[pre_key] = float(np.median(vals)) if len(vals) > 0 else 0.0
            if "czt1satctr1" in hk and len(hk["czt1satctr1"]) > 0:
                pre["hk_czt1satctr"] = float(np.max(hk["czt1satctr1"]))
            if "cdte1pilectr" in hk and len(hk["cdte1pilectr"]) > 0:
                pre["hk_cdte1pilectr"] = float(np.max(hk["cdte1pilectr"]))
        except Exception:
            pass

        # Spectral indices from HEL1OS spectra (all 4 detectors)
        for gamma_key in [
            "hxr_spectral_index_gamma",
            "hxr_gamma_czt2",
            "hxr_gamma_cdte1",
            "hxr_gamma_cdte2",
        ]:
            pre[gamma_key] = 0.0
        for det, num, gamma_key in [
            ("czt", 1, "hxr_spectral_index_gamma"),
            ("czt", 2, "hxr_gamma_czt2"),
            ("cdte", 1, "hxr_gamma_cdte1"),
            ("cdte", 2, "hxr_gamma_cdte2"),
        ]:
            try:
                spec = load_hel1os_spectra(d, detector=det, num=num)
                counts_spec = spec["counts"]
                if counts_spec.ndim == 2 and counts_spec.shape[0] > 0:
                    avg_spectrum = np.mean(counts_spec, axis=0)
                elif counts_spec.ndim == 1:
                    avg_spectrum = counts_spec
                else:
                    continue
                chans = spec.get("channel", np.arange(len(avg_spectrum)))
                if chans.ndim == 2:
                    chans = np.mean(chans, axis=0)
                if len(avg_spectrum) > 10 and np.sum(avg_spectrum > 0) > 5:
                    valid = avg_spectrum > 0
                    log_e = np.log(np.maximum(chans[valid].astype(float), 1.0))
                    log_c = np.log(avg_spectrum[valid])
                    if len(log_e) > 3:
                        A = np.vstack([log_e, np.ones_like(log_e)]).T
                        coeffs, *_ = np.linalg.lstsq(A, log_c, rcond=None)
                        pre[gamma_key] = float(-coeffs[0])
            except Exception:
                pass

        # Non-thermal parameters from combined spectrum
        for k in [
            "nonthermal_gamma",
            "nonthermal_ec",
            "nonthermal_n_nth",
            "thermal_fraction",
        ]:
            pre[k] = 0.0
        try:
            from bah2026.features.non_thermal import fit_combined_spectrum
            from bah2026.features.spectral_fitting import fit_temperature
            from bah2026.data.calibration import load_channel_energies

            energies = load_channel_energies()
            emin, emax = energies
            centroids = (emin + emax) / 2.0
            if pi_sum is not None and len(pi_sum) == 340:
                t_mk, em, chi2 = fit_temperature(pi_sum)
                pre["sxr_temperature_mk"] = t_mk
                pre["sxr_emission_measure"] = em
                pre["sxr_chi2_red"] = chi2
                # Use HEL1OS CdTe1 spectra (5-90 keV, 511 channels) for non-thermal
                try:
                    cdte_spec = load_hel1os_spectra(d, detector="cdte", num=1)
                    cdte_counts = cdte_spec["counts"]
                    cdte_chans = cdte_spec["channel"]
                    if cdte_counts.ndim == 2 and cdte_counts.shape[0] > 0:
                        avg_cdte = np.mean(cdte_counts, axis=0)
                        avg_cdte_chans = (
                            np.mean(cdte_chans, axis=0)
                            if cdte_chans.ndim == 2
                            else cdte_chans
                        )
                        # Convert channel numbers to keV: CdTe spans 5-90 keV over 511 channels
                        cdte_energies_kev = np.linspace(5.0, 90.0, len(avg_cdte))
                        # Use SoLEXS PI averaged spectrum
                        valid_pi = pi_sum > 0
                        if valid_pi.sum() > 10:
                            sep = fit_combined_spectrum(
                                centroids[valid_pi],
                                pi_sum[valid_pi],
                                cdte_energies_kev,
                                avg_cdte,
                                t_mk_init=max(t_mk, 5.0),
                            )
                            pre["nonthermal_gamma"] = sep.get("gamma", 0.0)
                            pre["nonthermal_ec"] = sep.get("ec", 0.0)
                            pre["nonthermal_n_nth"] = sep.get("n_nth", 0.0)
                            pre["thermal_fraction"] = sep.get("thermal_fraction", 0.0)
                except Exception:
                    pass
        except Exception:
            pre.setdefault("sxr_temperature_mk", 0.0)
            pre.setdefault("sxr_emission_measure", 0.0)
            pre.setdefault("sxr_chi2_red", 0.0)

        # CZT2 / CdTe2 aggregated features
        for k in [
            "czt2_total_mean",
            "czt2_total_max",
            "czt2_total_std",
            "cdte2_total_mean",
            "cdte2_total_max",
            "cdte2_total_std",
        ]:
            pre[k] = 0.0
        try:
            # CZT2 full band is index 9 (bands 5-9 = czt2)
            czt2_full = hxr4[:, 9]
            valid_czt2 = czt2_full[np.isfinite(czt2_full) & (czt2_full > 0)]
            if len(valid_czt2) > 0:
                pre["czt2_total_mean"] = float(np.mean(valid_czt2))
                pre["czt2_total_max"] = float(np.max(valid_czt2))
                pre["czt2_total_std"] = float(np.std(valid_czt2))
            # CdTe2 full band is index 19 (bands 15-19 = cdte2)
            cdte2_full = hxr4[:, 19]
            valid_cdte2 = cdte2_full[np.isfinite(cdte2_full) & (cdte2_full > 0)]
            if len(valid_cdte2) > 0:
                pre["cdte2_total_mean"] = float(np.mean(valid_cdte2))
                pre["cdte2_total_max"] = float(np.max(valid_cdte2))
                pre["cdte2_total_std"] = float(np.std(valid_cdte2))
        except Exception:
            pass

        # GOES features from netCDF files
        for k in ["goes_xrsb_flux", "goes_xrsa_flux", "goes_xrsa_xrsb_ratio"]:
            pre[k] = 0.0
        goes_xrsb_arr = None
        goes_xrsa_arr = None
        try:
            from pathlib import Path

            gdir = Path(__file__).resolve().parents[3] / "data" / "external" / "goes"
            for nc_file in gdir.glob(f"*g16_d{d.strftime('%Y%m%d')}_v*.nc"):
                from netCDF4 import Dataset as NCDataset

                with NCDataset(str(nc_file), "r") as nc:
                    gf_b = nc.variables["xrsb_flux"][:].astype(np.float64)
                    gf_a = nc.variables["xrsa_flux"][:].astype(np.float64)
                    gf_b = np.where(gf_b < 0, np.nan, gf_b)
                    gf_a = np.where(gf_a < 0, np.nan, gf_a)
                    if len(gf_b) > 10:
                        goes_xrsb_arr = gf_b
                        goes_xrsa_arr = gf_a
                        pre["goes_xrsb_flux"] = float(np.nanmean(gf_b))
                        pre["goes_xrsa_flux"] = float(np.nanmean(gf_a))
                        if pre["goes_xrsb_flux"] > 0:
                            pre["goes_xrsa_xrsb_ratio"] = (
                                pre["goes_xrsa_flux"] / pre["goes_xrsb_flux"]
                            )
                break
        except Exception:
            pass

        # Granger causality: HXR → SXR (raw, no threshold)
        for k in ["neupert_granger_improvement", "neupert_best_lag"]:
            pre[k] = 0.0
        try:
            from sklearn.linear_model import RidgeCV
            from sklearn.model_selection import TimeSeriesSplit
            from sklearn.metrics import r2_score

            hxr_full_1d = hxr4[:, 4]
            valid_mask = np.isfinite(counts) & np.isfinite(hxr_full_1d)
            if valid_mask.sum() > 500:
                hxr_v = hxr_full_1d[valid_mask]
                sxr_v = counts[valid_mask]
                n_gc = len(hxr_v)
                best_imp, best_lag_val = 0.0, 0
                tscv = TimeSeriesSplit(n_splits=3)
                for lag in [1, 5, 10, 20, 30, 60]:
                    if lag > n_gc // 4:
                        continue
                    X_r = np.column_stack(
                        [sxr_v[lag - j - 1 : n_gc - j - 1] for j in range(lag)]
                    )
                    X_f = np.column_stack(
                        [sxr_v[lag - j - 1 : n_gc - j - 1] for j in range(lag)]
                        + [hxr_v[lag - j - 1 : n_gc - j - 1] for j in range(lag)]
                    )
                    y_gc = sxr_v[lag:]
                    if len(y_gc) < lag * 2:
                        continue
                    r2_r_sum, r2_f_sum, nf = 0.0, 0.0, 0
                    for tr, te in tscv.split(X_r):
                        if len(te) < 5:
                            continue
                        try:
                            m_r = RidgeCV(alphas=[0.1, 1.0, 10.0]).fit(
                                X_r[tr], y_gc[tr]
                            )
                            m_f = RidgeCV(alphas=[0.1, 1.0, 10.0]).fit(
                                X_f[tr], y_gc[tr]
                            )
                            r2_r_sum += r2_score(y_gc[te], m_r.predict(X_r[te]))
                            r2_f_sum += r2_score(y_gc[te], m_f.predict(X_f[te]))
                            nf += 1
                        except Exception:
                            continue
                    if nf > 0:
                        imp = r2_f_sum / nf - r2_r_sum / nf
                        if imp > best_imp:
                            best_imp = imp
                            best_lag_val = lag
                pre["neupert_granger_improvement"] = best_imp
                pre["neupert_best_lag"] = float(best_lag_val)
        except Exception:
            pass

        # Mediation analysis: HXR(40-60) → CdTe(20-30) → SXR
        pre["max_mediation_proportion"] = 0.0
        try:
            from bah2026.features.causal_network import mediation_analysis

            treatment = hxr4[:, 1]  # CZT 40-60 keV
            mediator = hxr4[:, 6]  # CdTe 20-30 keV
            outcome = counts
            valid_mask = (
                np.isfinite(treatment) & np.isfinite(mediator) & np.isfinite(outcome)
            )
            if valid_mask.sum() > 50:
                ma = mediation_analysis(
                    treatment[valid_mask], mediator[valid_mask], outcome[valid_mask]
                )
                pre["max_mediation_proportion"] = ma.get("mediation_proportion", 0.0)
        except Exception:
            pass

        # ── Advanced features (GOES time-series, per-window spectral, wavelet) ──
        try:
            from bah2026.features.advanced_features import (
                extract_goes_timeseries_features,
                extract_per_window_spectral,
                extract_wavelet_scalogram_features,
            )

            goes_feats = extract_goes_timeseries_features(goes_xrsb_arr, goes_xrsa_arr)
            pre.update(goes_feats)

            # Compute goes_flare_history_24h: count C-class+ peaks in GOES
            if goes_xrsb_arr is not None:
                valid_goes = goes_xrsb_arr[np.isfinite(goes_xrsb_arr)]
                c_threshold = 1e-6  # C1.0 class = 1e-6 W/m^2
                flare_mask = valid_goes > c_threshold
                pre["goes_flare_history_24h"] = float(np.sum(flare_mask))
                peak_val = np.nanmax(valid_goes) if len(valid_goes) > 0 else 0.0
                if peak_val > 0 and len(valid_goes) > 0:
                    pre["goes_xrsb_prev_peak_ratio"] = float(valid_goes[-1] / peak_val)

            hxr_full_1d = hxr4[:, 4] if hxr4.shape[1] > 4 else hxr4[:, 0]
            try:
                czt_spec = load_hel1os_spectra(d, detector="czt", num=1)
                czt_spec_data = czt_spec["counts"]
            except Exception:
                czt_spec_data = None
            try:
                cdte_spec = load_hel1os_spectra(d, detector="cdte", num=1)
                cdte_spec_data = cdte_spec["counts"]
            except Exception:
                cdte_spec_data = None

            prev_gamma = pre.get("hxr_spectral_index_gamma", 0.0)
            pw_spec = extract_per_window_spectral(
                pi_sum if pi_sum is not None else None,
                czt_spec_data,
                cdte_spec_data,
                channel_energies=None,
                prev_gamma=prev_gamma,
            )
            pre.update(pw_spec)

            # Fix nonthermal_fraction_window: override with our precomputed value
            if pw_spec.get("nonthermal_fraction_window", 0.0) == 0.0:
                tf_clipped = float(np.clip(pre.get("thermal_fraction", 0.0), 0.0, 1.0))
                pre["nonthermal_fraction_window"] = float(max(0.0, 1.0 - tf_clipped))

            wavelet_feats = extract_wavelet_scalogram_features(
                counts, dt=1.0, hxr_signal=hxr_full_1d.astype(np.float64)
            )
            pre.update(wavelet_feats)
        except Exception:
            pass

        n_w = (len(counts) - LOOKBACK) // STEP + 1
        et = event_times.get(str(d), [])
        ys = np.zeros(n_w, dtype=np.int64)
        for wi in range(n_w):
            t = time_s[min(wi * STEP + LOOKBACK, len(time_s) - 1)]
            ys[wi] = 1 if any(0 < e - t <= 1800 for e in et) else 0
        return {
            "ok": True,
            "sxr": counts,
            "hxr": hxr4,
            "pi": pi_win,
            "y": ys,
            "n_w": n_w,
            "pre": pre,
        }
    except Exception as e:
        return {"ok": False, "d": str(d), "error": str(e)}


def main():
    print(
        f"[{time.strftime('%H:%M:%S')}] GPU extraction: {N_WORKERS} workers, {CHUNK} day chunks",
        flush=True,
    )

    # Resume handling
    processed = set()
    if PROCESSED_LOG.exists():
        with open(PROCESSED_LOG) as f:
            processed = {l.strip() for l in f if l.strip()}
        print(f"  Resume: {len(processed)} days already processed", flush=True)
        if (HDF5_DIR / "X_features.npy").exists():
            import numpy as _np

            existing_X = _np.load(HDF5_DIR / "X_features.npy")
            existing_y = _np.load(HDF5_DIR / "y_labels.npy")
            chunk_X, chunk_y = [existing_X], [existing_y]
            total_w = len(existing_y)
            print(
                f"  Loaded existing: X={existing_X.shape}, y={existing_y.shape}",
                flush=True,
            )
        else:
            chunk_X, chunk_y = [], []
            total_w = 0
    else:
        chunk_X, chunk_y = [], []
        total_w = 0

    csv = CATALOGS_DIR / "nowcast_catalogue.csv"
    df = pd.read_csv(csv) if csv.exists() else pd.DataFrame()
    event_times = {}
    if "date" in df.columns and "peak_time" in df.columns:
        for _, row in df.iterrows():
            event_times.setdefault(str(row["date"]), []).append(row["peak_time"])
    days = discover_combined_days()
    days = [d for d in days if str(d) not in processed]
    print(f"  Remaining: {len(days)} days", flush=True)
    if not days:
        print("  All days already processed!", flush=True)
        return

    from bah2026.features.gpu_features import (
        _batch_stats,
        _batch_acf,
        _batch_spectral_entropy,
        _batch_derivative_features,
        _batch_multiscale,
        _batch_neupert,
        _batch_hxr_features,
        _batch_pi_channel_features,
        FEATURE_AUTOCORR_LAGS,
        get_canonical_feature_names,
    )

    t_pipeline = time.time()

    for ci in range(0, len(days), CHUNK):
        chunk_days = days[ci : ci + CHUNK]
        cnum = ci // CHUNK + 1
        t_chunk = time.time()
        print(
            f"\n[{time.strftime('%H:%M:%S')}] Chunk {cnum}: {len(chunk_days)} days",
            flush=True,
        )

        with Pool(N_WORKERS) as pool:
            results = pool.map(_worker_load, [(d, event_times) for d in chunk_days])
        ok_r = [r for r in results if r.get("ok")]
        t_load = time.time() - t_chunk
        print(f"  Load: {len(ok_r)}/{len(chunk_days)} in {t_load:.1f}s", flush=True)
        if not ok_r:
            continue

        total_nw = sum(r["n_w"] for r in ok_r)
        sxr_np = np.zeros((total_nw, LOOKBACK), dtype=np.float32)
        hxr_np = np.zeros((total_nw, LOOKBACK, 20), dtype=np.float32)
        pi_np = np.zeros((total_nw, 340), dtype=np.float32)
        y_np = np.zeros(total_nw, dtype=np.int64)
        w_idx = 0
        day_ranges = []
        for r in ok_r:
            day_start = w_idx
            for wi in range(r["n_w"]):
                s = wi * STEP
                sxr_np[w_idx] = r["sxr"][s : s + LOOKBACK]
                hxr_np[w_idx] = r["hxr"][s : s + LOOKBACK]
                if r["pi"] is not None and wi < len(r["pi"]):
                    pi_np[w_idx] = r["pi"][wi]
                y_np[w_idx] = r["y"][wi]
                w_idx += 1
            day_ranges.append((day_start, w_idx, r))

        print(f"  GPU: {total_nw} windows...", flush=True)
        sxr_g = torch.from_numpy(sxr_np).to("cuda")
        hxr_g = torch.from_numpy(hxr_np).to("cuda")
        pi_g = torch.from_numpy(pi_np).to("cuda")
        torch.cuda.synchronize()
        vram = torch.cuda.memory_allocated() / 1024**3
        print(f"    VRAM={vram:.2f}GB", flush=True)

        feats = {}
        feats.update(_batch_stats(sxr_g))
        feats.update(_batch_acf(sxr_g, FEATURE_AUTOCORR_LAGS))
        feats.update(_batch_spectral_entropy(sxr_g))
        feats.update(_batch_derivative_features(sxr_g, hxr_g))
        feats.update(_batch_multiscale(sxr_g, hxr_g))
        feats.update(_batch_neupert(sxr_g, hxr_g))
        feats.update(_batch_hxr_features(hxr_g))
        feats.update(_batch_pi_channel_features(pi_g))
        torch.cuda.synchronize()
        t_comp = time.time() - t_chunk

        # Inject day-level precomputed features (broadcast to all windows of each day)
        canonical = get_canonical_feature_names()
        n_feat = len(canonical)
        row = torch.zeros(total_nw, n_feat, device="cuda", dtype=torch.float32)
        for fi, fn in enumerate(canonical):
            if (
                fn in feats
                and isinstance(feats[fn], torch.Tensor)
                and feats[fn].shape[0] == total_nw
            ):
                row[:, fi] = feats[fn]

        # Broadcast precomputed day-level features to all windows of each day
        for day_start, day_end, r in day_ranges:
            pre = r.get("pre", {})
            n_day_w = day_end - day_start
            for fi, fn in enumerate(canonical):
                if fn in pre and (row[day_start:day_end, fi] == 0).all():
                    val = float(pre[fn])
                    val = max(-1e6, min(1e6, val)) if np.isfinite(val) else 0.0
                    row[day_start:day_end, fi] = val

        # window_len = LOOKBACK for every window
        wl_idx = None
        for fi, fn in enumerate(canonical):
            if fn == "window_len":
                wl_idx = fi
                break
        if wl_idx is not None:
            row[:, wl_idx] = float(LOOKBACK)

        torch.cuda.synchronize()
        X_c = row.cpu().numpy().astype(np.float32)

        # CPU features: info-theory, QPP, causal — computed per day, broadcast
        t_cpu_feat = time.time()
        try:
            from bah2026.features.information_theory import (
                transfer_entropy,
                mutual_information,
                sample_entropy,
                lagged_cross_correlation,
            )
            from bah2026.features.qpp import detect_qpp
            from bah2026.features.causal_network import extract_causal_network_features

            it_keys = [
                "transfer_entropy_hxr_to_sxr",
                "mutual_information_sxr_hxr",
                "sample_entropy_sxr",
                "sample_entropy_hxr",
                "lagged_cross_corr",
                "lagged_cross_corr_lag",
            ]
            qpp_keys = [
                "qpp_detected",
                "qpp_period",
                "qpp_amplitude",
                "qpp_significance",
            ]
            causal_keys = [
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
            ]

            it_col = {fn: canonical.index(fn) for fn in it_keys if fn in canonical}
            qpp_col = {fn: canonical.index(fn) for fn in qpp_keys if fn in canonical}
            causal_col = {
                fn: canonical.index(fn) for fn in causal_keys if fn in canonical
            }

            for day_start, day_end, r in day_ranges:
                n_day_w = day_end - day_start
                if n_day_w < 1:
                    continue
                # Use the full-day SXR and HXR arrays
                sxr_day = r["sxr"]
                hxr_day = r["hxr"]
                hxr_full_1d = hxr_day[:, 4] if hxr_day.shape[1] > 4 else hxr_day[:, 0]

                it_vals = {k: 0.0 for k in it_keys}
                qpp_vals = {k: 0.0 for k in qpp_keys}
                causal_vals = {k: 0.0 for k in causal_keys}

                # Info-theory (downsampled for speed)
                try:
                    ds = 10
                    sxr_ds = sxr_day[::ds]
                    hxr_ds = hxr_full_1d[::ds]
                    valid_sxr = sxr_ds[np.isfinite(sxr_ds)]
                    valid_hxr = hxr_ds[np.isfinite(hxr_ds)]
                    ml = min(len(valid_sxr), len(valid_hxr))
                    if ml > 20:
                        it_vals["transfer_entropy_hxr_to_sxr"] = float(
                            transfer_entropy(
                                valid_hxr[:ml], valid_sxr[:ml], k=1, bins=8
                            )
                        )
                        it_vals["mutual_information_sxr_hxr"] = float(
                            mutual_information(valid_sxr[:ml], valid_hxr[:ml], bins=8)
                        )
                    if ml > 50:
                        it_vals["sample_entropy_sxr"] = float(
                            sample_entropy(valid_sxr[: min(ml, 500)], m=2, r_factor=0.2)
                        )
                        it_vals["sample_entropy_hxr"] = float(
                            sample_entropy(valid_hxr[: min(ml, 500)], m=2, r_factor=0.2)
                        )
                    if ml > 200:
                        corr, lag = lagged_cross_correlation(
                            valid_hxr[:ml], valid_sxr[:ml], max_lag=100
                        )
                        it_vals["lagged_cross_corr"] = float(corr)
                        it_vals["lagged_cross_corr_lag"] = float(lag)
                except Exception:
                    pass

                # QPP (on full-band HXR, downsampled)
                try:
                    hxr_qpp = hxr_full_1d[np.isfinite(hxr_full_1d)]
                    if len(hxr_qpp) > 100:
                        qpp = detect_qpp(hxr_qpp, dt=1.0, min_period=10, max_period=300)
                        qpp_vals["qpp_detected"] = 1.0 if qpp["detected"] else 0.0
                        qpp_vals["qpp_period"] = float(qpp["period"])
                        qpp_vals["qpp_amplitude"] = float(qpp["amplitude"])
                        qpp_vals["qpp_significance"] = float(qpp["significance"])
                except Exception:
                    pass

                # Causal network features
                try:
                    valid_mask = np.isfinite(sxr_day) & np.isfinite(hxr_full_1d)
                    if valid_mask.sum() > 200:
                        ds_c = 10
                        band_data = {
                            "SXR": sxr_day[valid_mask][::ds_c],
                            "CZT20": hxr_day[valid_mask, 0][::ds_c],
                            "CZT40": hxr_day[valid_mask, 1][::ds_c],
                            "CZT60": hxr_day[valid_mask, 2][::ds_c],
                            "CZT80": hxr_day[valid_mask, 3][::ds_c],
                            "CZT160": hxr_day[valid_mask, 4][::ds_c],
                        }
                        if hxr_day.shape[1] > 5:
                            band_data["CdTe5"] = hxr_day[valid_mask, 5][::ds_c]
                            band_data["CdTe20"] = hxr_day[valid_mask, 6][::ds_c]
                        cn = extract_causal_network_features(band_data, max_lag=20)
                        for k in causal_keys:
                            if k in cn:
                                causal_vals[k] = cn[k]
                except Exception:
                    pass

                # Broadcast to all windows of this day
                for fi, fn in enumerate(canonical):
                    if fn in it_col:
                        row[day_start:day_end, fi] = it_vals[fn]
                    elif fn in qpp_col:
                        row[day_start:day_end, fi] = qpp_vals[fn]
                    elif fn in causal_col:
                        row[day_start:day_end, fi] = causal_vals[fn]

            print(f"    CPU features: {time.time() - t_cpu_feat:.1f}s", flush=True)
        except Exception as e:
            print(f"    CPU features failed: {e}", flush=True)

        X_c = row.cpu().numpy().astype(np.float32)
        X_c = np.nan_to_num(X_c, nan=0.0, posinf=0.0, neginf=0.0)
        chunk_X.append(X_c)
        chunk_y.append(y_np)
        total_w += total_nw
        print(
            f"  Done: X={X_c.shape}, compute={t_comp - t_load:.1f}s, total={time.time() - t_chunk:.1f}s",
            flush=True,
        )

        with open(PROCESSED_LOG, "a") as f:
            for d in chunk_days:
                f.write(f"{d}\n")

        if chunk_X:
            X_tmp = np.vstack(chunk_X)
            y_tmp = np.concatenate(chunk_y)
            np.save(HDF5_DIR / "X_features.npy", X_tmp)
            np.save(HDF5_DIR / "y_labels.npy", y_tmp)
            print(f"  Saved checkpoint: X={X_tmp.shape}", flush=True)

        del sxr_g, hxr_g, pi_g, row, feats
        torch.cuda.empty_cache()

    X = np.vstack(chunk_X)
    y = np.concatenate(chunk_y)
    print(
        f"\n[{time.strftime('%H:%M:%S')}] X={X.shape}, y={y.shape}, pos={y.sum()} ({100 * y.mean():.2f}%) in {time.time() - t_pipeline:.0f}s",
        flush=True,
    )
    np.save(HDF5_DIR / "X_features.npy", X)
    np.save(HDF5_DIR / "y_labels.npy", y)
    (HDF5_DIR / "feature_names.json").write_text(
        json.dumps(get_canonical_feature_names())
    )
    print(f"Saved to {HDF5_DIR}", flush=True)


if __name__ == "__main__":
    main()
