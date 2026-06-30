"""GPU-accelerated batch feature extraction for solar flare forecasting.

Processes all 276 windows of a day in parallel on the A100 GPU.
Target: ~2-3 seconds per day (vs 5+ minutes on CPU).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import skew, kurtosis as sp_kurtosis
from bah2026.config import (
    FEATURE_AUTOCORR_LAGS,
    FEATURE_PERCENTILES,
    FEATURE_SPECTRAL_ENTROPY_NPERSEG,
)
from bah2026.features.engineering import (
    get_canonical_feature_names,
    pad_features_to_canonical,
)

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_DTYPE = torch.float32


def _to_gpu(x: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.asarray(x, dtype=np.float32)).to(_DEVICE)


def _batch_stats(windows: torch.Tensor) -> dict[str, torch.Tensor]:
    B, W = windows.shape
    feats = {}
    feats["sxr_mean"] = windows.mean(dim=1)
    feats["sxr_std"] = windows.std(dim=1)
    feats["sxr_max"] = windows.max(dim=1).values
    feats["sxr_min"] = windows.min(dim=1).values
    feats["sxr_median"] = windows.median(dim=1).values
    feats["sxr_range"] = feats["sxr_max"] - feats["sxr_min"]
    feats["sxr_cv"] = feats["sxr_std"] / feats["sxr_mean"].clamp(min=1e-6)

    diff = windows[:, 1:] - windows[:, :-1]
    feats["sxr_abs_slope"] = diff.abs().mean(dim=1)
    pos_diff = diff.clamp(min=0)
    neg_diff = diff.clamp(max=0)
    feats["sxr_rise_rate"] = pos_diff.sum(dim=1) / (pos_diff > 0).float().sum(
        dim=1
    ).clamp(min=1)
    feats["sxr_fall_rate"] = neg_diff.sum(dim=1) / (neg_diff < 0).float().sum(
        dim=1
    ).clamp(min=1)

    q75 = torch.quantile(windows, 0.75, dim=1)
    q25 = torch.quantile(windows, 0.25, dim=1)
    feats["sxr_iqr"] = q75 - q25

    for pct in FEATURE_PERCENTILES:
        feats[f"sxr_p{pct}"] = torch.quantile(windows, pct / 100.0, dim=1)

    diff_sq = diff**2
    feats["d2sxr_dt2_mean"] = (
        (diff_sq[:, 1:] - diff_sq[:, :-1]).mean(dim=1)
        if W > 3
        else torch.zeros(B, device=_DEVICE)
    )
    feats["d2sxr_dt2_std"] = (
        (diff_sq[:, 1:] - diff_sq[:, :-1]).std(dim=1)
        if W > 3
        else torch.zeros(B, device=_DEVICE)
    )

    return feats


def _batch_acf(windows: torch.Tensor, lags: list[int]) -> dict[str, torch.Tensor]:
    B, W = windows.shape
    feats = {}
    mean = windows.mean(dim=1, keepdim=True)
    var = windows.var(dim=1, keepdim=True).clamp(min=1e-10)
    centered = windows - mean
    for lag in lags:
        if lag < W:
            corr = (centered[:, lag:] * centered[:, : W - lag]).sum(dim=1) / (
                var.squeeze(1) * (W - lag)
            )
            feats[f"sxr_acf_{lag}s"] = corr
        else:
            feats[f"sxr_acf_{lag}s"] = torch.zeros(B, device=_DEVICE)
    return feats


def _batch_spectral_entropy(windows: torch.Tensor) -> dict[str, torch.Tensor]:
    B, W = windows.shape
    nperseg = min(FEATURE_SPECTRAL_ENTROPY_NPERSEG, W // 2)
    feats = {}
    try:
        segs = windows.unfold(1, nperseg, nperseg)
        psd = torch.fft.rfft(segs, dim=-1).abs() ** 2
        total = psd.sum(dim=-1, keepdim=True).clamp(min=1e-10)
        p = psd / total
        entropy = -(p * (p + 1e-30).log()).sum(dim=-1)
        feats["sxr_spec_entropy"] = entropy.mean(dim=1)
        feats["sxr_peak_freq"] = psd.argmax(dim=-1).float().mean(dim=1) / nperseg
    except Exception:
        feats["sxr_spec_entropy"] = torch.zeros(B, device=_DEVICE)
        feats["sxr_peak_freq"] = torch.zeros(B, device=_DEVICE)
    return feats


def _batch_hxr_features(hxr_windows: torch.Tensor) -> dict[str, torch.Tensor]:
    B, W, nb = hxr_windows.shape
    feats = {}
    nb_use = min(nb, 10)
    for b in range(nb_use):
        band = hxr_windows[:, :, b]
        feats[f"hxr_b{b}_mean"] = band.mean(dim=1)
        feats[f"hxr_b{b}_std"] = band.std(dim=1)
        feats[f"hxr_b{b}_max"] = band.max(dim=1).values

    if nb >= 2:
        lo, hi = hxr_windows[:, :, 0], hxr_windows[:, :, 1]
        hr = hi / lo.clamp(min=1e-6)
        feats["hxr_hardness_ratio"] = hr.mean(dim=1)
    if nb >= 5:
        tot = hxr_windows[:, :, :5].sum(dim=-1)
        feats["hxr_total_mean"] = tot.mean(dim=1)
    if nb >= 5:
        hxr_sum = hxr_windows[:, :, :5].sum(dim=-1)
        feats["soft_hard_ratio"] = (
            hxr_windows[:, :, 0] / hxr_sum.clamp(min=1e-6)
        ).mean(dim=1)

    if nb >= 10:
        thermal_lo = hxr_windows[:, :, 5]
        czt_full = hxr_windows[:, :, 4]
        feats["cdte_thermal_ratio"] = (thermal_lo / czt_full.clamp(min=1e-6)).mean(
            dim=1
        )
        bd_lo = hxr_windows[:, :, 5]
        bd_hi = hxr_windows[:, :, 6]
        feats["cdte_boundary_ratio"] = (bd_hi / bd_lo.clamp(min=1e-6)).mean(dim=1)

    if nb >= 2:
        hr_series = hxr_windows[:, :, 1] / hxr_windows[:, :, 0].clamp(min=1e-6)
        feats["hardness_ratio_mean"] = hr_series.mean(dim=1)
        feats["hardness_ratio_std"] = hr_series.std(dim=1)
        t_hr = torch.arange(W, device=_DEVICE, dtype=_DTYPE).unsqueeze(0).expand(B, -1)
        t_mean = t_hr.mean(dim=1, keepdim=True)
        t_var = t_hr.var(dim=1, keepdim=True).clamp(min=1e-10)
        feats["hardness_ratio_slope"] = (
            (hr_series - hr_series.mean(dim=1, keepdim=True)) * (t_hr - t_mean)
        ).sum(dim=1) / (t_var.squeeze(1) * W)

    return feats


def _batch_derivative_features(
    sxr_windows: torch.Tensor, hxr_windows: torch.Tensor | None
) -> dict[str, torch.Tensor]:
    B, W = sxr_windows.shape
    feats = {}
    dsxr = sxr_windows[:, 1:] - sxr_windows[:, :-1]
    feats["dsxr_dt_mean"] = dsxr.mean(dim=1)
    feats["dsxr_dt_std"] = dsxr.std(dim=1)
    feats["dsxr_dt_max"] = dsxr.max(dim=1).values
    feats["dsxr_dt_min"] = dsxr.min(dim=1).values

    if hxr_windows is not None and hxr_windows.shape[2] >= 5:
        hxr_full = hxr_windows[:, :, 4]
        dhxr = hxr_full[:, 1:] - hxr_full[:, :-1]
        feats["dhxr_dt_mean"] = dhxr.mean(dim=1)
        feats["dhxr_dt_std"] = dhxr.std(dim=1)
        feats["dhxr_dt_max"] = dhxr.max(dim=1).values
        hxr_lo = hxr_windows[:, :, 0]
        hxr_hi = hxr_windows[:, :, 1]
        hr = hxr_hi / hxr_lo.clamp(min=1e-6)
        dhr = hr[:, 1:] - hr[:, :-1]
        feats["dhr_dt_mean"] = dhr.mean(dim=1)
        feats["dhr_dt_max"] = dhr.max(dim=1).values
        feats["dsxr_dhxr_ratio_mean"] = dsxr.mean(dim=1) / hxr_full.mean(dim=1).clamp(
            min=1e-6
        )
    else:
        for k in [
            "dhxr_dt_mean",
            "dhxr_dt_std",
            "dhxr_dt_max",
            "dhr_dt_mean",
            "dhr_dt_max",
            "dsxr_dhxr_ratio_mean",
        ]:
            feats[k] = torch.zeros(B, device=_DEVICE)
    return feats


def _batch_multiscale(
    sxr_windows: torch.Tensor, hxr_windows: torch.Tensor | None
) -> dict[str, torch.Tensor]:
    B, W = sxr_windows.shape
    feats = {}
    scales = [(300, "5m"), (900, "15m"), (1800, "30m")]
    for sec, label in scales:
        n = min(sec, W)
        sxr_tail = sxr_windows[:, W - n :]
        feats[f"sxr_mean_{label}"] = sxr_tail.mean(dim=1)
        feats[f"sxr_std_{label}"] = sxr_tail.std(dim=1)
        feats[f"sxr_max_{label}"] = sxr_tail.max(dim=1).values
        if hxr_windows is not None and hxr_windows.shape[2] >= 5:
            hxr_full = hxr_windows[:, :, 4]
            hxr_tail = hxr_full[:, W - n : W]
            if hxr_tail.shape[1] > 0:
                feats[f"hxr_mean_{label}"] = hxr_tail.mean(dim=1)
                feats[f"hxr_std_{label}"] = hxr_tail.std(dim=1)
                feats[f"hxr_max_{label}"] = hxr_tail.max(dim=1).values
            else:
                feats[f"hxr_mean_{label}"] = torch.zeros(B, device=_DEVICE)
                feats[f"hxr_std_{label}"] = torch.zeros(B, device=_DEVICE)
                feats[f"hxr_max_{label}"] = torch.zeros(B, device=_DEVICE)
        else:
            for s in ["hxr_mean_", "hxr_std_", "hxr_max_"]:
                feats[f"{s}{label}"] = torch.zeros(B, device=_DEVICE)

    sxr_mean_full = sxr_windows.mean(dim=1).clamp(min=1e-6)
    sxr_mean_5m = feats.get("sxr_mean_5m", sxr_mean_full)
    sxr_mean_15m = feats.get("sxr_mean_15m", sxr_mean_full)
    feats["sxr_5m_to_60m_ratio"] = sxr_mean_5m / sxr_mean_full
    feats["sxr_acceleration_trend"] = (sxr_mean_5m - sxr_mean_15m) / sxr_mean_15m.clamp(
        min=1e-6
    )

    t = torch.arange(W, device=_DEVICE, dtype=_DTYPE)
    t_15m = t[W - 900 : W] if W >= 900 else t
    sxr_15m = sxr_windows[:, W - len(t_15m) : W]
    t_mean = t_15m.mean()
    t_var = t_15m.var().clamp(min=1e-10)
    feats["sxr_15m_slope"] = (
        (sxr_15m - sxr_15m.mean(dim=1, keepdim=True)) * (t_15m - t_mean)
    ).sum(dim=1) / (t_var * len(t_15m))

    if hxr_windows is not None and hxr_windows.shape[2] >= 5:
        hxr_mean_full = hxr_windows[:, :, 4].mean(dim=1).clamp(min=1e-6)
        hxr_mean_5m = feats.get("hxr_mean_5m", hxr_mean_full)
        hxr_mean_15m = feats.get("hxr_mean_15m", hxr_mean_full)
        feats["hxr_5m_to_60m_ratio"] = hxr_mean_5m / hxr_mean_full
        feats["hxr_acceleration_trend"] = (
            hxr_mean_5m - hxr_mean_15m
        ) / hxr_mean_15m.clamp(min=1e-6)
        hxr_15m = hxr_windows[:, W - len(t_15m) : W, 4]
        feats["hxr_15m_slope"] = (
            (hxr_15m - hxr_15m.mean(dim=1, keepdim=True)) * (t_15m - t_mean)
        ).sum(dim=1) / (t_var * len(t_15m))
    else:
        for k in ["hxr_5m_to_60m_ratio", "hxr_acceleration_trend", "hxr_15m_slope"]:
            feats[k] = torch.zeros(B, device=_DEVICE)

    return feats


def _batch_neupert(
    sxr_windows: torch.Tensor, hxr_windows: torch.Tensor | None
) -> dict[str, torch.Tensor]:
    B, W = sxr_windows.shape
    feats = {}
    dsxr = sxr_windows[:, 1:] - sxr_windows[:, :-1]
    if hxr_windows is not None and hxr_windows.shape[2] >= 5:
        hxr_full = hxr_windows[:, :, 4][:, 1:]
        ws = 300
        n_win = max(1, (W - 1 - ws) // 60)
        rhos = []
        for start in range(0, W - 1 - ws, 60):
            s = dsxr[:, start : start + ws]
            h = hxr_full[:, start : start + ws]
            s_std = s.std(dim=1).clamp(min=1e-10)
            h_std = h.std(dim=1).clamp(min=1e-10)
            r = (
                (s - s.mean(dim=1, keepdim=True)) * (h - h.mean(dim=1, keepdim=True))
            ).mean(dim=1) / (s_std * h_std)
            rhos.append(r)
        if rhos:
            rhos_t = torch.stack(rhos, dim=1)
            feats["neupert_rho_mean"] = rhos_t.mean(dim=1)
            feats["neupert_rho_std"] = rhos_t.std(dim=1)
        else:
            feats["neupert_rho_mean"] = torch.zeros(B, device=_DEVICE)
            feats["neupert_rho_std"] = torch.zeros(B, device=_DEVICE)
    else:
        feats["neupert_rho_mean"] = torch.zeros(B, device=_DEVICE)
        feats["neupert_rho_std"] = torch.zeros(B, device=_DEVICE)
    return feats


def _batch_qpp(hxr_windows: torch.Tensor | None) -> dict[str, torch.Tensor]:
    B, W = hxr_windows.shape if hxr_windows is not None else (0, 0)
    feats = {
        "qpp_detected": torch.zeros(B, device=_DEVICE),
        "qpp_period": torch.zeros(B, device=_DEVICE),
        "qpp_amplitude": torch.zeros(B, device=_DEVICE),
        "qpp_significance": torch.zeros(B, device=_DEVICE),
    }
    if hxr_windows is None or B == 0:
        return feats
    try:
        from bah2026.features.qpp import detect_qpp

        for i in range(B):
            sig = (
                hxr_windows[i, :, 4].cpu().numpy()
                if hxr_windows.shape[2] > 4
                else hxr_windows[i, :, 0].cpu().numpy()
            )
            result = detect_qpp(sig, dt=1.0, min_period=10, max_period=300)
            feats["qpp_detected"][i] = 1.0 if result["detected"] else 0.0
            feats["qpp_period"][i] = result["period"]
            feats["qpp_amplitude"][i] = result["amplitude"]
            feats["qpp_significance"][i] = result["significance"]
    except Exception:
        pass
    return feats


def _batch_causal(
    sxr_windows: torch.Tensor, hxr_windows: torch.Tensor | None
) -> dict[str, torch.Tensor]:
    B, W = sxr_windows.shape
    feats = {}
    keys = [
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
    for k in keys:
        feats[k] = torch.zeros(B, device=_DEVICE)

    if hxr_windows is None or hxr_windows.shape[2] < 5:
        return feats

    ds = 10
    sxr_ds = sxr_windows[:, ::ds]
    hxr_ds = hxr_windows[:, ::ds, :]
    W_ds = sxr_ds.shape[1]

    try:
        from bah2026.features.causal_network import extract_causal_network_features

        for i in range(B):
            band_data = {
                "SXR": sxr_ds[i].cpu().numpy(),
                "CZT20": hxr_ds[i, :, 0].cpu().numpy(),
                "CZT40": hxr_ds[i, :, 1].cpu().numpy(),
                "CZT60": hxr_ds[i, :, 2].cpu().numpy(),
                "CZT80": hxr_ds[i, :, 3].cpu().numpy(),
                "CZT160": hxr_ds[i, :, 4].cpu().numpy(),
            }
            if hxr_windows.shape[2] > 5:
                band_data["CdTe5"] = hxr_ds[i, :, 5].cpu().numpy()
                band_data["CdTe20"] = hxr_ds[i, :, 6].cpu().numpy()
            result = extract_causal_network_features(band_data, max_lag=20)
            for k in keys:
                if k in result:
                    feats[k][i] = result[k]
    except Exception:
        pass
    return feats


def _batch_info_theory(
    sxr_windows: torch.Tensor, hxr_windows: torch.Tensor | None
) -> dict[str, torch.Tensor]:
    B, W = sxr_windows.shape
    feats = {}
    keys = [
        "transfer_entropy_hxr_to_sxr",
        "mutual_information_sxr_hxr",
        "sample_entropy_sxr",
        "sample_entropy_hxr",
        "lagged_cross_corr",
        "lagged_cross_corr_lag",
    ]
    for k in keys:
        feats[k] = torch.zeros(B, device=_DEVICE)

    try:
        from bah2026.features.information_theory import (
            transfer_entropy,
            mutual_information,
            sample_entropy,
            lagged_cross_correlation,
        )

        ds = 10
        sxr_ds = sxr_windows[:, ::ds].cpu().numpy()
        hxr_full = None
        hxr_ds = None
        if hxr_windows is not None and hxr_windows.shape[2] >= 5:
            hxr_full = hxr_windows[:, :, 4].cpu().numpy()
            hxr_ds = hxr_full[:, ::ds]
        for i in range(B):
            sxr_i = sxr_ds[i]
            if hxr_ds is not None:
                hxr_i = hxr_ds[i]
                if len(sxr_i) > 20:
                    feats["transfer_entropy_hxr_to_sxr"][i] = transfer_entropy(
                        hxr_i, sxr_i, k=1, bins=8
                    )
                    feats["mutual_information_sxr_hxr"][i] = mutual_information(
                        sxr_i, hxr_i, bins=8
                    )
                if len(sxr_i) > 50:
                    feats["sample_entropy_sxr"][i] = sample_entropy(
                        sxr_i[:500], m=2, r_factor=0.2
                    )
                    feats["sample_entropy_hxr"][i] = sample_entropy(
                        hxr_i[:500], m=2, r_factor=0.2
                    )
                if hxr_full is not None:
                    ml = min(len(sxr_windows[i]), len(hxr_full[i]))
                    if ml > 200:
                        corr, lag = lagged_cross_correlation(
                            hxr_full[i, :ml],
                            sxr_windows[i, :ml].cpu().numpy(),
                            max_lag=100,
                        )
                        feats["lagged_cross_corr"][i] = corr
                        feats["lagged_cross_corr_lag"][i] = lag
    except Exception:
        pass
    return feats


def gpu_extract_features_batch(
    all_sxr: list[np.ndarray],
    all_hxr: list[np.ndarray | None],
    all_precomputed: list[dict],
    all_event_times: list[list[float]],
    all_time_s: list[np.ndarray],
    batch_size: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """Process multiple days on GPU in one batch. Returns (X_concat, y_concat)."""
    all_X, all_y = [], []
    for start in range(0, len(all_sxr), batch_size):
        end = min(start + batch_size, len(all_sxr))
        X_batch, y_batch = _gpu_batch_chunk(
            all_sxr[start:end],
            all_hxr[start:end],
            all_precomputed[start:end],
            all_event_times[start:end],
            all_time_s[start:end],
        )
        if X_batch is not None:
            all_X.append(X_batch)
            all_y.append(y_batch)
        torch.cuda.empty_cache()
    if not all_X:
        return np.empty((0, 0), dtype=np.float32), np.empty(0, dtype=int)
    return np.vstack(all_X), np.concatenate(all_y)


def _gpu_batch_chunk(
    sxr_list,
    hxr_list,
    precomputed_list,
    event_times_list,
    time_s_list,
) -> tuple[np.ndarray, np.ndarray]:
    lookback, step = 3600, 300
    window_feats = []
    window_ys = []

    sxr_batch = torch.from_numpy(
        np.array(
            [
                np.pad(s, (0, max(0, lookback + step - len(s))), mode="edge")[
                    : lookback + step
                ]
                for s in sxr_list
            ],
            dtype=np.float32,
        )
    ).to(_DEVICE)

    hxr_batch = None
    has_hxr = [h is not None and h.shape[1] >= 5 for h in hxr_list]
    if any(has_hxr):
        max_bands = max(h.shape[1] for h in hxr_list if h is not None)
        padded = []
        for h in hxr_list:
            if h is not None:
                p = np.pad(
                    h, ((0, max(0, lookback + step - h.shape[0])), (0, 0)), mode="edge"
                )[: lookback + step, :max_bands]
            else:
                p = np.zeros((lookback + step, max_bands), dtype=np.float32)
            padded.append(p)
        hxr_batch = torch.from_numpy(np.array(padded, dtype=np.float32)).to(_DEVICE)

    for i in range(len(sxr_list)):
        sxr_win = sxr_batch[i]
        hxr_win = hxr_batch[i] if hxr_batch is not None else None
        n_w = (len(sxr_list[i]) - lookback) // step + 1
        if n_w < 1:
            continue

        sxr_u = sxr_win.unsqueeze(0).expand(n_w, -1).unfold(1, lookback, step)[:, 0, :]
        hxr_u = None
        if hxr_win is not None and hxr_win.shape[1] >= 5:
            hxr_u = (
                hxr_win.unsqueeze(0)
                .expand(n_w, -1, -1)
                .unfold(1, lookback, step)[:, 0, :, :]
                .permute(0, 2, 1)
            )

        feats = {}
        feats.update(_batch_stats(sxr_u))
        feats.update(_batch_acf(sxr_u, FEATURE_AUTOCORR_LAGS))
        feats.update(_batch_spectral_entropy(sxr_u))
        feats.update(_batch_derivative_features(sxr_u, hxr_u))
        feats.update(_batch_multiscale(sxr_u, hxr_u))
        feats.update(_batch_neupert(sxr_u, hxr_u))
        if hxr_u is not None:
            feats.update(_batch_hxr_features(hxr_u))
        else:
            for k in list(
                _batch_hxr_features(torch.zeros(1, lookback, 10, device=_DEVICE)).keys()
            ):
                feats[k] = torch.zeros(n_w, device=_DEVICE)

        pre = precomputed_list[i]
        for k, v in pre.items():
            if isinstance(v, (int, float)):
                feats[k] = torch.full(
                    (n_w,), float(np.clip(v, -1e10, 1e10)), device=_DEVICE, dtype=_DTYPE
                )

        canonical = get_canonical_feature_names()
        n_feat = len(canonical)
        row = torch.zeros(n_w, n_feat, device=_DEVICE, dtype=_DTYPE)
        feat_keys = list(feats.keys())
        for fi, fn in enumerate(canonical):
            if fn in feat_keys:
                val = feats[fn]
                if isinstance(val, torch.Tensor) and val.shape[0] == n_w:
                    row[:, fi] = val

        X_day = row.cpu().numpy().astype(np.float32)
        X_day = np.nan_to_num(X_day, nan=0.0, posinf=0.0, neginf=0.0)

        y_day = np.zeros(n_w, dtype=int)
        et = event_times_list[i]
        ts = time_s_list[i]
        if ts is not None and et:
            for wi in range(n_w):
                t = ts[min(wi * step + lookback, len(ts) - 1)]
                y_day[wi] = 1 if any(0 < e - t <= 1800 for e in et) else 0

        window_feats.append(X_day)
        window_ys.append(y_day)

    if not window_feats:
        return None, None
    return np.vstack(window_feats), np.concatenate(window_ys)


def gpu_extract_features_day(
    counts: np.ndarray,
    hxr_bands: np.ndarray | None = None,
    precomputed: dict | None = None,
    event_times: list[float] | None = None,
    time_s: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    if precomputed is None:
        precomputed = {}

    lookback, step = 3600, 300
    N = len(counts)
    if N < lookback + step:
        return None

    n_windows = (N - lookback) // step + 1
    indices = torch.arange(n_windows, device=_DEVICE)
    starts = indices * step
    ends = starts + lookback

    sxr_windows = counts
    sxr_t = _to_gpu(sxr_windows)
    sxr_unfolded = sxr_t.unfold(0, lookback, step)[:n_windows]

    hxr_unfolded = None
    if hxr_bands is not None and hxr_bands.shape[1] >= 5:
        hxr_t = _to_gpu(hxr_bands)
        hxr_unfolded = hxr_t.unfold(0, lookback, step)[:n_windows].permute(0, 2, 1)

    all_feats = {}
    all_feats.update(_batch_stats(sxr_unfolded))
    all_feats.update(_batch_acf(sxr_unfolded, FEATURE_AUTOCORR_LAGS))
    all_feats.update(_batch_spectral_entropy(sxr_unfolded))
    all_feats.update(_batch_derivative_features(sxr_unfolded, hxr_unfolded))
    all_feats.update(_batch_multiscale(sxr_unfolded, hxr_unfolded))
    all_feats.update(_batch_neupert(sxr_unfolded, hxr_unfolded))
    if hxr_unfolded is not None:
        all_feats.update(_batch_hxr_features(hxr_unfolded))
    else:
        for k in list(
            _batch_hxr_features(torch.zeros(1, lookback, 10, device=_DEVICE)).keys()
        ):
            all_feats[k] = torch.zeros(n_windows, device=_DEVICE)

    for k, v in precomputed.items():
        if isinstance(v, (int, float)):
            fv = float(np.clip(v, -1e10, 1e10))
            all_feats[k] = torch.full((n_windows,), fv, device=_DEVICE, dtype=_DTYPE)
        elif isinstance(v, torch.Tensor):
            all_feats[k] = v.expand(n_windows).to(_DTYPE)

    canonical = get_canonical_feature_names()
    rows = []
    for wi in range(n_windows):
        feat = {}
        for k in canonical:
            if k in all_feats:
                val = all_feats[k]
                if isinstance(val, torch.Tensor):
                    feat[k] = float(val[wi].item()) if wi < len(val) else 0.0
                else:
                    feat[k] = float(val)
            else:
                feat[k] = 0.0
        rows.append(pad_features_to_canonical(feat, canonical))

    X = np.array(rows, dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    y = np.zeros(n_windows, dtype=int)
    if event_times is not None and time_s is not None:
        for wi in range(n_windows):
            t = time_s[min(wi * step + lookback, N - 1)]
            y[wi] = 1 if any(0 < et - t <= 1800 for et in event_times) else 0

    return X, y
