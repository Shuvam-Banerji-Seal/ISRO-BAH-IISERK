"""
Stage 2 — Correlation Analysis of Feature Matrix.
Full cross-correlation: Feature × Target + Feature × Feature (multicollinearity).
Nowcasting targets + Forecasting targets, Spearman rank ρ.
"""

import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from scipy.stats import spearmanr
from scipy.cluster.hierarchy import linkage, leaves_list
import warnings

warnings.filterwarnings("ignore")

SRC = Path("dist/stage2_feature_matrix_20260623.npz")
S1 = Path("data/processed/stage1_20260623.npz")
OUT_DIR = Path("dist/plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FIG_TARGETS = OUT_DIR / "correlation_targets.png"
FIG_TOPS = OUT_DIR / "correlation_top_features.png"
FIG_FEAT_FEAT = OUT_DIR / "correlation_feature_feature.png"

PHASE_COLORS = {
    "phase1_direct": "#ff9944",
    "phase2_perflare": "#44bbff",
    "phase3_tem_goes": "#ff6688",
    "phase4_tem_solexs": "#aa66ff",
    "phase5_wavelet": "#44dd88",
    "phase6_spectral": "#ffcc00",
    "phase7_nonlinear": "#66ccff",
    "phase8_event": "#ff88aa",
}


def _class_to_numeric(c):
    if not isinstance(c, str) or len(c) < 2:
        return 0.0
    try:
        p = c[0].upper()
        v = float(c[1:])
        return {"A": v / 1000, "B": v / 10, "C": v, "M": v * 10, "X": v * 100}.get(
            p, 0.0
        )
    except (ValueError, IndexError):
        return 0.0


def build_targets():
    """Build nowcast + forecast target arrays."""
    ds1 = np.load(S1, allow_pickle=True)
    fm = np.load(SRC, allow_pickle=True)
    meta = fm["__metadata__"].item()

    flare_id = ds1["flare_id"]
    flare_label = ds1["flare_label"]
    master_flag = ds1["master_flag"]
    goes_class = ds1["goes_class"]
    t = ds1["time"].astype(np.float64)

    good = master_flag == 0
    n = 86400

    # Nowcast targets
    y_in_flare = (flare_id > 0).astype(np.int8)
    y_flare_class = flare_label.copy()  # 0=quiet, 1=B, 2=C+

    # Forecasting targets: will a flare START within N seconds?
    # time_until_next_flare gives seconds until next active flare bin
    # We want seconds until next FLARE START, so we need to know
    # which bins are the start of a flare.
    in_flare = flare_id > 0
    flare_starts = np.diff(in_flare.astype(int)) == 1
    # For each bin, find seconds until next flare start
    start_times = np.where(flare_starts)[0] + 1  # indices of flare starts
    t_until_start = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        future = start_times[start_times > i]
        if len(future) > 0:
            t_until_start[i] = t[future[0]] - t[i]
        else:
            t_until_start[i] = np.nan  # no more flares today

    y_flare_30m = (t_until_start <= 1800).astype(np.int8)
    y_flare_1h = (t_until_start <= 3600).astype(np.int8)

    # Deep quiet: >30 min from any flare
    # Build distance to nearest flare
    flare_indices = np.where(in_flare)[0]
    dist_to_flare = np.full(n, np.inf, dtype=np.float64)
    for i in range(n):
        if len(flare_indices) > 0:
            dist_to_flare[i] = np.min(np.abs(flare_indices - i))
    y_deep_quiet = (dist_to_flare > 1800).astype(np.int8)  # >30 min from any flare

    targets = {
        "y_in_flare": y_in_flare,
        "y_flare_class": y_flare_class,
        "y_flare_30m": y_flare_30m,
        "y_flare_1h": y_flare_1h,
        "y_deep_quiet": y_deep_quiet,
    }

    # Print target distributions (GOOD bins only)
    print("Target distributions (GOOD bins only, master_flag==0):")
    for name, arr in targets.items():
        if good.sum() > 0:
            in_good = arr[good]
            if in_good.dtype.kind in ("i", "u", "b"):
                counts = np.bincount(in_good.astype(int))
                print(f"  {name}: {dict(enumerate(counts))}")
            else:
                print(f"  {name}: mean={in_good.mean():.3f}, std={in_good.std():.3f}")

    return targets, good, meta


def load_features():
    """Load feature matrix, return dict of arrays + metadata."""
    fm = np.load(SRC, allow_pickle=True)
    meta = fm["__metadata__"].item()
    phase_map = meta.get("phase_map", {})

    features = {}
    feature_info = []
    keys = sorted([k for k in fm.files if k != "__metadata__"])

    for k in keys:
        v = fm[k]
        if v.dtype.kind not in ("f", "i", "u", "b"):
            continue
        features[k] = v
        phase = phase_map.get(k, "unknown")
        # Determine if it's a "flare-internal" feature (only defined during flares)
        is_flare_internal = any(
            suffix in k
            for suffix in [
                "rise_time",
                "decay_time",
                "duration",
                "t_start",
                "t_peak",
                "t_end",
                "peak_flux",
                "bg_flux",
                "max_deriv",
                "dt_peak",
                "hxr_fluence",
                "peak_sxr",
                "peak_hxr",
                "T_peak_time",
                "EM_peak_time",
                "T_leads_EM",
                "reale_loop",
                "qpp_power_preflare",
                "qpp_power_onset",
                "qpp_power_decay",
                "T_MK_solexs",
                "EM_log10_solexs",
            ]
        )
        feature_info.append(
            {
                "name": k,
                "phase": phase,
                "is_flare_internal": is_flare_internal,
                "dtype": v.dtype.kind,
            }
        )

    return features, feature_info, meta


def compute_correlations(features, targets, good_mask):
    """Compute Spearman ρ between each feature and each target (GOOD bins only)."""
    good_idx = np.where(good_mask)[0]
    n_good = len(good_idx)

    target_names = list(targets.keys())
    target_arrays = {k: v[good_idx] for k, v in targets.items()}
    feat_names = list(features.keys())

    n_feat = len(feat_names)
    n_targ = len(target_names)

    rho = np.full((n_feat, n_targ), np.nan, dtype=np.float32)
    pval = np.full((n_feat, n_targ), np.nan, dtype=np.float32)
    valid_count = np.zeros(n_feat, dtype=int)

    print(f"\nComputing {n_feat} × {n_targ} Spearman correlations...")
    for fi, fname in enumerate(feat_names):
        fv = features[fname][good_idx].astype(np.float64)
        finf = fv[~np.isnan(fv)]
        valid_count[fi] = len(finf)
        if len(finf) < 50:
            continue
        for ti, tname in enumerate(target_names):
            tv = target_arrays[tname].astype(np.float64)
            # Pairwise complete
            mask = ~(np.isnan(fv) | np.isnan(tv))
            if mask.sum() < 50:
                continue
            r, p = spearmanr(fv[mask], tv[mask])
            rho[fi, ti] = r
            pval[fi, ti] = p

        if fi % 20 == 0:
            print(f"  {fi}/{n_feat} done")

    return rho, pval, feat_names, target_names, valid_count


def plot_target_correlations(rho, pval, feat_names, target_names, feature_info):
    """Figure 1: Feature × Target correlation heatmap."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list(
        "rdbu", ["#2166ac", "#f7f7f7", "#b2182b"], N=256
    )

    # Sort features by max absolute correlation across targets
    max_abs_rho = np.nanmax(np.abs(rho), axis=1)
    order = np.argsort(-max_abs_rho)
    rho_s = rho[order]
    pval_s = pval[order]
    feat_s = [feat_names[i] for i in order]

    # Only show top 50 features (otherwise too tall)
    n_show = min(50, len(feat_s))
    rho_s = rho_s[:n_show]
    pval_s = pval_s[:n_show]
    feat_s = feat_s[:n_show]

    n_targ = len(target_names)

    fig, ax = plt.subplots(figsize=(n_targ * 2.2 + 3, n_show * 0.45 + 2))
    fig.patch.set_facecolor("#0d0d1a")
    ax.set_facecolor("#1a1a2e")

    im = ax.imshow(rho_s, aspect="auto", cmap=cmap, vmin=-0.6, vmax=0.6)

    # Annotate significant correlations
    for i in range(n_show):
        for j in range(n_targ):
            if np.isnan(rho_s[i, j]):
                continue
            sig = pval_s[i, j] < 0.001
            val = rho_s[i, j]
            color = "white" if abs(val) > 0.3 else "#888"
            ax.text(
                j,
                i,
                f"{val:.2f}" + ("*" if sig else ""),
                ha="center",
                va="center",
                fontsize=5.5,
                color=color,
            )

    ax.set_xticks(range(n_targ))
    ax.set_xticklabels(target_names, rotation=30, ha="right", fontsize=8, color="#ccc")
    ax.set_yticks(range(n_show))
    ax.set_yticklabels(feat_s, fontsize=5.5, color="#ccc")
    ax.set_title(
        "Feature × Target Spearman ρ (GOOD bins only)",
        color="white",
        fontsize=11,
        pad=10,
    )

    plt.colorbar(im, ax=ax, label="ρ", shrink=0.6)
    for s in ax.spines.values():
        s.set_color("#444")
    ax.tick_params(colors="#aaa")

    plt.tight_layout()
    plt.savefig(FIG_TARGETS, dpi=150, facecolor="#0d0d1a")
    plt.close()
    print(f"Saved: {FIG_TARGETS}")


def plot_top_features(rho, pval, feat_names, target_names, feature_info):
    """Figure 2: Top-20 features for nowcast (y_in_flare) and forecast (y_flare_30m)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def top_n(feats, corrs, pvals, n=20):
        abs_c = np.abs(corrs)
        order = np.argsort(-abs_c)
        valid = ~np.isnan(corrs[order])
        order = order[valid]
        return [(feats[i], corrs[i], pvals[i]) for i in order[:n]]

    ti_now = target_names.index("y_in_flare")
    ti_fc = target_names.index("y_flare_30m")

    top_now = top_n(feat_names, rho[:, ti_now], pval[:, ti_now])
    top_fc = top_n(feat_names, rho[:, ti_fc], pval[:, ti_fc])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 10))
    fig.patch.set_facecolor("#0d0d1a")

    phase_map_info = {fi["name"]: fi["phase"] for fi in feature_info}

    for ax, top, title in [
        (ax1, top_now, "Top-20 NOWCAST (y_in_flare)"),
        (ax2, top_fc, "Top-20 FORECAST (y_flare_30m)"),
    ]:
        ax.set_facecolor("#1a1a2e")
        names = [t[0] for t in top][::-1]
        vals = [t[1] for t in top][::-1]
        phases = [phase_map_info.get(n, "unknown") for n in names]
        colors = [PHASE_COLORS.get(p, "#888888") for p in phases]

        ax.barh(range(len(names)), vals, color=colors, height=0.7, edgecolor="none")
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=7, color="#ccc")
        ax.axvline(0, color="#555", lw=0.5)
        ax.set_xlim(-0.6, 0.6)
        ax.set_title(title, color="white", fontsize=10)
        ax.tick_params(colors="#aaa")

        # Annotate value
        for i, v in enumerate(vals):
            ax.text(
                v + 0.01 if v >= 0 else v - 0.05,
                i,
                f"{v:.3f}",
                fontsize=6,
                color="#aaa",
                va="center",
            )

        for s in ax.spines.values():
            s.set_color("#444")

    # Legend for phases
    from matplotlib.patches import Patch

    legend_elements = [Patch(facecolor=c, label=p) for p, c in PHASE_COLORS.items()]
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        fontsize=7,
        ncol=4,
        bbox_to_anchor=(0.5, -0.02),
    )

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.savefig(FIG_TOPS, dpi=150, facecolor="#0d0d1a", bbox_inches="tight")
    plt.close()
    print(f"Saved: {FIG_TOPS}")

    return top_now, top_fc


def plot_feature_feature_correlation(features, feature_info, good_mask, max_show=30):
    """Figure 3: Feature × Feature Spearman ρ (multicollinearity heatmap)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    good_idx = np.where(good_mask)[0]
    feat_names = [fi["name"] for fi in feature_info]
    n_feat = len(feat_names)

    # Limit to features with enough valid data
    valid_floats = []
    for fi in feature_info:
        if fi["dtype"] != "f":
            continue
        fv = features[fi["name"]][good_idx]
        if np.sum(~np.isnan(fv)) > 1000:
            valid_floats.append(fi["name"])

    n_show = min(max_show, len(valid_floats))
    names = valid_floats[:n_show]

    # Compute correlation matrix (triangular)
    print(f"\nFeature-Feature correlation: {n_show} × {n_show} matrix...")
    data = np.column_stack([features[n][good_idx].astype(np.float64) for n in names])
    # Mean-impute NaN for correlation (pairwise complete is too slow for 60×60)
    data_mean = np.nanmean(data, axis=0)
    data_filled = np.where(np.isnan(data), data_mean, data)
    rho_ff, _ = spearmanr(data_filled)

    # Hierarchical clustering for reordering
    print("  Clustering...")
    dist = 1 - np.abs(rho_ff)
    try:
        Z = linkage(dist[np.triu_indices_from(dist, k=1)], method="average")
        order = leaves_list(Z)
    except Exception:
        order = np.arange(n_show)
    rho_ff_ord = rho_ff[order][:, order]
    names_ord = [names[i] for i in order]

    # FIGURE — limit size to avoid "Image size too large" error
    figsize_h = min(17, n_show * 0.45 + 2)
    fig, ax = plt.subplots(figsize=(18, figsize_h))
    fig.patch.set_facecolor("#0d0d1a")
    ax.set_facecolor("#1a1a2e")

    cmap = LinearSegmentedColormap.from_list(
        "rdbu", ["#2166ac", "#f7f7f7", "#b2182b"], N=256
    )
    im = ax.imshow(rho_ff_ord, cmap=cmap, vmin=-1, vmax=1)

    # Phase color bar on left side
    phase_order = [
        next((fi["phase"] for fi in feature_info if fi["name"] == n), "unknown")
        for n in names_ord
    ]
    phase_colors = [PHASE_COLORS.get(p, "#444") for p in phase_order]
    for i, c in enumerate(phase_colors):
        ax.barh(
            i,
            0.3,
            height=0.8,
            left=-0.35,
            color=c,
            transform=ax.get_yaxis_transform(),
            clip_on=False,
            align="center",
        )

    # Color stripe along the top for column phases (using scatter with squares)
    for j, c in enumerate(phase_colors):
        ax.plot(
            j,
            n_show + 0.5,
            marker="s",
            color=c,
            markersize=6,
            transform=ax.get_xaxis_transform(),
            clip_on=False,
        )

    ax.set_xticks(range(n_show))
    ax.set_yticks(range(n_show))
    ax.set_xticklabels(names_ord, rotation=90, fontsize=4, color="#aaa")
    ax.set_yticklabels(names_ord, fontsize=4, color="#aaa")
    ax.set_title(
        "Feature × Feature Spearman ρ (hierarchically clustered)",
        color="white",
        fontsize=10,
    )

    plt.colorbar(im, ax=ax, label="ρ", shrink=0.4, pad=0.02)
    for s in ax.spines.values():
        s.set_color("#444")

    plt.tight_layout()
    plt.savefig(FIG_FEAT_FEAT, dpi=150, facecolor="#0d0d1a", bbox_inches="tight")
    plt.close()
    print(f"Saved: {FIG_FEAT_FEAT}")

    # Find highly correlated groups
    print("\nFeature groups with |ρ| > 0.8 (potential redundancy):")
    high_corr_pairs = []
    for i in range(n_show):
        for j in range(i + 1, n_show):
            if abs(rho_ff_ord[i, j]) > 0.8:
                high_corr_pairs.append((names_ord[i], names_ord[j], rho_ff_ord[i, j]))
    high_corr_pairs.sort(key=lambda x: -abs(x[2]))
    for a, b, r in high_corr_pairs[:15]:
        print(f"  {a:35s} ~ {b:35s}  ρ={r:.3f}")

    return names_ord


def print_top_table(top_now, top_fc):
    """Print formatted top features table."""
    print("\n" + "=" * 80)
    print("TOP-15 FEATURES: NOWCAST vs FORECAST")
    print("=" * 80)
    print(
        f"{'#':<4} {'NOWCAST (y_in_flare)':<40s} {'ρ':<8} {'|':<3} {'FORECAST (y_flare_30m)':<40s} {'ρ':<8}"
    )
    print("-" * 80)
    for i in range(max(len(top_now), len(top_fc))):
        n = top_now[i] if i < len(top_now) else ("---", np.nan, np.nan)
        f = top_fc[i] if i < len(top_fc) else ("---", np.nan, np.nan)
        nr = f"{n[1]:.4f}" if not np.isnan(n[1]) else "---"
        fr = f"{f[1]:.4f}" if not np.isnan(f[1]) else "---"
        print(f"{i + 1:<4} {n[0]:<40s} {nr:<8} {'|':<3} {f[0]:<40s} {fr:<8}")


def main():
    print("=" * 60)
    print("STAGE 2 — CORRELATION ANALYSIS")
    print("=" * 60)

    targets, good_mask, meta = build_targets()
    features, feature_info, _ = load_features()
    rho, pval, feat_names, target_names, valid_count = compute_correlations(
        features, targets, good_mask
    )

    # Plots
    print("\nGenerating plots...")
    plot_target_correlations(rho, pval, feat_names, target_names, feature_info)
    top_now, top_fc = plot_top_features(
        rho, pval, feat_names, target_names, feature_info
    )
    feat_ord = plot_feature_feature_correlation(features, feature_info, good_mask)
    print_top_table(top_now, top_fc)

    # Summary console
    print("\n" + "=" * 60)
    print("KEY FINDINGS")
    print("=" * 60)

    # Best nowcast features (excluding flare-internal)
    ti = target_names.index("y_in_flare")
    ext_feats = [
        (feat_names[i], rho[i, ti], pval[i, ti])
        for i in range(len(feat_names))
        if not any(
            s in feat_names[i]
            for s in [
                "rise_time",
                "decay_time",
                "duration",
                "t_start",
                "t_peak",
                "t_end",
                "peak_flux",
                "bg_flux",
                "max_deriv",
                "dt_peak",
                "hxr_fluence",
                "peak_sxr",
                "peak_hxr",
                "T_peak_time",
                "EM_peak_time",
                "T_leads_EM",
                "reale_loop",
                "T_MK_solexs",
                "EM_log10_solexs",
                "qpp_power_pre",
                "qpp_power_on",
                "qpp_power_dec",
            ]
        )
    ]
    ext_feats.sort(key=lambda x: -abs(x[1]))
    print("\nTop-10 NOWCAST features (non-flare-internal):")
    for name, r, p in ext_feats[:10]:
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"  {name:40s}  ρ={r:+.4f}  {sig}")

    # Best forecast features
    ti_fc = target_names.index("y_flare_30m")
    fc_feats = [
        (feat_names[i], rho[i, ti_fc], pval[i, ti_fc])
        for i in range(len(feat_names))
        if not np.isnan(rho[i, ti_fc])
    ]
    fc_feats.sort(key=lambda x: -abs(x[1]))
    print("\nTop-10 FORECAST features (y_flare_30m):")
    for name, r, p in fc_feats[:10]:
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"  {name:40s}  ρ={r:+.4f}  {sig}")

    print(f"\nFigures saved:")
    print(f"  {FIG_TARGETS}")
    print(f"  {FIG_TOPS}")
    print(f"  {FIG_FEAT_FEAT}")


if __name__ == "__main__":
    main()
