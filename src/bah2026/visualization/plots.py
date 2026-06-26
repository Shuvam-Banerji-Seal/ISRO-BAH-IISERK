"""Publication-quality plots for flare analysis results."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from bah2026.config import (
    PLOTS_OVERVIEW, PLOTS_STATISTICS, PLOTS_NOWCAST, PLOTS_FORECAST,
    CZT_BANDS, CDTE_BANDS,
)
from bah2026.data.reader import (
    load_solexs_lc, load_hel1os_lc,
    discover_solexs_days, discover_hel1os_days,
)
from bah2026.data.preprocessing import background_subtract


# ── Single-day overview ─────────────────────────────────────────────────

def plot_day_overview(d: date, save: bool = True, show: bool = False) -> plt.Figure | None:
    """Multi-panel plot: SoLEXS SXR + HEL1OS HXR + hardness ratio."""
    try:
        solexs = load_solexs_lc(d)
    except FileNotFoundError:
        return None

    sxr = solexs["counts"]
    sxr_h = solexs["time"] / 3600.0
    sxr_plot = np.where(np.isfinite(sxr), sxr, np.nan)

    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(4, 1, height_ratios=[3, 2, 2, 1], hspace=0.3)

    ax1 = fig.add_subplot(gs[0])
    ax1.plot(sxr_h, sxr_plot, "b-", lw=0.3, alpha=0.7)
    ax1.set_ylabel("SoLEXS Counts/s", fontsize=11)
    ax1.set_title(f"Aditya-L1 X-ray Light Curves — {d}", fontsize=13, fontweight="bold")
    ax1.set_xlim(sxr_h[0], sxr_h[-1])
    ax1.grid(True, alpha=0.3)

    has_hxr = False
    for det, num, color, label in [
        ("czt", 1, "r", "CZT1 (18–160 keV)"),
        ("cdte", 1, "orange", "CdTe1 (1.8–90 keV)"),
    ]:
        try:
            hel = load_hel1os_lc(d, detector=det, num=num)
            if hel["ctr"].size == 0:
                continue
            mjd = hel["mjd"]
            t_h = (mjd - mjd[0]) * 24.0
            idx = -1
            ax = fig.add_subplot(gs[1] if det == "czt" else gs[2])
            ax.plot(t_h, hel["ctr"][:, idx], color=color, lw=0.3, alpha=0.7)
            ax.set_ylabel(label, fontsize=10)
            ax.set_xlim(0, max(t_h[-1], 24))
            ax.grid(True, alpha=0.3)
            has_hxr = True
        except Exception:
            continue

    if not has_hxr:
        for idx in (1, 2):
            ax = fig.add_subplot(gs[idx])
            ax.text(0.5, 0.5, "No HEL1OS data", transform=ax.transAxes,
                    ha="center", va="center", color="gray", fontsize=12)

    ax4 = fig.add_subplot(gs[3])
    ax4.set_xlabel("Time (hours from start)", fontsize=11)
    ax4.set_ylabel("SXR/HXR Ratio", fontsize=10)
    ax4.set_xlim(sxr_h[0], sxr_h[-1])
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    if save:
        fig.savefig(PLOTS_OVERVIEW / f"{d}_overview.png", dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return fig


# ── Coverage timeline ───────────────────────────────────────────────────

def plot_coverage_timeline(save: bool = True, show: bool = False) -> plt.Figure:
    """Bar chart of SoLEXS / HEL1OS / combined daily availability."""
    solexs = set(discover_solexs_days())
    hel1os = set(discover_hel1os_days())
    if not solexs and not hel1os:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data found", transform=ax.transAxes, ha="center")
        return fig

    all_dates = pd.date_range(
        min(min(solexs, default=date(2099,1,1)), min(hel1os, default=date(2099,1,1))),
        max(max(solexs, default=date(2000,1,1)), max(hel1os, default=date(2000,1,1))),
    )
    sx = [d.date() in solexs for d in all_dates]
    hx = [d.date() in hel1os for d in all_dates]
    both = [s and h for s, h in zip(sx, hx)]

    fig, axes = plt.subplots(3, 1, figsize=(16, 6), sharex=True,
                              gridspec_kw={"height_ratios": [1, 1, 1]})
    for ax, mask, label, color in [
        (axes[0], sx, "SoLEXS", "blue"),
        (axes[1], hx, "HEL1OS", "red"),
        (axes[2], both, "Both", "green"),
    ]:
        ax.bar(all_dates, mask, color=color, alpha=0.7, width=1.0)
        ax.set_ylabel(label)
        ax.set_ylim(0, 1.5)

    axes[0].set_title("Aditya-L1 Data Coverage", fontsize=13, fontweight="bold")
    axes[-1].set_xlabel("Date")
    plt.tight_layout()
    if save:
        fig.savefig(PLOTS_STATISTICS / "coverage_timeline.png", dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return fig


# ── Energy coverage diagram ─────────────────────────────────────────────

def plot_energy_coverage(save: bool = True, show: bool = False) -> plt.Figure:
    """Horizontal bar chart showing combined SoLEXS + HEL1OS energy coverage."""
    bands = [
        ("SoLEXS SDD2", 2, 22, "blue"),
        ("CdTe 5–20", 5, 20, "darkorange"),
        ("CdTe 20–30", 20, 30, "orange"),
        ("CdTe 30–40", 30, 40, "gold"),
        ("CdTe 40–60", 40, 60, "yellowgreen"),
        ("CZT 20–40", 20, 40, "red"),
        ("CZT 40–60", 40, 60, "darkred"),
        ("CZT 60–80", 60, 80, "brown"),
        ("CZT 80–150", 80, 150, "purple"),
    ]

    fig, ax = plt.subplots(figsize=(12, 3))
    for i, (name, elo, ehi, color) in enumerate(bands):
        ax.barh(i, ehi - elo, left=elo, height=0.6, color=color, alpha=0.7, label=name)

    ax.set_xlabel("Energy (keV)")
    ax.set_yticks(range(len(bands)))
    ax.set_yticklabels([b[0] for b in bands], fontsize=8)
    ax.set_xlim(1, 200)
    ax.set_xscale("log")
    ax.set_title("Combined Energy Coverage: 1.8–160 keV", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    if save:
        fig.savefig(PLOTS_STATISTICS / "energy_coverage.png", dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return fig


# ── Nowcast catalogue statistics ────────────────────────────────────────

def plot_flare_statistics(df: pd.DataFrame, save: bool = True, show: bool = False) -> plt.Figure:
    """Four-panel flare statistics from the nowcast catalogue."""
    class_colors = {"A": "blue", "B": "cyan", "C": "green", "M": "orange", "X": "red"}

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    cc = df["goes_class"].value_counts()
    ax.bar(cc.index, cc.values, color=[class_colors.get(c, "gray") for c in cc.index])
    ax.set_xlabel("GOES Class"); ax.set_ylabel("Count"); ax.set_title("Flare Class Distribution")
    ax.set_yscale("log")

    ax = axes[0, 1]
    for cls, color in class_colors.items():
        m = df["goes_class"] == cls
        if m.sum():
            ax.scatter(df.loc[m, "peak_flux"], df.loc[m, "duration_sec"] / 60,
                       c=color, alpha=0.5, s=15, label=cls)
    ax.set_xlabel("Peak Flux (cts/s)"); ax.set_ylabel("Duration (min)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.legend(fontsize=8); ax.set_title("Duration vs Peak Flux"); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    dates = pd.to_datetime(df["date"])
    monthly = dates.dt.to_period("M").value_counts().sort_index()
    ax.bar(range(len(monthly)), monthly.values, color="steelblue", alpha=0.7)
    ticks = list(range(0, len(monthly), max(1, len(monthly) // 10)))
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(monthly.index[i]) for i in ticks], rotation=45, fontsize=7)
    ax.set_xlabel("Month"); ax.set_ylabel("Flare Count"); ax.set_title("Monthly Flare Count")

    ax = axes[1, 1]
    if "has_hxr" in df.columns:
        vals = [int((~df["has_hxr"]).sum()), int(df["has_hxr"].sum())]
        ax.pie(vals, labels=["SXR only", "SXR + HXR"], autopct="%1.1f%%",
               colors=["skyblue", "salmon"])
        ax.set_title("HXR Confirmation")
    else:
        ax.text(0.5, 0.5, "No HXR data", transform=ax.transAxes, ha="center")

    plt.suptitle("Nowcast Catalogue Statistics", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    if save:
        fig.savefig(PLOTS_STATISTICS / "flare_statistics.png", dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return fig


# ── Flare examples with light curves ────────────────────────────────────

def plot_flare_examples(
    df: pd.DataFrame,
    n: int = 4,
    save: bool = True,
    show: bool = False,
) -> plt.Figure | None:
    """Plot light curves with detected flares overlaid."""
    flare_dates = df["date"].unique()
    sample = flare_dates[:n] if len(flare_dates) >= n else flare_dates
    if len(sample) == 0:
        return None

    fig, axes = plt.subplots(len(sample), 1, figsize=(16, 4 * len(sample)))
    if len(sample) == 1:
        axes = [axes]

    for ax, ds in zip(axes, sample):
        d = date.fromisoformat(ds)
        try:
            solexs = load_solexs_lc(d)
        except FileNotFoundError:
            ax.text(0.5, 0.5, f"No data for {d}", transform=ax.transAxes, ha="center")
            continue

        counts = solexs["counts"]
        t_h = solexs["time"] / 3600.0
        valid = np.where(np.isfinite(counts), counts, np.nanmedian(counts))
        bg, _ = background_subtract(valid)

        ax.plot(t_h, valid, "b-", lw=0.3, alpha=0.6, label="SoLEXS")
        ax.plot(t_h, bg, "gray", lw=1, alpha=0.5, label="Background")

        for _, evt in df[df["date"] == ds].iterrows():
            ax.axvspan(evt["start_time"] / 3600, evt["end_time"] / 3600,
                        alpha=0.3, color="red")
            ax.axvline(evt["peak_time"] / 3600, color="red", lw=1, ls="--")
            ax.annotate(f'{evt["goes_class"]}\n{evt["peak_flux"]:.0f}',
                        xy=(evt["peak_time"] / 3600, evt["peak_flux"]),
                        fontsize=7, color="red", ha="center", va="bottom")

        n_flares = len(df[df["date"] == ds])
        ax.set_ylabel("Counts/s")
        ax.set_title(f"{d} — {n_flares} flares detected")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (hours from start)")
    plt.tight_layout()
    if save:
        fig.savefig(PLOTS_NOWCAST / "flare_examples.png", dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return fig


# ── Feature importance ──────────────────────────────────────────────────

def plot_feature_importance(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    save: bool = True,
    show: bool = False,
) -> plt.Figure:
    """Mutual information + Random Forest feature importance."""
    from sklearn.feature_selection import mutual_info_classif
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    mi = mutual_info_classif(Xs, y, random_state=42, n_neighbors=5)
    top20 = np.argsort(mi)[::-1][:20]

    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(Xs, y)
    rf_imp = rf.feature_importances_
    rf20 = np.argsort(rf_imp)[::-1][:20]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    ax1.barh(range(20), mi[top20], color="steelblue", alpha=0.7)
    ax1.set_yticks(range(20))
    ax1.set_yticklabels([feature_names[i] for i in top20], fontsize=8)
    ax1.set_xlabel("Mutual Information")
    ax1.set_title("Top 20 Features (MI)")
    ax1.invert_yaxis()

    ax2.barh(range(20), rf_imp[rf20], color="forestgreen", alpha=0.7)
    ax2.set_yticks(range(20))
    ax2.set_yticklabels([feature_names[i] for i in rf20], fontsize=8)
    ax2.set_xlabel("Importance")
    ax2.set_title("Top 20 Features (Random Forest)")
    ax2.invert_yaxis()

    plt.suptitle("Feature Importance Analysis", fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save:
        fig.savefig(PLOTS_STATISTICS / "feature_importance.png", dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return fig


# ── Feature distributions ───────────────────────────────────────────────

def plot_feature_distributions(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    top_n: int = 6,
    save: bool = True,
    show: bool = False,
) -> plt.Figure:
    """Histograms of top features split by flare/no-flare."""
    from sklearn.feature_selection import mutual_info_classif
    from sklearn.preprocessing import StandardScaler

    Xs = StandardScaler().fit_transform(X)
    mi = mutual_info_classif(Xs, y, random_state=42, n_neighbors=5)
    top_idx = np.argsort(mi)[::-1][:top_n]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, idx in zip(axes.flatten(), top_idx):
        feat = X[:, idx]
        v = feat[np.isfinite(feat)]
        if len(v) < 10:
            continue
        p99, p01 = np.percentile(v, [1, 99])
        mask = (feat >= p01) & (feat <= p99) & np.isfinite(feat)
        ax.hist(feat[mask & (y == 0)], bins=50, alpha=0.5, color="blue", label="No flare", density=True)
        ax.hist(feat[mask & (y == 1)], bins=50, alpha=0.5, color="red", label="Flare", density=True)
        ax.set_title(f"{feature_names[idx]}\nMI={mi[idx]:.3f}", fontsize=9)
        ax.legend(fontsize=7)

    plt.suptitle("Feature Distributions: Flare vs No-Flare", fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save:
        fig.savefig(PLOTS_STATISTICS / "feature_distributions.png", dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return fig


# ── Model evaluation ────────────────────────────────────────────────────

def plot_model_evaluation(
    results: dict[str, dict],
    save: bool = True,
    show: bool = False,
) -> plt.Figure:
    """ROC curves, PR curves, and metric comparison bar chart."""
    from sklearn.metrics import roc_curve, precision_recall_curve

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    for name, r in results.items():
        fpr, tpr, _ = roc_curve(r["y_test"], r["y_pred_prob"])
        ax.plot(fpr, tpr, lw=2, label=f'{name} (AUC={r["auc_roc"]:.3f})')
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title("ROC Curves")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1]
    for name, r in results.items():
        prec, rec, _ = precision_recall_curve(r["y_test"], r["y_pred_prob"])
        ax.plot(rec, prec, lw=2, label=f'{name} (AP={r["auc_pr"]:.3f})')
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title("PR Curves")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[2]
    names = list(results.keys())
    metrics = ["precision", "recall", "f1"]
    x = np.arange(len(metrics))
    w = 0.25
    for i, name in enumerate(names):
        vals = [results[name].get(m, 0) for m in metrics]
        ax.bar(x + i * w, vals, w, label=name, alpha=0.7)
    ax.set_xticks(x + w); ax.set_xticklabels(["Precision", "Recall", "F1"])
    ax.set_ylim(0, 1); ax.set_title("Model Comparison")
    ax.legend(); ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle("Forecasting Model Evaluation", fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save:
        fig.savefig(PLOTS_FORECAST / "model_evaluation.png", dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return fig
