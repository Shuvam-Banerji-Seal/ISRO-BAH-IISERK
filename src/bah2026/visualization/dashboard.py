"""Streamlit dashboard for real-time flare monitoring and alerting.

Usage:
    streamlit run src/bah2026/visualization/dashboard.py

Features:
    - SoLEXS + HEL1OS light curve display
    - Nowcast overlay (detected flares highlighted)
    - Alert panel (GREEN/YELLOW/ORANGE/RED/FLARE NOW)
    - Time slider for navigating through days
    - Spectral viewer (when PI data available)
    - Multi-tier alerting system
"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# Page config must be first
st.set_page_config(
    page_title="BAH 2026 — Flare Monitor",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Paths ──────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bah2026.config import DATA_ROOT, CATALOGS_DIR
from bah2026.data.reader import (
    load_solexs_lc,
    load_hel1os_lc,
    load_solexs_gti,
    discover_solexs_days,
    discover_hel1os_days,
    discover_combined_days,
)
from bah2026.data.preprocessing import background_subtract, met_to_mjd


# ── Alert System ──────────────────────────────────────────────────────

ALERT_LEVELS = {
    "GREEN": {"color": "#00CC00", "icon": "✅", "desc": "No flare expected"},
    "YELLOW": {"color": "#FFCC00", "icon": "⚠️", "desc": "Precursor index > 0.6"},
    "ORANGE": {
        "color": "#FF8800",
        "icon": "🔶",
        "desc": "Probability > 70% for ≤15min",
    },
    "RED": {"color": "#FF0000", "icon": "🔴", "desc": "Probability > 90% for ≤5min"},
    "FLARE NOW": {"color": "#FF00FF", "icon": "💥", "desc": "Nowcast confirmed!"},
}


def compute_alert_level(
    flare_probability: float = 0.0,
    precursor_index: float = 0.0,
    is_nowcasted: bool = False,
) -> str:
    """Determine alert level from flare probability and precursor state."""
    if is_nowcasted:
        return "FLARE NOW"
    if flare_probability > 0.9:
        return "RED"
    if flare_probability > 0.7:
        return "ORANGE"
    if precursor_index > 0.6:
        return "YELLOW"
    return "GREEN"


# ── Data Loading ──────────────────────────────────────────────────────


@st.cache_data
def load_catalogue() -> pd.DataFrame:
    """Load the nowcast catalogue if available."""
    csv = CATALOGS_DIR / "nowcast_catalogue.csv"
    if csv.exists():
        return pd.read_csv(csv)
    return pd.DataFrame()


@st.cache_data
def load_day_data(d: date) -> dict:
    """Load all data for a single day."""
    result: dict = {"date": d, "solexs": None, "hel1os_czt": None, "hel1os_cdte": None}

    try:
        solexs = load_solexs_lc(d)
        gti = load_solexs_gti(d)
        result["solexs"] = solexs
        result["gti"] = gti
    except Exception:
        pass

    try:
        czt = load_hel1os_lc(d, detector="czt", num=1)
        result["hel1os_czt"] = czt
    except Exception:
        pass

    try:
        cdte = load_hel1os_lc(d, detector="cdte", num=1)
        result["hel1os_cdte"] = cdte
    except Exception:
        pass

    return result


# ── Plotting ──────────────────────────────────────────────────────────


def plot_combined_lc(
    d: date,
    solexs_data: dict | None,
    hel1os_czt: dict | None,
    hel1os_cdte: dict | None,
    flares: pd.DataFrame | None = None,
):
    """Render combined light curve plot with matplotlib in Streamlit."""
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(4, 1, height_ratios=[3, 2, 2, 1], hspace=0.3)

    # SoLEXS panel
    ax1 = fig.add_subplot(gs[0])
    if solexs_data is not None:
        t_h = solexs_data["time"] / 3600.0
        counts = np.where(
            np.isfinite(solexs_data["counts"]), solexs_data["counts"], np.nan
        )
        ax1.plot(t_h, counts, "b-", lw=0.3, alpha=0.7)
        bg, _ = background_subtract(
            np.where(
                np.isfinite(solexs_data["counts"]),
                solexs_data["counts"],
                np.nanmedian(solexs_data["counts"]),
            )
        )
        ax1.plot(t_h, bg, "gray", lw=1, alpha=0.5, label="Background")
        ax1.set_ylabel("SoLEXS Counts/s")

        # Overlay flares
        if flares is not None and len(flares) > 0:
            day_flares = flares[flares["date"] == str(d)]
            for _, evt in day_flares.iterrows():
                ax1.axvspan(
                    evt["start_time"] / 3600,
                    evt["end_time"] / 3600,
                    alpha=0.2,
                    color="red",
                )
                ax1.axvline(evt["peak_time"] / 3600, color="red", lw=1, ls="--")
                cls = evt.get("goes_class", "?")
                ax1.annotate(
                    cls,
                    xy=(evt["peak_time"] / 3600, evt["peak_flux"]),
                    fontsize=7,
                    color="red",
                    ha="center",
                    va="bottom",
                )
    else:
        ax1.text(
            0.5,
            0.5,
            "No SoLEXS data",
            transform=ax1.transAxes,
            ha="center",
            va="center",
            color="gray",
        )

    ax1.set_title(f"Aditya-L1 X-ray Light Curves — {d}", fontsize=13, fontweight="bold")
    ax1.set_xlim(0, 24)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8)

    # HEL1OS CZT panel
    ax2 = fig.add_subplot(gs[1])
    if hel1os_czt is not None:
        mjd = hel1os_czt["mjd"]
        t_h = (mjd - mjd[0]) * 24.0
        ax2.plot(t_h, hel1os_czt["ctr"][:, -1], "r-", lw=0.3, alpha=0.7)
        ax2.set_ylabel("CZT1 (18–160 keV)")
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(
            0.5,
            0.5,
            "No CZT data",
            transform=ax2.transAxes,
            ha="center",
            va="center",
            color="gray",
        )

    # HEL1OS CdTe panel
    ax3 = fig.add_subplot(gs[2])
    if hel1os_cdte is not None:
        mjd = hel1os_cdte["mjd"]
        t_h = (mjd - mjd[0]) * 24.0
        ax3.plot(t_h, hel1os_cdte["ctr"][:, -1], "orange", lw=0.3, alpha=0.7)
        ax3.set_ylabel("CdTe1 (1.8–90 keV)")
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(
            0.5,
            0.5,
            "No CdTe data",
            transform=ax3.transAxes,
            ha="center",
            va="center",
            color="gray",
        )

    # Hardness ratio panel
    ax4 = fig.add_subplot(gs[3])
    if hel1os_czt is not None and hel1os_cdte is not None:
        # Compute CZT / CdTe ratio where both available
        czt_full = hel1os_czt["ctr"][:, -1]
        cdte_full = hel1os_cdte["ctr"][:, -1]
        min_len = min(len(czt_full), len(cdte_full))
        ratio = np.where(
            cdte_full[:min_len] > 0, czt_full[:min_len] / cdte_full[:min_len], np.nan
        )
        mjd_czt = hel1os_czt["mjd"][:min_len]
        t_r = (mjd_czt - mjd_czt[0]) * 24.0
        ax4.plot(t_r, ratio, "g-", lw=0.3, alpha=0.7)
        ax4.set_ylabel("CZT/CdTe")
        ax4.grid(True, alpha=0.3)
    elif solexs_data is not None and hel1os_czt is not None:
        # SXR/HXR ratio
        mjd_solexs = met_to_mjd(
            solexs_data["time"], solexs_data["mjdrefi"], solexs_data["mjdreff"]
        )
        czt = hel1os_czt["ctr"][:, -1]
        min_len = min(len(mjd_solexs), len(czt))
        ratio = np.where(
            czt[:min_len] > 0, solexs_data["counts"][:min_len] / czt[:min_len], np.nan
        )
        t_r = np.arange(min_len) / 3600.0
        ax4.plot(t_r[:min_len:60], ratio[::60], "g.", ms=0.5, alpha=0.5)
        ax4.set_ylabel("SXR/HXR")
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(
            0.5,
            0.5,
            "No HXR data",
            transform=ax4.transAxes,
            ha="center",
            va="center",
            color="gray",
        )

    ax4.set_xlabel("Time (hours from start)")
    ax4.set_xlim(0, 24)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def render_alert_panel(alert_level: str):
    """Render the multi-tier alert panel."""
    info = ALERT_LEVELS[alert_level]
    st.markdown(
        f"""
        <div style="
            padding: 1.5rem;
            border-radius: 0.5rem;
            background-color: {info["color"]}22;
            border: 3px solid {info["color"]};
            text-align: center;
        ">
            <h1 style="margin: 0; font-size: 3rem;">{info["icon"]}</h1>
            <h2 style="margin: 0.5rem 0; color: {info["color"]};">
                {alert_level}
            </h2>
            <p style="margin: 0; font-size: 0.9rem;">{info["desc"]}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Main Dashboard ────────────────────────────────────────────────────


def main():
    st.title("☀️ BAH 2026 — Solar Flare Monitor")
    st.markdown("Aditya-L1 **SoLEXS** (2–22 keV) + **HEL1OS** (1.8–160 keV)")

    # Sidebar controls
    st.sidebar.header("Controls")

    # Date selection
    combined_days = discover_combined_days()
    if not combined_days:
        st.error("No combined SoLEXS + HEL1OS data found in data/processed/")
        st.info("Extract the datasets first, then restart this dashboard.")
        return

    # Date slider
    date_index = st.sidebar.slider(
        "Select Day",
        0,
        len(combined_days) - 1,
        len(combined_days) - 1,
        format="%d",
    )
    selected_date = combined_days[date_index]

    # Show date range
    st.sidebar.caption(
        f"Date range: {combined_days[0]} → {combined_days[-1]}"
        f"\n{len(combined_days)} combined days"
    )

    # Load catalogue
    catalogue = load_catalogue()

    # Load data for selected day
    with st.spinner("Loading data..."):
        day_data = load_day_data(selected_date)

    # ── Main layout ─────────────────────────────────────────────────
    col1, col2 = st.columns([3, 1])

    with col1:
        # Light curve plot
        flares = None if catalogue.empty else catalogue
        plot_combined_lc(
            selected_date,
            day_data.get("solexs"),
            day_data.get("hel1os_czt"),
            day_data.get("hel1os_cdte"),
            flares,
        )

    with col2:
        # Alert panel
        st.subheader("🚨 Alert Status")
        st.markdown("---")

        # Compute alert level from catalogue data
        alert_level = "GREEN"
        if not catalogue.empty:
            day_flares = catalogue[catalogue["date"] == str(selected_date)]
            if len(day_flares) > 0:
                alert_level = "FLARE NOW"
                # Check if any flares are high class
                for _, evt in day_flares.iterrows():
                    cls = evt.get("goes_class", "A")
                    if cls in ("M", "X"):
                        alert_level = "FLARE NOW"
                        break
        render_alert_panel(alert_level)

        # Flare statistics for the day
        st.subheader("📊 Today's Flares")
        if not catalogue.empty:
            day_flares = catalogue[catalogue["date"] == str(selected_date)]
            if len(day_flares) > 0:
                st.metric("Flare Count", len(day_flares))
                max_class = (
                    day_flares["goes_class"].max()
                    if "goes_class" in day_flares
                    else "?"
                )
                st.metric("Max Class", max_class)
                hxr_confirmed = (
                    day_flares["has_hxr"].sum() if "has_hxr" in day_flares else 0
                )
                st.metric("HXR Confirmed", f"{hxr_confirmed}/{len(day_flares)}")
            else:
                st.info("No flares detected this day")
        else:
            st.info("No nowcast catalogue available. Run `bah2026 nowcast` first.")

        # Quick stats
        st.subheader("📈 Overview")
        if not catalogue.empty:
            st.metric("Total Flares", len(catalogue))
            st.metric("Days with Flares", catalogue["date"].nunique())
            if "goes_class" in catalogue.columns:
                class_counts = catalogue["goes_class"].value_counts()
                for cls in ["X", "M", "C", "B", "A"]:
                    cnt = class_counts.get(cls, 0)
                    if cnt > 0:
                        st.metric(f"Class {cls}", int(cnt))

        # Instrument coverage
        st.subheader("🛰️ Coverage")
        if day_data.get("solexs"):
            st.success("SoLEXS: Available")
        else:
            st.error("SoLEXS: No data")
        if day_data.get("hel1os_czt"):
            st.success("HEL1OS CZT: Available")
        else:
            st.error("HEL1OS CZT: No data")
        if day_data.get("hel1os_cdte"):
            st.success("HEL1OS CdTe: Available")
        else:
            st.error("HEL1OS CdTe: No data")

    # ── Expanded sections ───────────────────────────────────────────
    with st.expander("🔬 Spectral Evolution Viewer"):
        st.info(
            "Full spectral viewer requires PI data loading — enable for detailed analysis"
        )
        try:
            pi_data = None  # Could load PI if needed: load_solexs_pi(selected_date)
            if pi_data is not None:
                st.write("Spectral data loaded")
        except Exception:
            st.write("Spectral data not available for this day")

    with st.expander("📋 Raw Data Table"):
        if catalogue is not None and not catalogue.empty:
            day_flares = catalogue[catalogue["date"] == str(selected_date)]
            if len(day_flares) > 0:
                st.dataframe(day_flares, use_container_width=True)
            else:
                st.write("No flares detected this day")
        else:
            st.write("No catalogue loaded")


if __name__ == "__main__":
    main()
