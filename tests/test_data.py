"""Tests for bah2026.data module (readers + preprocessing)."""

from datetime import date
from pathlib import Path

import numpy as np
import pytest

from bah2026.data.preprocessing import (
    compute_gti_mask, background_subtract, interpolate_to_common_grid,
    met_to_mjd, align_hel1os_to_solexs,
)


# ── Preprocessing tests ─────────────────────────────────────────────────

def test_compute_gti_mask():
    time_mjd = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    gti = np.array([[1.5, 3.5]])
    mask = compute_gti_mask(time_mjd, gti)
    expected = np.array([False, True, True, False, False])
    np.testing.assert_array_equal(mask, expected)


def test_compute_gti_mask_multiple():
    time_mjd = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    gti = np.array([[1.5, 2.5], [5.5, 6.5]])
    mask = compute_gti_mask(time_mjd, gti)
    expected = np.array([False, True, False, False, False, True, False])
    np.testing.assert_array_equal(mask, expected)


def test_background_subtract():
    counts = np.ones(1000) * 10.0
    counts[500:520] = 100.0
    bg, residual = background_subtract(counts, window_sec=100)
    assert bg.shape == counts.shape
    assert residual.shape == counts.shape
    assert np.median(residual) < 5.0
    assert residual[510] > 20.0


def test_background_subtract_default_window():
    counts = np.random.poisson(10, 1000).astype(float)
    bg, residual = background_subtract(counts)
    assert bg.shape == counts.shape


def test_background_subtract_nan_handling():
    counts = np.ones(1000) * 10.0
    counts[100] = np.nan
    bg, residual = background_subtract(counts, window_sec=50)
    assert np.isfinite(bg[100])
    assert np.isfinite(residual[100])


def test_interpolate_to_common_grid():
    src = np.array([0.0, 1.0, 2.0, 3.0])
    vals = np.array([10.0, 20.0, 30.0, 40.0])
    grid = np.array([0.5, 1.5, 2.5])
    result = interpolate_to_common_grid(src, vals, grid)
    np.testing.assert_allclose(result, [15.0, 25.0, 35.0])


def test_interpolate_out_of_range():
    src = np.array([1.0, 2.0])
    vals = np.array([10.0, 20.0])
    grid = np.array([0.5, 1.5, 2.5])
    result = interpolate_to_common_grid(src, vals, grid)
    assert np.isnan(result[0])
    assert np.isnan(result[2])


def test_met_to_mjd():
    met = np.array([0.0, 86400.0])
    mjd = met_to_mjd(met, 40587, 0.0)
    np.testing.assert_allclose(mjd, [40587.0, 40588.0])


def test_met_to_mjd_with_fractional():
    met = np.array([0.0])
    mjd = met_to_mjd(met, 40587, 0.25)
    np.testing.assert_allclose(mjd, [40587.25])


def test_align_hel1os_to_solexs():
    hel1os_mjd = np.array([40587.0, 40587.5, 40588.0])
    hel1os_ctr = np.array([[100.0, 200.0], [150.0, 250.0], [200.0, 300.0]])
    solexs_met = np.array([0.0, 43200.0, 86399.0])
    mjdrefi, mjdreff = 40587, 0.0

    aligned = align_hel1os_to_solexs(
        hel1os_mjd, hel1os_ctr, solexs_met, mjdrefi, mjdreff
    )
    assert aligned.shape == (3, 2)
    assert np.isfinite(aligned[0, 0])
    assert np.isfinite(aligned[2, 0])


# ── Reader tests (requires actual data) ─────────────────────────────────

@pytest.mark.skipif(
    not (Path(__file__).resolve().parents[1] / "data" / "processed" / "solexs").exists(),
    reason="No SoLEXS data available"
)
def test_load_solexs_lc():
    from bah2026.data.reader import load_solexs_lc
    days_path = Path(__file__).resolve().parents[1] / "data" / "processed" / "solexs"
    yd = sorted(d for d in days_path.iterdir() if d.is_dir())[0]
    md = sorted(d for d in yd.iterdir() if d.is_dir())[0]
    dd = sorted(d for d in md.iterdir() if d.is_dir())[0]
    d = date(int(yd.name), int(md.name), int(dd.name))
    result = load_solexs_lc(d)
    assert "time" in result
    assert "counts" in result
    assert len(result["time"]) == 86400
    assert result["mjdrefi"] > 0


@pytest.mark.skipif(
    not (Path(__file__).resolve().parents[1] / "data" / "processed" / "hel1os").exists(),
    reason="No HEL1OS data available"
)
def test_load_hel1os_lc():
    from bah2026.data.reader import load_hel1os_lc
    days_path = Path(__file__).resolve().parents[1] / "data" / "processed" / "hel1os"
    yd = sorted(d for d in days_path.iterdir() if d.is_dir())[0]
    md = sorted(d for d in yd.iterdir() if d.is_dir())[0]
    dd = sorted(d for d in md.iterdir() if d.is_dir())[0]
    d = date(int(yd.name), int(md.name), int(dd.name))
    result = load_hel1os_lc(d, detector="czt", num=1)
    assert "mjd" in result
    assert "ctr" in result
    assert result["ctr"].ndim == 2
    assert result["ctr"].shape[1] == 5
