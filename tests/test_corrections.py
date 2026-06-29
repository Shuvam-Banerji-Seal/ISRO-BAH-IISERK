"""Tests for bah2026.data.corrections module."""

import numpy as np
import pytest

from bah2026.data.corrections import (
    correct_solexs_deadtime,
    subtract_hel1os_background,
    subtract_solexs_spurious,
    correct_hel1os_deadtime_approx,
    apply_all_corrections,
    SOLEXS_TAU_SPECTRAL,
    SOLEXS_SPURIOUS_RATE,
    HEL1OS_BG_CZT_CPS,
    HEL1OS_BG_CDTE_CPS,
)


# ── SoLEXS deadtime correction ──────────────────────────────────────────


class TestSolexsDeadtime:
    def test_zero_counts(self):
        """Zero counts → zero corrected."""
        result = correct_solexs_deadtime(np.zeros(100))
        assert np.allclose(result, 0.0)

    def test_low_counts_minimal_correction(self):
        """Low count rates have minimal deadtime correction."""
        counts = np.full(100, 100.0)  # 100 cps
        result = correct_solexs_deadtime(counts)
        # At 100 cps with τ=13.65µs, correction should be < 0.2%
        pct = (result - counts) / counts * 100
        assert np.all(pct > 0)  # Corrected should be higher
        assert np.all(pct < 0.5)  # Less than 0.5% correction

    def test_high_counts_significant_correction(self):
        """High count rates have significant deadtime correction."""
        counts = np.array([20000.0])  # 20k cps
        result = correct_solexs_deadtime(counts)
        pct = (result[0] - counts[0]) / counts[0] * 100
        assert pct > 10.0  # At least 10% correction at 20k cps

    def test_paralyzable_limit(self):
        """Above paralyzable limit (1/τ), correction saturates."""
        n_max = 1.0 / SOLEXS_TAU_SPECTRAL  # ~73,260 cps
        counts = np.array([n_max * 0.99])
        result = correct_solexs_deadtime(counts)
        # Should not exceed the limit
        assert result[0] <= n_max * 1.01

    def test_preserves_shape(self):
        """Output shape matches input shape."""
        for shape in [(100,), (50, 5), (10, 20, 30)]:
            counts = np.random.poisson(100, shape).astype(float)
            result = correct_solexs_deadtime(counts)
            assert result.shape == counts.shape

    def test_monotonic_increase(self):
        """Higher measured rates → higher corrected rates (below limit)."""
        rates = np.array([100, 1000, 5000, 10000, 20000])
        result = correct_solexs_deadtime(rates)
        assert np.all(np.diff(result) > 0)


# ── HEL1OS background subtraction ───────────────────────────────────────


class TestHel1osBackground:
    def test_czt_single_band(self):
        """CZT single-band subtraction."""
        ctr = np.array([100.0, 150.0, 200.0])
        result = subtract_hel1os_background(ctr, "czt")
        assert np.allclose(result, [30.0, 80.0, 130.0])

    def test_cdte_single_band(self):
        """CdTe single-band subtraction."""
        ctr = np.array([1.0, 2.0, 0.5])
        result = subtract_hel1os_background(ctr, "cdte")
        assert np.allclose(result, [0.85, 1.85, 0.35])

    def test_czt_multi_band(self):
        """CZT multi-band subtraction uses per-band background."""
        ctr = np.array([[100.0, 100.0, 100.0, 100.0, 100.0]])
        result = subtract_hel1os_background(ctr, "czt")
        # Bands: [15, 12, 8, 5, 70] cps background
        assert result.shape == (1, 5)
        assert result[0, 0] == 85.0  # 100 - 15
        assert result[0, 4] == 30.0  # 100 - 70

    def test_no_negative(self):
        """Background-subtracted values are never negative."""
        ctr = np.array([10.0, 5.0, 1.0])
        result = subtract_hel1os_background(ctr, "czt")
        assert np.all(result >= 0.0)

    def test_zero_counts(self):
        """Zero counts → zero after subtraction."""
        result = subtract_hel1os_background(np.zeros(10), "czt")
        assert np.allclose(result, 0.0)


# ── Spurious count subtraction ──────────────────────────────────────────


class TestSpuriousSubtraction:
    def test_basic(self):
        """Spurious count subtraction."""
        counts = np.array([600.0, 500.0, 1000.0])
        result = subtract_solexs_spurious(counts)
        assert np.allclose(result, [100.0, 0.0, 500.0])

    def test_no_negative(self):
        """No negative values after subtraction."""
        counts = np.array([100.0, 200.0, 300.0])
        result = subtract_solexs_spurious(counts)
        assert np.all(result >= 0.0)


# ── Combined corrections ────────────────────────────────────────────────


class TestApplyAllCorrections:
    def test_solexs_only(self):
        """Apply corrections to SoLEXS only."""
        counts = np.full(100, 500.0)
        result = apply_all_corrections(solexs_counts=counts)
        assert "solexs_corrected" in result
        assert "stats" in result
        assert result["solexs_corrected"] is not None

    def test_hel1os_only(self):
        """Apply corrections to HEL1OS only."""
        ctr = np.full((100, 5), 100.0)
        result = apply_all_corrections(hel1os_ctr=ctr, hel1os_detector="czt")
        assert "hel1os_corrected" in result
        assert result["hel1os_corrected"].shape == (100, 5)

    def test_both(self):
        """Apply corrections to both instruments."""
        counts = np.full(100, 500.0)
        ctr = np.full((100, 5), 100.0)
        result = apply_all_corrections(
            solexs_counts=counts, hel1os_ctr=ctr, hel1os_detector="czt"
        )
        assert result["solexs_corrected"] is not None
        assert result["hel1os_corrected"] is not None
        assert "solexs_deadtime_max_corr_pct" in result["stats"]
        assert "hel1os_bg_fraction_pct" in result["stats"]
