"""Tests for bah2026.features.non_thermal module."""

import numpy as np

from bah2026.features.non_thermal import (
    thick_target_spectrum,
    thermal_bremsstrahlung,
    fit_non_thermal,
    separate_thermal_non_thermal,
    fit_combined_spectrum,
    compute_electron_column,
)


# ── Spectrum models ─────────────────────────────────────────────────────


class TestThickTargetSpectrum:
    def test_basic_power_law(self):
        """Thick-target spectrum follows power-law above cutoff."""
        e = np.array([20.0, 40.0, 60.0, 80.0, 100.0])
        spec = thick_target_spectrum(e, gamma=4.0, ec_kev=10.0, norm=1e5)
        assert spec.shape == (5,)
        assert np.all(spec > 0)

    def test_below_cutoff(self):
        """Below cutoff, spectrum is suppressed."""
        e = np.array([5.0, 8.0, 12.0])
        spec = thick_target_spectrum(e, gamma=4.0, ec_kev=10.0, norm=1e5)
        assert np.all(spec >= 0)

    def test_gamma_le_1(self):
        """gamma ≤ 1 returns zeros."""
        e = np.array([10.0, 20.0, 30.0])
        spec = thick_target_spectrum(e, gamma=0.5, ec_kev=5.0, norm=1e5)
        assert np.allclose(spec, 0.0)

    def test_steeper_gamma_lower_flux(self):
        """Steeper gamma → lower flux at high energy."""
        e = np.array([100.0])
        spec4 = thick_target_spectrum(e, gamma=4.0, ec_kev=10.0, norm=1e5)
        spec6 = thick_target_spectrum(e, gamma=6.0, ec_kev=10.0, norm=1e5)
        assert spec6[0] < spec4[0]


class TestThermalBremsstrahlung:
    def test_basic(self):
        """Thermal bremsstrahlung produces positive flux."""
        e = np.array([2.0, 5.0, 10.0, 15.0])
        spec = thermal_bremsstrahlung(e, t_mk=20.0, em=1e3)
        assert spec.shape == (4,)
        assert np.all(spec > 0)

    def test_zero_temperature(self):
        """Zero temperature returns zeros."""
        e = np.array([5.0, 10.0])
        spec = thermal_bremsstrahlung(e, t_mk=0.0, em=1e3)
        assert np.allclose(spec, 0.0)

    def test_exponential_decay(self):
        """Flux decreases with energy (exp(-E/kT))."""
        e = np.array([2.0, 5.0, 10.0, 20.0])
        spec = thermal_bremsstrahlung(e, t_mk=10.0, em=1e3)
        assert spec[0] > spec[1] > spec[2] > spec[3]


# ── Non-thermal fitting ─────────────────────────────────────────────────


class TestFitNonThermal:
    def test_recovery(self):
        """Recover known power-law parameters."""
        e = np.array([20.0, 40.0, 60.0, 80.0, 100.0, 120.0, 150.0])
        true_counts = thick_target_spectrum(e, gamma=4.0, ec_kev=10.0, norm=1e5)
        result = fit_non_thermal(e, true_counts)
        assert abs(result["gamma"] - 4.0) < 0.5
        assert result["chi2_red"] < 10.0
        assert result["delta"] == result["gamma"] + 1.0

    def test_insufficient_data(self):
        """Too few points returns zeros."""
        e = np.array([20.0, 30.0])
        c = np.array([100.0, 50.0])
        result = fit_non_thermal(e, c)
        assert result["gamma"] == 0.0
        assert result["chi2_red"] == 999.0

    def test_zero_counts(self):
        """Zero counts returns zeros."""
        e = np.array([20.0, 40.0, 60.0, 80.0])
        result = fit_non_thermal(e, np.zeros(4))
        assert result["gamma"] == 0.0


# ── Thermal/non-thermal separation ──────────────────────────────────────


class TestSeparateThermalNonThermal:
    def test_basic_separation(self):
        """Separate thermal + non-thermal components."""
        e = np.linspace(5, 150, 50)
        thermal = thermal_bremsstrahlung(e, 20.0, 1e3)
        nonthermal = thick_target_spectrum(e, 4.0, 10.0, 1e4)
        combined = thermal + nonthermal
        result = separate_thermal_non_thermal(e, combined, 20.0, 1e3)
        assert "t_mk" in result
        assert "gamma" in result
        assert "thermal_fraction" in result
        assert 0.0 <= result["thermal_fraction"] <= 1.0

    def test_returns_all_keys(self):
        """All expected keys are present."""
        e = np.linspace(5, 150, 50)
        c = np.random.poisson(100, 50).astype(float)
        result = separate_thermal_non_thermal(e, c, 15.0, 1e3)
        expected_keys = {
            "t_mk",
            "em",
            "gamma",
            "ec",
            "n_nth",
            "delta",
            "thermal_flux",
            "nonthermal_flux",
            "residual",
            "thermal_fraction",
            "nonthermal_chi2",
        }
        assert expected_keys.issubset(result.keys())


# ── Combined spectrum ───────────────────────────────────────────────────


class TestFitCombinedSpectrum:
    def test_basic(self):
        """Combined SoLEXS + HEL1OS fit."""
        solexs_e = np.linspace(2, 22, 50)
        solexs_c = thermal_bremsstrahlung(solexs_e, 15.0, 1e3)
        hel1os_e = np.array([20, 40, 60, 80, 100, 120, 150.0])
        hel1os_c = thick_target_spectrum(hel1os_e, 4.0, 10.0, 1e4)
        result = fit_combined_spectrum(solexs_e, solexs_c, hel1os_e, hel1os_c)
        assert "t_mk" in result
        assert "gamma" in result
        assert "combined_range_kev" in result


# ── Electron column ─────────────────────────────────────────────────────


class TestComputeElectronColumn:
    def test_basic(self):
        """Electron column computation."""
        n = compute_electron_column(gamma=4.0, ec_kev=10.0, flux_norm=1e5)
        assert n > 0

    def test_delta_le_1(self):
        """delta ≤ 1 returns 0."""
        n = compute_electron_column(gamma=4.0, ec_kev=10.0, flux_norm=1e5, delta=0.5)
        assert n == 0.0

    def test_zero_ec(self):
        """Zero Ec returns 0."""
        n = compute_electron_column(gamma=4.0, ec_kev=0.0, flux_norm=1e5)
        assert n == 0.0
