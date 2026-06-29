"""Tests for bah2026.features.response_convolution module."""

import numpy as np
import pytest

from bah2026.features.response_convolution import (
    build_response_matrix,
    convolve_model,
    deconvolve_spectrum,
    effective_area_at_energy,
    counts_to_energy_flux,
    has_caldb,
)


# ── CALDB availability ──────────────────────────────────────────────────


def test_has_caldb():
    """CALDB files are available."""
    assert has_caldb() is True


# ── Response matrix ─────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def response():
    """Build response matrix once for all tests."""
    return build_response_matrix()


class TestBuildResponseMatrix:
    def test_matrix_shape(self, response):
        """Response matrix has correct shape."""
        assert response["matrix"].shape[0] == 340  # channels
        assert response["matrix"].shape[1] > 0  # energy bins

    def test_arf_positive(self, response):
        """ARF values are non-negative."""
        assert np.all(response["arf"] >= 0.0)

    def test_channel_energies(self, response):
        """Channel energies are in correct range."""
        lo = response["channel_energies_lo"]
        hi = response["channel_energies_hi"]
        assert np.all(lo > 0)
        assert np.all(hi > lo)
        assert lo.min() < 5.0  # SoLEXS starts at ~2 keV
        assert hi.max() > 15.0  # Goes up to ~22 keV


# ── Forward folding ─────────────────────────────────────────────────────


class TestConvolveModel:
    def test_basic(self, response):
        """Forward folding produces counts."""
        n_energy = response["matrix"].shape[1]
        model = np.ones(n_energy) * 10.0
        counts = convolve_model(model, response)
        assert counts.shape[0] == 340
        assert np.all(counts >= 0.0)

    def test_zero_model(self, response):
        """Zero model → zero counts."""
        n_energy = response["matrix"].shape[1]
        model = np.zeros(n_energy)
        counts = convolve_model(model, response)
        assert np.allclose(counts, 0.0)


# ── Deconvolution ───────────────────────────────────────────────────────


class TestDeconvolveSpectrum:
    def test_nnls(self, response):
        """NNLS deconvolution produces non-negative flux."""
        counts = np.random.poisson(10, 340).astype(float)
        flux = deconvolve_spectrum(counts, response, method="nnls")
        # NNLS returns n_energy bins, fallback returns n_channels
        assert len(flux) > 0
        assert np.all(flux >= 0.0)

    def test_richardson_lucy(self, response):
        """Richardson-Lucy deconvolution produces non-negative flux."""
        counts = np.random.poisson(10, 340).astype(float)
        flux = deconvolve_spectrum(
            counts, response, method="richardson_lucy", max_iter=20
        )
        assert len(flux) > 0
        assert np.all(flux >= 0.0)

    def test_invalid_method(self, response):
        """Invalid method raises ValueError."""
        counts = np.zeros(340)
        with pytest.raises(ValueError):
            deconvolve_spectrum(counts, response, method="invalid")


# ── Effective area ──────────────────────────────────────────────────────


class TestEffectiveArea:
    def test_scalar(self):
        """Scalar energy input."""
        area = effective_area_at_energy(5.0)
        area_float = float(area)
        assert isinstance(area_float, float)

    def test_array(self):
        """Array energy input."""
        energies = np.array([2.0, 5.0, 10.0, 15.0, 20.0])
        areas = np.asarray(effective_area_at_energy(energies))
        assert areas.shape == (5,)

    def test_out_of_range(self):
        """Out-of-range energy returns 0."""
        area = effective_area_at_energy(1000.0)
        assert float(area) == 0.0


# ── Energy flux ─────────────────────────────────────────────────────────


class TestCountsToEnergyFlux:
    def test_basic(self, response):
        """Energy flux computation."""
        counts = np.random.poisson(100, 340).astype(float)
        flux = counts_to_energy_flux(counts, response=response)
        assert isinstance(flux, float)
        assert flux >= 0.0

    def test_zero_counts(self, response):
        """Zero counts → zero flux."""
        flux = counts_to_energy_flux(np.zeros(340), response=response)
        assert flux == 0.0
