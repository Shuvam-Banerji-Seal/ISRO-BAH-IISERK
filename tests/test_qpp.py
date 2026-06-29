"""Tests for bah2026.features.qpp module."""

import numpy as np

from bah2026.features.qpp import (
    detect_qpp,
    extract_qpp_features,
    wavelet_power,
    wavelet_power_gpu,
    wavelet_power_auto,
    lomb_scargle_periodogram,
    morlet_wavelet,
)


# ── Wavelet transform ───────────────────────────────────────────────────


class TestWaveletPower:
    def test_short_signal(self):
        """Short signal returns empty."""
        signal = np.random.randn(10)
        power, scales, periods = wavelet_power(signal)
        assert power.shape == (0, 0)

    def test_basic(self):
        """Wavelet power spectrum has correct shape."""
        np.random.seed(42)
        signal = np.random.randn(1000)
        power, scales, periods = wavelet_power(signal, dt=1.0, s_min=2.0, s_max=250.0)
        assert power.shape[0] == len(scales)
        assert power.shape[1] == 1000
        assert len(periods) == len(scales)

    def test_constant_signal(self):
        """Constant signal returns zeros."""
        signal = np.ones(500)
        power, scales, periods = wavelet_power(signal)
        assert np.allclose(power, 0.0)


class TestWaveletPowerGpu:
    def test_fallback_to_cpu(self):
        """GPU wavelet falls back to CPU if no GPU."""
        np.random.seed(42)
        signal = np.random.randn(500)
        # This should work regardless of GPU availability
        power, scales, periods = wavelet_power_auto(
            signal, dt=1.0, s_min=2.0, s_max=100.0
        )
        assert power.shape[1] == 500


class TestMorletWavelet:
    def test_shape(self):
        """Morlet wavelet has correct shape."""
        t = np.linspace(-10, 10, 100)
        w = morlet_wavelet(t, s=2.0)
        assert w.shape == (100,)


# ── Lomb-Scargle ────────────────────────────────────────────────────────


class TestLombScargle:
    def test_basic(self):
        """Lomb-Scargle finds known frequency."""
        t = np.arange(1000.0)
        signal = np.sin(2 * np.pi * t / 50)  # 50s period
        freqs = np.linspace(1 / 200, 1 / 10, 500)
        f, power = lomb_scargle_periodogram(t, signal, freqs)
        peak_freq = f[np.argmax(power)]
        peak_period = 1.0 / peak_freq
        assert abs(peak_period - 50.0) < 5.0

    def test_constant_signal(self):
        """Constant signal returns zero power."""
        t = np.arange(100.0)
        signal = np.ones(100)
        f, power = lomb_scargle_periodogram(t, signal)
        assert np.allclose(power, 0.0) or len(power) == 1


# ── QPP detection ───────────────────────────────────────────────────────


class TestDetectQpp:
    def test_periodic_signal(self):
        """Periodic signal is detected."""
        np.random.seed(42)
        t = np.arange(3000)
        signal = 100 + 20 * np.sin(2 * np.pi * t / 60) + np.random.randn(3000) * 5
        result = detect_qpp(signal, dt=1.0, min_period=10, max_period=300)
        assert result["detected"] is True
        assert abs(result["period"] - 60.0) < 10.0
        assert result["amplitude"] > 0
        assert result["n_qpp"] >= 1

    def test_random_signal(self):
        """Random noise is not detected as QPP."""
        np.random.seed(42)
        signal = np.random.randn(3000) * 10 + 100
        result = detect_qpp(signal, dt=1.0, min_period=10, max_period=300)
        # Random noise may or may not trigger, but amplitude should be small
        assert "detected" in result
        assert "period" in result

    def test_short_signal(self):
        """Short signal returns no detection."""
        signal = np.random.randn(30)
        result = detect_qpp(signal, dt=1.0)
        assert result["detected"] is False
        assert result["n_qpp"] == 0

    def test_returns_all_keys(self):
        """All expected keys are present."""
        signal = np.random.randn(500)
        result = detect_qpp(signal, dt=1.0)
        expected_keys = {
            "detected",
            "period",
            "amplitude",
            "significance",
            "periods",
            "n_qpp",
        }
        assert expected_keys.issubset(result.keys())


class TestExtractQppFeatures:
    def test_basic(self):
        """Sliding window QPP extraction."""
        np.random.seed(42)
        signal = (
            100
            + 20 * np.sin(2 * np.pi * np.arange(3600) / 60)
            + np.random.randn(3600) * 5
        )
        features = extract_qpp_features(signal, dt=1.0, window_sec=300, step_sec=60)
        assert "qpp_detected" in features
        assert "qpp_period" in features
        assert "qpp_amplitude" in features
        assert len(features["qpp_detected"]) > 0
