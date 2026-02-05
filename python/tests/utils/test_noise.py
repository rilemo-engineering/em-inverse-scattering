"""
Unit tests for utils/noise.py - Noise generation utilities.

Tests cover:
- AWGN (Additive White Gaussian Noise)
- SNR estimation
- Reproducibility with seeds
- Complex vs real signals

SNR Definition:
    SNR_dB = 10 * log10(P_signal / P_noise)
    P_noise = P_signal / 10^(SNR_dB/10)
"""

import numpy as np
import pytest

from inverse_scattering.utils.noise import (
    awgn,
    estimate_snr,
    add_noise_snr,
)


class TestAWGN:
    """Tests for awgn function."""

    def test_output_shape_unchanged(self):
        """Output should have same shape as input."""
        signal = np.random.randn(10, 8) + 1j * np.random.randn(10, 8)

        noisy = awgn(signal, snr_db=30)

        assert noisy.shape == signal.shape

    def test_output_is_complex_for_complex_input(self):
        """Complex input should give complex output."""
        signal = np.ones((5, 5), dtype=complex) * (1 + 1j)

        noisy = awgn(signal, snr_db=30)

        assert np.iscomplexobj(noisy)

    def test_output_is_real_for_real_input(self):
        """Real input should give real output."""
        signal = np.ones((5, 5))

        noisy = awgn(signal, snr_db=30)

        assert not np.iscomplexobj(noisy)

    def test_higher_snr_means_less_noise(self):
        """Higher SNR should result in less noise (closer to original)."""
        signal = np.ones((10, 10)) + 1j * np.ones((10, 10))

        noisy_high = awgn(signal, snr_db=40, seed=42)
        noisy_low = awgn(signal, snr_db=10, seed=42)

        error_high = np.mean(np.abs(noisy_high - signal)**2)
        error_low = np.mean(np.abs(noisy_low - signal)**2)

        assert error_high < error_low

    def test_achieves_target_snr(self):
        """Achieved SNR should be close to target (statistically)."""
        np.random.seed(12345)
        signal = np.random.randn(100, 100) + 1j * np.random.randn(100, 100)
        target_snr = 20

        noisy = awgn(signal, snr_db=target_snr)
        achieved_snr = estimate_snr(noisy, signal)

        # Should be within ~3 dB due to statistical variation
        assert abs(achieved_snr - target_snr) < 3

    def test_reproducibility_with_seed(self):
        """Same seed should give same noise."""
        signal = np.random.randn(10, 10) + 1j * np.random.randn(10, 10)

        noisy1 = awgn(signal, snr_db=20, seed=123)
        noisy2 = awgn(signal, snr_db=20, seed=123)

        np.testing.assert_array_equal(noisy1, noisy2)

    def test_different_seeds_give_different_noise(self):
        """Different seeds should give different noise."""
        signal = np.ones((10, 10), dtype=complex)

        noisy1 = awgn(signal, snr_db=20, seed=123)
        noisy2 = awgn(signal, snr_db=20, seed=456)

        assert not np.allclose(noisy1, noisy2)

    def test_measured_signal_power(self):
        """'measured' option should use actual signal power."""
        signal = 10 * np.ones((10, 10), dtype=complex)  # Strong signal
        weak_signal = 0.01 * np.ones((10, 10), dtype=complex)  # Weak signal

        # Same SNR, but different absolute noise levels
        noisy_strong = awgn(signal, snr_db=20, seed=42)
        noisy_weak = awgn(weak_signal, snr_db=20, seed=42)

        noise_strong = np.abs(noisy_strong - signal)
        noise_weak = np.abs(noisy_weak - weak_signal)

        # Strong signal should have proportionally stronger noise (same relative)
        assert np.mean(noise_strong) > np.mean(noise_weak)

    def test_explicit_signal_power(self):
        """Should accept explicit signal power value."""
        signal = np.ones((10, 10), dtype=complex)

        # Use explicit power value
        noisy = awgn(signal, snr_db=20, signal_power=1.0)

        assert noisy.shape == signal.shape

    def test_1d_signal(self):
        """Should work with 1D signals."""
        signal = np.random.randn(100) + 1j * np.random.randn(100)

        noisy = awgn(signal, snr_db=30)

        assert noisy.shape == signal.shape

    def test_3d_signal(self):
        """Should work with 3D signals."""
        signal = np.random.randn(5, 5, 5) + 1j * np.random.randn(5, 5, 5)

        noisy = awgn(signal, snr_db=30)

        assert noisy.shape == signal.shape

    def test_zero_snr(self):
        """SNR of 0 dB means signal power equals noise power."""
        np.random.seed(999)
        signal = np.random.randn(50, 50) + 1j * np.random.randn(50, 50)

        noisy = awgn(signal, snr_db=0)

        # Estimate achieved SNR
        achieved = estimate_snr(noisy, signal)

        # Should be close to 0 dB
        assert abs(achieved) < 3

    def test_negative_snr(self):
        """Negative SNR should add more noise than signal."""
        np.random.seed(111)
        signal = np.ones((50, 50), dtype=complex)

        noisy = awgn(signal, snr_db=-10)

        # Noise power should exceed signal power
        noise_power = np.mean(np.abs(noisy - signal)**2)
        signal_power = np.mean(np.abs(signal)**2)

        assert noise_power > signal_power


class TestEstimateSNR:
    """Tests for estimate_snr function."""

    def test_perfect_match_gives_inf(self):
        """Identical signals should give infinite SNR."""
        signal = np.random.randn(10, 10) + 1j * np.random.randn(10, 10)

        snr = estimate_snr(signal, signal)

        assert snr == np.inf

    def test_basic_estimation(self):
        """Should estimate SNR correctly for known noise level."""
        signal = np.ones((100, 100), dtype=complex)

        # Add known noise
        np.random.seed(42)
        target_snr = 20  # dB
        noise_power = 1.0 / (10 ** (target_snr / 10))
        noise = np.sqrt(noise_power / 2) * (np.random.randn(100, 100) +
                                             1j * np.random.randn(100, 100))
        noisy = signal + noise

        estimated_snr = estimate_snr(noisy, signal)

        # Should be within ~2 dB
        assert abs(estimated_snr - target_snr) < 2

    def test_higher_noise_lower_snr(self):
        """More noise should result in lower estimated SNR."""
        signal = np.ones((50, 50), dtype=complex)

        np.random.seed(42)
        noise_low = 0.01 * (np.random.randn(50, 50) + 1j * np.random.randn(50, 50))
        noise_high = 0.1 * (np.random.randn(50, 50) + 1j * np.random.randn(50, 50))

        snr_low_noise = estimate_snr(signal + noise_low, signal)
        snr_high_noise = estimate_snr(signal + noise_high, signal)

        assert snr_low_noise > snr_high_noise

    def test_real_signals(self):
        """Should work with real signals."""
        signal = np.ones((50, 50))

        np.random.seed(42)
        noise = 0.1 * np.random.randn(50, 50)
        noisy = signal + noise

        snr = estimate_snr(noisy, signal)

        assert snr > 0  # Some positive SNR

    def test_round_trip_with_awgn(self):
        """awgn then estimate_snr should recover approximate target SNR."""
        np.random.seed(54321)
        signal = np.random.randn(100, 100) + 1j * np.random.randn(100, 100)
        target_snr = 25

        noisy = awgn(signal, snr_db=target_snr)
        estimated_snr = estimate_snr(noisy, signal)

        # Should be within ~3 dB due to statistical variation
        assert abs(estimated_snr - target_snr) < 3


class TestAddNoiseSNR:
    """Tests for add_noise_snr function."""

    def test_matches_awgn_measured(self):
        """Should be equivalent to awgn with 'measured'."""
        signal = np.random.randn(10, 10) + 1j * np.random.randn(10, 10)

        result1 = add_noise_snr(signal, snr_db=20, seed=42)
        result2 = awgn(signal, snr_db=20, signal_power='measured', seed=42)

        np.testing.assert_array_equal(result1, result2)

    def test_output_shape(self):
        """Output should match input shape."""
        signal = np.random.randn(8, 12) + 1j * np.random.randn(8, 12)

        noisy = add_noise_snr(signal, snr_db=30)

        assert noisy.shape == signal.shape


class TestMATLABCompatibility:
    """Tests for MATLAB compatibility."""

    def test_matlab_style_call(self):
        """Test MATLAB-style function call."""
        # MATLAB: Escat_noisy = awgn(Escat, 30, 'measured', 345)
        Escat = np.random.randn(10, 10) + 1j * np.random.randn(10, 10)

        Escat_noisy = awgn(Escat, 30, 'measured', 345)

        assert Escat_noisy.shape == Escat.shape

    def test_typical_scenario_snr_40(self):
        """Test with typical scenario SNR of 40 dB."""
        np.random.seed(12345)
        signal = np.random.randn(25, 25) + 1j * np.random.randn(25, 25)

        noisy = awgn(signal, snr_db=40, signal_power='measured')
        estimated = estimate_snr(noisy, signal)

        # Should be close to 40 dB
        assert abs(estimated - 40) < 5

    def test_typical_scenario_snr_30(self):
        """Test with moderate SNR of 30 dB."""
        np.random.seed(12345)
        signal = np.random.randn(50, 50) + 1j * np.random.randn(50, 50)

        noisy = awgn(signal, snr_db=30, signal_power='measured')
        estimated = estimate_snr(noisy, signal)

        # Should be close to 30 dB
        assert abs(estimated - 30) < 3


class TestEdgeCases:
    """Tests for edge cases."""

    def test_scalar_signal(self):
        """Should handle scalar input."""
        signal = np.array(1.0 + 1j)

        noisy = awgn(signal, snr_db=30)

        assert noisy.shape == signal.shape

    def test_very_small_signal(self):
        """Should handle very small signals."""
        signal = 1e-10 * np.ones((10, 10), dtype=complex)

        noisy = awgn(signal, snr_db=20)

        assert noisy.shape == signal.shape
        assert not np.any(np.isnan(noisy))

    def test_very_large_signal(self):
        """Should handle very large signals."""
        signal = 1e10 * np.ones((10, 10), dtype=complex)

        noisy = awgn(signal, snr_db=20)

        assert noisy.shape == signal.shape
        assert not np.any(np.isnan(noisy))

    def test_very_high_snr(self):
        """Very high SNR should barely change signal."""
        signal = np.ones((10, 10), dtype=complex)

        noisy = awgn(signal, snr_db=100)  # 100 dB

        # Should be very close to original
        assert np.allclose(noisy, signal, rtol=0.01)

    def test_preserves_dtype(self):
        """Should preserve complex128 dtype."""
        signal = np.ones((5, 5), dtype=np.complex128)

        noisy = awgn(signal, snr_db=30)

        assert noisy.dtype == np.complex128


class TestIntegration:
    """Integration tests combining noise functions."""

    def test_full_workflow(self):
        """Test typical workflow: add noise, estimate SNR."""
        np.random.seed(99999)

        # Create signal
        signal = np.random.randn(100, 100) + 1j * np.random.randn(100, 100)

        # Add noise at various SNR levels and verify
        for target_snr in [10, 20, 30, 40]:
            noisy = add_noise_snr(signal, snr_db=target_snr, seed=None)
            estimated = estimate_snr(noisy, signal)

            # Should be within reasonable range
            assert abs(estimated - target_snr) < 5

    def test_statistical_properties(self):
        """Noise should have expected statistical properties."""
        np.random.seed(888)
        signal = np.ones((100, 100), dtype=complex)

        noisy = awgn(signal, snr_db=20)
        noise = noisy - signal

        # Noise should have zero mean (approximately)
        assert abs(np.mean(noise.real)) < 0.05
        assert abs(np.mean(noise.imag)) < 0.05

        # Real and imaginary should have similar variance
        var_real = np.var(noise.real)
        var_imag = np.var(noise.imag)
        assert abs(var_real - var_imag) / var_real < 0.2  # Within 20%
