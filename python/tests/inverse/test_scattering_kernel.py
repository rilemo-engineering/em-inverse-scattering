"""
Unit tests for inverse_scattering.inverse.scattering_kernel module.

Tests verify that the scattering kernel is built correctly and
produces the correct mapping from contrast to scattered field.

Theory:
    Under the Born approximation (E_tot ≈ E_inc):
        E_scat = S @ τ
    where:
        S[m,n] = k² * G(r_m, r_n) * E_inc(r_n) * dx * dy
"""

import numpy as np
import pytest
from scipy import special

from inverse_scattering.inverse.scattering_kernel import (
    build_scattering_kernel,
    kernel_scattering,
    kernel_scattering_exp,
    apply_scattering_kernel,
    reshape_escat_to_matrix,
)
from inverse_scattering.core.utils import create_grid, compute_wavenumber
from inverse_scattering.forward.incident_field import (
    compute_incident_field_all_views,
    setup_transmitters,
)
from inverse_scattering.forward.profiles import create_circular_profile


class TestBuildScatteringKernel:
    """Test scattering kernel construction."""

    def test_output_shape(self):
        """Kernel should be (Nm*Nv × Nx*Ny)."""
        lx, ly, nx, ny = 0.5, 0.5, 8, 8
        Nm, Nv = 6, 4
        freq = 300e6

        X, Y, xvec, yvec, dx, dy = create_grid(lx, ly, nx, ny)
        k = compute_wavenumber(freq, epsilon_r=1.0, sigma=0.0)

        tx_pos = setup_transmitters(Nv, radius=0.5)
        Einc = compute_incident_field_all_views(X, Y, k, tx_pos)

        S = build_scattering_kernel(
            Einc, nx, ny, lx, ly, Nv, eb=1.0, sb=0.0, freq=freq, Nm=Nm, Rm=0.5
        )

        expected_shape = (Nm * Nv, nx * ny)
        assert S.shape == expected_shape

    def test_complex_dtype(self):
        """Kernel should be complex."""
        lx, ly, nx, ny = 0.5, 0.5, 8, 8
        Nm, Nv = 4, 4
        freq = 300e6

        X, Y, xvec, yvec, dx, dy = create_grid(lx, ly, nx, ny)
        k = compute_wavenumber(freq, epsilon_r=1.0, sigma=0.0)
        tx_pos = setup_transmitters(Nv, radius=0.5)
        Einc = compute_incident_field_all_views(X, Y, k, tx_pos)

        S = build_scattering_kernel(
            Einc, nx, ny, lx, ly, Nv, eb=1.0, sb=0.0, freq=freq, Nm=Nm, Rm=0.5
        )

        assert np.issubdtype(S.dtype, np.complexfloating)

    def test_no_nan_or_inf(self):
        """Kernel should not contain NaN or Inf."""
        lx, ly, nx, ny = 0.5, 0.5, 8, 8
        Nm, Nv = 4, 4
        freq = 300e6

        X, Y, xvec, yvec, dx, dy = create_grid(lx, ly, nx, ny)
        k = compute_wavenumber(freq, epsilon_r=1.0, sigma=0.0)
        tx_pos = setup_transmitters(Nv, radius=0.5)
        Einc = compute_incident_field_all_views(X, Y, k, tx_pos)

        S = build_scattering_kernel(
            Einc, nx, ny, lx, ly, Nv, eb=1.0, sb=0.0, freq=freq, Nm=Nm, Rm=0.5
        )

        assert np.all(np.isfinite(S))

    def test_formula_verification(self):
        """Verify S[row, col] = k² * G * E_inc * cell_area."""
        lx, ly, nx, ny = 0.5, 0.5, 6, 6
        Nm, Nv = 4, 2
        freq = 300e6
        Rm = 0.5

        X, Y, xvec, yvec, dx, dy = create_grid(lx, ly, nx, ny)
        k = compute_wavenumber(freq, epsilon_r=1.0, sigma=0.0)
        tx_pos = setup_transmitters(Nv, radius=Rm)
        Einc = compute_incident_field_all_views(X, Y, k, tx_pos)

        S = build_scattering_kernel(
            Einc, nx, ny, lx, ly, Nv, eb=1.0, sb=0.0, freq=freq, Nm=Nm, Rm=Rm
        )

        # Verify a few elements manually
        cell_area = dx * dy
        meas_theta = np.linspace(0, 2*np.pi - 2*np.pi/Nm, Nm)
        rx_x = Rm * np.cos(meas_theta)
        rx_y = Rm * np.sin(meas_theta)
        x_flat = X.ravel()
        y_flat = Y.ravel()

        # Check element (view=0, meas=0, cell=0)
        # Note: kernel uses Fortran order for grid flattening
        x_flat_F = X.ravel(order='F')
        y_flat_F = Y.ravel(order='F')
        v, m, n = 0, 0, 0
        row_idx = v * Nm + m
        R = np.sqrt((rx_x[m] - x_flat_F[n])**2 + (rx_y[m] - y_flat_F[n])**2)
        # MATLAB uses exp(-jωt) convention, so Green's function is -(j/4)*H_0^(2)
        G = -(1j / 4) * special.hankel2(0, k * R)
        E_n = Einc[:, :, v].ravel(order='F')[n]
        expected = k**2 * G * E_n * cell_area

        np.testing.assert_allclose(S[row_idx, n], expected, rtol=1e-10)


class TestKernelScattering:
    """Test MATLAB-compatible kernel_scattering function."""

    def test_matlab_interface(self):
        """Test MATLAB-compatible interface."""
        lx, ly, nx, ny = 0.5, 0.5, 8, 8
        Nm, Nv = 4, 4
        freq = 300e6

        X, Y, xvec, yvec, dx, dy = create_grid(lx, ly, nx, ny)
        k = compute_wavenumber(freq, epsilon_r=1.0, sigma=0.0)
        tx_pos = setup_transmitters(Nv, radius=0.5)
        Einc = compute_incident_field_all_views(X, Y, k, tx_pos)

        # MATLAB call: kernel_scattering(Etot, Nx, Ny, lx, ly, 1, eb, sb, freq, Nm, Rm)
        S = kernel_scattering(Einc, nx, ny, lx, ly, 1, 1.0, 0.0, freq, Nm, 0.5)

        assert S.shape == (Nm * Nv, nx * ny)

    def test_matches_build_function(self):
        """kernel_scattering should match build_scattering_kernel."""
        lx, ly, nx, ny = 0.5, 0.5, 8, 8
        Nm, Nv = 4, 4
        freq = 300e6

        X, Y, xvec, yvec, dx, dy = create_grid(lx, ly, nx, ny)
        k = compute_wavenumber(freq, epsilon_r=1.0, sigma=0.0)
        tx_pos = setup_transmitters(Nv, radius=0.5)
        Einc = compute_incident_field_all_views(X, Y, k, tx_pos)

        S1 = kernel_scattering(Einc, nx, ny, lx, ly, 1, 1.0, 0.0, freq, Nm, 0.5)
        S2 = build_scattering_kernel(
            Einc, nx, ny, lx, ly, Nv, 1.0, 0.0, freq, Nm, 0.5
        )

        # Use allclose for numerical precision (machine epsilon level)
        np.testing.assert_allclose(S1, S2, rtol=1e-14, atol=1e-18)


class TestKernelScatteringExp:
    """Test experimental data kernel."""

    def test_matches_kernel_scattering(self):
        """kernel_scattering_exp should match kernel_scattering."""
        lx, ly, nx, ny = 0.5, 0.5, 8, 8
        Nm, Nv = 4, 4
        freq = 300e6

        X, Y, xvec, yvec, dx, dy = create_grid(lx, ly, nx, ny)
        k = compute_wavenumber(freq, epsilon_r=1.0, sigma=0.0)
        tx_pos = setup_transmitters(Nv, radius=0.5)
        Einc = compute_incident_field_all_views(X, Y, k, tx_pos)

        S1 = kernel_scattering(Einc, nx, ny, lx, ly, 1, 1.0, 0.0, freq, Nm, 0.5)
        S2 = kernel_scattering_exp(Einc, nx, ny, lx, ly, 1, 1.0, 0.0, freq, Nm, 0.5)

        # Use allclose for numerical precision (machine epsilon level)
        np.testing.assert_allclose(S1, S2, rtol=1e-14, atol=1e-18)


class TestApplyScatteringKernel:
    """Test kernel application to contrast."""

    def test_output_shape(self):
        """Output should be (Nm*Nv,)."""
        lx, ly, nx, ny = 0.5, 0.5, 8, 8
        Nm, Nv = 4, 4
        freq = 300e6

        X, Y, xvec, yvec, dx, dy = create_grid(lx, ly, nx, ny)
        k = compute_wavenumber(freq, epsilon_r=1.0, sigma=0.0)
        tx_pos = setup_transmitters(Nv, radius=0.5)
        Einc = compute_incident_field_all_views(X, Y, k, tx_pos)

        S = build_scattering_kernel(
            Einc, nx, ny, lx, ly, Nv, 1.0, 0.0, freq, Nm, 0.5
        )

        tau = create_circular_profile(X, Y, (0.0, 0.0), 0.1, epsilon_r=1.5)
        Escat_vec = apply_scattering_kernel(S, tau)

        assert Escat_vec.shape == (Nm * Nv,)

    def test_complex_output(self):
        """Output should be complex."""
        lx, ly, nx, ny = 0.5, 0.5, 8, 8
        Nm, Nv = 4, 4
        freq = 300e6

        X, Y, xvec, yvec, dx, dy = create_grid(lx, ly, nx, ny)
        k = compute_wavenumber(freq, epsilon_r=1.0, sigma=0.0)
        tx_pos = setup_transmitters(Nv, radius=0.5)
        Einc = compute_incident_field_all_views(X, Y, k, tx_pos)

        S = build_scattering_kernel(
            Einc, nx, ny, lx, ly, Nv, 1.0, 0.0, freq, Nm, 0.5
        )

        tau = create_circular_profile(X, Y, (0.0, 0.0), 0.1, epsilon_r=1.5)
        Escat_vec = apply_scattering_kernel(S, tau)

        assert np.issubdtype(Escat_vec.dtype, np.complexfloating)

    def test_zero_contrast_gives_zero(self):
        """Zero contrast should give zero scattered field."""
        lx, ly, nx, ny = 0.5, 0.5, 8, 8
        Nm, Nv = 4, 4
        freq = 300e6

        X, Y, xvec, yvec, dx, dy = create_grid(lx, ly, nx, ny)
        k = compute_wavenumber(freq, epsilon_r=1.0, sigma=0.0)
        tx_pos = setup_transmitters(Nv, radius=0.5)
        Einc = compute_incident_field_all_views(X, Y, k, tx_pos)

        S = build_scattering_kernel(
            Einc, nx, ny, lx, ly, Nv, 1.0, 0.0, freq, Nm, 0.5
        )

        tau = np.zeros((ny, nx), dtype=complex)
        Escat_vec = apply_scattering_kernel(S, tau)

        np.testing.assert_allclose(Escat_vec, 0.0, atol=1e-14)

    def test_linearity(self):
        """Operator should be linear: S(a*τ1 + b*τ2) = a*S(τ1) + b*S(τ2)."""
        lx, ly, nx, ny = 0.5, 0.5, 8, 8
        Nm, Nv = 4, 4
        freq = 300e6

        X, Y, xvec, yvec, dx, dy = create_grid(lx, ly, nx, ny)
        k = compute_wavenumber(freq, epsilon_r=1.0, sigma=0.0)
        tx_pos = setup_transmitters(Nv, radius=0.5)
        Einc = compute_incident_field_all_views(X, Y, k, tx_pos)

        S = build_scattering_kernel(
            Einc, nx, ny, lx, ly, Nv, 1.0, 0.0, freq, Nm, 0.5
        )

        tau1 = create_circular_profile(X, Y, (-0.1, 0.0), 0.1, epsilon_r=1.5)
        tau2 = create_circular_profile(X, Y, (0.1, 0.0), 0.1, epsilon_r=2.0)
        a, b = 2.0 + 0.5j, 1.5 - 0.3j

        # S(a*τ1 + b*τ2)
        Escat_combined = apply_scattering_kernel(S, a * tau1 + b * tau2)

        # a*S(τ1) + b*S(τ2)
        Escat_separate = (a * apply_scattering_kernel(S, tau1) +
                         b * apply_scattering_kernel(S, tau2))

        np.testing.assert_allclose(Escat_combined, Escat_separate, rtol=1e-10)

    def test_accepts_flattened_tau(self):
        """Should accept flattened contrast vector."""
        lx, ly, nx, ny = 0.5, 0.5, 8, 8
        Nm, Nv = 4, 4
        freq = 300e6

        X, Y, xvec, yvec, dx, dy = create_grid(lx, ly, nx, ny)
        k = compute_wavenumber(freq, epsilon_r=1.0, sigma=0.0)
        tx_pos = setup_transmitters(Nv, radius=0.5)
        Einc = compute_incident_field_all_views(X, Y, k, tx_pos)

        S = build_scattering_kernel(
            Einc, nx, ny, lx, ly, Nv, 1.0, 0.0, freq, Nm, 0.5
        )

        tau = create_circular_profile(X, Y, (0.0, 0.0), 0.1, epsilon_r=1.5)

        Escat1 = apply_scattering_kernel(S, tau)
        Escat2 = apply_scattering_kernel(S, tau.ravel())

        np.testing.assert_array_equal(Escat1, Escat2)


class TestReshapeEscatToMatrix:
    """Test reshaping scattered field to matrix form."""

    def test_output_shape(self):
        """Output should be (Nm × Nv)."""
        Nm, Nv = 8, 6
        escat_vec = np.random.rand(Nm * Nv) + 1j * np.random.rand(Nm * Nv)

        Escat_mat = reshape_escat_to_matrix(escat_vec, Nm, Nv)

        assert Escat_mat.shape == (Nm, Nv)

    def test_preserves_values(self):
        """Total elements should be preserved."""
        Nm, Nv = 4, 3
        escat_vec = np.arange(Nm * Nv) + 0j

        Escat_mat = reshape_escat_to_matrix(escat_vec, Nm, Nv)

        # All values should be present
        assert set(Escat_mat.ravel().real.astype(int)) == set(range(Nm * Nv))


class TestMATLABCompatibility:
    """Tests for MATLAB compatibility."""

    def test_scenario_parameters(self):
        """Test with parameters similar to MATLAB DATA_scenario.mat."""
        lx, ly = 0.1, 0.1
        nx, ny = 30, 30  # Smaller for speed
        Nm, Nv = 12, 12
        freq = 1e9
        Rm = 0.1

        X, Y, xvec, yvec, dx, dy = create_grid(lx, ly, nx, ny)
        k = compute_wavenumber(freq, epsilon_r=1.0, sigma=0.0)
        tx_pos = setup_transmitters(Nv, radius=Rm)
        Einc = compute_incident_field_all_views(X, Y, k, tx_pos)

        S = kernel_scattering(Einc, nx, ny, lx, ly, 1, 1.0, 0.0, freq, Nm, Rm)

        # Verify shape
        assert S.shape == (Nm * Nv, nx * ny)

        # Verify finite values
        assert np.all(np.isfinite(S))


class TestEdgeCases:
    """Test edge cases."""

    def test_single_measurement(self):
        """Test with single receiver and transmitter."""
        lx, ly, nx, ny = 0.5, 0.5, 8, 8
        Nm, Nv = 1, 1
        freq = 300e6

        X, Y, xvec, yvec, dx, dy = create_grid(lx, ly, nx, ny)
        k = compute_wavenumber(freq, epsilon_r=1.0, sigma=0.0)
        tx_pos = setup_transmitters(Nv, radius=0.5)
        Einc = compute_incident_field_all_views(X, Y, k, tx_pos)

        S = build_scattering_kernel(
            Einc, nx, ny, lx, ly, Nv, 1.0, 0.0, freq, Nm, 0.5
        )

        assert S.shape == (1, nx * ny)

    def test_many_measurements(self):
        """Test with many receivers and transmitters."""
        lx, ly, nx, ny = 0.5, 0.5, 8, 8
        Nm, Nv = 24, 24
        freq = 300e6

        X, Y, xvec, yvec, dx, dy = create_grid(lx, ly, nx, ny)
        k = compute_wavenumber(freq, epsilon_r=1.0, sigma=0.0)
        tx_pos = setup_transmitters(Nv, radius=0.5)
        Einc = compute_incident_field_all_views(X, Y, k, tx_pos)

        S = build_scattering_kernel(
            Einc, nx, ny, lx, ly, Nv, 1.0, 0.0, freq, Nm, 0.5
        )

        assert S.shape == (Nm * Nv, nx * ny)

    def test_rectangular_grid(self):
        """Test with non-square grid."""
        lx, ly, nx, ny = 0.5, 0.8, 6, 10
        Nm, Nv = 4, 4
        freq = 300e6

        X, Y, xvec, yvec, dx, dy = create_grid(lx, ly, nx, ny)
        k = compute_wavenumber(freq, epsilon_r=1.0, sigma=0.0)
        tx_pos = setup_transmitters(Nv, radius=0.5)
        Einc = compute_incident_field_all_views(X, Y, k, tx_pos)

        S = build_scattering_kernel(
            Einc, nx, ny, lx, ly, Nv, 1.0, 0.0, freq, Nm, 0.5
        )

        assert S.shape == (Nm * Nv, nx * ny)
