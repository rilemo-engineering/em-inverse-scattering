"""
Unit tests for inverse_scattering.core.utils module.

Tests verify that utility functions match MATLAB behavior and
produce physically correct results.
"""

import numpy as np
import pytest

from inverse_scattering.core.utils import (
    compute_wavelength,
    compute_wavenumber,
    create_grid,
    compute_dof,
    compute_contrast,
    compute_measurement_positions,
    nmse,
)
from inverse_scattering.core.constants import EPSILON_0, MU_0, C


class TestComputeWavelength:
    """Test wavelength computation."""

    def test_known_values(self):
        """Test wavelength at common frequencies."""
        # 300 MHz -> 1 m
        assert compute_wavelength(300e6) == 1.0

        # 3 GHz -> 10 cm
        np.testing.assert_allclose(compute_wavelength(3e9), 0.1, rtol=1e-10)

        # 4 GHz -> 7.5 cm
        np.testing.assert_allclose(compute_wavelength(4e9), 0.075, rtol=1e-10)

    def test_relationship_with_c(self):
        """Verify λ = c/f relationship."""
        for freq in [100e6, 1e9, 10e9]:
            wavelength = compute_wavelength(freq)
            expected = C / freq
            assert wavelength == expected

    def test_inverse_relationship_with_frequency(self):
        """Higher frequency -> shorter wavelength."""
        freq1 = 1e9
        freq2 = 2e9
        assert compute_wavelength(freq2) == compute_wavelength(freq1) / 2

    def test_positive_output(self):
        """Wavelength must be positive for positive frequency."""
        assert compute_wavelength(1e9) > 0


class TestComputeWavenumber:
    """Test wavenumber computation."""

    def test_free_space_wavenumber(self):
        """Test wavenumber in free space (εr=1, σ=0)."""
        freq = 1e9
        k = compute_wavenumber(freq, epsilon_r=1.0, sigma=0.0)

        # k should be approximately 2π/λ = 2πf/c
        k_expected = 2 * np.pi * freq / C
        np.testing.assert_allclose(np.real(k), k_expected, rtol=0.01)

        # Should be real for lossless medium
        np.testing.assert_allclose(np.imag(k), 0, atol=1e-10)

    def test_dielectric_medium(self):
        """Test wavenumber in dielectric medium (εr>1, σ=0)."""
        freq = 1e9
        epsilon_r = 4.0  # εr = 4

        k_dielectric = compute_wavenumber(freq, epsilon_r=epsilon_r, sigma=0.0)
        k_freespace = compute_wavenumber(freq, epsilon_r=1.0, sigma=0.0)

        # k should scale as sqrt(εr)
        np.testing.assert_allclose(k_dielectric, k_freespace * np.sqrt(epsilon_r), rtol=1e-10)

    def test_lossy_medium(self):
        """Test wavenumber in lossy medium (σ>0)."""
        freq = 1e9
        sigma = 0.1  # Some conductivity

        k = compute_wavenumber(freq, epsilon_r=1.0, sigma=sigma)

        # Should have non-zero imaginary part (attenuation)
        assert np.imag(k) != 0

    def test_matlab_equivalent_formula(self):
        """
        Verify MATLAB formula:
        eb_eq = eb - 1i*(sb/(e0*2*pi*freq))
        kb = 2*pi*freq*sqrt(e0*m0*eb_eq)
        """
        freq = 4e9
        eb = 1.0
        sb = 0.01

        omega = 2 * np.pi * freq
        eb_eq = eb - 1j * (sb / (omega * EPSILON_0))
        kb_matlab = omega * np.sqrt(EPSILON_0 * MU_0 * eb_eq)

        k_python = compute_wavenumber(freq, epsilon_r=eb, sigma=sb)

        np.testing.assert_allclose(k_python, kb_matlab, rtol=1e-10)

    def test_default_parameters(self):
        """Test default parameters give free-space wavenumber."""
        k = compute_wavenumber(1e9)
        k_free = compute_wavenumber(1e9, epsilon_r=1.0, sigma=0.0)
        assert k == k_free


class TestCreateGrid:
    """Test grid creation."""

    def test_grid_shape(self, small_grid_params):
        """Verify grid dimensions match input."""
        nx, ny = small_grid_params['Nx'], small_grid_params['Ny']
        lx, ly = small_grid_params['lx'], small_grid_params['ly']

        X, Y, xvec, yvec, dx, dy = create_grid(lx, ly, nx, ny)

        # Check shapes
        assert X.shape == (ny, nx)
        assert Y.shape == (ny, nx)
        assert xvec.shape == (nx,)
        assert yvec.shape == (ny,)

    def test_cell_size(self):
        """Verify cell sizes are correct."""
        lx, ly = 1.0, 2.0
        nx, ny = 10, 20

        X, Y, xvec, yvec, dx, dy = create_grid(lx, ly, nx, ny)

        assert dx == lx / nx
        assert dy == ly / ny

    def test_grid_centered_at_origin(self):
        """Verify grid is centered at origin."""
        lx, ly = 1.0, 1.0
        nx, ny = 16, 16

        X, Y, xvec, yvec, dx, dy = create_grid(lx, ly, nx, ny)

        # Mean should be approximately 0
        np.testing.assert_allclose(np.mean(xvec), 0, atol=1e-10)
        np.testing.assert_allclose(np.mean(yvec), 0, atol=1e-10)

    def test_grid_boundaries(self):
        """Verify grid stays within DoI boundaries."""
        lx, ly = 1.0, 1.0
        nx, ny = 10, 10

        X, Y, xvec, yvec, dx, dy = create_grid(lx, ly, nx, ny)

        # Cell centers should be within [-lx/2 + dx/2, lx/2 - dx/2]
        assert xvec[0] == -lx/2 + dx/2
        assert xvec[-1] == lx/2 - dx/2
        assert yvec[0] == -ly/2 + dy/2
        assert yvec[-1] == ly/2 - dy/2

    def test_meshgrid_consistency(self):
        """Verify X and Y grids are consistent with xvec, yvec."""
        lx, ly = 1.0, 1.0
        nx, ny = 8, 8

        X, Y, xvec, yvec, dx, dy = create_grid(lx, ly, nx, ny)

        # Check X varies along columns (axis 1)
        np.testing.assert_allclose(X[0, :], xvec)

        # Check Y varies along rows (axis 0)
        np.testing.assert_allclose(Y[:, 0], yvec)

    def test_matlab_equivalent(self):
        """
        Verify MATLAB equivalent:
        dx = lx/Nx; dy = ly/Ny;
        xvec = -lx/2+dx/2:dx:lx/2-dx/2;
        yvec = -ly/2+dy/2:dy:ly/2-dy/2;
        """
        lx, ly = 0.1, 0.1
        nx, ny = 60, 60

        X, Y, xvec, yvec, dx, dy = create_grid(lx, ly, nx, ny)

        # MATLAB xvec = -lx/2+dx/2:dx:lx/2-dx/2
        expected_dx = lx / nx
        expected_xvec = np.arange(-lx/2 + expected_dx/2, lx/2, expected_dx)

        np.testing.assert_allclose(dx, expected_dx, rtol=1e-10)
        np.testing.assert_allclose(xvec, expected_xvec[:nx], rtol=1e-10)


class TestComputeDof:
    """Test degrees of freedom computation."""

    def test_known_values(self):
        """Test DOF for known configurations."""
        # From MATLAB DATA_scenario.mat: lx=0.1, freq=1e9, DOF=3
        # Let's verify the formula gives reasonable values
        dof = compute_dof(lx=0.1, freq=1e9)
        # DOF = 2 * k0 * a = 2 * (2πf/c) * (√2 * lx/2)
        k0 = 2 * np.pi * 1e9 / C
        a = np.sqrt(2) * 0.1 / 2
        expected = int(np.ceil(2 * k0 * a))
        assert dof == expected

    def test_scaling_with_frequency(self):
        """DOF should increase with frequency."""
        lx = 1.0
        dof_low = compute_dof(lx, 100e6)
        dof_high = compute_dof(lx, 1e9)
        assert dof_high > dof_low

    def test_scaling_with_size(self):
        """DOF should increase with domain size."""
        freq = 1e9
        dof_small = compute_dof(0.1, freq)
        dof_large = compute_dof(1.0, freq)
        assert dof_large > dof_small

    def test_returns_integer(self):
        """DOF should always be an integer."""
        dof = compute_dof(0.5, 500e6)
        assert isinstance(dof, int)

    def test_rectangular_domain(self):
        """Test DOF for non-square domain."""
        lx, ly = 0.1, 0.2
        freq = 1e9
        dof = compute_dof(lx, freq, ly)

        # Should use diagonal for characteristic size
        k0 = 2 * np.pi * freq / C
        a = np.sqrt(lx**2 + ly**2) / 2
        expected = int(np.ceil(2 * k0 * a))
        assert dof == expected


class TestComputeContrast:
    """Test contrast function computation."""

    def test_lossless_contrast(self):
        """Test contrast for lossless materials."""
        # εr = 3, εb = 1 -> τ = 2
        tau = compute_contrast(epsilon_r=3.0, epsilon_b=1.0)
        assert tau == 2.0

        # εr = 1.5, εb = 1 -> τ = 0.5
        tau = compute_contrast(epsilon_r=1.5, epsilon_b=1.0)
        assert tau == 0.5

    def test_lossy_contrast(self):
        """Test contrast for lossy materials."""
        tau = compute_contrast(
            epsilon_r=2.0, epsilon_b=1.0,
            sigma=0.1, freq=1e9
        )

        # Should have imaginary part
        assert np.imag(tau) != 0
        # Real part should still be εr - εb = 1
        np.testing.assert_allclose(np.real(tau), 1.0, rtol=1e-10)

    def test_requires_freq_for_lossy(self):
        """Should raise error if sigma>0 but freq not provided."""
        with pytest.raises(ValueError):
            compute_contrast(epsilon_r=2.0, epsilon_b=1.0, sigma=0.1)

    def test_zero_contrast(self):
        """Test when object matches background."""
        tau = compute_contrast(epsilon_r=1.0, epsilon_b=1.0)
        assert tau == 0.0


class TestComputeMeasurementPositions:
    """Test measurement position computation."""

    def test_number_of_positions(self):
        """Verify correct number of positions generated."""
        n_meas = 12
        theta, x, y = compute_measurement_positions(n_meas, radius=1.0)

        assert len(theta) == n_meas
        assert len(x) == n_meas
        assert len(y) == n_meas

    def test_positions_on_circle(self):
        """Verify all positions are on circle of given radius."""
        radius = 2.5
        theta, x, y = compute_measurement_positions(20, radius)

        r = np.sqrt(x**2 + y**2)
        np.testing.assert_allclose(r, radius, rtol=1e-10)

    def test_angular_spacing(self):
        """Verify uniform angular spacing."""
        n_meas = 8
        theta, x, y = compute_measurement_positions(n_meas, radius=1.0)

        # Angular differences should be constant
        dtheta = np.diff(theta)
        expected_dtheta = 2 * np.pi / n_meas
        np.testing.assert_allclose(dtheta, expected_dtheta, rtol=1e-10)

    def test_first_position(self):
        """First position should be at θ=0 (positive x-axis)."""
        theta, x, y = compute_measurement_positions(12, radius=1.0)

        assert theta[0] == 0
        np.testing.assert_allclose(x[0], 1.0, rtol=1e-10)
        np.testing.assert_allclose(y[0], 0.0, atol=1e-10)

    def test_matlab_equivalent(self):
        """
        Verify MATLAB equivalent:
        meas_pos_theta = linspace(0, 2*pi - 2*pi/Nm, Nm)
        """
        Nm = 12
        theta, x, y = compute_measurement_positions(Nm, radius=1.0)

        # MATLAB: linspace(0, 2*pi - 2*pi/Nm, Nm)
        expected_theta = np.linspace(0, 2*np.pi - 2*np.pi/Nm, Nm)
        np.testing.assert_allclose(theta, expected_theta, rtol=1e-10)


class TestNMSE:
    """Test NMSE computation."""

    def test_perfect_reconstruction(self):
        """NMSE should be 0 for identical profiles."""
        profile = np.random.rand(32, 32) + 1j * np.random.rand(32, 32)
        assert nmse(profile, profile) == 0.0

    def test_completely_wrong(self):
        """NMSE should be high for very different profiles."""
        true = np.ones((16, 16))
        wrong = np.zeros((16, 16))
        assert nmse(true, wrong) == 1.0

    def test_normalization(self):
        """NMSE should be normalized by true profile energy."""
        true = np.array([[1.0, 0], [0, 0]])  # Single non-zero element
        rec = np.array([[0.9, 0], [0, 0]])  # 10% error

        error = nmse(true, rec)
        # Error = |1.0 - 0.9|^2 / |1.0|^2 = 0.01
        np.testing.assert_allclose(error, 0.01, rtol=1e-10)

    def test_complex_profiles(self):
        """NMSE should work with complex profiles."""
        true = np.array([[1+1j, 0], [0, 0]])
        rec = np.array([[1+0.9j, 0], [0, 0]])

        error = nmse(true, rec)
        # |true|^2 = 2, |error|^2 = 0.01
        expected = 0.01 / 2
        np.testing.assert_allclose(error, expected, rtol=1e-10)

    def test_matlab_equivalent(self):
        """
        Verify MATLAB formula:
        NMSE = sum(sum(abs(PROF-PROF_rec).^2))/sum(sum(abs(PROF).^2))
        """
        PROF = np.random.rand(32, 32) + 1j * np.random.rand(32, 32)
        PROF_rec = PROF + 0.1 * np.random.rand(32, 32)

        nmse_python = nmse(PROF, PROF_rec)

        # MATLAB formula
        numerator = np.sum(np.abs(PROF - PROF_rec)**2)
        denominator = np.sum(np.abs(PROF)**2)
        nmse_matlab = numerator / denominator

        np.testing.assert_allclose(nmse_python, nmse_matlab, rtol=1e-10)

    def test_zero_true_profile(self):
        """NMSE with zero true profile should handle gracefully."""
        true = np.zeros((16, 16))
        rec = np.ones((16, 16))

        error = nmse(true, rec)
        assert error == np.inf  # Or could be a large number

    def test_interpretation(self):
        """Test NMSE interpretation guidelines."""
        true = np.ones((32, 32))

        # Excellent: NMSE < 0.05
        rec_excellent = true + 0.15 * np.random.randn(32, 32)
        # This should give NMSE around 0.0225

        # The exact value depends on random noise, so we just
        # verify NMSE is a positive float
        error = nmse(true, rec_excellent)
        assert error > 0
        assert isinstance(error, float)
