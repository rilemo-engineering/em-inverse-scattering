"""
Unit tests for inverse_scattering.forward.incident_field module.

Tests verify that incident field computations match theory and MATLAB behavior.

Theory:
    - Plane wave: E_inc = exp(jk·r)
    - Line source (cylindrical wave): E_inc = (i/4) * H₀⁽¹⁾(k|r - r_s|)
"""

import numpy as np
import pytest
from scipy import special

from inverse_scattering.forward.incident_field import (
    compute_incident_field_plane_wave,
    compute_incident_field_line_source,
    compute_incident_field_all_views,
    setup_transmitters,
    setup_receivers,
)
from inverse_scattering.core.utils import create_grid, compute_wavenumber


class TestPlaneWaveIncidentField:
    """Test plane wave incident field computation."""

    def test_basic_shape(self):
        """Output should match grid shape."""
        lx, ly, nx, ny = 1.0, 1.0, 32, 32
        X, Y, xvec, yvec, dx, dy = create_grid(lx, ly, nx, ny)
        k = 20.0

        E_inc = compute_incident_field_plane_wave(X, Y, k, direction=0.0)

        assert E_inc.shape == (ny, nx)

    def test_complex_dtype(self):
        """Output should be complex."""
        lx, ly, nx, ny = 1.0, 1.0, 16, 16
        X, Y, xvec, yvec, dx, dy = create_grid(lx, ly, nx, ny)
        k = 10.0

        E_inc = compute_incident_field_plane_wave(X, Y, k, direction=0.0)

        assert np.issubdtype(E_inc.dtype, np.complexfloating)

    def test_formula_x_direction(self):
        """Verify E_inc = exp(jkx) for wave in +x direction."""
        lx, ly, nx, ny = 1.0, 1.0, 32, 32
        X, Y, xvec, yvec, dx, dy = create_grid(lx, ly, nx, ny)
        k = 15.0

        # Direction θ = 0 means wave propagates in +x
        E_inc = compute_incident_field_plane_wave(X, Y, k, direction=0.0)

        # E_inc = exp(jkx)
        expected = np.exp(1j * k * X)
        np.testing.assert_allclose(E_inc, expected, rtol=1e-14)

    def test_formula_y_direction(self):
        """Verify E_inc = exp(jky) for wave in +y direction."""
        lx, ly, nx, ny = 1.0, 1.0, 32, 32
        X, Y, xvec, yvec, dx, dy = create_grid(lx, ly, nx, ny)
        k = 15.0

        # Direction θ = π/2 means wave propagates in +y
        E_inc = compute_incident_field_plane_wave(X, Y, k, direction=np.pi/2)

        expected = np.exp(1j * k * Y)
        np.testing.assert_allclose(E_inc, expected, rtol=1e-14)

    def test_formula_diagonal_direction(self):
        """Test wave propagating at 45 degrees."""
        lx, ly, nx, ny = 1.0, 1.0, 32, 32
        X, Y, xvec, yvec, dx, dy = create_grid(lx, ly, nx, ny)
        k = 10.0

        # Direction θ = π/4 (45 degrees)
        theta = np.pi / 4
        E_inc = compute_incident_field_plane_wave(X, Y, k, direction=theta)

        # E_inc = exp(jk(x*cos(θ) + y*sin(θ)))
        expected = np.exp(1j * k * (X * np.cos(theta) + Y * np.sin(theta)))
        np.testing.assert_allclose(E_inc, expected, rtol=1e-14)

    def test_direction_as_vector(self):
        """Test with direction specified as (kx, ky) vector."""
        lx, ly, nx, ny = 1.0, 1.0, 32, 32
        X, Y, xvec, yvec, dx, dy = create_grid(lx, ly, nx, ny)
        k = 10.0

        # Direction (1, 1) should be normalized to 45 degrees
        E_inc = compute_incident_field_plane_wave(X, Y, k, direction=(1.0, 1.0))

        # Same as θ = π/4
        E_inc_angle = compute_incident_field_plane_wave(X, Y, k, direction=np.pi/4)
        np.testing.assert_allclose(E_inc, E_inc_angle, rtol=1e-14)

    def test_unit_magnitude(self):
        """Plane wave should have |E_inc| = 1 everywhere."""
        lx, ly, nx, ny = 1.0, 1.0, 32, 32
        X, Y, xvec, yvec, dx, dy = create_grid(lx, ly, nx, ny)
        k = 20.0

        E_inc = compute_incident_field_plane_wave(X, Y, k, direction=0.3)

        # For real k, |exp(jkx)| = 1
        np.testing.assert_allclose(np.abs(E_inc), 1.0, rtol=1e-14)

    def test_phase_variation(self):
        """Phase should vary linearly with position."""
        lx, ly, nx, ny = 1.0, 1.0, 64, 64
        X, Y, xvec, yvec, dx, dy = create_grid(lx, ly, nx, ny)
        k = 10.0

        E_inc = compute_incident_field_plane_wave(X, Y, k, direction=0.0)

        # Phase should be k*x
        phase = np.angle(E_inc)
        expected_phase = k * X

        # Unwrap to handle phase wrapping
        # Along center row (y=0)
        mid_row = ny // 2
        actual = np.unwrap(phase[mid_row, :])
        expected = expected_phase[mid_row, :]

        # Check phase increases linearly
        diff = np.diff(actual)
        np.testing.assert_allclose(diff, k * dx, rtol=1e-10)


class TestLineSourceIncidentField:
    """Test line source (cylindrical wave) incident field computation."""

    def test_basic_shape(self):
        """Output should match grid shape."""
        lx, ly, nx, ny = 1.0, 1.0, 32, 32
        X, Y, xvec, yvec, dx, dy = create_grid(lx, ly, nx, ny)
        k = 20.0

        E_inc = compute_incident_field_line_source(X, Y, k, source_position=(1.0, 0.0))

        assert E_inc.shape == (ny, nx)

    def test_complex_dtype(self):
        """Output should be complex."""
        lx, ly, nx, ny = 1.0, 1.0, 16, 16
        X, Y, xvec, yvec, dx, dy = create_grid(lx, ly, nx, ny)
        k = 10.0

        E_inc = compute_incident_field_line_source(X, Y, k, source_position=(0.5, 0.0))

        assert np.issubdtype(E_inc.dtype, np.complexfloating)

    def test_formula_verification(self):
        """Verify E_inc = (i/4) * H₀⁽¹⁾(k|r - r_s|)."""
        lx, ly, nx, ny = 1.0, 1.0, 32, 32
        X, Y, xvec, yvec, dx, dy = create_grid(lx, ly, nx, ny)
        k = 15.0
        x_s, y_s = 0.8, 0.3

        E_inc = compute_incident_field_line_source(X, Y, k, source_position=(x_s, y_s))

        # Manual calculation
        R = np.sqrt((X - x_s)**2 + (Y - y_s)**2)
        R = np.maximum(R, 1e-10)
        expected = (1j / 4) * special.hankel1(0, k * R)

        np.testing.assert_allclose(E_inc, expected, rtol=1e-10)

    def test_radial_symmetry(self):
        """Field should be radially symmetric around source."""
        lx, ly, nx, ny = 2.0, 2.0, 64, 64
        X, Y, xvec, yvec, dx, dy = create_grid(lx, ly, nx, ny)
        k = 10.0
        x_s, y_s = 0.0, 0.0  # Source at origin

        E_inc = compute_incident_field_line_source(X, Y, k, source_position=(x_s, y_s))

        # Points at same distance should have same magnitude
        r1 = np.sqrt(0.3**2 + 0.4**2)  # Distance 0.5

        # Find grid points at approximately this distance
        R = np.sqrt(X**2 + Y**2)
        mask = np.abs(R - r1) < 0.05  # Within tolerance

        if np.any(mask):
            values = np.abs(E_inc[mask])
            # All should be approximately equal
            np.testing.assert_allclose(values, values[0], rtol=0.1)

    def test_decay_with_distance(self):
        """Field magnitude should decay with distance from source."""
        lx, ly, nx, ny = 2.0, 2.0, 64, 64
        X, Y, xvec, yvec, dx, dy = create_grid(lx, ly, nx, ny)
        k = 10.0
        x_s, y_s = 1.5, 0.0  # Source outside DoI

        E_inc = compute_incident_field_line_source(X, Y, k, source_position=(x_s, y_s))

        # Near point (inside DoI, close to source)
        near_idx = np.argmin(np.abs(xvec - 0.5))
        near_val = np.abs(E_inc[ny//2, near_idx])

        # Far point (inside DoI, far from source)
        far_idx = np.argmin(np.abs(xvec + 0.5))
        far_val = np.abs(E_inc[ny//2, far_idx])

        assert near_val > far_val

    def test_no_nan_at_source(self):
        """Should handle source at or near grid point gracefully."""
        lx, ly, nx, ny = 1.0, 1.0, 32, 32
        X, Y, xvec, yvec, dx, dy = create_grid(lx, ly, nx, ny)
        k = 10.0

        # Source near center of grid
        E_inc = compute_incident_field_line_source(X, Y, k, source_position=(0.0, 0.0))

        # Should not have NaN or Inf
        assert np.all(np.isfinite(E_inc))


class TestAllViewsIncidentField:
    """Test incident field computation for all views."""

    def test_output_shape(self):
        """Output should be (Ny, Nx, Nv)."""
        lx, ly, nx, ny = 1.0, 1.0, 32, 32
        X, Y, xvec, yvec, dx, dy = create_grid(lx, ly, nx, ny)
        k = 20.0
        n_tx = 12

        tx_pos = setup_transmitters(n_tx, radius=0.5)
        E_inc = compute_incident_field_all_views(X, Y, k, tx_pos)

        assert E_inc.shape == (ny, nx, n_tx)

    def test_complex_dtype(self):
        """Output should be complex."""
        lx, ly, nx, ny = 1.0, 1.0, 16, 16
        X, Y, xvec, yvec, dx, dy = create_grid(lx, ly, nx, ny)
        k = 10.0

        tx_pos = setup_transmitters(8, radius=0.5)
        E_inc = compute_incident_field_all_views(X, Y, k, tx_pos)

        assert np.issubdtype(E_inc.dtype, np.complexfloating)

    def test_each_view_matches_single(self):
        """Each view should match individual computation."""
        lx, ly, nx, ny = 1.0, 1.0, 32, 32
        X, Y, xvec, yvec, dx, dy = create_grid(lx, ly, nx, ny)
        k = 15.0
        n_tx = 6

        tx_pos = setup_transmitters(n_tx, radius=0.8)
        E_inc_all = compute_incident_field_all_views(X, Y, k, tx_pos, source_type='line')

        # Check each view
        for v in range(n_tx):
            E_inc_single = compute_incident_field_line_source(X, Y, k, tx_pos[v])
            np.testing.assert_allclose(E_inc_all[:, :, v], E_inc_single, rtol=1e-14)

    def test_plane_wave_source_type(self):
        """Test plane wave source type."""
        lx, ly, nx, ny = 1.0, 1.0, 32, 32
        X, Y, xvec, yvec, dx, dy = create_grid(lx, ly, nx, ny)
        k = 10.0
        n_tx = 4

        tx_pos = setup_transmitters(n_tx, radius=0.5)
        E_inc = compute_incident_field_all_views(X, Y, k, tx_pos, source_type='plane')

        # Should have unit magnitude (plane waves with real k)
        np.testing.assert_allclose(np.abs(E_inc), 1.0, rtol=1e-14)

    def test_unknown_source_type_raises(self):
        """Unknown source type should raise ValueError."""
        lx, ly, nx, ny = 1.0, 1.0, 16, 16
        X, Y, xvec, yvec, dx, dy = create_grid(lx, ly, nx, ny)
        k = 10.0

        tx_pos = setup_transmitters(4, radius=0.5)

        with pytest.raises(ValueError, match="Unknown source type"):
            compute_incident_field_all_views(X, Y, k, tx_pos, source_type='unknown')


class TestSetupTransmitters:
    """Test transmitter position setup."""

    def test_number_of_transmitters(self):
        """Should return correct number of positions."""
        n_tx = 12
        positions = setup_transmitters(n_tx, radius=1.0)

        assert positions.shape == (n_tx, 2)

    def test_positions_on_circle(self):
        """All positions should be on circle of given radius."""
        radius = 0.5
        positions = setup_transmitters(8, radius)

        r = np.sqrt(positions[:, 0]**2 + positions[:, 1]**2)
        np.testing.assert_allclose(r, radius, rtol=1e-14)

    def test_uniform_angular_spacing(self):
        """Positions should be uniformly spaced in angle."""
        n_tx = 12
        positions = setup_transmitters(n_tx, radius=1.0)

        theta = np.arctan2(positions[:, 1], positions[:, 0])
        # Sort and handle angle wrapping
        theta_sorted = np.sort(theta)
        d_theta = np.diff(theta_sorted)

        expected_spacing = 2 * np.pi / n_tx
        np.testing.assert_allclose(d_theta, expected_spacing, rtol=1e-10)

    def test_first_position(self):
        """First transmitter should be at θ=0."""
        positions = setup_transmitters(12, radius=1.0)

        # First position should be (radius, 0)
        np.testing.assert_allclose(positions[0, 0], 1.0, rtol=1e-14)
        np.testing.assert_allclose(positions[0, 1], 0.0, atol=1e-14)

    def test_matlab_equivalent(self):
        """
        Verify MATLAB formula:
            meas_pos_theta = linspace(0, 2*pi - 2*pi/Nm, Nm)
            tx_x = Rm * cos(meas_pos_theta)
            tx_y = Rm * sin(meas_pos_theta)
        """
        Nm = 12
        Rm = 0.1

        positions = setup_transmitters(Nm, Rm)

        # MATLAB formula
        theta = np.linspace(0, 2*np.pi - 2*np.pi/Nm, Nm)
        expected_x = Rm * np.cos(theta)
        expected_y = Rm * np.sin(theta)

        np.testing.assert_allclose(positions[:, 0], expected_x, rtol=1e-14)
        np.testing.assert_allclose(positions[:, 1], expected_y, rtol=1e-14)

    def test_no_overlap(self):
        """Full circle should not include both 0 and 2π."""
        n_tx = 12
        positions = setup_transmitters(n_tx, radius=1.0, full_circle=True)

        # First and last should NOT be the same
        assert not np.allclose(positions[0], positions[-1])


class TestSetupReceivers:
    """Test receiver position setup."""

    def test_same_as_transmitters(self):
        """setup_receivers should give same result as setup_transmitters."""
        n = 12
        radius = 0.5

        tx_pos = setup_transmitters(n, radius)
        rx_pos = setup_receivers(n, radius)

        np.testing.assert_allclose(tx_pos, rx_pos, rtol=1e-14)

    def test_output_shape(self):
        """Should return (n_rx, 2) array."""
        n_rx = 8
        positions = setup_receivers(n_rx, radius=0.3)

        assert positions.shape == (n_rx, 2)


class TestMATLABCompatibility:
    """Tests for MATLAB compatibility."""

    def test_scenario_parameters(self):
        """Test with parameters from DATA_scenario.mat."""
        # Parameters matching MATLAB reference
        lx, ly = 0.1, 0.1
        nx, ny = 60, 60
        freq = 1e9
        Rm = 0.1
        Nv = 12

        X, Y, xvec, yvec, dx, dy = create_grid(lx, ly, nx, ny)
        k = compute_wavenumber(freq, epsilon_r=1.0, sigma=0.0)

        tx_pos = setup_transmitters(Nv, Rm)
        E_inc = compute_incident_field_all_views(X, Y, k, tx_pos, source_type='line')

        # Verify dimensions
        assert E_inc.shape == (ny, nx, Nv)

        # Verify finite values
        assert np.all(np.isfinite(E_inc))

    def test_experimental_parameters(self):
        """Test with experimental data parameters (4 GHz)."""
        # Parameters from experimental setup
        lx, ly = 0.15, 0.15
        nx, ny = 64, 64
        freq = 4e9
        Rm = 0.76  # ~76 cm measurement radius

        X, Y, xvec, yvec, dx, dy = create_grid(lx, ly, nx, ny)
        k = compute_wavenumber(freq, epsilon_r=1.0, sigma=0.0)

        tx_pos = setup_transmitters(36, Rm)
        E_inc = compute_incident_field_all_views(X, Y, k, tx_pos)

        assert E_inc.shape == (ny, nx, 36)
        assert np.all(np.isfinite(E_inc))


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_transmitter(self):
        """Test with single transmitter."""
        lx, ly, nx, ny = 1.0, 1.0, 32, 32
        X, Y, xvec, yvec, dx, dy = create_grid(lx, ly, nx, ny)
        k = 10.0

        tx_pos = setup_transmitters(1, radius=0.5)
        E_inc = compute_incident_field_all_views(X, Y, k, tx_pos)

        assert E_inc.shape == (ny, nx, 1)

    def test_large_number_of_transmitters(self):
        """Test with large number of transmitters."""
        lx, ly, nx, ny = 1.0, 1.0, 16, 16
        X, Y, xvec, yvec, dx, dy = create_grid(lx, ly, nx, ny)
        k = 10.0

        tx_pos = setup_transmitters(360, radius=0.5)
        E_inc = compute_incident_field_all_views(X, Y, k, tx_pos)

        assert E_inc.shape == (ny, nx, 360)

    def test_complex_wavenumber(self):
        """Test with lossy medium (complex wavenumber)."""
        lx, ly, nx, ny = 1.0, 1.0, 32, 32
        X, Y, xvec, yvec, dx, dy = create_grid(lx, ly, nx, ny)
        k = 10.0 + 0.5j  # Lossy medium

        E_inc = compute_incident_field_line_source(X, Y, k, source_position=(0.5, 0.0))

        # Should be finite
        assert np.all(np.isfinite(E_inc))

        # Magnitude should vary (decay due to loss)
        mags = np.abs(E_inc)
        assert mags.max() > mags.min()

    def test_source_at_origin(self):
        """Test line source at origin."""
        lx, ly, nx, ny = 1.0, 1.0, 32, 32
        X, Y, xvec, yvec, dx, dy = create_grid(lx, ly, nx, ny)
        k = 10.0

        E_inc = compute_incident_field_line_source(X, Y, k, source_position=(0.0, 0.0))

        # Should handle gracefully
        assert np.all(np.isfinite(E_inc))

    def test_negative_angle_direction(self):
        """Test plane wave with negative angle."""
        lx, ly, nx, ny = 1.0, 1.0, 32, 32
        X, Y, xvec, yvec, dx, dy = create_grid(lx, ly, nx, ny)
        k = 10.0

        E_inc = compute_incident_field_plane_wave(X, Y, k, direction=-np.pi/4)

        # Should be valid
        assert np.all(np.isfinite(E_inc))
        np.testing.assert_allclose(np.abs(E_inc), 1.0, rtol=1e-14)
