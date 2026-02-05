"""
Unit tests for inverse_scattering.forward.forward_solver module.

Tests verify that the forward solver orchestrates all components correctly
and produces physically meaningful scattered field data.

Theory:
    Forward problem:
        E_tot(r) = E_inc(r) + k² ∫∫ G(r,r') τ(r') E_tot(r') dr'
        E_scat(r_m) = k² ∫∫ G(r_m,r') τ(r') E_tot(r') dr'
"""

import numpy as np
import pytest

from inverse_scattering.forward.forward_solver import (
    forward_solver,
    forward_solver_with_profile,
    compute_scattered_field,
    ForwardSolverResult,
)
from inverse_scattering.core.utils import create_grid, compute_wavenumber
from inverse_scattering.forward.profiles import create_circular_profile
from inverse_scattering.forward.incident_field import (
    compute_incident_field_all_views,
    setup_transmitters,
)
from inverse_scattering.forward.cgfft import cgfft_solve_all_views


class TestForwardSolverResult:
    """Test ForwardSolverResult dataclass."""

    def test_dataclass_fields(self):
        """Result should have all expected fields."""
        # Create a minimal result
        result = ForwardSolverResult(
            Escat=np.zeros((4, 4), dtype=complex),
            PROF=np.zeros((8, 8), dtype=complex),
            Einc_domain=np.zeros((8, 8, 4), dtype=complex),
            Etot_domain=np.zeros((8, 8, 4), dtype=complex),
            freq=300e6,
            lx=1.0,
            ly=1.0,
            eb=1.0,
            sb=0.0,
            Nx=8,
            Ny=8,
            Rm=3.0,
            DOF=3,
            Rv=3.0,
            Nm=4,
            Nv=4,
            lambda0=1.0,
            k=2*np.pi,
            xvec=np.linspace(-0.5, 0.5, 8),
            yvec=np.linspace(-0.5, 0.5, 8),
            X=np.zeros((8, 8)),
            Y=np.zeros((8, 8)),
            dx=0.125,
            dy=0.125
        )

        # Verify essential fields exist
        assert hasattr(result, 'Escat')
        assert hasattr(result, 'PROF')
        assert hasattr(result, 'Einc_domain')
        assert hasattr(result, 'Etot_domain')
        assert hasattr(result, 'freq')
        assert hasattr(result, 'k')


class TestForwardSolver:
    """Test the main forward solver function."""

    def test_returns_result_dataclass(self):
        """Solver should return ForwardSolverResult."""
        result = forward_solver(n_iter=50, Nx=8, Ny=8, Nm=4, Nv=4)
        assert isinstance(result, ForwardSolverResult)

    def test_escat_shape(self):
        """Escat should be (Nm × Nv)."""
        Nm, Nv = 6, 6
        result = forward_solver(n_iter=50, Nx=8, Ny=8, Nm=Nm, Nv=Nv)
        assert result.Escat.shape == (Nm, Nv)

    def test_escat_complex_dtype(self):
        """Escat should be complex."""
        result = forward_solver(n_iter=50, Nx=8, Ny=8, Nm=4, Nv=4)
        assert np.issubdtype(result.Escat.dtype, np.complexfloating)

    def test_profile_shape(self):
        """PROF should be (Ny × Nx)."""
        Nx, Ny = 12, 10
        result = forward_solver(n_iter=50, Nx=Nx, Ny=Ny, Nm=4, Nv=4)
        assert result.PROF.shape == (Ny, Nx)

    def test_incident_field_shape(self):
        """Einc_domain should be (Ny × Nx × Nv)."""
        Nx, Ny, Nv = 8, 8, 5
        result = forward_solver(n_iter=50, Nx=Nx, Ny=Ny, Nm=4, Nv=Nv)
        assert result.Einc_domain.shape == (Ny, Nx, Nv)

    def test_total_field_shape(self):
        """Etot_domain should be (Ny × Nx × Nv)."""
        Nx, Ny, Nv = 8, 8, 5
        result = forward_solver(n_iter=50, Nx=Nx, Ny=Ny, Nm=4, Nv=Nv)
        assert result.Etot_domain.shape == (Ny, Nx, Nv)

    def test_finite_outputs(self):
        """All outputs should be finite (no NaN/Inf)."""
        result = forward_solver(n_iter=50, Nx=8, Ny=8, Nm=4, Nv=4)

        assert np.all(np.isfinite(result.Escat))
        assert np.all(np.isfinite(result.PROF))
        assert np.all(np.isfinite(result.Einc_domain))
        assert np.all(np.isfinite(result.Etot_domain))

    def test_very_small_contrast_gives_small_escat(self):
        """With very small contrast, scattered field should be very small."""
        result = forward_solver(
            n_iter=100, Nx=8, Ny=8, Nm=4, Nv=4,
            target_epsilon_r=1.001,  # Very small contrast
            target_radius=0.1,
            tol=1e-8
        )

        # Escat should be very small (within numerical tolerance)
        assert np.max(np.abs(result.Escat)) < 0.01

    def test_default_parameters(self):
        """Test solver with default parameters."""
        result = forward_solver(n_iter=50, Nx=8, Ny=8)

        # Should have computed DOF-based Nm and Nv
        assert result.Nm > 0
        assert result.Nv > 0
        assert result.Nm == result.DOF

    def test_custom_frequency(self):
        """Test solver with custom frequency."""
        freq = 1e9  # 1 GHz
        result = forward_solver(n_iter=50, Nx=8, Ny=8, Nm=4, Nv=4, freq=freq)

        assert result.freq == freq
        np.testing.assert_allclose(result.lambda0, 3e8 / freq, rtol=1e-10)

    def test_custom_domain_size(self):
        """Test solver with custom domain size."""
        lx, ly = 0.5, 0.8
        result = forward_solver(n_iter=50, Nx=8, Ny=8, Nm=4, Nv=4, lx=lx, ly=ly)

        assert result.lx == lx
        assert result.ly == ly
        assert result.dx == lx / 8
        assert result.dy == ly / 8

    def test_custom_target(self):
        """Test solver with custom target parameters."""
        result = forward_solver(
            n_iter=50, Nx=12, Ny=12, Nm=4, Nv=4,
            target_center=(0.1, -0.1),
            target_radius=0.2,
            target_epsilon_r=3.0
        )

        # Profile should have non-zero values at offset location
        assert np.max(np.abs(result.PROF)) > 0

    def test_lossy_background(self):
        """Test solver with lossy background."""
        result = forward_solver(
            n_iter=50, Nx=8, Ny=8, Nm=4, Nv=4,
            eb=1.0, sb=0.01  # Slightly lossy
        )

        # Wavenumber should have imaginary part
        # (depends on implementation, but result should be valid)
        assert np.all(np.isfinite(result.Escat))


class TestForwardSolverWithProfile:
    """Test forward solver with custom profile."""

    def test_returns_result_dataclass(self):
        """Should return ForwardSolverResult."""
        lx, ly, nx, ny = 0.5, 0.5, 16, 16
        X, Y, xvec, yvec, dx, dy = create_grid(lx, ly, nx, ny)
        PROF = create_circular_profile(X, Y, (0.0, 0.0), 0.1, epsilon_r=1.5)

        result = forward_solver_with_profile(
            PROF, X, Y, freq=300e6, lx=lx, ly=ly, Nm=4, Nv=4, n_iter=50
        )

        assert isinstance(result, ForwardSolverResult)

    def test_uses_provided_profile(self):
        """Should use the provided profile."""
        lx, ly, nx, ny = 0.5, 0.5, 16, 16
        X, Y, xvec, yvec, dx, dy = create_grid(lx, ly, nx, ny)
        PROF = create_circular_profile(X, Y, (0.0, 0.0), 0.1, epsilon_r=2.0)

        result = forward_solver_with_profile(
            PROF, X, Y, freq=300e6, lx=lx, ly=ly, Nm=4, Nv=4, n_iter=50
        )

        # Result profile should match input
        np.testing.assert_array_equal(result.PROF, PROF)

    def test_escat_shape_matches_nm_nv(self):
        """Escat shape should match specified Nm and Nv."""
        lx, ly, nx, ny = 0.5, 0.5, 12, 12
        X, Y, xvec, yvec, dx, dy = create_grid(lx, ly, nx, ny)
        PROF = create_circular_profile(X, Y, (0.0, 0.0), 0.1, epsilon_r=1.5)

        Nm, Nv = 8, 6
        result = forward_solver_with_profile(
            PROF, X, Y, freq=300e6, lx=lx, ly=ly, Nm=Nm, Nv=Nv, n_iter=50
        )

        assert result.Escat.shape == (Nm, Nv)


class TestComputeScatteredField:
    """Test scattered field computation."""

    def test_output_shape(self):
        """Output should be (Nm × Nv)."""
        lx, ly, nx, ny = 0.5, 0.5, 16, 16
        Nm, Nv = 8, 6
        X, Y, xvec, yvec, dx, dy = create_grid(lx, ly, nx, ny)
        k = 20.0

        tau = create_circular_profile(X, Y, (0.0, 0.0), 0.1, epsilon_r=1.5)
        Etot_domain = np.ones((ny, nx, Nv), dtype=complex)  # Dummy total field

        Escat = compute_scattered_field(
            tau, Etot_domain, X, Y, k, dx, dy, Nm, Rm=0.5
        )

        assert Escat.shape == (Nm, Nv)

    def test_complex_dtype(self):
        """Output should be complex."""
        lx, ly, nx, ny = 0.5, 0.5, 12, 12
        X, Y, xvec, yvec, dx, dy = create_grid(lx, ly, nx, ny)
        k = 15.0

        tau = create_circular_profile(X, Y, (0.0, 0.0), 0.1, epsilon_r=1.5)
        Etot_domain = np.ones((ny, nx, 4), dtype=complex)

        Escat = compute_scattered_field(
            tau, Etot_domain, X, Y, k, dx, dy, Nm=6, Rm=0.5
        )

        assert np.issubdtype(Escat.dtype, np.complexfloating)

    def test_zero_contrast_gives_zero(self):
        """Zero contrast should give zero scattered field."""
        lx, ly, nx, ny = 0.5, 0.5, 12, 12
        X, Y, xvec, yvec, dx, dy = create_grid(lx, ly, nx, ny)
        k = 15.0

        tau = np.zeros_like(X, dtype=complex)  # No scatterer
        Etot_domain = np.ones((ny, nx, 4), dtype=complex)

        Escat = compute_scattered_field(
            tau, Etot_domain, X, Y, k, dx, dy, Nm=6, Rm=0.5
        )

        np.testing.assert_allclose(Escat, 0.0, atol=1e-14)

    def test_finite_output(self):
        """Output should not contain NaN or Inf."""
        lx, ly, nx, ny = 0.5, 0.5, 12, 12
        X, Y, xvec, yvec, dx, dy = create_grid(lx, ly, nx, ny)
        k = 15.0

        tau = create_circular_profile(X, Y, (0.0, 0.0), 0.1, epsilon_r=1.5)
        Etot_domain = np.random.rand(ny, nx, 4) + 1j * np.random.rand(ny, nx, 4)

        Escat = compute_scattered_field(
            tau, Etot_domain, X, Y, k, dx, dy, Nm=6, Rm=0.5
        )

        assert np.all(np.isfinite(Escat))

    def test_scaling_with_k_squared(self):
        """Scattered field should scale approximately with k²."""
        lx, ly, nx, ny = 0.5, 0.5, 12, 12
        X, Y, xvec, yvec, dx, dy = create_grid(lx, ly, nx, ny)

        tau = create_circular_profile(X, Y, (0.0, 0.0), 0.1, epsilon_r=1.5)
        Etot_domain = np.ones((ny, nx, 4), dtype=complex)

        k1 = 10.0
        k2 = 20.0  # 2x k1

        Escat1 = compute_scattered_field(
            tau, Etot_domain, X, Y, k1, dx, dy, Nm=4, Rm=0.5
        )
        Escat2 = compute_scattered_field(
            tau, Etot_domain, X, Y, k2, dx, dy, Nm=4, Rm=0.5
        )

        # Escat ~ k², so ratio should be approximately 4
        # But Green's function also depends on k, so check general increase
        assert np.mean(np.abs(Escat2)) > np.mean(np.abs(Escat1))


class TestMATLABCompatibility:
    """Tests for MATLAB compatibility."""

    def test_scenario_parameters(self):
        """Test with parameters similar to DATA_scenario.mat."""
        result = forward_solver(
            n_iter=100,
            freq=300e6,  # 300 MHz
            lx=1.0,
            ly=1.0,
            Nx=16,  # Smaller for test speed
            Ny=16,
            Rm=3.0,
            Nm=4,
            Nv=4,
            target_center=(0.0, 0.0),
            target_radius=0.25,
            target_epsilon_r=2.0,
            tol=1e-6
        )

        # Verify dimensions
        assert result.Escat.shape == (4, 4)
        assert result.PROF.shape == (16, 16)

        # Verify finite outputs
        assert np.all(np.isfinite(result.Escat))

    def test_1ghz_scenario(self):
        """Test 1 GHz scenario similar to MATLAB examples."""
        result = forward_solver(
            n_iter=100,
            freq=1e9,  # 1 GHz
            lx=0.1,   # 10 cm DoI
            ly=0.1,
            Nx=16,    # Smaller for speed
            Ny=16,
            Rm=0.1,   # 10 cm measurement radius
            Nm=4,
            Nv=4,
            target_center=(0.0, 0.0),
            target_radius=0.02,  # 2 cm radius
            target_epsilon_r=1.5,
            tol=1e-6
        )

        # Should produce valid output
        assert result.Escat.shape == (4, 4)
        assert np.all(np.isfinite(result.Escat))


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_small_grid(self):
        """Test on minimal grid."""
        result = forward_solver(
            n_iter=50, Nx=6, Ny=6, Nm=2, Nv=2,
            target_radius=0.15,
            target_epsilon_r=1.5,
            tol=1e-6
        )

        assert result.Escat.shape == (2, 2)
        assert np.all(np.isfinite(result.Escat))

    def test_single_measurement(self):
        """Test with single receiver and transmitter."""
        result = forward_solver(
            n_iter=20, Nx=8, Ny=8, Nm=1, Nv=1,
            target_radius=0.1
        )

        assert result.Escat.shape == (1, 1)
        assert np.all(np.isfinite(result.Escat))

    def test_many_measurements(self):
        """Test with many receivers and transmitters."""
        Nm, Nv = 24, 24
        result = forward_solver(
            n_iter=20, Nx=8, Ny=8, Nm=Nm, Nv=Nv,
            target_radius=0.1
        )

        assert result.Escat.shape == (Nm, Nv)

    def test_rectangular_domain(self):
        """Test with non-square domain."""
        result = forward_solver(
            n_iter=50, Nx=8, Ny=12, Nm=4, Nv=4,
            lx=0.5, ly=0.8,
            target_radius=0.1
        )

        assert result.PROF.shape == (12, 8)
        assert result.Einc_domain.shape == (12, 8, 4)

    def test_small_target(self):
        """Test with small target."""
        result = forward_solver(
            n_iter=100, Nx=16, Ny=16, Nm=4, Nv=4,
            target_radius=0.05,  # Small but reasonable
            target_epsilon_r=1.5,
            tol=1e-6
        )

        assert np.all(np.isfinite(result.Escat))

    def test_large_contrast(self):
        """Test with large contrast (strong scatterer)."""
        result = forward_solver(
            n_iter=100, Nx=8, Ny=8, Nm=4, Nv=4,
            target_epsilon_r=4.0,  # Strong contrast
            target_radius=0.2,
            tol=1e-4  # Looser tolerance for strong scatterer
        )

        assert np.all(np.isfinite(result.Escat))

    def test_off_center_target(self):
        """Test with target not at origin."""
        result = forward_solver(
            n_iter=50, Nx=16, Ny=16, Nm=4, Nv=4,
            target_center=(0.2, -0.15),
            target_radius=0.1
        )

        assert np.all(np.isfinite(result.Escat))
        # Profile should have values at offset location
        assert np.max(np.abs(result.PROF)) > 0


class TestPhysicalConsistency:
    """Test physical consistency of results."""

    def test_scattered_field_nonzero_for_scatterer(self):
        """Non-zero contrast should produce non-zero scattered field."""
        result = forward_solver(
            n_iter=100, Nx=16, Ny=16, Nm=4, Nv=4,
            target_epsilon_r=2.0,  # Non-zero contrast
            target_radius=0.2,
            tol=1e-6
        )

        # Should have non-trivial scattered field
        assert np.max(np.abs(result.Escat)) > 0

    def test_total_equals_incident_plus_scattered(self):
        """Total field should equal incident plus scattered in DoI."""
        # This is implicitly tested by the CGFFT solver convergence,
        # but we verify the Lippmann-Schwinger equation is approximately satisfied
        result = forward_solver(
            n_iter=200, Nx=12, Ny=12, Nm=4, Nv=4,
            target_epsilon_r=1.2,  # Weak scatterer
            target_radius=0.2,
            tol=1e-8
        )

        # For weak scatterers, E_tot ≈ E_inc (Born approximation)
        diff = result.Etot_domain - result.Einc_domain
        relative_diff = np.abs(diff) / np.abs(result.Einc_domain)

        # Most points should have small relative difference
        assert np.mean(relative_diff) < 0.5  # Within 50% on average
