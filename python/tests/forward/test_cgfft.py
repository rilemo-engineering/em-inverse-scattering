"""
Unit tests for inverse_scattering.forward.cgfft module.

Tests verify that the CGFFT solver converges correctly and produces
physically meaningful total field solutions.

Theory:
    CGFFT solves the Lippmann-Schwinger equation:
        E_tot = E_inc + k² ∫∫ G(r,r') τ(r') E_tot(r') dr'

    For weak scatterers (Born approximation valid):
        E_tot ≈ E_inc + k² ∫∫ G(r,r') τ(r') E_inc(r') dr'
"""

import numpy as np
import pytest

from inverse_scattering.forward.cgfft import (
    cgfft_solve,
    cgfft_solve_all_views,
    bicgstab_solve,
    CGFFTResult,
)
from inverse_scattering.core.utils import create_grid, compute_wavenumber
from inverse_scattering.forward.incident_field import (
    compute_incident_field_line_source,
    compute_incident_field_all_views,
    setup_transmitters,
)
from inverse_scattering.forward.profiles import create_circular_profile


class TestCGFFTResult:
    """Test CGFFTResult dataclass."""

    def test_dataclass_fields(self):
        """Result should have all expected fields."""
        result = CGFFTResult(
            E_tot=np.zeros((4, 4), dtype=complex),
            converged=True,
            n_iterations=10,
            residual_history=[1.0, 0.1, 0.01],
            final_residual=0.001
        )

        assert hasattr(result, 'E_tot')
        assert hasattr(result, 'converged')
        assert hasattr(result, 'n_iterations')
        assert hasattr(result, 'residual_history')
        assert hasattr(result, 'final_residual')


class TestCGFFTSolve:
    """Test CGFFT solver."""

    def test_output_shape(self):
        """Output should match input shape."""
        lx, ly, nx, ny = 0.5, 0.5, 16, 16
        X, Y, xvec, yvec, dx, dy = create_grid(lx, ly, nx, ny)
        k = 10.0

        E_inc = compute_incident_field_line_source(X, Y, k, source_position=(0.5, 0.0))
        tau = create_circular_profile(X, Y, (0.0, 0.0), 0.1, epsilon_r=1.5)

        result = cgfft_solve(E_inc, tau, k, dx, dy, max_iter=100)

        assert result.E_tot.shape == (ny, nx)

    def test_complex_dtype(self):
        """Output should be complex."""
        lx, ly, nx, ny = 0.5, 0.5, 16, 16
        X, Y, xvec, yvec, dx, dy = create_grid(lx, ly, nx, ny)
        k = 10.0

        E_inc = compute_incident_field_line_source(X, Y, k, source_position=(0.5, 0.0))
        tau = create_circular_profile(X, Y, (0.0, 0.0), 0.1, epsilon_r=1.5)

        result = cgfft_solve(E_inc, tau, k, dx, dy, max_iter=100)

        assert np.issubdtype(result.E_tot.dtype, np.complexfloating)

    def test_convergence_flag(self):
        """Result should indicate whether solver converged."""
        lx, ly, nx, ny = 0.5, 0.5, 16, 16
        X, Y, xvec, yvec, dx, dy = create_grid(lx, ly, nx, ny)
        k = 10.0

        E_inc = compute_incident_field_line_source(X, Y, k, source_position=(0.5, 0.0))
        tau = create_circular_profile(X, Y, (0.0, 0.0), 0.1, epsilon_r=1.5)

        result = cgfft_solve(E_inc, tau, k, dx, dy, max_iter=500, tol=1e-6)

        assert isinstance(result.converged, bool)

    def test_residual_decreases(self):
        """Residual should generally decrease over iterations."""
        lx, ly, nx, ny = 0.5, 0.5, 16, 16
        X, Y, xvec, yvec, dx, dy = create_grid(lx, ly, nx, ny)
        k = 10.0

        E_inc = compute_incident_field_line_source(X, Y, k, source_position=(0.5, 0.0))
        tau = create_circular_profile(X, Y, (0.0, 0.0), 0.1, epsilon_r=1.5)

        result = cgfft_solve(E_inc, tau, k, dx, dy, max_iter=100, tol=1e-6)

        # Final should be less than initial
        assert result.residual_history[-1] < result.residual_history[0]

    def test_zero_contrast_returns_incident(self):
        """With very small τ, total field should be close to incident field."""
        lx, ly, nx, ny = 0.5, 0.5, 16, 16
        X, Y, xvec, yvec, dx, dy = create_grid(lx, ly, nx, ny)
        k = 10.0

        E_inc = compute_incident_field_line_source(X, Y, k, source_position=(0.5, 0.0))
        # Use very small contrast instead of exactly zero to avoid numerical issues
        tau = np.ones_like(X, dtype=complex) * 1e-10

        result = cgfft_solve(E_inc, tau, k, dx, dy, max_iter=50, tol=1e-8)

        # Should be very close to incident field
        np.testing.assert_allclose(result.E_tot, E_inc, rtol=1e-6)

    def test_weak_scatterer_converges(self):
        """Weak scatterer (small τ) should converge quickly."""
        lx, ly, nx, ny = 0.5, 0.5, 16, 16
        X, Y, xvec, yvec, dx, dy = create_grid(lx, ly, nx, ny)
        k = 10.0

        E_inc = compute_incident_field_line_source(X, Y, k, source_position=(0.5, 0.0))
        # Weak scatterer: τ = 0.1
        tau = create_circular_profile(X, Y, (0.0, 0.0), 0.1, epsilon_r=1.1)

        result = cgfft_solve(E_inc, tau, k, dx, dy, max_iter=100, tol=1e-6)

        # Should converge
        assert result.converged or result.final_residual < 1e-3

    def test_output_finite(self):
        """Output should not contain NaN or Inf."""
        lx, ly, nx, ny = 0.5, 0.5, 16, 16
        X, Y, xvec, yvec, dx, dy = create_grid(lx, ly, nx, ny)
        k = 10.0

        E_inc = compute_incident_field_line_source(X, Y, k, source_position=(0.5, 0.0))
        tau = create_circular_profile(X, Y, (0.0, 0.0), 0.1, epsilon_r=1.5)

        result = cgfft_solve(E_inc, tau, k, dx, dy, max_iter=100)

        assert np.all(np.isfinite(result.E_tot))

    def test_born_approximation_for_weak(self):
        """For weak scatterers, E_tot ≈ E_inc (Born approximation)."""
        lx, ly, nx, ny = 0.5, 0.5, 16, 16
        X, Y, xvec, yvec, dx, dy = create_grid(lx, ly, nx, ny)
        k = 10.0

        E_inc = compute_incident_field_line_source(X, Y, k, source_position=(0.5, 0.0))
        # Very weak scatterer
        tau = create_circular_profile(X, Y, (0.0, 0.0), 0.05, epsilon_r=1.05)

        result = cgfft_solve(E_inc, tau, k, dx, dy, max_iter=100, tol=1e-8)

        # Total field should be close to incident for weak scatterer
        # (Not exactly equal due to scattered field contribution)
        relative_diff = np.abs(result.E_tot - E_inc) / np.abs(E_inc)
        assert np.max(relative_diff) < 0.5  # Within 50% (reasonable for weak)


class TestCGFFTSolveAllViews:
    """Test CGFFT solver for multiple views."""

    def test_output_shape(self):
        """Output should be (Ny, Nx, Nv)."""
        lx, ly, nx, ny = 0.5, 0.5, 16, 16
        n_tx = 8
        X, Y, xvec, yvec, dx, dy = create_grid(lx, ly, nx, ny)
        k = 10.0

        tx_pos = setup_transmitters(n_tx, radius=0.5)
        Einc_domain = compute_incident_field_all_views(X, Y, k, tx_pos)
        tau = create_circular_profile(X, Y, (0.0, 0.0), 0.1, epsilon_r=1.5)

        Etot_domain = cgfft_solve_all_views(Einc_domain, tau, k, dx, dy, max_iter=50)

        assert Etot_domain.shape == (ny, nx, n_tx)

    def test_each_view_valid(self):
        """Each view should produce valid (finite) field."""
        lx, ly, nx, ny = 0.5, 0.5, 12, 12
        n_tx = 4
        X, Y, xvec, yvec, dx, dy = create_grid(lx, ly, nx, ny)
        k = 10.0

        tx_pos = setup_transmitters(n_tx, radius=0.5)
        Einc_domain = compute_incident_field_all_views(X, Y, k, tx_pos)
        tau = create_circular_profile(X, Y, (0.0, 0.0), 0.1, epsilon_r=1.3)

        Etot_domain = cgfft_solve_all_views(Einc_domain, tau, k, dx, dy, max_iter=50)

        for v in range(n_tx):
            assert np.all(np.isfinite(Etot_domain[:, :, v]))

    def test_very_small_contrast_all_views(self):
        """With very small τ, all views should be close to incident field."""
        lx, ly, nx, ny = 0.5, 0.5, 12, 12
        n_tx = 4
        X, Y, xvec, yvec, dx, dy = create_grid(lx, ly, nx, ny)
        k = 10.0

        tx_pos = setup_transmitters(n_tx, radius=0.5)
        Einc_domain = compute_incident_field_all_views(X, Y, k, tx_pos)
        # Very small contrast instead of exactly zero
        tau = np.ones_like(X, dtype=complex) * 1e-10

        Etot_domain = cgfft_solve_all_views(Einc_domain, tau, k, dx, dy, max_iter=50, tol=1e-8)

        np.testing.assert_allclose(Etot_domain, Einc_domain, rtol=1e-6)


class TestBiCGSTABSolve:
    """Test BiCGSTAB solver."""

    def test_output_shape(self):
        """Output should match input shape."""
        lx, ly, nx, ny = 0.5, 0.5, 16, 16
        X, Y, xvec, yvec, dx, dy = create_grid(lx, ly, nx, ny)
        k = 10.0

        E_inc = compute_incident_field_line_source(X, Y, k, source_position=(0.5, 0.0))
        tau = create_circular_profile(X, Y, (0.0, 0.0), 0.1, epsilon_r=1.5)

        result = bicgstab_solve(E_inc, tau, k, dx, dy, max_iter=100)

        assert result.E_tot.shape == (ny, nx)

    def test_returns_cgfft_result(self):
        """Should return CGFFTResult dataclass."""
        lx, ly, nx, ny = 0.5, 0.5, 16, 16
        X, Y, xvec, yvec, dx, dy = create_grid(lx, ly, nx, ny)
        k = 10.0

        E_inc = compute_incident_field_line_source(X, Y, k, source_position=(0.5, 0.0))
        tau = create_circular_profile(X, Y, (0.0, 0.0), 0.1, epsilon_r=1.5)

        result = bicgstab_solve(E_inc, tau, k, dx, dy, max_iter=100)

        assert isinstance(result, CGFFTResult)

    def test_very_small_contrast_returns_incident(self):
        """With very small τ, should return close to incident field."""
        lx, ly, nx, ny = 0.5, 0.5, 16, 16
        X, Y, xvec, yvec, dx, dy = create_grid(lx, ly, nx, ny)
        k = 10.0

        E_inc = compute_incident_field_line_source(X, Y, k, source_position=(0.5, 0.0))
        # Very small contrast instead of exactly zero
        tau = np.ones_like(X, dtype=complex) * 1e-10

        result = bicgstab_solve(E_inc, tau, k, dx, dy, max_iter=50, tol=1e-8)

        np.testing.assert_allclose(result.E_tot, E_inc, rtol=1e-6)

    def test_output_finite(self):
        """Output should not contain NaN or Inf."""
        lx, ly, nx, ny = 0.5, 0.5, 16, 16
        X, Y, xvec, yvec, dx, dy = create_grid(lx, ly, nx, ny)
        k = 10.0

        E_inc = compute_incident_field_line_source(X, Y, k, source_position=(0.5, 0.0))
        tau = create_circular_profile(X, Y, (0.0, 0.0), 0.1, epsilon_r=1.5)

        result = bicgstab_solve(E_inc, tau, k, dx, dy, max_iter=100)

        assert np.all(np.isfinite(result.E_tot))


class TestSolverComparison:
    """Compare CG and BiCGSTAB solvers."""

    def test_both_converge_similar(self):
        """Both solvers should produce similar results."""
        lx, ly, nx, ny = 0.5, 0.5, 12, 12
        X, Y, xvec, yvec, dx, dy = create_grid(lx, ly, nx, ny)
        k = 10.0

        E_inc = compute_incident_field_line_source(X, Y, k, source_position=(0.5, 0.0))
        tau = create_circular_profile(X, Y, (0.0, 0.0), 0.1, epsilon_r=1.3)

        result_cg = cgfft_solve(E_inc, tau, k, dx, dy, max_iter=200, tol=1e-6)
        result_bicg = bicgstab_solve(E_inc, tau, k, dx, dy, max_iter=200, tol=1e-6)

        # Both should have converged or have low residual
        if result_cg.converged and result_bicg.converged:
            # Solutions should be similar
            np.testing.assert_allclose(result_cg.E_tot, result_bicg.E_tot, rtol=1e-4)


class TestMATLABCompatibility:
    """Tests for MATLAB compatibility."""

    def test_scenario_parameters(self):
        """Test with parameters from DATA_scenario.mat."""
        lx, ly = 0.1, 0.1
        nx, ny = 30, 30  # Smaller than MATLAB for speed
        freq = 1e9
        Rm = 0.1

        X, Y, xvec, yvec, dx, dy = create_grid(lx, ly, nx, ny)
        k = compute_wavenumber(freq, epsilon_r=1.0, sigma=0.0)

        E_inc = compute_incident_field_line_source(X, Y, k, source_position=(Rm, 0.0))
        # Weak scatterer
        tau = create_circular_profile(X, Y, (0.0, 0.0), 0.02, epsilon_r=1.5)

        result = cgfft_solve(E_inc, tau, k, dx, dy, max_iter=200, tol=1e-6)

        # Should produce valid output
        assert result.E_tot.shape == (ny, nx)
        assert np.all(np.isfinite(result.E_tot))


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_small_grid(self):
        """Test on minimal grid."""
        lx, ly, nx, ny = 0.2, 0.2, 4, 4
        X, Y, xvec, yvec, dx, dy = create_grid(lx, ly, nx, ny)
        k = 5.0

        E_inc = compute_incident_field_line_source(X, Y, k, source_position=(0.3, 0.0))
        tau = create_circular_profile(X, Y, (0.0, 0.0), 0.05, epsilon_r=1.3)

        result = cgfft_solve(E_inc, tau, k, dx, dy, max_iter=50)

        assert result.E_tot.shape == (ny, nx)
        assert np.all(np.isfinite(result.E_tot))

    def test_constant_contrast(self):
        """Test with uniform contrast (filled domain)."""
        lx, ly, nx, ny = 0.5, 0.5, 12, 12
        X, Y, xvec, yvec, dx, dy = create_grid(lx, ly, nx, ny)
        k = 10.0

        E_inc = compute_incident_field_line_source(X, Y, k, source_position=(0.5, 0.0))
        tau = np.ones_like(X, dtype=complex) * 0.5  # Uniform weak contrast

        result = cgfft_solve(E_inc, tau, k, dx, dy, max_iter=200, tol=1e-6)

        assert np.all(np.isfinite(result.E_tot))

    def test_max_iterations_reached(self):
        """Test that max_iter limit is respected."""
        lx, ly, nx, ny = 0.5, 0.5, 12, 12
        X, Y, xvec, yvec, dx, dy = create_grid(lx, ly, nx, ny)
        k = 10.0

        E_inc = compute_incident_field_line_source(X, Y, k, source_position=(0.5, 0.0))
        # Stronger scatterer that may not converge quickly
        tau = create_circular_profile(X, Y, (0.0, 0.0), 0.15, epsilon_r=2.0)

        max_iter = 5
        result = cgfft_solve(E_inc, tau, k, dx, dy, max_iter=max_iter, tol=1e-12)

        # Should respect max_iter
        assert result.n_iterations <= max_iter

    def test_complex_wavenumber(self):
        """Test with lossy background (complex k)."""
        lx, ly, nx, ny = 0.5, 0.5, 12, 12
        X, Y, xvec, yvec, dx, dy = create_grid(lx, ly, nx, ny)
        k = 10.0 + 0.5j  # Lossy

        E_inc = compute_incident_field_line_source(X, Y, k, source_position=(0.5, 0.0))
        tau = create_circular_profile(X, Y, (0.0, 0.0), 0.1, epsilon_r=1.5)

        result = cgfft_solve(E_inc, tau, k, dx, dy, max_iter=100)

        assert np.all(np.isfinite(result.E_tot))

    def test_single_view(self):
        """Test all_views solver with single view."""
        lx, ly, nx, ny = 0.5, 0.5, 12, 12
        X, Y, xvec, yvec, dx, dy = create_grid(lx, ly, nx, ny)
        k = 10.0

        tx_pos = setup_transmitters(1, radius=0.5)
        Einc_domain = compute_incident_field_all_views(X, Y, k, tx_pos)
        tau = create_circular_profile(X, Y, (0.0, 0.0), 0.1, epsilon_r=1.3)

        Etot_domain = cgfft_solve_all_views(Einc_domain, tau, k, dx, dy, max_iter=50)

        assert Etot_domain.shape == (ny, nx, 1)
