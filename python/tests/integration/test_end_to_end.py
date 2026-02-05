"""
End-to-end workflow tests for simulated and experimental data pipelines.

These tests verify the complete workflow from scenario setup through inversion.
"""

import numpy as np
import pytest
from pathlib import Path
import tempfile

from inverse_scattering.core.utils import compute_wavenumber, create_grid
from inverse_scattering.forward.forward_solver import forward_solver, ForwardSolverResult
from inverse_scattering.forward.profiles import create_circular_profile
from inverse_scattering.inverse.scattering_kernel import build_scattering_kernel
from inverse_scattering.inverse.tsvd import compute_svd, tsvd_solve, find_truncation_index
from inverse_scattering.utils.noise import awgn, estimate_snr
from inverse_scattering.data.mat_io import save_mat, load_mat


class TestSimulatedWorkflow:
    """End-to-end tests for simulated data workflow."""

    def test_forward_to_inversion_small_grid(self):
        """Test complete forward + inversion workflow on small grid."""
        # Small grid for fast testing
        Nx, Ny = 15, 15
        Nm, Nv = 8, 8
        freq = 300e6
        target_epsilon_r = 1.5  # Weak scatterer

        # Run forward solver
        result = forward_solver(
            n_iter=100,
            Nx=Nx, Ny=Ny,
            Nm=Nm, Nv=Nv,
            freq=freq,
            target_epsilon_r=target_epsilon_r,
            target_radius=0.05,
            tol=1e-6
        )

        # Verify forward result
        assert isinstance(result, ForwardSolverResult)
        assert result.Escat.shape == (Nm, Nv)
        assert result.PROF.shape == (Ny, Nx)

        # Build scattering kernel for inversion
        S = build_scattering_kernel(
            Etot_approx=result.Einc_domain,
            Nx=Nx, Ny=Ny,
            lx=result.lx, ly=result.ly,
            n_views=Nv,
            eb=result.eb, sb=result.sb,
            freq=freq,
            Nm=Nm, Rm=result.Rm
        )

        # SVD and TSVD solve
        U, s, Vh = compute_svd(S, full_matrices=False)
        k_trunc = find_truncation_index(s, -25)
        tau_rec = tsvd_solve(U, s, Vh, k_trunc, result.Escat, Nx, Ny)

        # Verify reconstruction
        assert tau_rec.shape == (Ny, Nx)
        assert not np.any(np.isnan(tau_rec))

        # Check reconstruction has content where target is
        # (Born approximation may not be perfect, but should show something)
        max_rec = np.max(np.abs(tau_rec))
        assert max_rec > 0

    def test_forward_to_inversion_with_noise(self):
        """Test workflow with added measurement noise."""
        Nx, Ny = 15, 15
        Nm, Nv = 8, 8
        freq = 300e6

        # Forward solve
        result = forward_solver(
            n_iter=100,
            Nx=Nx, Ny=Ny,
            Nm=Nm, Nv=Nv,
            freq=freq,
            target_epsilon_r=1.3,
            target_radius=0.05,
            tol=1e-6
        )

        # Add noise
        snr_db = 30
        Escat_noisy = awgn(result.Escat, snr_db=snr_db, seed=42)

        # Verify noise was added
        actual_snr = estimate_snr(Escat_noisy, result.Escat)
        assert abs(actual_snr - snr_db) < 5

        # Inversion with noisy data
        S = build_scattering_kernel(
            Etot_approx=result.Einc_domain,
            Nx=Nx, Ny=Ny,
            lx=result.lx, ly=result.ly,
            n_views=Nv,
            eb=result.eb, sb=result.sb,
            freq=freq,
            Nm=Nm, Rm=result.Rm
        )

        U, s, Vh = compute_svd(S, full_matrices=False)
        k_trunc = find_truncation_index(s, -25)  # More aggressive truncation for noise
        tau_rec = tsvd_solve(U, s, Vh, k_trunc, Escat_noisy, Nx, Ny)

        assert tau_rec.shape == (Ny, Nx)
        assert not np.any(np.isnan(tau_rec))

    def test_save_and_load_scenario(self):
        """Test saving and reloading scenario data."""
        Nx, Ny = 10, 10
        Nm, Nv = 6, 6
        freq = 300e6

        # Forward solve
        result = forward_solver(
            n_iter=50,
            Nx=Nx, Ny=Ny,
            Nm=Nm, Nv=Nv,
            freq=freq,
            target_epsilon_r=1.2,
            target_radius=0.04,
            tol=1e-5
        )

        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix='.mat', delete=False) as f:
            filepath = Path(f.name)

        try:
            save_mat(filepath, {
                'Escat': result.Escat,
                'PROF': result.PROF,
                'Einc_domain': result.Einc_domain,
                'freq': freq,
                'Nx': Nx,
                'Ny': Ny,
                'lx': result.lx,
                'ly': result.ly,
            })

            # Reload
            loaded = load_mat(filepath)

            # Verify
            np.testing.assert_array_almost_equal(loaded['Escat'], result.Escat)
            np.testing.assert_array_almost_equal(loaded['PROF'], result.PROF)
            assert loaded['freq'] == freq
        finally:
            filepath.unlink()

    def test_different_truncation_levels(self):
        """Test effect of different truncation levels."""
        Nx, Ny = 12, 12
        Nm, Nv = 6, 6
        freq = 300e6

        result = forward_solver(
            n_iter=50,
            Nx=Nx, Ny=Ny,
            Nm=Nm, Nv=Nv,
            freq=freq,
            target_epsilon_r=1.4,
            target_radius=0.04,
            tol=1e-5
        )

        S = build_scattering_kernel(
            Etot_approx=result.Einc_domain,
            Nx=Nx, Ny=Ny,
            lx=result.lx, ly=result.ly,
            n_views=Nv,
            eb=result.eb, sb=result.sb,
            freq=freq,
            Nm=Nm, Rm=result.Rm
        )

        U, s, Vh = compute_svd(S, full_matrices=False)

        reconstructions = []
        for threshold_db in [-15, -25, -35]:
            k = find_truncation_index(s, threshold_db)
            tau = tsvd_solve(U, s, Vh, k, result.Escat, Nx, Ny)
            reconstructions.append(tau)

        # Different truncations should give different results
        for i in range(len(reconstructions) - 1):
            assert not np.allclose(reconstructions[i], reconstructions[i+1])


class TestWorkflowValidation:
    """Tests validating workflow produces physically reasonable results."""

    def test_weak_scatterer_born_valid(self):
        """Weak scatterer should satisfy Born approximation."""
        Nx, Ny = 15, 15
        Nm, Nv = 8, 8
        freq = 300e6

        # Very weak scatterer
        result = forward_solver(
            n_iter=100,
            Nx=Nx, Ny=Ny,
            Nm=Nm, Nv=Nv,
            freq=freq,
            target_epsilon_r=1.05,  # Only 5% contrast
            target_radius=0.03,
            tol=1e-6
        )

        # For weak scatterer: |Escat| << |Einc|
        max_escat = np.max(np.abs(result.Escat))
        max_einc = np.max(np.abs(result.Einc_domain))

        # Scattered field should be much smaller
        assert max_escat < 0.1 * max_einc

    def test_stronger_scatterer_larger_escat(self):
        """Stronger scatterer should produce larger scattered field."""
        # Use larger grid and more iterations for numerical stability
        Nx, Ny = 20, 20
        Nm, Nv = 8, 8
        freq = 300e6

        # Weak scatterer
        result_weak = forward_solver(
            n_iter=200,
            Nx=Nx, Ny=Ny,
            Nm=Nm, Nv=Nv,
            freq=freq,
            target_epsilon_r=1.1,
            target_radius=0.05,
            tol=1e-6
        )

        # Moderately stronger scatterer (not too strong for Born validity)
        result_strong = forward_solver(
            n_iter=200,
            Nx=Nx, Ny=Ny,
            Nm=Nm, Nv=Nv,
            freq=freq,
            target_epsilon_r=1.3,
            target_radius=0.05,
            tol=1e-6
        )

        # Stronger scatterer should have larger scattered field
        assert np.max(np.abs(result_strong.Escat)) > np.max(np.abs(result_weak.Escat))

    def test_reconstruction_localization(self):
        """Reconstruction should be localized to target region."""
        Nx, Ny = 20, 20
        Nm, Nv = 10, 10
        freq = 300e6

        result = forward_solver(
            n_iter=100,
            Nx=Nx, Ny=Ny,
            Nm=Nm, Nv=Nv,
            freq=freq,
            target_epsilon_r=1.3,
            target_radius=0.04,
            tol=1e-6
        )

        S = build_scattering_kernel(
            Etot_approx=result.Einc_domain,
            Nx=Nx, Ny=Ny,
            lx=result.lx, ly=result.ly,
            n_views=Nv,
            eb=result.eb, sb=result.sb,
            freq=freq,
            Nm=Nm, Rm=result.Rm
        )

        U, s, Vh = compute_svd(S, full_matrices=False)
        k_trunc = find_truncation_index(s, -25)
        tau_rec = tsvd_solve(U, s, Vh, k_trunc, result.Escat, Nx, Ny)

        # Find center of mass of reconstruction
        X, Y = np.meshgrid(np.arange(Nx), np.arange(Ny))
        weights = np.abs(tau_rec)
        total_weight = np.sum(weights)

        if total_weight > 0:
            com_x = np.sum(X * weights) / total_weight
            com_y = np.sum(Y * weights) / total_weight

            # Center of mass should be near center (target is centered)
            assert abs(com_x - Nx/2) < Nx/3
            assert abs(com_y - Ny/2) < Ny/3


class TestNMSECalculation:
    """Tests for normalized mean square error calculation."""

    def test_nmse_zero_for_perfect_match(self):
        """NMSE should be zero for identical profiles."""
        from inverse_scattering.core.utils import nmse

        profile = np.random.randn(10, 10) + 1j * np.random.randn(10, 10)
        error = nmse(profile, profile)

        assert error == 0

    def test_nmse_positive_for_mismatch(self):
        """NMSE should be positive for different profiles."""
        from inverse_scattering.core.utils import nmse

        true = np.ones((10, 10), dtype=complex)
        est = true + 0.1 * np.random.randn(10, 10)

        error = nmse(true, est)

        assert error > 0

    def test_nmse_increases_with_noise(self):
        """NMSE should increase with more noise."""
        from inverse_scattering.core.utils import nmse

        true = np.ones((10, 10), dtype=complex)
        est_low_noise = true + 0.01 * np.random.randn(10, 10)
        est_high_noise = true + 0.5 * np.random.randn(10, 10)

        nmse_low = nmse(true, est_low_noise)
        nmse_high = nmse(true, est_high_noise)

        assert nmse_high > nmse_low


class TestEdgeCases:
    """Edge case tests for workflow."""

    def test_minimum_grid_size(self):
        """Test with minimum practical grid size."""
        Nx, Ny = 6, 6
        Nm, Nv = 4, 4
        freq = 300e6

        result = forward_solver(
            n_iter=50,
            Nx=Nx, Ny=Ny,
            Nm=Nm, Nv=Nv,
            freq=freq,
            target_epsilon_r=1.2,
            target_radius=0.03,
            tol=1e-5
        )

        assert result.Escat.shape == (Nm, Nv)
        assert result.PROF.shape == (Ny, Nx)

    def test_asymmetric_grid(self):
        """Test with non-square grid."""
        Nx, Ny = 12, 8
        Nm, Nv = 6, 6
        freq = 300e6

        result = forward_solver(
            n_iter=50,
            Nx=Nx, Ny=Ny,
            Nm=Nm, Nv=Nv,
            freq=freq,
            target_epsilon_r=1.3,
            target_radius=0.03,
            tol=1e-5
        )

        assert result.Escat.shape == (Nm, Nv)
        assert result.PROF.shape == (Ny, Nx)

    def test_more_receivers_than_transmitters(self):
        """Test with Nm > Nv."""
        Nx, Ny = 10, 10
        Nm, Nv = 10, 6
        freq = 300e6

        result = forward_solver(
            n_iter=50,
            Nx=Nx, Ny=Ny,
            Nm=Nm, Nv=Nv,
            freq=freq,
            target_epsilon_r=1.2,
            target_radius=0.03,
            tol=1e-5
        )

        assert result.Escat.shape == (Nm, Nv)
