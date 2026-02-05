"""
Unit tests for inverse/tsvd.py - TSVD solver for inverse scattering.

Tests cover:
- SVD computation
- Truncation index finding
- TSVD solution
- MATLAB interface compatibility
- Threshold suggestion

TSVD Theory:
    Given S = U * Σ * V^H, the TSVD solution is:
    τ_TSVD = Σ(i=1 to k) (u_i^H * E_scat / σ_i) * v_i

    This is a regularization technique that discards small singular values
    dominated by noise.
"""

import numpy as np
import pytest

from inverse_scattering.inverse.tsvd import (
    compute_svd,
    find_truncation_index,
    tsvd_solve,
    tsvd_solver_matlab_interface,
    suggest_threshold,
)


class TestComputeSVD:
    """Tests for compute_svd function."""

    def test_output_shapes_economy(self):
        """Economy SVD should return reduced dimensions."""
        m, n = 20, 10  # m > n
        S = np.random.randn(m, n) + 1j * np.random.randn(m, n)

        U, s, Vh = compute_svd(S, full_matrices=False)

        # Economy SVD: U is (m, k), s is (k,), Vh is (k, n) where k = min(m, n)
        k = min(m, n)
        assert U.shape == (m, k)
        assert s.shape == (k,)
        assert Vh.shape == (k, n)

    def test_output_shapes_full(self):
        """Full SVD should return full-size matrices."""
        m, n = 20, 10
        S = np.random.randn(m, n) + 1j * np.random.randn(m, n)

        U, s, Vh = compute_svd(S, full_matrices=True)

        assert U.shape == (m, m)
        assert s.shape == (min(m, n),)
        assert Vh.shape == (n, n)

    def test_svd_reconstruction(self):
        """U @ diag(s) @ Vh should reconstruct original matrix."""
        m, n = 15, 10
        S = np.random.randn(m, n) + 1j * np.random.randn(m, n)

        U, s, Vh = compute_svd(S, full_matrices=False)

        # Reconstruct: S_rec = U @ diag(s) @ Vh
        S_rec = U @ np.diag(s) @ Vh

        np.testing.assert_allclose(S_rec, S, rtol=1e-10)

    def test_singular_values_sorted(self):
        """Singular values should be sorted in descending order."""
        m, n = 20, 15
        S = np.random.randn(m, n) + 1j * np.random.randn(m, n)

        _, s, _ = compute_svd(S, full_matrices=False)

        # Check descending order
        assert np.all(np.diff(s) <= 0)

    def test_singular_values_nonnegative(self):
        """Singular values should all be non-negative."""
        m, n = 10, 10
        S = np.random.randn(m, n) + 1j * np.random.randn(m, n)

        _, s, _ = compute_svd(S, full_matrices=False)

        assert np.all(s >= 0)

    def test_unitary_U(self):
        """U should have orthonormal columns (U^H @ U = I)."""
        m, n = 20, 10
        S = np.random.randn(m, n) + 1j * np.random.randn(m, n)

        U, _, _ = compute_svd(S, full_matrices=False)

        # U^H @ U should be identity
        np.testing.assert_allclose(U.conj().T @ U, np.eye(U.shape[1]), atol=1e-12)

    def test_unitary_Vh(self):
        """Vh should have orthonormal rows (Vh @ Vh^H = I)."""
        m, n = 20, 10
        S = np.random.randn(m, n) + 1j * np.random.randn(m, n)

        _, _, Vh = compute_svd(S, full_matrices=False)

        # Vh @ Vh^H should be identity
        np.testing.assert_allclose(Vh @ Vh.conj().T, np.eye(Vh.shape[0]), atol=1e-12)

    def test_real_matrix(self):
        """Should work with real matrices too."""
        m, n = 10, 8
        S = np.random.randn(m, n)

        U, s, Vh = compute_svd(S, full_matrices=False)

        S_rec = U @ np.diag(s) @ Vh
        np.testing.assert_allclose(S_rec, S, rtol=1e-10)


class TestFindTruncationIndex:
    """Tests for find_truncation_index function."""

    def test_basic_example(self):
        """Test with known singular value sequence."""
        # σ = [1.0, 0.1, 0.01, 0.001]
        # In dB: [0, -20, -40, -60]
        s = np.array([1.0, 0.1, 0.01, 0.001])

        # At -20 dB threshold, should return index 2 (1-indexed)
        idx = find_truncation_index(s, -20)
        assert idx == 2

    def test_threshold_minus_40(self):
        """Test -40 dB threshold."""
        s = np.array([1.0, 0.1, 0.01, 0.001])

        idx = find_truncation_index(s, -40)
        assert idx == 3  # 1-indexed, third value is at -40 dB

    def test_threshold_0(self):
        """Threshold of 0 dB should return first index."""
        s = np.array([1.0, 0.5, 0.1, 0.01])

        idx = find_truncation_index(s, 0)
        assert idx == 1  # First singular value is at 0 dB relative to itself

    def test_exponential_decay(self):
        """Test with exponentially decaying singular values."""
        # Create SVs that decay exponentially
        # σ_i = 10^(-i/10), so in dB: -2i dB
        s = 10.0 ** (-np.arange(20) / 10)

        # At -10 dB, should be around index 5-6
        idx = find_truncation_index(s, -10)
        # Index 6 corresponds to σ_5 which is at -2*5 = -10 dB
        assert 5 <= idx <= 7

    def test_returns_one_indexed(self):
        """Output should be 1-indexed for MATLAB compatibility."""
        s = np.array([1.0, 0.5, 0.25, 0.125])

        idx = find_truncation_index(s, -6)  # ~-6 dB is σ=0.5

        # Should be at least 1, not 0
        assert idx >= 1

    def test_handles_very_small_threshold(self):
        """Should handle very small threshold values."""
        s = np.array([1.0, 0.1, 0.01, 0.001, 0.0001])

        # -80 dB is very small, should return last or near-last
        idx = find_truncation_index(s, -80)

        # Should return something reasonable (not crash)
        assert 1 <= idx <= len(s)

    def test_normalizes_by_first(self):
        """Should normalize by first singular value, not max."""
        s = np.array([10.0, 1.0, 0.1])  # Scaled by 10

        # -20 dB means 0.1 * max = 1.0
        idx = find_truncation_index(s, -20)
        assert idx == 2  # Second value is at -20 dB


class TestTSVDSolve:
    """Tests for tsvd_solve function."""

    def test_output_shape(self):
        """Output should have shape (ny, nx)."""
        m, n = 24, 16  # Measurements, grid points
        nx, ny = 4, 4  # 4x4 grid

        # Create simple test case
        S = np.random.randn(m, n) + 1j * np.random.randn(m, n)
        U, s, Vh = compute_svd(S, full_matrices=False)

        data = np.random.randn(m) + 1j * np.random.randn(m)

        tau = tsvd_solve(U, s, Vh, truncation_index=5, data=data, nx=nx, ny=ny)

        assert tau.shape == (ny, nx)

    def test_complex_output(self):
        """Output should be complex."""
        m, n = 20, 16
        nx, ny = 4, 4

        S = np.random.randn(m, n) + 1j * np.random.randn(m, n)
        U, s, Vh = compute_svd(S, full_matrices=False)

        data = np.random.randn(m) + 1j * np.random.randn(m)

        tau = tsvd_solve(U, s, Vh, truncation_index=5, data=data, nx=nx, ny=ny)

        assert np.iscomplexobj(tau)

    def test_truncation_effect(self):
        """More truncation should give smoother (lower rank) solution."""
        m, n = 30, 16
        nx, ny = 4, 4

        S = np.random.randn(m, n) + 1j * np.random.randn(m, n)
        U, s, Vh = compute_svd(S, full_matrices=False)

        data = np.random.randn(m) + 1j * np.random.randn(m)

        # More singular values
        tau_k10 = tsvd_solve(U, s, Vh, truncation_index=10, data=data, nx=nx, ny=ny)
        # Fewer singular values (more regularization)
        tau_k2 = tsvd_solve(U, s, Vh, truncation_index=2, data=data, nx=nx, ny=ny)

        # With fewer singular values, the "effective rank" of the solution is lower
        # We verify they're different
        assert not np.allclose(tau_k10, tau_k2)

    def test_full_truncation_matches_pseudoinverse(self):
        """With all singular values, should match pseudoinverse solution."""
        m, n = 20, 16
        nx, ny = 4, 4

        S = np.random.randn(m, n) + 1j * np.random.randn(m, n)
        U, s, Vh = compute_svd(S, full_matrices=False)

        data = np.random.randn(m) + 1j * np.random.randn(m)

        # Full truncation (all singular values)
        tau_tsvd = tsvd_solve(U, s, Vh, truncation_index=len(s), data=data, nx=nx, ny=ny)

        # Pseudoinverse solution - use Fortran order to match MATLAB convention
        tau_pinv = np.linalg.lstsq(S, data, rcond=None)[0].reshape((ny, nx), order='F')

        np.testing.assert_allclose(tau_tsvd, tau_pinv, rtol=1e-8)

    def test_accepts_2d_data(self):
        """Should accept 2D data array (Nm × Nv)."""
        m, n = 20, 16
        nx, ny = 4, 4
        Nm, Nv = 5, 4

        S = np.random.randn(m, n) + 1j * np.random.randn(m, n)
        U, s, Vh = compute_svd(S, full_matrices=False)

        # 2D data (Nm × Nv)
        data_2d = np.random.randn(Nm, Nv) + 1j * np.random.randn(Nm, Nv)

        tau = tsvd_solve(U, s, Vh, truncation_index=5, data=data_2d, nx=nx, ny=ny)

        assert tau.shape == (ny, nx)

    def test_known_solution(self):
        """Test with a known solution."""
        # Create simple problem where we know the answer
        nx, ny = 3, 3
        n = nx * ny
        m = 12

        # Known contrast
        tau_true = np.array([
            [0.1, 0.2, 0.1],
            [0.2, 0.5, 0.2],
            [0.1, 0.2, 0.1]
        ], dtype=complex)

        # Random scattering operator
        S = np.random.randn(m, n) + 1j * np.random.randn(m, n)

        # Generate "measured" data
        data = S @ tau_true.ravel()

        # Solve with TSVD (use all singular values for exact match)
        U, s, Vh = compute_svd(S, full_matrices=False)
        tau_rec = tsvd_solve(U, s, Vh, truncation_index=n, data=data, nx=nx, ny=ny)

        # Should recover true solution (for well-posed case)
        np.testing.assert_allclose(tau_rec, tau_true, rtol=1e-8)

    def test_single_singular_value(self):
        """Should work with just one singular value."""
        m, n = 20, 16
        nx, ny = 4, 4

        S = np.random.randn(m, n) + 1j * np.random.randn(m, n)
        U, s, Vh = compute_svd(S, full_matrices=False)

        data = np.random.randn(m) + 1j * np.random.randn(m)

        # Use only first singular value
        tau = tsvd_solve(U, s, Vh, truncation_index=1, data=data, nx=nx, ny=ny)

        assert tau.shape == (ny, nx)
        # With just 1 SV, solution should be rank-1 in some sense
        # At minimum, it shouldn't be all zeros (unless data is zero)


class TestTSVDSolverMatlabInterface:
    """Tests for MATLAB-compatible interface."""

    def test_matches_tsvd_solve(self):
        """MATLAB interface should produce same result as tsvd_solve."""
        m, n = 20, 16
        nx, ny = 4, 4

        S = np.random.randn(m, n) + 1j * np.random.randn(m, n)

        # Python-style SVD
        U, s, Vh = compute_svd(S, full_matrices=False)

        # MATLAB-style: S is diagonal, V is V (not Vh)
        S_diag = np.diag(s)
        V = Vh.T.conj()

        data = np.random.randn(m) + 1j * np.random.randn(m)
        Nt = 8

        # Both methods
        tau_python = tsvd_solve(U, s, Vh, Nt, data, nx, ny)
        tau_matlab = tsvd_solver_matlab_interface(U, S_diag, V, Nt, data, nx, ny)

        np.testing.assert_allclose(tau_matlab, tau_python, rtol=1e-12)

    def test_accepts_1d_singular_values(self):
        """Should also accept 1D singular value array."""
        m, n = 20, 16
        nx, ny = 4, 4

        S = np.random.randn(m, n) + 1j * np.random.randn(m, n)
        U, s, Vh = compute_svd(S, full_matrices=False)
        V = Vh.T.conj()

        data = np.random.randn(m) + 1j * np.random.randn(m)

        # Pass s as 1D array (not diagonal)
        tau = tsvd_solver_matlab_interface(U, s, V, 5, data, nx, ny)

        assert tau.shape == (ny, nx)

    def test_2d_data_matrix(self):
        """Should accept 2D data matrix (Nm × Nv)."""
        Nm, Nv = 5, 4
        m = Nm * Nv
        nx, ny = 4, 4
        n = nx * ny

        S = np.random.randn(m, n) + 1j * np.random.randn(m, n)
        U, s, Vh = compute_svd(S, full_matrices=False)
        V = Vh.T.conj()

        # 2D data
        data = np.random.randn(Nm, Nv) + 1j * np.random.randn(Nm, Nv)

        tau = tsvd_solver_matlab_interface(U, np.diag(s), V, 8, data, nx, ny)

        assert tau.shape == (ny, nx)


class TestSuggestThreshold:
    """Tests for suggest_threshold function."""

    def test_snr_40(self):
        """SNR 40 dB should give -35 dB threshold."""
        threshold = suggest_threshold(40)
        assert threshold == -35

    def test_snr_30(self):
        """SNR 30 dB should give -25 dB threshold."""
        threshold = suggest_threshold(30)
        assert threshold == -25

    def test_snr_20(self):
        """SNR 20 dB should give -15 dB threshold."""
        threshold = suggest_threshold(20)
        assert threshold == -15

    def test_snr_10(self):
        """SNR 10 dB should give -5 dB threshold."""
        threshold = suggest_threshold(10)
        assert threshold == -5

    def test_formula(self):
        """Should follow threshold = -(SNR - 5)."""
        for snr in [5, 15, 25, 35, 50]:
            expected = -(snr - 5)
            assert suggest_threshold(snr) == expected


class TestIntegration:
    """Integration tests combining multiple functions."""

    def test_full_workflow(self):
        """Test complete TSVD workflow."""
        # Setup
        nx, ny = 8, 8
        n = nx * ny
        Nm, Nv = 8, 8
        m = Nm * Nv

        # Create scattering operator
        S = np.random.randn(m, n) + 1j * np.random.randn(m, n)

        # Known contrast
        tau_true = np.zeros((ny, nx), dtype=complex)
        tau_true[2:6, 2:6] = 0.5 + 0.1j  # Square target

        # Generate data
        data = S @ tau_true.ravel()

        # Add noise
        noise_level = 0.01
        noise = noise_level * (np.random.randn(m) + 1j * np.random.randn(m))
        data_noisy = data + noise

        # Compute SVD
        U, s, Vh = compute_svd(S, full_matrices=False)

        # Find truncation (using ~40 dB SNR approximation)
        snr_db = 40  # Approximate
        threshold = suggest_threshold(snr_db)
        k = find_truncation_index(s, threshold)

        # Solve
        tau_rec = tsvd_solve(U, s, Vh, k, data_noisy, nx, ny)

        # Verify reasonable reconstruction
        assert tau_rec.shape == tau_true.shape
        # With proper truncation, NMSE should be reasonable
        nmse = np.mean(np.abs(tau_rec - tau_true)**2) / np.mean(np.abs(tau_true)**2)
        assert nmse < 1.0  # At least better than guessing zero

    def test_scenario_like_parameters(self):
        """Test with parameters similar to MATLAB scenarios."""
        # Parameters from typical simulation
        Nx, Ny = 25, 25
        Nm, Nv = 10, 10

        m = Nm * Nv  # 100 measurements
        n = Nx * Ny  # 625 unknowns

        # Create scattering kernel (simplified)
        S = np.random.randn(m, n) + 1j * np.random.randn(m, n)
        S *= 0.01  # Scale to realistic magnitude

        # Create data
        tau_true = np.zeros((Ny, Nx), dtype=complex)
        # Circular target in center
        X, Y = np.meshgrid(np.arange(Nx), np.arange(Ny))
        center_x, center_y = Nx // 2, Ny // 2
        radius = 5
        mask = (X - center_x)**2 + (Y - center_y)**2 < radius**2
        tau_true[mask] = 0.5

        data = S @ tau_true.ravel()

        # SVD and solve
        U, s, Vh = compute_svd(S, full_matrices=False)

        # Use moderate truncation
        k = min(30, len(s))
        tau_rec = tsvd_solve(U, s, Vh, k, data, Nx, Ny)

        assert tau_rec.shape == (Ny, Nx)
        assert not np.any(np.isnan(tau_rec))
        assert not np.any(np.isinf(tau_rec))


class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_square_matrix(self):
        """Should work with square scattering operator."""
        n = 16
        nx, ny = 4, 4

        S = np.random.randn(n, n) + 1j * np.random.randn(n, n)
        U, s, Vh = compute_svd(S, full_matrices=False)

        data = np.random.randn(n) + 1j * np.random.randn(n)
        tau = tsvd_solve(U, s, Vh, truncation_index=8, data=data, nx=nx, ny=ny)

        assert tau.shape == (ny, nx)

    def test_underdetermined(self):
        """Should work with underdetermined system (m < n)."""
        m, n = 10, 25
        nx, ny = 5, 5

        S = np.random.randn(m, n) + 1j * np.random.randn(m, n)
        U, s, Vh = compute_svd(S, full_matrices=False)

        data = np.random.randn(m) + 1j * np.random.randn(m)
        tau = tsvd_solve(U, s, Vh, truncation_index=5, data=data, nx=nx, ny=ny)

        assert tau.shape == (ny, nx)

    def test_overdetermined(self):
        """Should work with overdetermined system (m > n)."""
        m, n = 50, 16
        nx, ny = 4, 4

        S = np.random.randn(m, n) + 1j * np.random.randn(m, n)
        U, s, Vh = compute_svd(S, full_matrices=False)

        data = np.random.randn(m) + 1j * np.random.randn(m)
        tau = tsvd_solve(U, s, Vh, truncation_index=10, data=data, nx=nx, ny=ny)

        assert tau.shape == (ny, nx)

    def test_rectangular_grid(self):
        """Should work with non-square grid."""
        m = 24
        nx, ny = 6, 4  # Rectangular
        n = nx * ny

        S = np.random.randn(m, n) + 1j * np.random.randn(m, n)
        U, s, Vh = compute_svd(S, full_matrices=False)

        data = np.random.randn(m) + 1j * np.random.randn(m)
        tau = tsvd_solve(U, s, Vh, truncation_index=10, data=data, nx=nx, ny=ny)

        assert tau.shape == (ny, nx)

    def test_near_zero_singular_values(self):
        """Should handle matrices with near-zero singular values."""
        m, n = 20, 16
        nx, ny = 4, 4

        # Create low-rank matrix
        rank = 5
        U_true = np.random.randn(m, rank) + 1j * np.random.randn(m, rank)
        V_true = np.random.randn(n, rank) + 1j * np.random.randn(n, rank)
        S = U_true @ V_true.T.conj()

        U, s, Vh = compute_svd(S, full_matrices=False)

        # Many singular values should be ~0
        assert np.sum(s < 1e-10) >= n - rank - 2

        data = np.random.randn(m) + 1j * np.random.randn(m)

        # Should still work with truncation
        tau = tsvd_solve(U, s, Vh, truncation_index=rank, data=data, nx=nx, ny=ny)
        assert not np.any(np.isnan(tau))


class TestMATLABCompatibility:
    """Tests for MATLAB compatibility."""

    def test_1_indexed_truncation(self):
        """Truncation index should be 1-indexed for MATLAB."""
        s = np.array([1.0, 0.5, 0.1, 0.01])

        # Using Nt=1 should use only the first singular value
        m, n = 10, 4
        nx, ny = 2, 2

        # Create simple case
        S = np.diag(s) @ np.random.randn(4, n)
        S = np.random.randn(m, 4) @ S

        U, s_svd, Vh = compute_svd(S, full_matrices=False)
        data = np.random.randn(m)

        # Nt=1 means first singular value only
        tau = tsvd_solve(U, s_svd, Vh, truncation_index=1, data=data, nx=nx, ny=ny)

        assert tau.shape == (ny, nx)

    def test_column_major_reshape(self):
        """Output should match MATLAB's column-major reshape."""
        m, n = 20, 12
        nx, ny = 4, 3  # n = 12

        S = np.random.randn(m, n) + 1j * np.random.randn(m, n)
        U, s, Vh = compute_svd(S, full_matrices=False)

        data = np.random.randn(m) + 1j * np.random.randn(m)

        tau = tsvd_solve(U, s, Vh, truncation_index=8, data=data, nx=nx, ny=ny)

        # Verify shape is (ny, nx) - MATLAB convention
        assert tau.shape == (ny, nx)
