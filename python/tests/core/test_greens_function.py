"""
Unit tests for inverse_scattering.core.greens_function module.

Tests verify that the 2D Green's function implementation matches
the theoretical formula and produces physically correct results.

Theory:
    G(r, r') = (i/4) * H₀⁽¹⁾(k|r - r'|)

    where H₀⁽¹⁾ is the Hankel function of the first kind, order 0.
"""

import numpy as np
import pytest
from scipy import special

from inverse_scattering.core.greens_function import (
    hankel1_0,
    greens_function_2d,
    greens_function_matrix,
    _compute_self_term,
    greens_function_grad,
    SINGULARITY_THRESHOLD,
)
from inverse_scattering.core.utils import compute_wavenumber


class TestHankel10:
    """Test Hankel function of first kind, order 0."""

    def test_matches_scipy(self):
        """Verify our wrapper matches scipy.special.hankel1."""
        z_values = [0.1, 1.0, 5.0, 10.0, 100.0]
        for z in z_values:
            result = hankel1_0(z)
            expected = special.hankel1(0, z)
            np.testing.assert_allclose(result, expected, rtol=1e-14)

    def test_array_input(self):
        """Test with array input."""
        z = np.array([0.1, 1.0, 5.0, 10.0])
        result = hankel1_0(z)
        expected = special.hankel1(0, z)
        np.testing.assert_allclose(result, expected, rtol=1e-14)

    def test_complex_output(self):
        """Hankel function should return complex values."""
        result = hankel1_0(1.0)
        assert isinstance(result, (complex, np.complexfloating))

    def test_bessel_decomposition(self):
        """Verify H₀⁽¹⁾(z) = J₀(z) + i*Y₀(z)."""
        z = 2.5
        H0 = hankel1_0(z)
        J0 = special.jv(0, z)
        Y0 = special.yv(0, z)
        expected = J0 + 1j * Y0
        np.testing.assert_allclose(H0, expected, rtol=1e-14)

    def test_small_argument_approximation(self):
        """
        For small z: H₀⁽¹⁾(z) ≈ 1 + (2i/π) * (ln(z/2) + γ)
        where γ ≈ 0.5772 is Euler's constant.
        """
        gamma_euler = 0.5772156649015329
        z = 0.01  # Small argument

        H0_exact = hankel1_0(z)
        H0_approx = 1 + (2j / np.pi) * (np.log(z / 2) + gamma_euler)

        # Approximation should be close for small z
        np.testing.assert_allclose(H0_exact, H0_approx, rtol=0.01)

    def test_large_argument_asymptotic(self):
        """
        For large z: H₀⁽¹⁾(z) ≈ √(2/(πz)) * exp(i(z - π/4))
        """
        z = 100.0  # Large argument

        H0_exact = hankel1_0(z)
        H0_asymp = np.sqrt(2 / (np.pi * z)) * np.exp(1j * (z - np.pi / 4))

        # Asymptotic approximation should be close for large z
        np.testing.assert_allclose(H0_exact, H0_asymp, rtol=0.01)


class TestGreensFunction2D:
    """Test 2D Green's function for single point pairs."""

    def test_formula_verification(self):
        """Verify G(r, r') = (i/4) * H₀⁽¹⁾(k|r - r'|)."""
        k = 20.0  # wavenumber
        r_obs = np.array([1.0, 0.0])
        r_src = np.array([0.0, 0.0])

        G = greens_function_2d(k, r_obs, r_src)

        # Manual calculation
        distance = np.linalg.norm(r_obs - r_src)
        G_expected = (1j / 4) * special.hankel1(0, k * distance)

        np.testing.assert_allclose(G, G_expected, rtol=1e-14)

    def test_symmetry(self):
        """Green's function should be symmetric: G(r,r') = G(r',r)."""
        k = 15.0
        r1 = np.array([0.5, 0.3])
        r2 = np.array([-0.2, 0.7])

        G12 = greens_function_2d(k, r1, r2)
        G21 = greens_function_2d(k, r2, r1)

        np.testing.assert_allclose(G12, G21, rtol=1e-14)

    def test_distance_dependence(self):
        """G should decrease with distance (in magnitude)."""
        k = 10.0
        r_src = np.array([0.0, 0.0])

        r_near = np.array([0.1, 0.0])
        r_far = np.array([1.0, 0.0])

        G_near = greens_function_2d(k, r_near, r_src)
        G_far = greens_function_2d(k, r_far, r_src)

        # Near field should have larger magnitude
        assert np.abs(G_near) > np.abs(G_far)

    def test_self_term_returns_zero(self):
        """Self-term (r=r') should return zero (handled separately)."""
        k = 10.0
        r = np.array([0.5, 0.5])

        G_self = greens_function_2d(k, r, r)

        assert G_self == 0.0 + 0.0j

    def test_singularity_threshold(self):
        """Points closer than threshold should be treated as singular."""
        k = 10.0
        r1 = np.array([0.0, 0.0])
        r2 = np.array([SINGULARITY_THRESHOLD / 2, 0.0])

        G = greens_function_2d(k, r1, r2)
        assert G == 0.0 + 0.0j

    def test_complex_wavenumber(self):
        """Test with lossy medium (complex wavenumber)."""
        k = 10.0 + 0.5j  # Complex wavenumber (lossy)
        r_obs = np.array([0.5, 0.0])
        r_src = np.array([0.0, 0.0])

        G = greens_function_2d(k, r_obs, r_src)

        # Should return valid complex number
        assert np.isfinite(G)
        assert isinstance(G, (complex, np.complexfloating))

    def test_different_distances(self):
        """Test Green's function at various distances."""
        k = 20.0
        r_src = np.array([0.0, 0.0])
        distances = [0.1, 0.5, 1.0, 2.0, 5.0]

        for d in distances:
            r_obs = np.array([d, 0.0])
            G = greens_function_2d(k, r_obs, r_src)

            # Verify formula
            expected = (1j / 4) * special.hankel1(0, k * d)
            np.testing.assert_allclose(G, expected, rtol=1e-14)

    def test_diagonal_distance(self):
        """Test with non-axis-aligned points."""
        k = 15.0
        r_obs = np.array([1.0, 1.0])
        r_src = np.array([0.0, 0.0])

        G = greens_function_2d(k, r_obs, r_src)

        distance = np.sqrt(2)  # Diagonal
        expected = (1j / 4) * special.hankel1(0, k * distance)
        np.testing.assert_allclose(G, expected, rtol=1e-14)


class TestGreensMatrix:
    """Test Green's function matrix computation."""

    def test_matrix_shape(self):
        """Verify output matrix dimensions."""
        k = 10.0
        n_obs, n_src = 5, 8

        x_obs = np.linspace(-1, 1, n_obs)
        y_obs = np.linspace(-1, 1, n_obs)
        x_src = np.linspace(-0.5, 0.5, n_src)
        y_src = np.linspace(-0.5, 0.5, n_src)

        G = greens_function_matrix(k, x_obs, y_obs, x_src, y_src)

        assert G.shape == (n_obs, n_src)

    def test_matrix_shape_2d_grids(self):
        """Test with 2D meshgrid inputs."""
        k = 10.0
        n_obs_x, n_obs_y = 3, 4
        n_src_x, n_src_y = 5, 6

        x_obs, y_obs = np.meshgrid(
            np.linspace(-1, 1, n_obs_x),
            np.linspace(-1, 1, n_obs_y)
        )
        x_src, y_src = np.meshgrid(
            np.linspace(-0.5, 0.5, n_src_x),
            np.linspace(-0.5, 0.5, n_src_y)
        )

        G = greens_function_matrix(k, x_obs, y_obs, x_src, y_src)

        n_obs = n_obs_x * n_obs_y
        n_src = n_src_x * n_src_y
        assert G.shape == (n_obs, n_src)

    def test_matrix_symmetry_same_grid(self):
        """Matrix should be symmetric when obs == src grid."""
        k = 10.0
        n = 6

        x = np.linspace(-1, 1, n)
        y = np.linspace(-1, 1, n)

        G = greens_function_matrix(k, x, y, x, y)

        # Should be symmetric (G_ij = G_ji)
        np.testing.assert_allclose(G, G.T, rtol=1e-14)

    def test_matrix_elements_match_pointwise(self):
        """Matrix elements should match point-by-point computation."""
        k = 15.0
        x_obs = np.array([0.0, 0.5, 1.0])
        y_obs = np.array([0.0, 0.3, 0.6])
        x_src = np.array([-0.2, 0.2])
        y_src = np.array([0.1, -0.1])

        G_matrix = greens_function_matrix(k, x_obs, y_obs, x_src, y_src)

        # Verify each element
        for i in range(len(x_obs)):
            for j in range(len(x_src)):
                r_obs = np.array([x_obs[i], y_obs[i]])
                r_src = np.array([x_src[j], y_src[j]])
                G_expected = greens_function_2d(k, r_obs, r_src)
                np.testing.assert_allclose(G_matrix[i, j], G_expected, rtol=1e-14)

    def test_self_terms_without_cell_size(self):
        """Self-terms should be zero when cell sizes not provided."""
        k = 10.0
        x = np.array([0.0, 0.5, 1.0])
        y = np.array([0.0, 0.5, 1.0])

        G = greens_function_matrix(k, x, y, x, y)

        # Diagonal elements (self-terms) should be zero
        for i in range(len(x)):
            assert G[i, i] == 0.0 + 0.0j

    def test_self_terms_with_cell_size(self):
        """Self-terms should be computed when cell sizes provided."""
        k = 10.0
        x = np.array([0.0, 0.5, 1.0])
        y = np.array([0.0, 0.5, 1.0])
        dx, dy = 0.1, 0.1

        G = greens_function_matrix(k, x, y, x, y, dx=dx, dy=dy)

        # Diagonal elements should now have non-zero self-term
        expected_self = _compute_self_term(k, dx, dy)
        for i in range(len(x)):
            np.testing.assert_allclose(G[i, i], expected_self, rtol=1e-14)

    def test_complex_dtype(self):
        """Output matrix should be complex."""
        k = 10.0
        x = np.linspace(-1, 1, 5)
        y = np.linspace(-1, 1, 5)

        G = greens_function_matrix(k, x, y, x, y)

        assert G.dtype == complex or np.issubdtype(G.dtype, np.complexfloating)

    def test_no_nan_or_inf(self):
        """Matrix should not contain NaN or Inf values."""
        k = 10.0
        x = np.linspace(-1, 1, 10)
        y = np.linspace(-1, 1, 10)

        G = greens_function_matrix(k, x, y, x, y, dx=0.2, dy=0.2)

        assert np.all(np.isfinite(G))


class TestSelfTerm:
    """Test self-term (singularity) computation."""

    def test_returns_complex(self):
        """Self-term should be complex."""
        k = 20.0
        dx, dy = 0.1, 0.1

        self_term = _compute_self_term(k, dx, dy)

        assert isinstance(self_term, (complex, np.complexfloating))

    def test_nonzero_value(self):
        """Self-term should be non-zero for reasonable parameters."""
        k = 20.0
        dx, dy = 0.05, 0.05

        self_term = _compute_self_term(k, dx, dy)

        assert np.abs(self_term) > 0

    def test_scaling_with_cell_size(self):
        """Self-term magnitude should scale with cell area."""
        k = 10.0

        self_small = _compute_self_term(k, 0.05, 0.05)
        self_large = _compute_self_term(k, 0.1, 0.1)

        # Larger cell should give larger magnitude
        assert np.abs(self_large) > np.abs(self_small)

    def test_square_vs_rectangular_cell(self):
        """Test rectangular vs square cells of same area."""
        k = 10.0
        area = 0.01  # Same area

        # Square cell
        side = np.sqrt(area)
        self_square = _compute_self_term(k, side, side)

        # Rectangular cell with same area
        dx, dy = 0.05, 0.2
        self_rect = _compute_self_term(k, dx, dy)

        # Both should be close since area is similar
        np.testing.assert_allclose(
            np.abs(self_square),
            np.abs(self_rect),
            rtol=0.1  # Allow some difference due to geometry
        )

    def test_zero_wavenumber(self):
        """Self-term should be zero for k=0 (static limit)."""
        k = 0.0
        dx, dy = 0.1, 0.1

        self_term = _compute_self_term(k, dx, dy)

        assert self_term == 0.0 + 0.0j

    def test_very_small_wavenumber(self):
        """Self-term should be near zero for very small k."""
        k = 1e-11
        dx, dy = 0.1, 0.1

        self_term = _compute_self_term(k, dx, dy)

        assert self_term == 0.0 + 0.0j

    def test_physical_units(self):
        """Test with typical physical parameters."""
        # Frequency = 1 GHz, cell size in DoI
        freq = 1e9
        k = compute_wavenumber(freq, epsilon_r=1.0, sigma=0.0)
        dx, dy = 0.001, 0.001  # 1 mm cells

        self_term = _compute_self_term(k, dx, dy)

        # Should be finite and reasonably small
        assert np.isfinite(self_term)
        assert np.abs(self_term) < 1.0


class TestGreensGradient:
    """Test Green's function gradient computation."""

    def test_gradient_formula(self):
        """
        Verify ∇G = -(ik/4) * H₁⁽¹⁾(kr) * r̂.
        """
        k = 15.0
        x_obs, y_obs = 1.0, 0.0
        x_src, y_src = 0.0, 0.0

        dG_dx, dG_dy = greens_function_grad(k, x_obs, y_obs, x_src, y_src)

        # Manual calculation
        dx = x_obs - x_src
        dy = y_obs - y_src
        r = np.sqrt(dx**2 + dy**2)
        H1 = special.hankel1(1, k * r)
        factor = -(1j * k / 4) * H1 / r

        expected_dG_dx = factor * dx
        expected_dG_dy = factor * dy

        np.testing.assert_allclose(dG_dx, expected_dG_dx, rtol=1e-14)
        np.testing.assert_allclose(dG_dy, expected_dG_dy, rtol=1e-14)

    def test_gradient_at_diagonal(self):
        """Test gradient for diagonal direction."""
        k = 10.0
        d = 1.0  # Distance along diagonal
        x_obs, y_obs = d / np.sqrt(2), d / np.sqrt(2)
        x_src, y_src = 0.0, 0.0

        dG_dx, dG_dy = greens_function_grad(k, x_obs, y_obs, x_src, y_src)

        # For symmetric diagonal, |dG/dx| should equal |dG/dy|
        np.testing.assert_allclose(np.abs(dG_dx), np.abs(dG_dy), rtol=1e-14)

    def test_gradient_self_term(self):
        """Gradient at self-point should return zeros."""
        k = 10.0
        x, y = 0.5, 0.5

        dG_dx, dG_dy = greens_function_grad(k, x, y, x, y)

        assert dG_dx == 0.0 + 0.0j
        assert dG_dy == 0.0 + 0.0j

    def test_gradient_direction(self):
        """Gradient should point in radial direction."""
        k = 10.0
        x_src, y_src = 0.0, 0.0

        # Test point on x-axis
        dG_dx, dG_dy = greens_function_grad(k, 1.0, 0.0, x_src, y_src)
        # dG_dy should be zero on x-axis
        np.testing.assert_allclose(dG_dy, 0.0, atol=1e-14)

        # Test point on y-axis
        dG_dx, dG_dy = greens_function_grad(k, 0.0, 1.0, x_src, y_src)
        # dG_dx should be zero on y-axis
        np.testing.assert_allclose(dG_dx, 0.0, atol=1e-14)

    def test_gradient_magnitude_decay(self):
        """Gradient magnitude should generally decay with distance."""
        k = 10.0
        x_src, y_src = 0.0, 0.0

        dG_dx_near, _ = greens_function_grad(k, 0.2, 0.0, x_src, y_src)
        dG_dx_far, _ = greens_function_grad(k, 2.0, 0.0, x_src, y_src)

        # Near field gradient should be larger
        assert np.abs(dG_dx_near) > np.abs(dG_dx_far)


class TestMATLABCompatibility:
    """Tests for MATLAB compatibility."""

    def test_matlab_formula_match(self):
        """
        Verify matches MATLAB formula from theory document:
        G = (i/4) * H_0^(1)(k_b * ||r - r'||)
        """
        # Parameters similar to MATLAB DATA_scenario.mat
        freq = 1e9
        k = compute_wavenumber(freq, epsilon_r=1.0, sigma=0.0)

        r_obs = np.array([0.05, 0.0])  # 5 cm from origin
        r_src = np.array([0.0, 0.0])

        G = greens_function_2d(k, r_obs, r_src)

        # MATLAB formula
        distance = np.linalg.norm(r_obs - r_src)
        G_matlab = (1j / 4) * special.hankel1(0, k * distance)

        np.testing.assert_allclose(G, G_matlab, rtol=1e-14)

    def test_typical_scattering_scenario(self):
        """Test Green's function in typical scattering setup."""
        # Setup similar to DATA_scenario.mat
        freq = 1e9
        lx = 0.1  # 10 cm DoI
        Rm = 0.1  # Measurement radius

        k = compute_wavenumber(freq, epsilon_r=1.0, sigma=0.0)

        # Measurement point at radius Rm
        theta = np.pi / 4
        r_meas = np.array([Rm * np.cos(theta), Rm * np.sin(theta)])

        # Source point inside DoI
        r_src = np.array([0.02, 0.01])

        G = greens_function_2d(k, r_meas, r_src)

        # Should be valid complex number
        assert np.isfinite(G)
        assert np.abs(G) > 0


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_very_small_wavenumber(self):
        """Test with very small (but non-zero) wavenumber."""
        k = 1e-5
        r_obs = np.array([1.0, 0.0])
        r_src = np.array([0.0, 0.0])

        G = greens_function_2d(k, r_obs, r_src)

        assert np.isfinite(G)

    def test_large_wavenumber(self):
        """Test with large wavenumber (high frequency)."""
        k = 1000.0
        r_obs = np.array([0.1, 0.0])
        r_src = np.array([0.0, 0.0])

        G = greens_function_2d(k, r_obs, r_src)

        assert np.isfinite(G)

    def test_very_close_points(self):
        """Test with points very close but not identical."""
        k = 10.0
        r_obs = np.array([1e-9, 0.0])  # Just above threshold
        r_src = np.array([0.0, 0.0])

        G = greens_function_2d(k, r_obs, r_src)

        # Should be handled without overflow
        assert np.isfinite(G) or G == 0.0 + 0.0j

    def test_negative_coordinates(self):
        """Test with negative coordinate values."""
        k = 15.0
        r_obs = np.array([-0.5, -0.3])
        r_src = np.array([0.2, -0.1])

        G = greens_function_2d(k, r_obs, r_src)

        # Verify against direct formula
        distance = np.linalg.norm(r_obs - r_src)
        expected = (1j / 4) * special.hankel1(0, k * distance)
        np.testing.assert_allclose(G, expected, rtol=1e-14)

    def test_empty_matrix(self):
        """Test matrix computation with empty arrays."""
        k = 10.0
        x_obs = np.array([])
        y_obs = np.array([])
        x_src = np.array([0.0])
        y_src = np.array([0.0])

        G = greens_function_matrix(k, x_obs, y_obs, x_src, y_src)

        assert G.shape == (0, 1)

    def test_single_point(self):
        """Test matrix with single observation and source point."""
        k = 10.0
        x_obs = np.array([1.0])
        y_obs = np.array([0.0])
        x_src = np.array([0.0])
        y_src = np.array([0.0])

        G = greens_function_matrix(k, x_obs, y_obs, x_src, y_src)

        assert G.shape == (1, 1)
        expected = greens_function_2d(k, np.array([1.0, 0.0]), np.array([0.0, 0.0]))
        np.testing.assert_allclose(G[0, 0], expected, rtol=1e-14)
