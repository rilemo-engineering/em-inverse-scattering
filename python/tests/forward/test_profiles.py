"""
Unit tests for inverse_scattering.forward.profiles module.

Tests verify that profile creation matches MATLAB behavior and
produces physically correct contrast functions.

Theory:
    Contrast function: τ(r) = ε_r(r) - ε_b - jσ(r)/(ωε₀)
    For lossless materials: τ = ε_r - ε_b
"""

import numpy as np
import pytest

from inverse_scattering.forward.profiles import (
    create_circular_profile,
    create_square_profile,
    create_multi_object_profile,
    create_fresnel_single_target,
    create_fresnel_two_targets,
)
from inverse_scattering.core.utils import create_grid
from inverse_scattering.core.constants import EPSILON_0


class TestCircularProfile:
    """Test circular cylinder profile creation."""

    def test_basic_shape(self):
        """Verify circular profile has correct shape."""
        lx, ly, nx, ny = 1.0, 1.0, 32, 32
        X, Y, xvec, yvec, dx, dy = create_grid(lx, ly, nx, ny)

        PROF = create_circular_profile(
            X, Y, center=(0.0, 0.0), radius=0.2, epsilon_r=2.0
        )

        assert PROF.shape == (ny, nx)

    def test_contrast_value_lossless(self):
        """Test contrast value τ = ε_r - ε_b for lossless material."""
        lx, ly, nx, ny = 1.0, 1.0, 64, 64
        X, Y, xvec, yvec, dx, dy = create_grid(lx, ly, nx, ny)

        epsilon_r = 3.0
        epsilon_b = 1.0
        expected_tau = epsilon_r - epsilon_b  # τ = 2.0

        PROF = create_circular_profile(
            X, Y, center=(0.0, 0.0), radius=0.3,
            epsilon_r=epsilon_r, epsilon_b=epsilon_b
        )

        # Check center point has correct contrast
        center_value = PROF[ny // 2, nx // 2]
        np.testing.assert_allclose(center_value.real, expected_tau, rtol=1e-10)
        np.testing.assert_allclose(center_value.imag, 0.0, atol=1e-10)

    def test_background_is_zero(self):
        """Background (outside object) should have τ = 0."""
        lx, ly, nx, ny = 1.0, 1.0, 32, 32
        X, Y, xvec, yvec, dx, dy = create_grid(lx, ly, nx, ny)

        PROF = create_circular_profile(
            X, Y, center=(0.0, 0.0), radius=0.1, epsilon_r=2.0
        )

        # Corner points should be background
        assert PROF[0, 0] == 0.0
        assert PROF[0, -1] == 0.0
        assert PROF[-1, 0] == 0.0
        assert PROF[-1, -1] == 0.0

    def test_circle_boundary(self):
        """Verify points are correctly classified as inside/outside."""
        lx, ly, nx, ny = 1.0, 1.0, 100, 100
        X, Y, xvec, yvec, dx, dy = create_grid(lx, ly, nx, ny)

        radius = 0.3
        PROF = create_circular_profile(
            X, Y, center=(0.0, 0.0), radius=radius, epsilon_r=2.0
        )

        # Count non-zero cells
        nonzero_mask = np.abs(PROF) > 0
        inside_count = np.sum(nonzero_mask)

        # Approximate expected count from circle area
        cell_area = dx * dy
        circle_area = np.pi * radius**2
        expected_count = circle_area / cell_area

        # Should be within reasonable tolerance (discretization error)
        np.testing.assert_allclose(inside_count, expected_count, rtol=0.1)

    def test_off_center_circle(self):
        """Test circle centered away from origin."""
        lx, ly, nx, ny = 1.0, 1.0, 50, 50
        X, Y, xvec, yvec, dx, dy = create_grid(lx, ly, nx, ny)

        x0, y0 = 0.2, -0.1
        radius = 0.1
        tau = 1.5  # ε_r - ε_b

        PROF = create_circular_profile(
            X, Y, center=(x0, y0), radius=radius,
            epsilon_r=tau + 1.0, epsilon_b=1.0
        )

        # Find nearest grid point to center
        i_center = np.argmin(np.abs(yvec - y0))
        j_center = np.argmin(np.abs(xvec - x0))

        # Center should have non-zero contrast
        assert np.abs(PROF[i_center, j_center]) > 0

    def test_lossy_material(self):
        """Test contrast for lossy material (σ > 0)."""
        lx, ly, nx, ny = 1.0, 1.0, 32, 32
        X, Y, xvec, yvec, dx, dy = create_grid(lx, ly, nx, ny)

        epsilon_r = 2.0
        epsilon_b = 1.0
        sigma = 0.1
        freq = 1e9

        PROF = create_circular_profile(
            X, Y, center=(0.0, 0.0), radius=0.2,
            epsilon_r=epsilon_r, epsilon_b=epsilon_b,
            sigma=sigma, freq=freq
        )

        center_value = PROF[ny // 2, nx // 2]

        # Real part: ε_r - ε_b = 1.0
        np.testing.assert_allclose(center_value.real, epsilon_r - epsilon_b, rtol=1e-10)

        # Imaginary part: -σ/(ω*ε₀)
        omega = 2 * np.pi * freq
        expected_imag = -sigma / (omega * EPSILON_0)
        np.testing.assert_allclose(center_value.imag, expected_imag, rtol=1e-10)

    def test_zero_radius(self):
        """Zero radius should give all-zero profile."""
        lx, ly, nx, ny = 1.0, 1.0, 16, 16
        X, Y, xvec, yvec, dx, dy = create_grid(lx, ly, nx, ny)

        PROF = create_circular_profile(
            X, Y, center=(0.0, 0.0), radius=0.0, epsilon_r=2.0
        )

        assert np.all(PROF == 0)

    def test_dtype_is_complex(self):
        """Profile should be complex dtype."""
        lx, ly, nx, ny = 1.0, 1.0, 16, 16
        X, Y, xvec, yvec, dx, dy = create_grid(lx, ly, nx, ny)

        PROF = create_circular_profile(
            X, Y, center=(0.0, 0.0), radius=0.2, epsilon_r=2.0
        )

        assert np.issubdtype(PROF.dtype, np.complexfloating)


class TestSquareProfile:
    """Test square object profile creation."""

    def test_basic_shape(self):
        """Verify square profile has correct shape."""
        lx, ly, nx, ny = 1.0, 1.0, 32, 32
        X, Y, xvec, yvec, dx, dy = create_grid(lx, ly, nx, ny)

        PROF = create_square_profile(
            X, Y, center=(0.0, 0.0), side=0.4, epsilon_r=2.0
        )

        assert PROF.shape == (ny, nx)

    def test_contrast_value(self):
        """Test contrast value inside square."""
        lx, ly, nx, ny = 1.0, 1.0, 64, 64
        X, Y, xvec, yvec, dx, dy = create_grid(lx, ly, nx, ny)

        epsilon_r = 4.0
        epsilon_b = 1.0
        expected_tau = epsilon_r - epsilon_b

        PROF = create_square_profile(
            X, Y, center=(0.0, 0.0), side=0.3,
            epsilon_r=epsilon_r, epsilon_b=epsilon_b
        )

        center_value = PROF[ny // 2, nx // 2]
        np.testing.assert_allclose(center_value.real, expected_tau, rtol=1e-10)

    def test_square_boundary(self):
        """Verify square boundary is correct."""
        lx, ly, nx, ny = 1.0, 1.0, 100, 100
        X, Y, xvec, yvec, dx, dy = create_grid(lx, ly, nx, ny)

        side = 0.4
        PROF = create_square_profile(
            X, Y, center=(0.0, 0.0), side=side, epsilon_r=2.0
        )

        nonzero_mask = np.abs(PROF) > 0
        inside_count = np.sum(nonzero_mask)

        # Approximate expected count from square area
        cell_area = dx * dy
        square_area = side**2
        expected_count = square_area / cell_area

        np.testing.assert_allclose(inside_count, expected_count, rtol=0.1)

    def test_off_center_square(self):
        """Test square centered away from origin."""
        lx, ly, nx, ny = 1.0, 1.0, 50, 50
        X, Y, xvec, yvec, dx, dy = create_grid(lx, ly, nx, ny)

        x0, y0 = -0.2, 0.15
        side = 0.2

        PROF = create_square_profile(
            X, Y, center=(x0, y0), side=side, epsilon_r=2.0
        )

        # Find grid point at center
        i_center = np.argmin(np.abs(yvec - y0))
        j_center = np.argmin(np.abs(xvec - x0))

        assert np.abs(PROF[i_center, j_center]) > 0

    def test_square_lossy(self):
        """Test lossy square object."""
        lx, ly, nx, ny = 1.0, 1.0, 32, 32
        X, Y, xvec, yvec, dx, dy = create_grid(lx, ly, nx, ny)

        sigma = 0.05
        freq = 2e9

        PROF = create_square_profile(
            X, Y, center=(0.0, 0.0), side=0.2,
            epsilon_r=2.5, sigma=sigma, freq=freq
        )

        center_value = PROF[ny // 2, nx // 2]

        # Should have imaginary part
        assert center_value.imag != 0


class TestMultiObjectProfile:
    """Test multiple object profile creation."""

    def test_two_circles(self):
        """Test profile with two circular objects."""
        lx, ly, nx, ny = 1.0, 1.0, 64, 64
        X, Y, xvec, yvec, dx, dy = create_grid(lx, ly, nx, ny)

        objects = [
            {'shape': 'circle', 'center': (-0.2, 0.0), 'size': 0.1, 'epsilon_r': 2.0},
            {'shape': 'circle', 'center': (0.2, 0.0), 'size': 0.1, 'epsilon_r': 3.0},
        ]

        PROF = create_multi_object_profile(X, Y, objects)

        # Check both objects exist
        left_idx = np.argmin(np.abs(xvec + 0.2))
        right_idx = np.argmin(np.abs(xvec - 0.2))
        mid_y = ny // 2

        left_value = PROF[mid_y, left_idx]
        right_value = PROF[mid_y, right_idx]

        np.testing.assert_allclose(left_value.real, 1.0, rtol=1e-10)  # τ = 2 - 1
        np.testing.assert_allclose(right_value.real, 2.0, rtol=1e-10)  # τ = 3 - 1

    def test_mixed_shapes(self):
        """Test profile with circle and square."""
        lx, ly, nx, ny = 1.0, 1.0, 64, 64
        X, Y, xvec, yvec, dx, dy = create_grid(lx, ly, nx, ny)

        objects = [
            {'shape': 'circle', 'center': (-0.2, 0.0), 'size': 0.1, 'epsilon_r': 2.0},
            {'shape': 'square', 'center': (0.2, 0.0), 'size': 0.15, 'epsilon_r': 3.0},
        ]

        PROF = create_multi_object_profile(X, Y, objects)

        # Verify both regions have non-zero contrast
        left_idx = np.argmin(np.abs(xvec + 0.2))
        right_idx = np.argmin(np.abs(xvec - 0.2))
        mid_y = ny // 2

        assert np.abs(PROF[mid_y, left_idx]) > 0
        assert np.abs(PROF[mid_y, right_idx]) > 0

    def test_empty_object_list(self):
        """Empty object list should give all-zero profile."""
        lx, ly, nx, ny = 1.0, 1.0, 16, 16
        X, Y, xvec, yvec, dx, dy = create_grid(lx, ly, nx, ny)

        PROF = create_multi_object_profile(X, Y, [])

        assert np.all(PROF == 0)

    def test_overlapping_objects(self):
        """Overlapping objects: last one wins."""
        lx, ly, nx, ny = 1.0, 1.0, 64, 64
        X, Y, xvec, yvec, dx, dy = create_grid(lx, ly, nx, ny)

        # Second object overlaps center of first
        objects = [
            {'shape': 'circle', 'center': (0.0, 0.0), 'size': 0.2, 'epsilon_r': 2.0},
            {'shape': 'circle', 'center': (0.05, 0.0), 'size': 0.1, 'epsilon_r': 4.0},
        ]

        PROF = create_multi_object_profile(X, Y, objects)

        # Center should have second object's contrast
        center_value = PROF[ny // 2, nx // 2]
        np.testing.assert_allclose(center_value.real, 3.0, rtol=1e-10)  # τ = 4 - 1

    def test_unknown_shape_raises(self):
        """Unknown shape should raise ValueError."""
        lx, ly, nx, ny = 1.0, 1.0, 16, 16
        X, Y, xvec, yvec, dx, dy = create_grid(lx, ly, nx, ny)

        objects = [
            {'shape': 'triangle', 'center': (0.0, 0.0), 'size': 0.1, 'epsilon_r': 2.0},
        ]

        with pytest.raises(ValueError, match="Unknown shape"):
            create_multi_object_profile(X, Y, objects)


class TestFresnelSingleTarget:
    """Test Fresnel Institute single target creation."""

    def test_returns_profile_and_params(self):
        """Should return tuple of profile and parameters."""
        lx, ly, nx, ny = 0.15, 0.15, 64, 64
        X, Y, xvec, yvec, dx, dy = create_grid(lx, ly, nx, ny)

        result = create_fresnel_single_target(X, Y)

        assert isinstance(result, tuple)
        assert len(result) == 2
        PROF, params = result
        assert isinstance(PROF, np.ndarray)
        assert isinstance(params, dict)

    def test_profile_shape(self):
        """Profile should match grid dimensions."""
        lx, ly, nx, ny = 0.15, 0.15, 64, 64
        X, Y, xvec, yvec, dx, dy = create_grid(lx, ly, nx, ny)

        PROF, params = create_fresnel_single_target(X, Y)

        assert PROF.shape == (ny, nx)

    def test_fresnel_specifications(self):
        """Verify Fresnel target specifications."""
        lx, ly, nx, ny = 0.15, 0.15, 64, 64
        X, Y, xvec, yvec, dx, dy = create_grid(lx, ly, nx, ny)

        PROF, params = create_fresnel_single_target(X, Y)

        # Verify parameters
        assert params['r0'] == 0.015      # 15 mm radius
        assert params['x0'] == 0.025      # 25 mm offset
        assert params['y0'] == 0.0

    def test_contrast_value(self):
        """Contrast should be τ = 2 (ε_r=3, ε_b=1)."""
        lx, ly, nx, ny = 0.15, 0.15, 128, 128
        X, Y, xvec, yvec, dx, dy = create_grid(lx, ly, nx, ny)

        PROF, params = create_fresnel_single_target(X, Y)

        # Find point inside cylinder
        x0, y0 = params['x0'], params['y0']
        i = np.argmin(np.abs(yvec - y0))
        j = np.argmin(np.abs(xvec - x0))

        np.testing.assert_allclose(PROF[i, j].real, 2.0, rtol=1e-10)

    def test_matlab_equivalent(self):
        """
        Verify MATLAB formula:
            r0 = 0.015; x0 = 0.025; y0 = 0.0;
            rxy = sqrt((X-x0).^2 + (Y-y0).^2);
            PROF(rxy <= r0) = 2;
        """
        lx, ly, nx, ny = 0.15, 0.15, 64, 64
        X, Y, xvec, yvec, dx, dy = create_grid(lx, ly, nx, ny)

        PROF, params = create_fresnel_single_target(X, Y)

        # MATLAB formula
        r0 = 0.015
        x0 = 0.025
        y0 = 0.0
        rxy = np.sqrt((X - x0)**2 + (Y - y0)**2)
        PROF_matlab = np.zeros_like(X, dtype=complex)
        PROF_matlab[rxy <= r0] = 2.0

        np.testing.assert_allclose(PROF, PROF_matlab, rtol=1e-10)


class TestFresnelTwoTargets:
    """Test Fresnel Institute two targets creation."""

    def test_returns_profile_and_params(self):
        """Should return tuple of profile and parameters."""
        lx, ly, nx, ny = 0.15, 0.15, 64, 64
        X, Y, xvec, yvec, dx, dy = create_grid(lx, ly, nx, ny)

        result = create_fresnel_two_targets(X, Y)

        assert isinstance(result, tuple)
        assert len(result) == 2
        PROF, params = result
        assert isinstance(PROF, np.ndarray)
        assert isinstance(params, dict)

    def test_fresnel_specifications(self):
        """Verify two-target specifications."""
        lx, ly, nx, ny = 0.15, 0.15, 64, 64
        X, Y, xvec, yvec, dx, dy = create_grid(lx, ly, nx, ny)

        PROF, params = create_fresnel_two_targets(X, Y)

        assert params['r0'] == 0.015        # Both 15 mm radius
        assert params['x0_l'] == -0.045     # Left: x = -45 mm
        assert params['y0_l'] == 0.015      # Left: y = +15 mm
        assert params['x0_r'] == 0.045      # Right: x = +45 mm
        assert params['y0_r'] == 0.005      # Right: y = +5 mm

    def test_two_separate_regions(self):
        """Verify two distinct regions have contrast."""
        lx, ly, nx, ny = 0.15, 0.15, 128, 128
        X, Y, xvec, yvec, dx, dy = create_grid(lx, ly, nx, ny)

        PROF, params = create_fresnel_two_targets(X, Y)

        # Find points inside each cylinder
        i_l = np.argmin(np.abs(yvec - params['y0_l']))
        j_l = np.argmin(np.abs(xvec - params['x0_l']))
        i_r = np.argmin(np.abs(yvec - params['y0_r']))
        j_r = np.argmin(np.abs(xvec - params['x0_r']))

        # Both should have τ = 2
        np.testing.assert_allclose(PROF[i_l, j_l].real, 2.0, rtol=1e-10)
        np.testing.assert_allclose(PROF[i_r, j_r].real, 2.0, rtol=1e-10)

    def test_center_is_zero(self):
        """Origin should have zero contrast (between cylinders)."""
        lx, ly, nx, ny = 0.15, 0.15, 64, 64
        X, Y, xvec, yvec, dx, dy = create_grid(lx, ly, nx, ny)

        PROF, params = create_fresnel_two_targets(X, Y)

        # Center of grid should be background
        assert PROF[ny // 2, nx // 2] == 0.0

    def test_matlab_equivalent(self):
        """
        Verify MATLAB formula:
            r0 = 0.015;
            x0_l = -0.045; y0_l = 0.015;
            x0_r = 0.045;  y0_r = 0.005;
            PROF = zeros(Ny, Nx);
            rxy = sqrt((X-x0_l).^2 + (Y-y0_l).^2); PROF(rxy <= r0) = 2;
            rxy = sqrt((X-x0_r).^2 + (Y-y0_r).^2); PROF(rxy <= r0) = 2;
        """
        lx, ly, nx, ny = 0.15, 0.15, 64, 64
        X, Y, xvec, yvec, dx, dy = create_grid(lx, ly, nx, ny)

        PROF, params = create_fresnel_two_targets(X, Y)

        # MATLAB formula
        r0 = 0.015
        x0_l, y0_l = -0.045, 0.015
        x0_r, y0_r = 0.045, 0.005

        PROF_matlab = np.zeros_like(X, dtype=complex)

        rxy_l = np.sqrt((X - x0_l)**2 + (Y - y0_l)**2)
        PROF_matlab[rxy_l <= r0] = 2.0

        rxy_r = np.sqrt((X - x0_r)**2 + (Y - y0_r)**2)
        PROF_matlab[rxy_r <= r0] = 2.0

        np.testing.assert_allclose(PROF, PROF_matlab, rtol=1e-10)


class TestMATLABCompatibility:
    """Tests for MATLAB compatibility of profile functions."""

    def test_grid_indexing_convention(self):
        """Verify Python grid matches MATLAB indexing (row = y, col = x)."""
        lx, ly, nx, ny = 0.1, 0.1, 32, 32
        X, Y, xvec, yvec, dx, dy = create_grid(lx, ly, nx, ny)

        # In MATLAB: PROF(row, col) where row ~ y, col ~ x
        # Profile shape should be (Ny, Nx)
        PROF = create_circular_profile(
            X, Y, center=(0.0, 0.0), radius=0.02, epsilon_r=2.0
        )

        assert PROF.shape == (ny, nx)

    def test_scenario_parameters(self):
        """Test with parameters from DATA_scenario.mat."""
        # Parameters from MATLAB reference
        lx, ly = 0.1, 0.1
        nx, ny = 60, 60
        X, Y, xvec, yvec, dx, dy = create_grid(lx, ly, nx, ny)

        # Create circular scatterer
        PROF = create_circular_profile(
            X, Y, center=(0.0, 0.0), radius=0.02,
            epsilon_r=1.5, epsilon_b=1.0  # Weak scatterer
        )

        # Verify dimensions
        assert PROF.shape == (ny, nx)

        # Verify contrast at center
        np.testing.assert_allclose(PROF[ny//2, nx//2].real, 0.5, rtol=1e-10)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_large_object_covering_domain(self):
        """Object larger than DoI should fill everything."""
        lx, ly, nx, ny = 1.0, 1.0, 16, 16
        X, Y, xvec, yvec, dx, dy = create_grid(lx, ly, nx, ny)

        PROF = create_circular_profile(
            X, Y, center=(0.0, 0.0), radius=2.0, epsilon_r=2.0
        )

        # All cells should be inside
        assert np.all(np.abs(PROF) > 0)

    def test_object_outside_domain(self):
        """Object completely outside DoI should give all zeros."""
        lx, ly, nx, ny = 1.0, 1.0, 16, 16
        X, Y, xvec, yvec, dx, dy = create_grid(lx, ly, nx, ny)

        PROF = create_circular_profile(
            X, Y, center=(5.0, 5.0), radius=0.1, epsilon_r=2.0
        )

        assert np.all(PROF == 0)

    def test_very_small_object(self):
        """Very small object might not cover any grid points."""
        lx, ly, nx, ny = 1.0, 1.0, 10, 10
        X, Y, xvec, yvec, dx, dy = create_grid(lx, ly, nx, ny)

        # Object smaller than cell size
        PROF = create_circular_profile(
            X, Y, center=(0.0, 0.0), radius=dx / 10, epsilon_r=2.0
        )

        # Might have zero or one cell depending on center alignment
        assert np.sum(np.abs(PROF) > 0) <= 1

    def test_negative_contrast(self):
        """Contrast can be negative (ε_r < ε_b)."""
        lx, ly, nx, ny = 1.0, 1.0, 32, 32
        X, Y, xvec, yvec, dx, dy = create_grid(lx, ly, nx, ny)

        PROF = create_circular_profile(
            X, Y, center=(0.0, 0.0), radius=0.2,
            epsilon_r=0.5, epsilon_b=1.0  # τ = -0.5
        )

        center_value = PROF[ny // 2, nx // 2]
        np.testing.assert_allclose(center_value.real, -0.5, rtol=1e-10)

    def test_different_background(self):
        """Test with non-unity background permittivity."""
        lx, ly, nx, ny = 1.0, 1.0, 32, 32
        X, Y, xvec, yvec, dx, dy = create_grid(lx, ly, nx, ny)

        epsilon_r = 4.0
        epsilon_b = 2.0  # Non-unity background

        PROF = create_circular_profile(
            X, Y, center=(0.0, 0.0), radius=0.2,
            epsilon_r=epsilon_r, epsilon_b=epsilon_b
        )

        center_value = PROF[ny // 2, nx // 2]
        np.testing.assert_allclose(center_value.real, epsilon_r - epsilon_b, rtol=1e-10)
