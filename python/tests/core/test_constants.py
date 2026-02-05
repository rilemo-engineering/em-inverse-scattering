"""
Unit tests for inverse_scattering.core.constants module.

Tests verify that physical constants match expected values and are
consistent with fundamental physics relationships.
"""

import numpy as np
import pytest

from inverse_scattering.core.constants import (
    EPSILON_0,
    MU_0,
    C,
    e0,
    m0,
    c,
)


class TestPhysicalConstants:
    """Test physical constants values and relationships."""

    def test_epsilon_0_value(self, physical_constants):
        """Verify vacuum permittivity matches expected value."""
        # MATLAB uses e0 = 8.85e-12
        assert EPSILON_0 == 8.85e-12
        # Also check it's reasonably close to CODATA value
        np.testing.assert_allclose(
            EPSILON_0,
            physical_constants['epsilon_0'],
            rtol=1e-3  # Within 0.1%
        )

    def test_mu_0_value(self, physical_constants):
        """Verify vacuum permeability matches expected value."""
        # MATLAB uses m0 = 4*pi*1e-7
        expected_mu0 = 4.0 * np.pi * 1e-7
        np.testing.assert_allclose(MU_0, expected_mu0, rtol=1e-15)
        # Also verify against CODATA
        np.testing.assert_allclose(
            MU_0,
            physical_constants['mu_0'],
            rtol=1e-10
        )

    def test_c_value(self, physical_constants):
        """Verify speed of light matches expected value."""
        # MATLAB uses c = 3e8 (approximate)
        assert C == 3.0e8
        # Check it's within 1% of exact value
        np.testing.assert_allclose(
            C,
            physical_constants['c'],
            rtol=0.01
        )

    def test_speed_of_light_relationship(self):
        """
        Verify c ≈ 1/sqrt(ε₀ * μ₀).

        This is a fundamental relationship from Maxwell's equations.
        Note: MATLAB uses approximations, so there's a small discrepancy.
        """
        computed_c = 1.0 / np.sqrt(EPSILON_0 * MU_0)
        # With MATLAB's approximate constants, computed c ≈ 2.998e8
        np.testing.assert_allclose(
            computed_c,
            C,
            rtol=0.01  # Within 1% due to MATLAB's approximate values
        )

    def test_constants_are_positive(self):
        """All physical constants must be positive."""
        assert EPSILON_0 > 0
        assert MU_0 > 0
        assert C > 0

    def test_constants_are_real(self):
        """All physical constants must be real (no imaginary part)."""
        assert np.isreal(EPSILON_0)
        assert np.isreal(MU_0)
        assert np.isreal(C)


class TestMATLABCompatibility:
    """Test MATLAB-compatible aliases."""

    def test_lowercase_aliases_exist(self):
        """Verify lowercase aliases are defined for MATLAB compatibility."""
        # These should be importable
        assert e0 is not None
        assert m0 is not None
        assert c is not None

    def test_lowercase_aliases_match_constants(self):
        """Verify lowercase aliases equal uppercase constants."""
        assert e0 == EPSILON_0
        assert m0 == MU_0
        assert c == C

    def test_matlab_exact_values(self):
        """
        Verify constants match MATLAB code exactly.

        From MATLAB:
            e0 = 8.85e-12
            m0 = 4*pi*1e-7
            c = 3e8
        """
        assert EPSILON_0 == 8.85e-12
        assert MU_0 == pytest.approx(4 * np.pi * 1e-7, rel=1e-15)
        assert C == 3.0e8


class TestConstantTypes:
    """Test that constants have correct types."""

    def test_epsilon_0_is_float(self):
        """EPSILON_0 should be a float."""
        assert isinstance(EPSILON_0, float)

    def test_mu_0_is_float(self):
        """MU_0 should be a float."""
        assert isinstance(MU_0, float)

    def test_c_is_float(self):
        """C should be a float."""
        assert isinstance(C, float)


class TestConstantUsage:
    """Test that constants work correctly in typical usage."""

    def test_wavenumber_computation(self):
        """
        Test constants in wavenumber computation.

        k = ω * sqrt(ε₀ * μ₀ * εᵣ) = 2πf * sqrt(ε₀ * μ₀ * εᵣ)
        """
        freq = 1e9  # 1 GHz
        epsilon_r = 1.0  # Free space

        omega = 2 * np.pi * freq
        k = omega * np.sqrt(EPSILON_0 * MU_0 * epsilon_r)

        # k should be approximately 2πf/c
        k_expected = 2 * np.pi * freq / C
        np.testing.assert_allclose(k, k_expected, rtol=0.01)

    def test_wavelength_computation(self):
        """
        Test constants in wavelength computation.

        λ = c / f
        """
        freq = 300e6  # 300 MHz
        wavelength = C / freq

        # At 300 MHz, wavelength should be 1 meter
        assert wavelength == 1.0

    def test_impedance_computation(self):
        """
        Test constants in free-space impedance computation.

        η₀ = sqrt(μ₀ / ε₀) ≈ 377 Ω
        """
        eta_0 = np.sqrt(MU_0 / EPSILON_0)

        # Free-space impedance should be approximately 377 Ω
        np.testing.assert_allclose(eta_0, 377, rtol=0.01)
