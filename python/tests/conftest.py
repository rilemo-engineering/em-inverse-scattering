"""
Pytest configuration and shared fixtures for inverse scattering tests.

This module provides:
- Paths to reference MATLAB data files
- Common test parameters
- Shared fixtures for grid, profile, and solver setup
"""

import pytest
import numpy as np
from pathlib import Path


# ============================================================
# Path fixtures
# ============================================================

@pytest.fixture
def project_root():
    """Return the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def matlab_data_root():
    """Return path to MATLAB simulated data root."""
    project = Path(__file__).parent.parent.parent
    return project / "matlab" / "simulated"


@pytest.fixture
def experimental_data_root():
    """Return path to experimental data root."""
    project = Path(__file__).parent.parent.parent
    return project / "matlab" / "experimental"


@pytest.fixture
def reference_scenario_mat(matlab_data_root):
    """Path to reference DATA_scenario.mat from MATLAB."""
    return matlab_data_root / "scenario" / "DATA_scenario.mat"


@pytest.fixture
def reference_noweak_mat(matlab_data_root):
    """Path to reference strong scatterer scenario."""
    return matlab_data_root / "inversion" / "DATA_scenario_noweak.mat"


@pytest.fixture
def reference_square_mat(matlab_data_root):
    """Path to reference square target scenario."""
    return matlab_data_root / "inversion" / "DATA_scenario_square.mat"


@pytest.fixture
def fixtures_dir():
    """Return path to test fixtures directory."""
    return Path(__file__).parent / "fixtures"


# ============================================================
# Physical constants fixtures
# ============================================================

@pytest.fixture
def physical_constants():
    """Standard physical constants for validation."""
    return {
        'epsilon_0': 8.854187817e-12,  # F/m (exact CODATA value)
        'mu_0': 4 * np.pi * 1e-7,      # H/m (exact definition)
        'c': 299792458.0,               # m/s (exact definition)
    }


# ============================================================
# Grid fixtures
# ============================================================

@pytest.fixture
def small_grid_params():
    """Small grid parameters for fast unit tests."""
    return {
        'Nx': 16,
        'Ny': 16,
        'lx': 1.0,
        'ly': 1.0,
    }


@pytest.fixture
def medium_grid_params():
    """Medium grid parameters for integration tests."""
    return {
        'Nx': 32,
        'Ny': 32,
        'lx': 1.0,
        'ly': 1.0,
    }


@pytest.fixture
def reference_grid_params():
    """Grid parameters matching MATLAB DATA_scenario.mat."""
    return {
        'Nx': 60,
        'Ny': 60,
        'lx': 0.1,
        'ly': 0.1,
        'freq': 1e9,
        'Rm': 0.1,
        'Nm': 12,
        'Nv': 12,
        'eb': 1.0,
        'sb': 0.0,
    }


@pytest.fixture
def small_grid(small_grid_params):
    """Create a small test grid."""
    from inverse_scattering.core.utils import create_grid
    return create_grid(
        small_grid_params['lx'],
        small_grid_params['ly'],
        small_grid_params['Nx'],
        small_grid_params['Ny']
    )


# ============================================================
# Profile fixtures
# ============================================================

@pytest.fixture
def weak_scatterer_params():
    """Parameters for a weak scatterer (Born approximation valid)."""
    return {
        'epsilon_r': 1.5,  # Contrast τ = 0.5
        'sigma': 0.0,
        'radius': 0.1,
        'center': (0.0, 0.0),
    }


@pytest.fixture
def strong_scatterer_params():
    """Parameters for a strong scatterer (Born approximation breaks down)."""
    return {
        'epsilon_r': 4.0,  # Contrast τ = 3.0
        'sigma': 0.0,
        'radius': 0.2,
        'center': (0.0, 0.0),
    }


@pytest.fixture
def weak_profile(small_grid, weak_scatterer_params):
    """Create a weak scatterer profile on small grid."""
    from inverse_scattering.forward.profiles import create_circular_profile
    X, Y, xvec, yvec, dx, dy = small_grid
    return create_circular_profile(
        X, Y,
        center=weak_scatterer_params['center'],
        radius=weak_scatterer_params['radius'],
        epsilon_r=weak_scatterer_params['epsilon_r'],
        epsilon_b=1.0,
        sigma=weak_scatterer_params['sigma'],
    )


# ============================================================
# Solver fixtures
# ============================================================

@pytest.fixture
def default_freq():
    """Default frequency for tests (300 MHz)."""
    return 300e6


@pytest.fixture
def default_wavenumber(default_freq):
    """Default wavenumber for free space."""
    from inverse_scattering.core.utils import compute_wavenumber
    return compute_wavenumber(default_freq, epsilon_r=1.0, sigma=0.0)


# ============================================================
# Tolerance fixtures
# ============================================================

@pytest.fixture
def analytical_tolerance():
    """Tolerance for comparing analytical formulas (should be exact)."""
    return 1e-10


@pytest.fixture
def numerical_tolerance():
    """Tolerance for numerical computations (some error expected)."""
    return 1e-6


@pytest.fixture
def integration_tolerance():
    """Tolerance for comparing against MATLAB reference (larger due to implementation differences)."""
    return 1e-3


# ============================================================
# Helper functions
# ============================================================

def assert_allclose_complex(actual, expected, rtol=1e-7, atol=0):
    """Assert that complex arrays are close, handling phase differences."""
    np.testing.assert_allclose(np.abs(actual), np.abs(expected), rtol=rtol, atol=atol)
    # For phase, allow wrapping
    phase_diff = np.angle(actual) - np.angle(expected)
    phase_diff_wrapped = np.mod(phase_diff + np.pi, 2*np.pi) - np.pi
    np.testing.assert_allclose(phase_diff_wrapped, 0, atol=rtol)


@pytest.fixture
def assert_complex_close():
    """Provide complex array comparison function."""
    return assert_allclose_complex


# ============================================================
# Skip conditions
# ============================================================

def matlab_reference_available(path):
    """Check if MATLAB reference file exists."""
    return Path(path).exists()


def fresnel_data_available(root):
    """Check if Fresnel experimental data is available."""
    # Look for any .txt files in expected locations
    exp_data = root / "scenario"
    return exp_data.exists() and any(exp_data.glob("*.txt"))
