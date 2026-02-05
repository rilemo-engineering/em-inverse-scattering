"""
Inverse Scattering - Python Port

A Python implementation of 2D TM electromagnetic inverse scattering exercises
using Born approximation and TSVD regularization.

This package is a 1:1 port of the MATLAB exercises for studying inverse
scattering problems in electromagnetics.

Modules:
    core: Physical constants, utilities, and Green's functions
    forward: Forward problem solvers (profile generation, CGFFT, etc.)
    inverse: Inverse problem solvers (scattering kernel, TSVD)
    data: Data loading utilities (Fresnel experimental data, .mat files)
    utils: Helper utilities (noise generation, etc.)
    visualization: Plotting functions matching MATLAB figures
    scripts: Main executable scripts for running exercises

Quick Start:
    # Run simulated scenario (forward problem)
    poetry run run-scenario

    # Run Born inversion on simulated data
    poetry run run-inversion

    # Run experimental data scenario
    poetry run run-exp-scenario

    # Run Born inversion on experimental data
    poetry run run-exp-inversion

Example Usage:
    from inverse_scattering.forward import forward_solver, ForwardSolverResult
    from inverse_scattering.inverse import tsvd_solve, compute_svd

    # Run forward problem
    result = forward_solver(n_iter=1000, verbose=True)

    # Access results
    Escat = result.Escat  # Scattered field (Nm x Nv)
    PROF = result.PROF    # True contrast profile (Ny x Nx)
"""

__version__ = "0.1.0"
__author__ = "Inverse Scattering Team"

# Core imports for convenience
from inverse_scattering.core.constants import (
    EPSILON_0,
    MU_0,
    C,
)
from inverse_scattering.core.utils import (
    compute_wavelength,
    compute_wavenumber,
    create_grid,
    compute_dof,
    nmse,
)

# Forward problem
from inverse_scattering.forward import (
    forward_solver,
    ForwardSolverResult,
    cgfft_solve,
    create_circular_profile,
)

# Inverse problem
from inverse_scattering.inverse import (
    kernel_scattering,
    tsvd_solve,
    compute_svd,
)

__all__ = [
    # Version
    "__version__",
    # Constants
    "EPSILON_0",
    "MU_0",
    "C",
    # Core utilities
    "compute_wavelength",
    "compute_wavenumber",
    "create_grid",
    "compute_dof",
    "nmse",
    # Forward problem
    "forward_solver",
    "ForwardSolverResult",
    "cgfft_solve",
    "create_circular_profile",
    # Inverse problem
    "kernel_scattering",
    "tsvd_solve",
    "compute_svd",
]
