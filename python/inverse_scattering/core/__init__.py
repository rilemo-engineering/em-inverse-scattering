"""
Core module: Physical constants, utilities, and Green's functions.
"""

from inverse_scattering.core.constants import EPSILON_0, MU_0, C
from inverse_scattering.core.utils import (
    compute_wavelength,
    compute_wavenumber,
    create_grid,
    compute_dof,
)

__all__ = [
    "EPSILON_0",
    "MU_0",
    "C",
    "compute_wavelength",
    "compute_wavenumber",
    "create_grid",
    "compute_dof",
]
