"""
Data module: Fresnel experimental data loader and .mat file I/O.

This module provides utilities for:
- Loading Fresnel Institute experimental data files
- Reading/writing MATLAB .mat files for compatibility
"""

from inverse_scattering.data.fresnel_loader import (
    load_fresnel_data,
    load_data_fr2001,
    get_fresnel_parameters,
)
from inverse_scattering.data.mat_io import (
    load_mat,
    save_mat,
)

__all__ = [
    # Fresnel data
    "load_fresnel_data",
    "load_data_fr2001",
    "get_fresnel_parameters",
    # MATLAB I/O
    "load_mat",
    "save_mat",
]
