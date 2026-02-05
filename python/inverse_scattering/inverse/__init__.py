"""
Inverse problem module: Scattering kernel and TSVD solver.

This module contains components for solving the inverse scattering problem:
- Scattering kernel (Born approximation operator)
- TSVD (Truncated Singular Value Decomposition) solver
"""

from inverse_scattering.inverse.scattering_kernel import (
    build_scattering_kernel,
    kernel_scattering,
    kernel_scattering_exp,
    apply_scattering_kernel,
    reshape_escat_to_matrix,
)
from inverse_scattering.inverse.tsvd import (
    compute_svd,
    find_truncation_index,
    tsvd_solve,
    tsvd_solver_matlab_interface,
)

__all__ = [
    # Scattering kernel
    "build_scattering_kernel",
    "kernel_scattering",
    "kernel_scattering_exp",
    "apply_scattering_kernel",
    "reshape_escat_to_matrix",
    # TSVD solver
    "compute_svd",
    "find_truncation_index",
    "tsvd_solve",
    "tsvd_solver_matlab_interface",
]
