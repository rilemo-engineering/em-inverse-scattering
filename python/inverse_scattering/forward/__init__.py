"""
Forward problem module: Profile generation, incident field, CGFFT solver.

This module contains all components needed to solve the forward scattering problem:
- Profile generation (contrast functions)
- Incident field computation
- CGFFT solver for total field
- Internal field operator
- Forward solver orchestration
"""

from inverse_scattering.forward.profiles import (
    create_circular_profile,
    create_square_profile,
    create_fresnel_single_target,
    create_fresnel_two_targets,
)
from inverse_scattering.forward.incident_field import (
    compute_incident_field_line_source,
    compute_incident_field_plane_wave,
    compute_incident_field_all_views,
    setup_transmitters,
)
from inverse_scattering.forward.cgfft import (
    cgfft_solve,
    cgfft_solve_all_views,
    CGFFTResult,
)
from inverse_scattering.forward.forward_solver import (
    forward_solver,
    forward_solver_with_profile,
    compute_scattered_field,
    ForwardSolverResult,
)

__all__ = [
    # Profiles
    "create_circular_profile",
    "create_square_profile",
    "create_fresnel_single_target",
    "create_fresnel_two_targets",
    # Incident field
    "compute_incident_field_line_source",
    "compute_incident_field_plane_wave",
    "compute_incident_field_all_views",
    "setup_transmitters",
    # CGFFT
    "cgfft_solve",
    "cgfft_solve_all_views",
    "CGFFTResult",
    # Forward solver
    "forward_solver",
    "forward_solver_with_profile",
    "compute_scattered_field",
    "ForwardSolverResult",
]
