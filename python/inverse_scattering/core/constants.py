"""
Physical constants for electromagnetic wave propagation.

These constants are fundamental to the inverse scattering problem and match
the MATLAB implementation exactly.

Constants:
    EPSILON_0: Vacuum permittivity (F/m)
    MU_0: Vacuum permeability (H/m)
    C: Speed of light in vacuum (m/s)
"""

import numpy as np

# Vacuum dielectric permittivity [F/m]
# MATLAB: e0 = 8.85e-12
EPSILON_0: float = 8.85e-12

# Vacuum magnetic permeability [H/m]
# MATLAB: m0 = 4*pi*1e-7
MU_0: float = 4.0 * np.pi * 1e-7

# Speed of light in vacuum [m/s]
# MATLAB: c = 3e8 (approximate)
# Note: Exact value is c = 1/sqrt(e0*m0) ≈ 299792458 m/s
# We use 3e8 for consistency with MATLAB code
C: float = 3.0e8


# Aliases for MATLAB compatibility (lowercase)
e0 = EPSILON_0
m0 = MU_0
c = C
