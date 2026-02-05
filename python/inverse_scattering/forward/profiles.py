"""
Profile generation for object contrast functions.

This module creates contrast profiles τ(x,y) for various object shapes,
replacing the MATLAB Profili.p function.

The contrast function is defined as:
    τ(r) = ε_r(r) - ε_b - jσ(r)/(ωε₀)

For lossless dielectrics (σ = 0):
    τ = ε_r - ε_b

MATLAB equivalent: Profili.p
"""

import numpy as np
from typing import Tuple, List, Optional, Union


def create_circular_profile(
    X: np.ndarray,
    Y: np.ndarray,
    center: Tuple[float, float],
    radius: float,
    epsilon_r: float,
    epsilon_b: float = 1.0,
    sigma: float = 0.0,
    freq: float = None
) -> np.ndarray:
    """
    Create a circular cylinder contrast profile.

    MATLAB equivalent in c1_Scenario_ExpData.m:
        rxy = sqrt((X-x0).^2 + (Y-y0).^2);
        PROF = zeros(Ny, Nx);
        PROF(rxy <= r0) = 2;  % tau = eps_r - eps_b = 3 - 1 = 2

    Args:
        X: 2D meshgrid of x-coordinates (Ny × Nx)
        Y: 2D meshgrid of y-coordinates (Ny × Nx)
        center: (x0, y0) center of cylinder in meters
        radius: Cylinder radius in meters
        epsilon_r: Relative permittivity of cylinder
        epsilon_b: Background relative permittivity (default: 1.0)
        sigma: Conductivity in S/m (default: 0.0 for lossless)
        freq: Frequency in Hz (required if sigma > 0)

    Returns:
        Contrast profile τ(x,y) as 2D array (Ny × Nx)

    Example:
        >>> # Single Fresnel cylinder at (25mm, 0), r=15mm, eps_r=3
        >>> PROF = create_circular_profile(X, Y, (0.025, 0.0), 0.015, 3.0)
    """
    x0, y0 = center

    # Distance from center
    # MATLAB: rxy = sqrt((X-x0).^2 + (Y-y0).^2)
    rxy = np.sqrt((X - x0)**2 + (Y - y0)**2)

    # Initialize profile with zeros (background has τ = 0)
    PROF = np.zeros_like(X, dtype=complex)

    # Compute contrast
    # τ = ε_r - ε_b for lossless
    tau_real = epsilon_r - epsilon_b

    if sigma > 0 and freq is not None:
        omega = 2 * np.pi * freq
        from inverse_scattering.core.constants import EPSILON_0
        tau_imag = -sigma / (omega * EPSILON_0)
        tau = tau_real + 1j * tau_imag
    else:
        tau = tau_real

    # Set contrast inside cylinder
    # MATLAB: PROF(rxy <= r0) = 2
    PROF[rxy <= radius] = tau

    return PROF


def create_square_profile(
    X: np.ndarray,
    Y: np.ndarray,
    center: Tuple[float, float],
    side: float,
    epsilon_r: float,
    epsilon_b: float = 1.0,
    sigma: float = 0.0,
    freq: float = None
) -> np.ndarray:
    """
    Create a square object contrast profile.

    Args:
        X: 2D meshgrid of x-coordinates (Ny × Nx)
        Y: 2D meshgrid of y-coordinates (Ny × Nx)
        center: (x0, y0) center of square in meters
        side: Side length of square in meters
        epsilon_r: Relative permittivity
        epsilon_b: Background relative permittivity (default: 1.0)
        sigma: Conductivity in S/m (default: 0.0)
        freq: Frequency in Hz (required if sigma > 0)

    Returns:
        Contrast profile τ(x,y) as 2D array (Ny × Nx)
    """
    x0, y0 = center
    half_side = side / 2

    PROF = np.zeros_like(X, dtype=complex)

    # Compute contrast
    tau_real = epsilon_r - epsilon_b
    if sigma > 0 and freq is not None:
        omega = 2 * np.pi * freq
        from inverse_scattering.core.constants import EPSILON_0
        tau_imag = -sigma / (omega * EPSILON_0)
        tau = tau_real + 1j * tau_imag
    else:
        tau = tau_real

    # Set contrast inside square
    inside = (np.abs(X - x0) <= half_side) & (np.abs(Y - y0) <= half_side)
    PROF[inside] = tau

    return PROF


def create_multi_object_profile(
    X: np.ndarray,
    Y: np.ndarray,
    objects: List[dict],
    epsilon_b: float = 1.0
) -> np.ndarray:
    """
    Create a contrast profile with multiple objects.

    Each object is specified by a dictionary with keys:
        - 'shape': 'circle' or 'square'
        - 'center': (x0, y0) tuple
        - 'size': radius (for circle) or side (for square)
        - 'epsilon_r': relative permittivity
        - 'sigma': conductivity (optional, default 0)
        - 'freq': frequency (required if sigma > 0)

    MATLAB equivalent for two cylinders (from c1_Scenario_ExpData.m):
        % leftmost cylinder
        rxy = sqrt((X-x0_l).^2 + (Y-y0_l).^2);
        PROF(rxy <= r0) = 2;
        % rightmost cylinder
        rxy = sqrt((X-x0_r).^2 + (Y-y0_r).^2);
        PROF(rxy <= r0) = 2;

    Args:
        X: 2D meshgrid of x-coordinates (Ny × Nx)
        Y: 2D meshgrid of y-coordinates (Ny × Nx)
        objects: List of object dictionaries
        epsilon_b: Background relative permittivity

    Returns:
        Contrast profile τ(x,y) as 2D array (Ny × Nx)

    Example:
        >>> # Two Fresnel cylinders
        >>> objects = [
        ...     {'shape': 'circle', 'center': (-0.045, 0.015), 'size': 0.015, 'epsilon_r': 3.0},
        ...     {'shape': 'circle', 'center': (0.045, 0.005), 'size': 0.015, 'epsilon_r': 3.0},
        ... ]
        >>> PROF = create_multi_object_profile(X, Y, objects)
    """
    PROF = np.zeros_like(X, dtype=complex)

    for obj in objects:
        shape = obj.get('shape', 'circle')
        center = obj['center']
        size = obj['size']
        epsilon_r = obj['epsilon_r']
        sigma = obj.get('sigma', 0.0)
        freq = obj.get('freq', None)

        if shape == 'circle':
            obj_prof = create_circular_profile(
                X, Y, center, size, epsilon_r, epsilon_b, sigma, freq
            )
        elif shape == 'square':
            obj_prof = create_square_profile(
                X, Y, center, size, epsilon_r, epsilon_b, sigma, freq
            )
        else:
            raise ValueError(f"Unknown shape: {shape}")

        # Combine (objects can overlap; last one wins)
        PROF = np.where(obj_prof != 0, obj_prof, PROF)

    return PROF


def create_fresnel_single_target(
    X: np.ndarray,
    Y: np.ndarray
) -> Tuple[np.ndarray, dict]:
    """
    Create the Fresnel Institute single dielectric cylinder target.

    Specifications from experimental data documentation:
        - Radius: r0 = 15 mm
        - Center: (x0, y0) = (25 mm, 0 mm)
        - Permittivity: ε_r = 3.0 ± 0.3
        - Contrast: τ = 2.0 ± 0.3

    MATLAB equivalent (c1_Scenario_ExpData.m):
        r0 = 0.015;
        x0 = 0.025;
        y0 = 0.0;
        rxy = sqrt((X-x0).^2 + (Y-y0).^2);
        PROF = zeros(Ny, Nx);
        PROF(rxy <= r0) = 2;

    Args:
        X: 2D meshgrid of x-coordinates
        Y: 2D meshgrid of y-coordinates

    Returns:
        Tuple of (PROF, params) where:
            PROF: Contrast profile (Ny × Nx)
            params: Dictionary with r0, x0, y0
    """
    r0 = 0.015      # 15 mm radius
    x0 = 0.025      # 25 mm from center
    y0 = 0.0        # On x-axis

    PROF = create_circular_profile(X, Y, (x0, y0), r0, epsilon_r=3.0, epsilon_b=1.0)

    params = {'r0': r0, 'x0': x0, 'y0': y0}

    return PROF, params


def create_fresnel_two_targets(
    X: np.ndarray,
    Y: np.ndarray
) -> Tuple[np.ndarray, dict]:
    """
    Create the Fresnel Institute two dielectric cylinders target.

    Specifications:
        - Both radius: r0 = 15 mm
        - Left cylinder: (x, y) = (-45 mm, +15 mm)
        - Right cylinder: (x, y) = (+45 mm, +5 mm)
        - Both permittivity: ε_r = 3.0 ± 0.3
        - Separation: 90 mm center-to-center

    MATLAB equivalent (c1_Scenario_ExpData.m):
        r0 = 0.015;
        x0_l = -0.045; y0_l = 0.015;
        x0_r = 0.045;  y0_r = 0.005;

    Args:
        X: 2D meshgrid of x-coordinates
        Y: 2D meshgrid of y-coordinates

    Returns:
        Tuple of (PROF, params) where:
            PROF: Contrast profile (Ny × Nx)
            params: Dictionary with r0, x0_l, y0_l, x0_r, y0_r
    """
    r0 = 0.015      # 15 mm radius

    # Left cylinder
    x0_l = -0.045   # -45 mm
    y0_l = 0.015    # +15 mm

    # Right cylinder
    x0_r = 0.045    # +45 mm
    y0_r = 0.005    # +5 mm

    objects = [
        {'shape': 'circle', 'center': (x0_l, y0_l), 'size': r0, 'epsilon_r': 3.0},
        {'shape': 'circle', 'center': (x0_r, y0_r), 'size': r0, 'epsilon_r': 3.0},
    ]

    PROF = create_multi_object_profile(X, Y, objects, epsilon_b=1.0)

    params = {
        'r0': r0,
        'x0_l': x0_l, 'y0_l': y0_l,
        'x0_r': x0_r, 'y0_r': y0_r
    }

    return PROF, params
