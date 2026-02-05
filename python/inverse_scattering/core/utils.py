"""
Core utility functions for inverse scattering calculations.

This module provides fundamental calculations used throughout the forward
and inverse problems, matching MATLAB implementations exactly.

Functions:
    compute_wavelength: Calculate wavelength from frequency
    compute_wavenumber: Calculate wavenumber from frequency and medium
    create_grid: Create spatial discretization grid
    compute_dof: Calculate degrees of freedom
    compute_contrast: Calculate contrast function from material properties
"""

import numpy as np
from typing import Tuple

from inverse_scattering.core.constants import EPSILON_0, MU_0, C


def compute_wavelength(freq: float) -> float:
    """
    Compute free-space wavelength from frequency.

    MATLAB equivalent: lambda0 = 3*1e8/freq

    Args:
        freq: Frequency in Hz

    Returns:
        Wavelength in meters (λ₀ = c/f)

    Example:
        >>> compute_wavelength(4e9)  # 4 GHz
        0.075  # 7.5 cm
    """
    return C / freq


def compute_wavenumber(
    freq: float,
    epsilon_r: float = 1.0,
    sigma: float = 0.0
) -> complex:
    """
    Compute wavenumber in a medium.

    For a medium with relative permittivity ε_r and conductivity σ:
        k = ω√(ε₀μ₀ε_eq)
    where:
        ω = 2πf
        ε_eq = ε_r - jσ/(ωε₀)  (complex equivalent permittivity)

    MATLAB equivalent:
        eb_eq = eb - 1i*(sb/(e0*2*pi*freq))
        kb = 2*pi*freq*sqrt(e0*m0*eb_eq)

    Args:
        freq: Frequency in Hz
        epsilon_r: Relative permittivity of the medium (default: 1.0 for free space)
        sigma: Conductivity of the medium in S/m (default: 0.0)

    Returns:
        Complex wavenumber k in rad/m

    Example:
        >>> compute_wavenumber(4e9, 1.0, 0.0)  # Free space at 4 GHz
        (83.776... + 0j)  # k₀ ≈ 2π/λ₀
    """
    omega = 2 * np.pi * freq

    # Complex equivalent permittivity
    # MATLAB: eb_eq = eb - 1i*(sb/(e0*2*pi*freq))
    epsilon_eq = epsilon_r - 1j * (sigma / (omega * EPSILON_0))

    # Wavenumber: k = ω√(ε₀μ₀ε_eq)
    # MATLAB: kb = 2*pi*freq*sqrt(e0*m0*eb_eq)
    k = omega * np.sqrt(EPSILON_0 * MU_0 * epsilon_eq)

    return k


def create_grid(
    lx: float,
    ly: float,
    nx: int,
    ny: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    """
    Create a spatial discretization grid for the Domain of Investigation (DoI).

    The grid is centered at the origin with cell centers at:
        x: from -lx/2 + dx/2 to lx/2 - dx/2
        y: from -ly/2 + dy/2 to ly/2 - dy/2

    MATLAB equivalent:
        dx = lx/Nx
        dy = ly/Ny
        xvec = -lx/2+dx/2:dx:lx/2-dx/2
        yvec = -ly/2+dy/2:dy:ly/2-dy/2
        [X, Y] = meshgrid(xvec, yvec)

    Args:
        lx: Domain size in x-direction [m]
        ly: Domain size in y-direction [m]
        nx: Number of cells in x-direction
        ny: Number of cells in y-direction

    Returns:
        Tuple of (X, Y, xvec, yvec, dx, dy) where:
            X: 2D meshgrid of x-coordinates (ny × nx)
            Y: 2D meshgrid of y-coordinates (ny × nx)
            xvec: 1D array of x-coordinates (nx,)
            yvec: 1D array of y-coordinates (ny,)
            dx: Cell size in x-direction [m]
            dy: Cell size in y-direction [m]

    Note:
        MATLAB's meshgrid produces (ny, nx) arrays when given xvec and yvec,
        which matches numpy's default behavior.
    """
    dx = lx / nx
    dy = ly / ny

    # Cell-centered coordinates
    # MATLAB: xvec = -lx/2+dx/2:dx:lx/2-dx/2
    xvec = np.linspace(-lx/2 + dx/2, lx/2 - dx/2, nx)
    yvec = np.linspace(-ly/2 + dy/2, ly/2 - dy/2, ny)

    # 2D meshgrid
    # MATLAB: [X, Y] = meshgrid(xvec, yvec)
    X, Y = np.meshgrid(xvec, yvec)

    return X, Y, xvec, yvec, dx, dy


def compute_dof(lx: float, freq: float, ly: float = None) -> int:
    """
    Compute the Degrees of Freedom (DoF) for the inverse problem.

    DoF ≈ 2 * k₀ * a

    where a is the characteristic size of the DoI. For a square DoI:
        a = √2 * lx / 2  (half-diagonal)

    The DoF represents the maximum number of independent parameters
    that can be recovered from the scattered field data.

    MATLAB comment: DoF = 2*beta*a, where a = sqrt(2)*lx/2

    Args:
        lx: Domain size in x-direction [m]
        freq: Frequency in Hz
        ly: Domain size in y-direction [m] (default: same as lx)

    Returns:
        Degrees of freedom (rounded up to nearest integer)

    Example:
        >>> compute_dof(0.15, 4e9)  # 15cm DoI at 4 GHz
        ~18  # Depends on exact geometry
    """
    if ly is None:
        ly = lx

    # Free-space wavenumber
    k0 = 2 * np.pi / compute_wavelength(freq)

    # Characteristic size (half-diagonal for rectangular DoI)
    a = np.sqrt(lx**2 + ly**2) / 2

    # DoF = 2 * k0 * a
    dof = 2 * k0 * a

    # Round up (MATLAB comment says "Round up the DoF number")
    return int(np.ceil(dof))


def compute_contrast(
    epsilon_r: float,
    epsilon_b: float,
    sigma: float = 0.0,
    freq: float = None
) -> complex:
    """
    Compute the contrast function τ for a material.

    τ(r) = ε_r(r) - ε_b - jσ(r)/(ωε₀)

    For lossless materials (σ = 0):
        τ = ε_r - ε_b

    MATLAB comments show:
        τ = 2 means ε_r = 3 (with ε_b = 1)
        τ = 0.5 means ε_r = 1.5 (with ε_b = 1)

    Args:
        epsilon_r: Object relative permittivity
        epsilon_b: Background relative permittivity
        sigma: Object conductivity in S/m (default: 0.0)
        freq: Frequency in Hz (required if sigma > 0)

    Returns:
        Complex contrast function value

    Raises:
        ValueError: If sigma > 0 but freq is not provided
    """
    # Real part: permittivity contrast
    tau = epsilon_r - epsilon_b

    # Imaginary part: conductivity contribution
    if sigma > 0:
        if freq is None:
            raise ValueError("Frequency required for lossy materials (sigma > 0)")
        omega = 2 * np.pi * freq
        tau = tau - 1j * sigma / (omega * EPSILON_0)

    return tau


def compute_measurement_positions(
    n_meas: int,
    radius: float,
    full_circle: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute measurement (receiver/transmitter) positions on a circle.

    MATLAB equivalent:
        meas_pos_theta = linspace(0, 2*pi - 2*pi/Nm, Nm)
        x_meas = Rm * cos(meas_pos_theta)
        y_meas = Rm * sin(meas_pos_theta)

    Args:
        n_meas: Number of measurement points
        radius: Radius of measurement circle [m]
        full_circle: If True, positions span [0, 2π). If False, [0, 2π].

    Returns:
        Tuple of (theta, x_pos, y_pos) where:
            theta: Angular positions in radians (n_meas,)
            x_pos: x-coordinates of positions (n_meas,)
            y_pos: y-coordinates of positions (n_meas,)
    """
    if full_circle:
        # Exclude last point to avoid overlap at 0 and 2π
        # MATLAB: linspace(0, 2*pi - 2*pi/Nm, Nm)
        theta = np.linspace(0, 2*np.pi - 2*np.pi/n_meas, n_meas)
    else:
        theta = np.linspace(0, 2*np.pi, n_meas)

    x_pos = radius * np.cos(theta)
    y_pos = radius * np.sin(theta)

    return theta, x_pos, y_pos


def nmse(true_profile: np.ndarray, reconstructed_profile: np.ndarray) -> float:
    """
    Compute Normalized Mean Square Error between true and reconstructed profiles.

    NMSE = ||τ_true - τ_rec||² / ||τ_true||²

    MATLAB equivalent:
        NMSE_BORN = sum(sum(abs(PROF-PROF_rec_BORN).^2))/sum(sum(abs(PROF).^2))

    Args:
        true_profile: Ground truth contrast profile (Ny × Nx)
        reconstructed_profile: Reconstructed contrast profile (Ny × Nx)

    Returns:
        NMSE value (0 = perfect, 1 = poor, >1 = very poor)

    Interpretation:
        NMSE < 0.05: Excellent
        0.05 - 0.15: Good
        0.15 - 0.30: Acceptable
        0.30 - 0.50: Marginal
        > 0.50: Poor
    """
    numerator = np.sum(np.abs(true_profile - reconstructed_profile)**2)
    denominator = np.sum(np.abs(true_profile)**2)

    if denominator == 0:
        return np.inf if numerator > 0 else 0.0

    return float(numerator / denominator)
