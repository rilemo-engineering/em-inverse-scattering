"""
2D Green's function for electromagnetic wave propagation.

The 2D Green's function for the Helmholtz equation represents the field
produced by a line source. It is fundamental to both forward and inverse
scattering problems.

Theory:
    For a 2D problem (TM polarization), the Green's function satisfies:
        ∇²G + k²G = -δ(r - r')

    Solution:
        G(r, r') = (i/4) * H₀⁽¹⁾(k|r - r'|)

    where H₀⁽¹⁾ is the Hankel function of the first kind, order 0.

    For small arguments (near singularity):
        H₀⁽¹⁾(z) ≈ 1 + (2i/π) * (ln(z/2) + γ)
        where γ ≈ 0.5772... is Euler's constant

References:
    - Theory document: 01_THEORY_AND_CONCEPTS.md, Section 3
    - MATLAB: Uses Bessel/Hankel functions from special functions
"""

import numpy as np
from scipy import special
from typing import Union, Tuple

# Small value threshold for singularity handling
SINGULARITY_THRESHOLD = 1e-10


def hankel1_0(z: Union[float, np.ndarray]) -> Union[complex, np.ndarray]:
    """
    Compute Hankel function of the first kind, order 0.

    H₀⁽¹⁾(z) = J₀(z) + i*Y₀(z)

    where J₀ is the Bessel function of the first kind and
    Y₀ is the Bessel function of the second kind (Neumann function).

    Args:
        z: Argument (can be scalar or array)

    Returns:
        H₀⁽¹⁾(z) as complex number(s)

    Note:
        scipy.special.hankel1(0, z) computes this directly.
    """
    return special.hankel1(0, z)


def greens_function_2d(
    k: complex,
    r_obs: np.ndarray,
    r_src: np.ndarray
) -> complex:
    """
    Compute 2D Green's function between observation and source points.

    G(r_obs, r_src) = (i/4) * H₀⁽¹⁾(k|r_obs - r_src|)

    MATLAB equivalent in theory document:
        G = (i/4) * H_0^(1)(k_b * ||r - r'||)

    Args:
        k: Wavenumber (complex, rad/m)
        r_obs: Observation point [x, y] in meters
        r_src: Source point [x', y'] in meters

    Returns:
        Complex Green's function value

    Note:
        When r_obs = r_src, the function is singular. This case should be
        handled separately (e.g., using cell integration for self-terms).
    """
    # Distance between points
    distance = np.sqrt((r_obs[0] - r_src[0])**2 + (r_obs[1] - r_src[1])**2)

    if distance < SINGULARITY_THRESHOLD:
        # Return a regularized value for self-term
        # In practice, this should be handled by cell integration
        return 0.0 + 0.0j

    # Green's function: G = (i/4) * H_0^(1)(k*r)
    return (1j / 4) * hankel1_0(k * distance)


def greens_function_matrix(
    k: complex,
    x_obs: np.ndarray,
    y_obs: np.ndarray,
    x_src: np.ndarray,
    y_src: np.ndarray,
    dx: float = None,
    dy: float = None
) -> np.ndarray:
    """
    Compute Green's function matrix between observation and source grids.

    G[i,j] = G(r_obs[i], r_src[j])

    This is the core computation for building scattering operators.

    Args:
        k: Complex wavenumber (rad/m)
        x_obs: x-coordinates of observation points (M,) or (My, Mx)
        y_obs: y-coordinates of observation points (M,) or (My, Mx)
        x_src: x-coordinates of source points (N,) or (Ny, Nx)
        y_src: y-coordinates of source points (N,) or (Ny, Nx)
        dx: Cell size in x (for self-term integration, optional)
        dy: Cell size in y (for self-term integration, optional)

    Returns:
        Green's function matrix (n_obs × n_src)
    """
    # Flatten coordinates to 1D
    x_obs_flat = np.ravel(x_obs)
    y_obs_flat = np.ravel(y_obs)
    x_src_flat = np.ravel(x_src)
    y_src_flat = np.ravel(y_src)

    n_obs = len(x_obs_flat)
    n_src = len(x_src_flat)

    # Compute distance matrix using broadcasting
    # r[i,j] = |r_obs[i] - r_src[j]|
    dx_mat = x_obs_flat[:, np.newaxis] - x_src_flat[np.newaxis, :]
    dy_mat = y_obs_flat[:, np.newaxis] - y_src_flat[np.newaxis, :]
    distance = np.sqrt(dx_mat**2 + dy_mat**2)

    # Initialize Green's function matrix
    G = np.zeros((n_obs, n_src), dtype=complex)

    # Compute for non-singular points
    non_singular = distance > SINGULARITY_THRESHOLD
    G[non_singular] = (1j / 4) * hankel1_0(k * distance[non_singular])

    # Handle self-terms (diagonal) with cell integration if cell sizes provided
    if dx is not None and dy is not None:
        self_term = _compute_self_term(k, dx, dy)
        # Find diagonal elements where obs == src
        for i in range(min(n_obs, n_src)):
            if distance[i, i] < SINGULARITY_THRESHOLD:
                G[i, i] = self_term

    return G


def _compute_self_term(k: complex, dx: float, dy: float) -> complex:
    """
    Compute the self-term (singularity) of the Green's function.

    For a square cell, the self-term involves integrating the singular
    Green's function over the cell area. Using the small-argument
    approximation:

        G_self ≈ (1/(2π)) * [ln(k*a_eq/2) + γ - iπ/2 + 1]

    where a_eq = √(dx*dy/π) is the equivalent radius and γ is Euler's constant.

    A simpler approximation used in many codes:
        G_self ≈ (i/4) * H₀⁽¹⁾(k * a_eq) * area / (π * a_eq²)

    Or the widely used approximation:
        G_self = (dx*dy) * (1 - 1j*(2/π)*(log(γ_e*k*a_eq/2) + 1))/(2π)

    where γ_e ≈ 1.781 is the exponential of Euler's constant.

    Args:
        k: Wavenumber
        dx: Cell size in x
        dy: Cell size in y

    Returns:
        Self-term value (complex)

    Reference:
        Richmond, J.H., "Scattering by a dielectric cylinder of arbitrary
        cross section shape," IEEE Trans. Antennas Propagat., 1965
    """
    # Equivalent radius for a rectangular cell
    cell_area = dx * dy
    a_eq = np.sqrt(cell_area / np.pi)

    # Euler's constant
    gamma_euler = 0.5772156649015329

    # Using the standard approximation for the self-term integral
    # This accounts for the 1/r singularity in the 2D Green's function
    if np.abs(k) < 1e-10:
        # Static limit
        return 0.0 + 0.0j

    # Small argument expansion of H_0^(1)
    # H_0^(1)(z) ≈ 1 + (2i/π) * [ln(z/2) + γ]
    z = k * a_eq

    # Self-term integral over circular cell of radius a_eq
    # ∫∫ G(r,r') dr' ≈ (i/4) * (2π) * a_eq² / a_eq * [complex correction]
    # Simplified form used in MoM codes:
    self_term = (cell_area / (2 * np.pi)) * (
        1 - 1j * (2 / np.pi) * (np.log(1.781 * np.abs(k) * a_eq / 2) + 1)
    )

    return self_term


def greens_function_grad(
    k: complex,
    x_obs: float,
    y_obs: float,
    x_src: float,
    y_src: float
) -> Tuple[complex, complex]:
    """
    Compute gradient of 2D Green's function.

    ∇G = (∂G/∂x, ∂G/∂y)

    For G = (i/4) * H₀⁽¹⁾(kr):
        ∇G = -(ik/4) * H₁⁽¹⁾(kr) * r̂

    where r̂ = (r_obs - r_src)/|r_obs - r_src|

    Args:
        k: Wavenumber
        x_obs, y_obs: Observation point
        x_src, y_src: Source point

    Returns:
        Tuple of (dG/dx, dG/dy)
    """
    dx = x_obs - x_src
    dy = y_obs - y_src
    r = np.sqrt(dx**2 + dy**2)

    if r < SINGULARITY_THRESHOLD:
        return 0.0 + 0.0j, 0.0 + 0.0j

    # H_1^(1) (Hankel first kind, order 1)
    H1 = special.hankel1(1, k * r)

    # ∇G = -(ik/4) * H_1^(1)(kr) * r̂
    factor = -(1j * k / 4) * H1 / r

    dG_dx = factor * dx
    dG_dy = factor * dy

    return dG_dx, dG_dy
