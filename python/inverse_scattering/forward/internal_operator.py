"""
Internal field operator for forward scattering.

The internal field operator A relates the contrast τ to the scattered field
contribution from within the domain. It is used to solve the
Lippmann-Schwinger equation:

    E_tot(r) = E_inc(r) + k₀² ∫∫ τ(r') E_tot(r') G(r, r') dr'

In discrete form:
    E_tot = E_inc + A @ (τ * E_tot)

where A_ij represents the contribution from cell j to field at cell i.

MATLAB equivalent: ainterno.p
"""

import numpy as np
from typing import Tuple
from scipy import special

from inverse_scattering.core.greens_function import (
    greens_function_matrix,
    _compute_self_term
)


def build_internal_operator(
    X: np.ndarray,
    Y: np.ndarray,
    k: complex,
    dx: float,
    dy: float
) -> np.ndarray:
    """
    Build the internal field operator matrix A.

    The operator maps contrast*field to scattered field contribution:
        A_ij = k₀² * G(r_i, r_j) * dx * dy

    For self-terms (i=j), special integration is used.

    MATLAB equivalent: ainterno.p

    Args:
        X: 2D meshgrid of x-coordinates (Ny × Nx)
        Y: 2D meshgrid of y-coordinates (Ny × Nx)
        k: Wavenumber (complex)
        dx: Cell size in x [m]
        dy: Cell size in y [m]

    Returns:
        Internal operator matrix A (N × N) where N = Nx * Ny
    """
    Ny, Nx = X.shape
    N = Nx * Ny

    # Flatten coordinates
    x_flat = X.ravel()
    y_flat = Y.ravel()

    # Build Green's function matrix
    G = greens_function_matrix(
        k, x_flat, y_flat, x_flat, y_flat, dx, dy
    )

    # Multiply by k² * cell_area to get operator
    cell_area = dx * dy
    k_squared = k**2

    A = k_squared * G * cell_area

    # Handle self-terms more carefully
    self_term = _compute_internal_self_term(k, dx, dy)
    for i in range(N):
        A[i, i] = k_squared * self_term * cell_area

    return A


def _compute_internal_self_term(k: complex, dx: float, dy: float) -> complex:
    """
    Compute the self-term for the internal operator.

    For a square cell, the self-interaction integral:
        ∫∫_cell G(r, r') dr' ≈ G_self

    This uses the equivalent circular cell approximation.

    Args:
        k: Wavenumber
        dx: Cell size in x
        dy: Cell size in y

    Returns:
        Self-term value
    """
    # Use the same self-term computation as in greens_function module
    return _compute_self_term(k, dx, dy)


def build_toeplitz_green(
    Nx: int,
    Ny: int,
    k: complex,
    dx: float,
    dy: float
) -> np.ndarray:
    """
    Build the Green's function in Toeplitz form for FFT acceleration.

    For a uniform grid, the Green's function G(r_i, r_j) depends only on
    (r_i - r_j), making the matrix Toeplitz (actually block-Toeplitz-Toeplitz-block).

    This structure enables FFT-based fast multiplication.

    The extended grid is (2Ny-1) × (2Nx-1) to handle circular convolution.

    Args:
        Nx: Number of x grid points
        Ny: Number of y grid points
        k: Wavenumber
        dx: Cell size in x
        dy: Cell size in y

    Returns:
        Green's function on extended grid (2Ny-1, 2Nx-1)
    """
    # Extended grid for circular convolution
    Nx_ext = 2 * Nx - 1
    Ny_ext = 2 * Ny - 1

    # Create coordinate differences
    # x_diff[j] = j*dx for j in range(-(Nx-1), Nx)
    x_diff = np.arange(-(Nx-1), Nx) * dx
    y_diff = np.arange(-(Ny-1), Ny) * dy

    X_diff, Y_diff = np.meshgrid(x_diff, y_diff)
    R = np.sqrt(X_diff**2 + Y_diff**2)

    # Compute Green's function
    G_toeplitz = np.zeros((Ny_ext, Nx_ext), dtype=complex)

    # Non-zero distances
    nonzero = R > 1e-10
    G_toeplitz[nonzero] = (1j / 4) * special.hankel1(0, k * R[nonzero])

    # Self-term at center
    center_y = Ny - 1
    center_x = Nx - 1
    G_toeplitz[center_y, center_x] = _compute_internal_self_term(k, dx, dy)

    # Multiply by k² * cell_area
    cell_area = dx * dy
    G_toeplitz *= k**2 * cell_area

    return G_toeplitz


def fft_green(G_toeplitz: np.ndarray) -> np.ndarray:
    """
    Compute 2D FFT of the Toeplitz Green's function.

    This is precomputed once for efficient CGFFT iterations.

    Args:
        G_toeplitz: Green's function on extended grid (2Ny-1, 2Nx-1)

    Returns:
        FFT of Green's function
    """
    return np.fft.fft2(G_toeplitz)


def apply_operator_fft(
    tau_E: np.ndarray,
    G_fft: np.ndarray,
    Nx: int,
    Ny: int
) -> np.ndarray:
    """
    Apply internal operator to (τ * E) using FFT convolution.

    This computes A @ (τ * E) efficiently via:
        1. Zero-pad (τ * E) to extended grid
        2. FFT
        3. Multiply by G_fft
        4. Inverse FFT
        5. Extract valid region

    Args:
        tau_E: Product of contrast and field (Ny, Nx)
        G_fft: FFT of Toeplitz Green's function
        Nx: Number of x grid points
        Ny: Number of y grid points

    Returns:
        Result of A @ (τ * E) shaped (Ny, Nx)
    """
    Ny_ext, Nx_ext = G_fft.shape

    # Zero-pad input to extended grid
    tau_E_ext = np.zeros((Ny_ext, Nx_ext), dtype=complex)
    tau_E_ext[:Ny, :Nx] = tau_E

    # FFT convolution
    tau_E_fft = np.fft.fft2(tau_E_ext)
    result_fft = tau_E_fft * G_fft
    result_ext = np.fft.ifft2(result_fft)

    # Extract valid region (first Ny×Nx of the result)
    result = result_ext[:Ny, :Nx]

    return result
