"""
Scattering kernel (operator) builder for inverse scattering.

The scattering kernel S maps the contrast function τ to the scattered field
at measurement points under the Born approximation:

    E_scat = S @ τ

where S incorporates the incident field, Green's function, and measurement geometry.

For Born approximation: E_tot ≈ E_inc, so:
    S_mn = k₀² * G(r_m, r_n) * E_inc(r_n) * dx * dy

MATLAB equivalent: kernel_scattering.p, kernel_scattering_exp.p
"""

import numpy as np
from typing import Tuple, Optional
from scipy import special

from inverse_scattering.core.constants import EPSILON_0, MU_0
from inverse_scattering.core.utils import compute_wavenumber


def build_scattering_kernel(
    Etot_approx: np.ndarray,
    Nx: int,
    Ny: int,
    lx: float,
    ly: float,
    n_views: int,
    eb: float,
    sb: float,
    freq: float,
    Nm: int,
    Rm: float,
    Rv: Optional[float] = None
) -> np.ndarray:
    """
    Build the Born scattering operator matrix.

    The scattering operator S maps contrast τ to scattered field E_scat:
        E_scat = S @ τ (vectorized)

    Matrix form: S is (Nm*Nv) × (Nx*Ny)

    Each row of S corresponds to one measurement (receiver m, transmitter v).
    Each column corresponds to one DoI cell.

    S[m*Nv + v, n] = k² * G(r_m, r_n) * E_tot_approx[n, v] * dx * dy

    MATLAB equivalent:
        S_BORN = kernel_scattering(Etot_approx_BORN, Nx, Ny, lx, ly, 1, eb, sb, freq, Nm, Rm)

    Args:
        Etot_approx: Approximate total field (Ny × Nx × Nv)
                     For Born: E_tot ≈ E_inc
        Nx: Number of x grid points
        Ny: Number of y grid points
        lx: DoI x-dimension [m]
        ly: DoI y-dimension [m]
        n_views: Number of views (transmitters), should match Etot_approx.shape[2]
        eb: Background relative permittivity
        sb: Background conductivity [S/m]
        freq: Frequency [Hz]
        Nm: Number of measurement points (receivers)
        Rm: Measurement radius [m]
        Rv: Transmitter radius [m] (default: same as Rm)

    Returns:
        Scattering operator S of shape (Nm × Nv) × (Nx × Ny)
        When flattened properly: (Nm*Nv, Nx*Ny)
    """
    if Rv is None:
        Rv = Rm

    # Grid parameters
    dx = lx / Nx
    dy = ly / Ny
    cell_area = dx * dy

    # Cell-centered coordinates
    xvec = np.linspace(-lx/2 + dx/2, lx/2 - dx/2, Nx)
    yvec = np.linspace(-ly/2 + dy/2, ly/2 - dy/2, Ny)
    X, Y = np.meshgrid(xvec, yvec)

    # Wavenumber in background medium
    k = compute_wavenumber(freq, eb, sb)

    # Measurement positions (receivers on circle)
    meas_theta = np.linspace(0, 2*np.pi - 2*np.pi/Nm, Nm)
    rx_x = Rm * np.cos(meas_theta)
    rx_y = Rm * np.sin(meas_theta)

    # Get number of views from input
    Nv = Etot_approx.shape[2]

    # Total number of measurements and unknowns
    n_meas_total = Nm * Nv
    n_unknowns = Nx * Ny

    # Initialize scattering operator
    S = np.zeros((n_meas_total, n_unknowns), dtype=complex)

    # Flatten grid coordinates - use Fortran order to match MATLAB's column-major convention
    # MATLAB: X(:), Y(:) vectorize column-by-column
    x_flat = X.ravel(order='F')  # (N,) where N = Nx*Ny
    y_flat = Y.ravel(order='F')

    # Build operator row by row
    # Each row corresponds to one (receiver, view) combination
    for v in range(Nv):
        # Get E_tot for this view and flatten - Fortran order for MATLAB compatibility
        E_v = Etot_approx[:, :, v].ravel(order='F')  # (N,)

        for m in range(Nm):
            # Row index in S
            row_idx = m + v * Nm  # or v + m*Nv depending on MATLAB ordering

            # Actually MATLAB uses: Escat(m, v), so likely row = m + v*Nm
            # But for vectorized Escat(:), the ordering might differ
            # Let's use standard ordering: row = v*Nm + m (all receivers for view 0, then view 1, etc.)
            row_idx = v * Nm + m

            # Distance from receiver to each DoI cell
            R_m = np.sqrt((rx_x[m] - x_flat)**2 + (rx_y[m] - y_flat)**2)

            # Green's function from DoI cells to receiver m
            # MATLAB uses exp(-jωt) time convention, which requires H_0^(2) instead of H_0^(1)
            # G(r_m, r_n) = -(j/4) * H_0^(2)(k * |r_m - r_n|)
            # Equivalently: G = conj((j/4) * H_0^(1)(k * r))
            # Using H_0^(2) directly for clarity:
            G_m = -(1j / 4) * special.hankel2(0, k * R_m)

            # S row: S[row, n] = k² * G(r_m, r_n) * E_tot(r_n, v) * cell_area
            S[row_idx, :] = (k**2) * G_m * E_v * cell_area

    return S


def kernel_scattering(
    Etot_approx: np.ndarray,
    Nx: int,
    Ny: int,
    lx: float,
    ly: float,
    _unused: int,  # For MATLAB compatibility (was 1 in calls)
    eb: float,
    sb: float,
    freq: float,
    Nm: int,
    Rm: float
) -> np.ndarray:
    """
    MATLAB-compatible interface for scattering kernel.

    MATLAB call:
        S_BORN = kernel_scattering(Etot_approx_BORN, Nx, Ny, lx, ly, 1, eb, sb, freq, Nm, Rm)

    Args:
        Etot_approx: Approximate total field (Ny × Nx × Nv)
        Nx, Ny: Grid dimensions
        lx, ly: DoI dimensions [m]
        _unused: Unused parameter (was '1' in MATLAB)
        eb: Background permittivity
        sb: Background conductivity
        freq: Frequency [Hz]
        Nm: Number of receivers
        Rm: Measurement radius [m]

    Returns:
        Scattering operator S
    """
    Nv = Etot_approx.shape[2]
    return build_scattering_kernel(
        Etot_approx, Nx, Ny, lx, ly, Nv, eb, sb, freq, Nm, Rm
    )


def kernel_scattering_exp(
    Etot_approx: np.ndarray,
    Nx: int,
    Ny: int,
    lx: float,
    ly: float,
    _unused: int,
    eb: float,
    sb: float,
    freq: float,
    Nm: int,
    Rm: float
) -> np.ndarray:
    """
    Scattering kernel for experimental data.

    Same as kernel_scattering but explicitly named for experimental data.

    MATLAB call:
        S_BORN = kernel_scattering_exp(Etot_approx_BORN, Nx, Ny, lx, ly, 1, eb, sb, freq, Nm, Rm)

    Args:
        Same as kernel_scattering

    Returns:
        Scattering operator S
    """
    return kernel_scattering(
        Etot_approx, Nx, Ny, lx, ly, _unused, eb, sb, freq, Nm, Rm
    )


def apply_scattering_kernel(
    S: np.ndarray,
    tau: np.ndarray
) -> np.ndarray:
    """
    Apply scattering kernel to contrast to get scattered field.

    E_scat_vec = S @ tau_vec

    Args:
        S: Scattering operator (Nm*Nv × Nx*Ny)
        tau: Contrast profile (Ny × Nx) or flattened (Nx*Ny,)

    Returns:
        Scattered field vector (Nm*Nv,)
    """
    # Use Fortran order to match MATLAB's column-major convention
    tau_flat = tau.ravel(order='F')
    return S @ tau_flat


def reshape_escat_to_matrix(
    escat_vec: np.ndarray,
    Nm: int,
    Nv: int
) -> np.ndarray:
    """
    Reshape scattered field vector to MVMS matrix form.

    Args:
        escat_vec: Scattered field vector (Nm*Nv,)
        Nm: Number of receivers
        Nv: Number of transmitters

    Returns:
        Escat matrix (Nm × Nv)
    """
    # The vector ordering is [all receivers for view 0, all receivers for view 1, ...]
    return escat_vec.reshape((Nv, Nm)).T
