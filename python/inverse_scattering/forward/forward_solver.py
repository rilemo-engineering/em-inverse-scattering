"""
Forward scattering solver - Python port of forward_solver.p

This module orchestrates the complete forward scattering problem:
1. Set up scenario parameters
2. Generate scatterer profile
3. Compute incident fields for all transmitters
4. Solve for total field using CGFFT (Lippmann-Schwinger equation)
5. Compute scattered field at measurement points

MATLAB equivalent: forward_solver.p

The forward problem is:
    E_tot(r) = E_inc(r) + k² ∫∫ G(r,r') τ(r') E_tot(r') dr'

    E_scat(r_m) = k² ∫∫ G(r_m,r') τ(r') E_tot(r') dr'
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass
from scipy import special

from inverse_scattering.core.constants import C, EPSILON_0
from inverse_scattering.core.utils import (
    compute_wavenumber,
    compute_wavelength,
    create_grid,
    compute_dof
)
from inverse_scattering.forward.profiles import create_circular_profile
from inverse_scattering.forward.incident_field import (
    setup_transmitters,
    compute_incident_field_all_views
)
from inverse_scattering.forward.cgfft import cgfft_solve_all_views


@dataclass
class ForwardSolverResult:
    """Result of forward solver containing all scenario data."""
    Escat: np.ndarray          # Scattered field data matrix (Nm × Nv)
    PROF: np.ndarray           # Contrast profile (Ny × Nx)
    Einc_domain: np.ndarray    # Incident field on DoI (Ny × Nx × Nv)
    Etot_domain: np.ndarray    # Total field on DoI (Ny × Nx × Nv)
    freq: float                # Frequency [Hz]
    lx: float                  # DoI x-dimension [m]
    ly: float                  # DoI y-dimension [m]
    eb: float                  # Background permittivity
    sb: float                  # Background conductivity [S/m]
    Nx: int                    # Number of x cells
    Ny: int                    # Number of y cells
    Rm: float                  # Measurement radius [m]
    DOF: int                   # Degrees of freedom
    Rv: float                  # Transmitter radius [m]
    Nm: int                    # Number of receivers
    Nv: int                    # Number of transmitters
    lambda0: float             # Wavelength [m]
    k: complex                 # Wavenumber
    xvec: np.ndarray           # x-coordinates
    yvec: np.ndarray           # y-coordinates
    X: np.ndarray              # x-meshgrid
    Y: np.ndarray              # y-meshgrid
    dx: float                  # Cell size x
    dy: float                  # Cell size y


def forward_solver(
    n_iter: int = 1000,
    # Optional parameters to override defaults
    freq: Optional[float] = None,
    lx: Optional[float] = None,
    ly: Optional[float] = None,
    Nx: Optional[int] = None,
    Ny: Optional[int] = None,
    eb: float = 1.0,
    sb: float = 0.0,
    Rm: Optional[float] = None,
    Rv: Optional[float] = None,
    Nm: Optional[int] = None,
    Nv: Optional[int] = None,
    # Profile parameters
    target_center: Tuple[float, float] = (0.0, 0.0),
    target_radius: Optional[float] = None,
    target_epsilon_r: float = 2.0,
    target_sigma: float = 0.0,
    # CGFFT parameters
    tol: float = 1e-6,
    verbose: bool = False
) -> ForwardSolverResult:
    """
    Solve the forward scattering problem.

    MATLAB equivalent: forward_solver(n_iter)

    This function sets up the complete forward problem with default parameters
    matching the MATLAB exercises, computes the total field using CGFFT,
    and evaluates the scattered field at measurement points.

    Args:
        n_iter: Maximum CGFFT iterations
        freq: Frequency [Hz] (default: 300 MHz)
        lx: DoI x-dimension [m] (default: 1.0)
        ly: DoI y-dimension [m] (default: same as lx)
        Nx: Number of x cells (default: 32)
        Ny: Number of y cells (default: same as Nx)
        eb: Background relative permittivity (default: 1.0, free space)
        sb: Background conductivity [S/m] (default: 0.0)
        Rm: Measurement radius [m] (default: 3.0)
        Rv: Transmitter radius [m] (default: same as Rm)
        Nm: Number of receivers (default: computed from DOF)
        Nv: Number of transmitters (default: same as Nm)
        target_center: (x0, y0) target center [m]
        target_radius: Target radius [m] (default: lambda0/4)
        target_epsilon_r: Target relative permittivity
        target_sigma: Target conductivity [S/m]
        tol: CGFFT convergence tolerance
        verbose: Print progress information

    Returns:
        ForwardSolverResult with all scenario data
    """
    # ========================================
    # Default parameters (matching MATLAB)
    # ========================================
    if freq is None:
        freq = 300e6  # 300 MHz

    lambda0 = compute_wavelength(freq)

    if lx is None:
        lx = 1.0  # 1 meter DoI
    if ly is None:
        ly = lx

    if Nx is None:
        Nx = 32
    if Ny is None:
        Ny = Nx

    if Rm is None:
        Rm = 3.0  # 3 meters
    if Rv is None:
        Rv = Rm

    # Compute degrees of freedom
    # DOF = 2 * beta * a, where a = sqrt(2) * lx/2 (diagonal of square DoI)
    DOF = compute_dof(lx, freq, ly)

    if Nm is None:
        Nm = DOF  # Round up DOF
    if Nv is None:
        Nv = Nm

    if target_radius is None:
        target_radius = lambda0 / 4  # Quarter wavelength

    if verbose:
        print("=" * 60)
        print("Forward Solver")
        print("=" * 60)
        print(f"  Frequency: {freq/1e6:.1f} MHz")
        print(f"  Wavelength: {lambda0:.4f} m")
        print(f"  DoI size: {lx} x {ly} m")
        print(f"  Grid: {Nx} x {Ny}")
        print(f"  DOF: {DOF}")
        print(f"  Nm (receivers): {Nm}")
        print(f"  Nv (transmitters): {Nv}")
        print(f"  Rm: {Rm} m, Rv: {Rv} m")

    # ========================================
    # Create grid
    # ========================================
    X, Y, xvec, yvec, dx, dy = create_grid(lx, ly, Nx, Ny)

    # ========================================
    # Compute wavenumber
    # ========================================
    k = compute_wavenumber(freq, eb, sb)

    # ========================================
    # Create profile (contrast function)
    # ========================================
    # MATLAB: Profili.p creates the profile
    PROF = create_circular_profile(
        X, Y,
        center=target_center,
        radius=target_radius,
        epsilon_r=target_epsilon_r,
        epsilon_b=eb,
        sigma=target_sigma,
        freq=freq
    )

    if verbose:
        print(f"\nProfile created:")
        print(f"  Target: circle at ({target_center[0]}, {target_center[1]})")
        print(f"  Radius: {target_radius:.4f} m ({target_radius/lambda0:.2f} λ)")
        print(f"  εr = {target_epsilon_r}, σ = {target_sigma} S/m")
        print(f"  Max |τ|: {np.max(np.abs(PROF)):.4f}")

    # ========================================
    # Set up transmitters and compute incident field
    # ========================================
    tx_positions = setup_transmitters(Nv, Rv)
    Einc_domain = compute_incident_field_all_views(X, Y, k, tx_positions, 'line')

    if verbose:
        print(f"\nIncident field computed:")
        print(f"  Shape: {Einc_domain.shape}")

    # ========================================
    # Solve for total field using CGFFT
    # ========================================
    if verbose:
        print(f"\nSolving for total field (CGFFT)...")

    Etot_domain = cgfft_solve_all_views(
        Einc_domain, PROF, k, dx, dy,
        max_iter=n_iter, tol=tol, verbose=verbose
    )

    if verbose:
        print(f"  Total field computed: shape = {Etot_domain.shape}")

    # ========================================
    # Compute scattered field at measurement points
    # ========================================
    Escat = compute_scattered_field(
        PROF, Etot_domain, X, Y, k, dx, dy, Nm, Rm
    )

    if verbose:
        print(f"\nScattered field computed:")
        print(f"  Escat shape: {Escat.shape}")
        print(f"  Max |Escat|: {np.max(np.abs(Escat)):.6e}")

    return ForwardSolverResult(
        Escat=Escat,
        PROF=PROF,
        Einc_domain=Einc_domain,
        Etot_domain=Etot_domain,
        freq=freq,
        lx=lx,
        ly=ly,
        eb=eb,
        sb=sb,
        Nx=Nx,
        Ny=Ny,
        Rm=Rm,
        DOF=DOF,
        Rv=Rv,
        Nm=Nm,
        Nv=Nv,
        lambda0=lambda0,
        k=k,
        xvec=xvec,
        yvec=yvec,
        X=X,
        Y=Y,
        dx=dx,
        dy=dy
    )


def compute_scattered_field(
    tau: np.ndarray,
    Etot_domain: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    k: complex,
    dx: float,
    dy: float,
    Nm: int,
    Rm: float
) -> np.ndarray:
    """
    Compute scattered field at measurement points.

    The scattered field at receiver position r_m is:
        E_scat(r_m) = k² ∫∫ G(r_m, r') τ(r') E_tot(r') dr'

    In discrete form:
        E_scat[m, v] = k² Σ_n G(r_m, r_n) τ[n] E_tot[n, v] * cell_area

    Args:
        tau: Contrast profile (Ny × Nx)
        Etot_domain: Total field (Ny × Nx × Nv)
        X: x-meshgrid (Ny × Nx)
        Y: y-meshgrid (Ny × Nx)
        k: Wavenumber
        dx, dy: Cell sizes [m]
        Nm: Number of measurement points (receivers)
        Rm: Measurement radius [m]

    Returns:
        Escat: Scattered field matrix (Nm × Nv)
    """
    Ny, Nx, Nv = Etot_domain.shape
    cell_area = dx * dy

    # Measurement positions (receivers on circle)
    meas_theta = np.linspace(0, 2*np.pi - 2*np.pi/Nm, Nm)
    rx_x = Rm * np.cos(meas_theta)
    rx_y = Rm * np.sin(meas_theta)

    # Flatten grid
    x_flat = X.ravel()
    y_flat = Y.ravel()
    tau_flat = tau.ravel()

    # Initialize scattered field matrix
    Escat = np.zeros((Nm, Nv), dtype=complex)

    # For each receiver
    for m in range(Nm):
        # Distance from all DoI cells to receiver m
        R = np.sqrt((rx_x[m] - x_flat)**2 + (rx_y[m] - y_flat)**2)

        # Green's function from DoI to receiver
        G_m = (1j / 4) * special.hankel1(0, k * R)

        # For each transmitter view
        for v in range(Nv):
            Etot_v = Etot_domain[:, :, v].ravel()

            # Scattered field: k² * Σ G(r_m, r_n) * τ[n] * E_tot[n, v] * cell_area
            Escat[m, v] = k**2 * np.sum(G_m * tau_flat * Etot_v) * cell_area

    return Escat


def forward_solver_with_profile(
    PROF: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    freq: float,
    lx: float,
    ly: float,
    eb: float = 1.0,
    sb: float = 0.0,
    Rm: float = 3.0,
    Rv: Optional[float] = None,
    Nm: Optional[int] = None,
    Nv: Optional[int] = None,
    n_iter: int = 1000,
    tol: float = 1e-6,
    verbose: bool = False
) -> ForwardSolverResult:
    """
    Forward solver with user-provided profile.

    This is useful when you have a custom profile and want to compute
    the scattered field without regenerating the profile.

    Args:
        PROF: Contrast profile (Ny × Nx)
        X, Y: Meshgrids (Ny × Nx)
        freq: Frequency [Hz]
        lx, ly: DoI dimensions [m]
        eb: Background permittivity
        sb: Background conductivity [S/m]
        Rm: Measurement radius [m]
        Rv: Transmitter radius [m] (default: Rm)
        Nm: Number of receivers
        Nv: Number of transmitters
        n_iter: Maximum CGFFT iterations
        tol: Convergence tolerance
        verbose: Print progress

    Returns:
        ForwardSolverResult
    """
    Ny, Nx = PROF.shape

    if Rv is None:
        Rv = Rm

    lambda0 = compute_wavelength(freq)
    DOF = compute_dof(lx, freq, ly)

    if Nm is None:
        Nm = DOF
    if Nv is None:
        Nv = Nm

    # Grid parameters
    dx = lx / Nx
    dy = ly / Ny
    xvec = np.linspace(-lx/2 + dx/2, lx/2 - dx/2, Nx)
    yvec = np.linspace(-ly/2 + dy/2, ly/2 - dy/2, Ny)

    # Wavenumber
    k = compute_wavenumber(freq, eb, sb)

    # Transmitters and incident field
    tx_positions = setup_transmitters(Nv, Rv)
    Einc_domain = compute_incident_field_all_views(X, Y, k, tx_positions, 'line')

    if verbose:
        print(f"Forward solver (custom profile)")
        print(f"  Grid: {Nx} x {Ny}, DOF: {DOF}")
        print(f"  Nm: {Nm}, Nv: {Nv}")

    # Solve for total field
    Etot_domain = cgfft_solve_all_views(
        Einc_domain, PROF, k, dx, dy,
        max_iter=n_iter, tol=tol, verbose=verbose
    )

    # Compute scattered field
    Escat = compute_scattered_field(
        PROF, Etot_domain, X, Y, k, dx, dy, Nm, Rm
    )

    return ForwardSolverResult(
        Escat=Escat,
        PROF=PROF,
        Einc_domain=Einc_domain,
        Etot_domain=Etot_domain,
        freq=freq,
        lx=lx,
        ly=ly,
        eb=eb,
        sb=sb,
        Nx=Nx,
        Ny=Ny,
        Rm=Rm,
        DOF=DOF,
        Rv=Rv,
        Nm=Nm,
        Nv=Nv,
        lambda0=lambda0,
        k=k,
        xvec=xvec,
        yvec=yvec,
        X=X,
        Y=Y,
        dx=dx,
        dy=dy
    )
