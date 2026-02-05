"""
Incident field computation for forward scattering problems.

The incident field E_inc is the electromagnetic field that would exist
in the absence of the scatterer. For 2D TM problems, this is typically:
- A plane wave: E_inc = exp(jk·r)
- A cylindrical wave from a line source: E_inc = (i/4) * H_0^(1)(k|r - r_s|)

The MATLAB code uses cylindrical wave illumination from transmitters
positioned on a circle around the Domain of Investigation.
"""

import numpy as np
from typing import Tuple
from scipy import special

from inverse_scattering.core.constants import EPSILON_0, MU_0
from inverse_scattering.core.utils import compute_wavenumber


def compute_incident_field_plane_wave(
    X: np.ndarray,
    Y: np.ndarray,
    k: complex,
    direction: Tuple[float, float]
) -> np.ndarray:
    """
    Compute plane wave incident field.

    E_inc(r) = exp(jk · r) = exp(jk(x*cos(θ) + y*sin(θ)))

    where θ is the direction of propagation.

    Args:
        X: 2D meshgrid of x-coordinates (Ny × Nx)
        Y: 2D meshgrid of y-coordinates (Ny × Nx)
        k: Wavenumber (complex)
        direction: (kx, ky) direction vector or angle θ in radians

    Returns:
        Incident field on the grid (Ny × Nx)
    """
    if isinstance(direction, (int, float)):
        # Interpret as angle
        theta = direction
        kx = np.cos(theta)
        ky = np.sin(theta)
    else:
        kx, ky = direction
        # Normalize
        norm = np.sqrt(kx**2 + ky**2)
        kx, ky = kx/norm, ky/norm

    # E_inc = exp(jk(x*kx + y*ky))
    E_inc = np.exp(1j * k * (X * kx + Y * ky))

    return E_inc


def compute_incident_field_line_source(
    X: np.ndarray,
    Y: np.ndarray,
    k: complex,
    source_position: Tuple[float, float]
) -> np.ndarray:
    """
    Compute incident field from a line source (cylindrical wave).

    E_inc(r) = (i/4) * H_0^(1)(k|r - r_s|)

    This is the field produced by an infinite line current at r_s.

    Args:
        X: 2D meshgrid of x-coordinates (Ny × Nx)
        Y: 2D meshgrid of y-coordinates (Ny × Nx)
        k: Wavenumber (complex)
        source_position: (x_s, y_s) position of line source

    Returns:
        Incident field on the grid (Ny × Nx)
    """
    x_s, y_s = source_position

    # Distance from source to each grid point
    R = np.sqrt((X - x_s)**2 + (Y - y_s)**2)

    # Avoid division by zero at source location
    R = np.maximum(R, 1e-10)

    # Hankel function of first kind, order 0
    H0 = special.hankel1(0, k * R)

    # Incident field: E_inc = (i/4) * H_0^(1)(k*R)
    E_inc = (1j / 4) * H0

    return E_inc


def compute_incident_field_all_views(
    X: np.ndarray,
    Y: np.ndarray,
    k: complex,
    tx_positions: np.ndarray,
    source_type: str = 'line'
) -> np.ndarray:
    """
    Compute incident fields for all transmitter positions.

    MATLAB equivalent (from forward_solver.p):
        Einc_domain: Incident field on the RoI. Dimension: Ny x Nx x Nv

    Args:
        X: 2D meshgrid of x-coordinates (Ny × Nx)
        Y: 2D meshgrid of y-coordinates (Ny × Nx)
        k: Wavenumber (complex)
        tx_positions: Transmitter positions (Nv × 2) array of [x, y] coords
        source_type: 'line' for cylindrical wave, 'plane' for plane wave

    Returns:
        Incident field array (Ny × Nx × Nv)
    """
    Ny, Nx = X.shape
    Nv = len(tx_positions)

    Einc_domain = np.zeros((Ny, Nx, Nv), dtype=complex)

    for kv in range(Nv):
        if source_type == 'line':
            Einc_domain[:, :, kv] = compute_incident_field_line_source(
                X, Y, k, tx_positions[kv]
            )
        elif source_type == 'plane':
            # For plane wave, use angle from origin to Tx position
            theta = np.arctan2(tx_positions[kv, 1], tx_positions[kv, 0])
            # Wave propagates toward the object (opposite direction)
            Einc_domain[:, :, kv] = compute_incident_field_plane_wave(
                X, Y, k, theta + np.pi
            )
        else:
            raise ValueError(f"Unknown source type: {source_type}")

    return Einc_domain


def setup_transmitters(
    n_tx: int,
    radius: float,
    full_circle: bool = True
) -> np.ndarray:
    """
    Set up transmitter positions on a circle.

    MATLAB equivalent:
        meas_pos_theta = linspace(0, 2*pi - 2*pi/Nm, Nm)
        tx_x = Rm * cos(meas_pos_theta)
        tx_y = Rm * sin(meas_pos_theta)

    Args:
        n_tx: Number of transmitters
        radius: Radius of transmitter circle [m]
        full_circle: If True, span [0, 2π) without overlap

    Returns:
        Transmitter positions (n_tx × 2) array
    """
    if full_circle:
        theta = np.linspace(0, 2*np.pi - 2*np.pi/n_tx, n_tx)
    else:
        theta = np.linspace(0, 2*np.pi, n_tx, endpoint=False)

    tx_x = radius * np.cos(theta)
    tx_y = radius * np.sin(theta)

    return np.column_stack([tx_x, tx_y])


def setup_receivers(
    n_rx: int,
    radius: float,
    full_circle: bool = True
) -> np.ndarray:
    """
    Set up receiver positions on a circle.

    Same as setup_transmitters but named for clarity.

    Args:
        n_rx: Number of receivers
        radius: Radius of receiver circle [m]
        full_circle: If True, span [0, 2π) without overlap

    Returns:
        Receiver positions (n_rx × 2) array
    """
    return setup_transmitters(n_rx, radius, full_circle)
