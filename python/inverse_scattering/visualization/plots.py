"""
Visualization functions for inverse scattering exercises.

This module provides plotting functions that replicate the MATLAB figures
from the exercises, using matplotlib.

MATLAB equivalents: Various figure() and plotting commands in the scripts.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from typing import Optional, Tuple, List, Union


def setup_figure_style():
    """Set up matplotlib style to match MATLAB appearance."""
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.grid': True,
        'font.size': 12,
        'figure.figsize': (10, 8),
    })


def plot_scenario(
    xvec: np.ndarray,
    yvec: np.ndarray,
    PROF: np.ndarray,
    Rm: float,
    meas_positions: np.ndarray,
    title: str = 'Simulated Scenario',
    fignum: int = 1
) -> plt.Figure:
    """
    Plot the scenario overview with DoI, object, and measurement points.

    MATLAB equivalent (c1_Scenario.m Figure 1):
        figure(1), clf, set(gcf,'color','w'), hold on, box on, grid on
        imagesc(xvec, yvec, abs(PROF)), colormap(flipud(gray))
        plot(Rm*cos(linspace(0,2*pi,100)), Rm*sin(linspace(0,2*pi,100)), '--k')
        plot(Rm*cos(meas_pos_theta), Rm*sin(meas_pos_theta), '.r', 'markersize', 20)

    Args:
        xvec: x-coordinates (Nx,)
        yvec: y-coordinates (Ny,)
        PROF: Contrast profile (Ny × Nx)
        Rm: Measurement radius [m]
        meas_positions: Measurement positions (Nm × 2)
        title: Figure title
        fignum: Figure number

    Returns:
        matplotlib Figure object
    """
    fig = plt.figure(fignum, figsize=(8, 8))
    fig.clf()
    ax = fig.add_subplot(111)

    # Plot object profile
    im = ax.imshow(
        np.abs(PROF),
        extent=[xvec[0], xvec[-1], yvec[0], yvec[-1]],
        origin='lower',
        cmap='gray_r',
        aspect='equal'
    )

    # DoI boundary
    lx = xvec[-1] - xvec[0] + (xvec[1] - xvec[0])
    ly = yvec[-1] - yvec[0] + (yvec[1] - yvec[0])
    rect_x = [-lx/2, lx/2, lx/2, -lx/2, -lx/2]
    rect_y = [-ly/2, -ly/2, ly/2, ly/2, -ly/2]
    ax.plot(rect_x, rect_y, 'k-', linewidth=2, label='RoI')

    # Measurement circle
    theta_circle = np.linspace(0, 2*np.pi, 100)
    ax.plot(Rm*np.cos(theta_circle), Rm*np.sin(theta_circle),
            '--k', linewidth=1, label='Measurement Surface')

    # Measurement points
    ax.plot(meas_positions[:, 0], meas_positions[:, 1],
            'r.', markersize=15, label='Measurement Points')

    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title(title)
    ax.legend(loc='best')
    ax.set_aspect('equal')
    ax.grid(True)

    plt.tight_layout()
    return fig


def plot_profile(
    xvec: np.ndarray,
    yvec: np.ndarray,
    PROF: np.ndarray,
    lambda0: float,
    fignum: int = 2
) -> plt.Figure:
    """
    Plot contrast profile (real and imaginary parts).

    MATLAB equivalent (c1_Scenario.m Figure 2):
        subplot(1,2,1), imagesc(xvec/lambda0, yvec/lambda0, real(PROF)), colorbar
        subplot(1,2,2), imagesc(xvec/lambda0, yvec/lambda0, imag(PROF)), colorbar

    Args:
        xvec: x-coordinates [m]
        yvec: y-coordinates [m]
        PROF: Contrast profile (Ny × Nx)
        lambda0: Wavelength [m]
        fignum: Figure number

    Returns:
        matplotlib Figure object
    """
    fig = plt.figure(fignum, figsize=(12, 5))
    fig.clf()

    # Normalized coordinates
    xvec_norm = xvec / lambda0
    yvec_norm = yvec / lambda0

    # Real part
    ax1 = fig.add_subplot(1, 2, 1)
    im1 = ax1.imshow(
        np.real(PROF),
        extent=[xvec_norm[0], xvec_norm[-1], yvec_norm[0], yvec_norm[-1]],
        origin='lower',
        aspect='equal'
    )
    plt.colorbar(im1, ax=ax1)
    ax1.set_xlabel(r'$x/\lambda_0$')
    ax1.set_ylabel(r'$y/\lambda_0$')
    ax1.set_title(r'Re[$\tau$]')

    # Imaginary part
    ax2 = fig.add_subplot(1, 2, 2)
    im2 = ax2.imshow(
        np.imag(PROF),
        extent=[xvec_norm[0], xvec_norm[-1], yvec_norm[0], yvec_norm[-1]],
        origin='lower',
        aspect='equal'
    )
    plt.colorbar(im2, ax=ax2)
    ax2.set_xlabel(r'$x/\lambda_0$')
    ax2.set_ylabel(r'$y/\lambda_0$')
    ax2.set_title(r'Im[$\tau$]')

    plt.tight_layout()
    return fig


def plot_mvms_matrix(
    Escat: np.ndarray,
    fignum: int = 3
) -> plt.Figure:
    """
    Plot the MVMS (Multi-View Multi-Static) data matrix.

    MATLAB equivalent (c1_Scenario.m Figure 3):
        imagesc(1:1:Nv, 1:1:Nm, abs(Escat)), colorbar
        xlabel('nv'), ylabel('nm'), title('MVMS Data Matrix (amplitude)')

    Args:
        Escat: Scattered field matrix (Nm × Nv)
        fignum: Figure number

    Returns:
        matplotlib Figure object
    """
    fig = plt.figure(fignum, figsize=(8, 6))
    fig.clf()
    ax = fig.add_subplot(111)

    Nm, Nv = Escat.shape
    im = ax.imshow(
        np.abs(Escat),
        extent=[0.5, Nv+0.5, 0.5, Nm+0.5],
        origin='lower',
        aspect='equal'
    )
    plt.colorbar(im, ax=ax)
    ax.set_xlabel('$n_v$ (transmitter)')
    ax.set_ylabel('$n_m$ (receiver)')
    ax.set_title('MVMS Data Matrix\n(amplitude)')

    plt.tight_layout()
    return fig


def plot_singular_values(
    singular_values: np.ndarray,
    threshold_db: Optional[float] = None,
    truncation_index: Optional[int] = None,
    fignum: int = 1
) -> plt.Figure:
    """
    Plot normalized singular value spectrum.

    MATLAB equivalent (c2_Inversion_BORN.m Figure 1):
        plot(20*log10(norm_sing_val), 'b', 'linewidth', 2)
        hold on, plot(ones(1,100)*Nt, linspace(-100, threshold_dB, 100), '--r')

    Args:
        singular_values: Array of singular values
        threshold_db: Truncation threshold in dB (optional)
        truncation_index: Truncation index (optional)
        fignum: Figure number

    Returns:
        matplotlib Figure object
    """
    fig = plt.figure(fignum, figsize=(10, 6))
    fig.clf()
    ax = fig.add_subplot(111)

    # Normalize and convert to dB
    norm_sv = np.abs(singular_values) / np.abs(singular_values[0])
    sv_db = 20 * np.log10(norm_sv + 1e-15)

    ax.plot(range(1, len(sv_db)+1), sv_db, 'b-', linewidth=2)

    # Add threshold lines if provided
    if threshold_db is not None and truncation_index is not None:
        ax.axhline(y=threshold_db, color='r', linestyle='--', linewidth=1.5)
        ax.axvline(x=truncation_index, color='r', linestyle='--', linewidth=1.5)
        ax.legend([f'Truncation index: {truncation_index} @ {threshold_db}dB'])

    ax.set_xlabel('n')
    ax.set_ylabel('Normalized Singular Values [dB]')
    ax.set_title('Singular Value Spectrum')
    ax.set_ylim([-80, 1])
    ax.grid(True)

    plt.tight_layout()
    return fig


def plot_reconstruction_comparison(
    xvec: np.ndarray,
    yvec: np.ndarray,
    PROF_true: np.ndarray,
    PROF_rec: np.ndarray,
    lambda0: float,
    fignum: int = 2
) -> plt.Figure:
    """
    Plot comparison between true and reconstructed profiles.

    MATLAB equivalent (c2_Inversion_BORN.m Figure 2):
        subplot(2,2,1), imagesc(..., real(PROF))
        subplot(2,2,2), imagesc(..., imag(PROF))
        subplot(2,2,3), imagesc(..., real(PROF_rec_BORN))
        subplot(2,2,4), imagesc(..., imag(PROF_rec_BORN))

    Args:
        xvec: x-coordinates [m]
        yvec: y-coordinates [m]
        PROF_true: True contrast profile
        PROF_rec: Reconstructed contrast profile
        lambda0: Wavelength [m]
        fignum: Figure number

    Returns:
        matplotlib Figure object
    """
    fig = plt.figure(fignum, figsize=(12, 10))
    fig.clf()

    # Normalized coordinates
    xvec_norm = xvec / lambda0
    yvec_norm = yvec / lambda0
    extent = [xvec_norm[0], xvec_norm[-1], yvec_norm[0], yvec_norm[-1]]

    # Color limits (shared for comparison)
    vmin_r = min(np.min(np.real(PROF_true)), np.min(np.real(PROF_rec)))
    vmax_r = max(np.max(np.real(PROF_true)), np.max(np.real(PROF_rec)))
    vmin_i = min(np.min(np.imag(PROF_true)), np.min(np.imag(PROF_rec)))
    vmax_i = max(np.max(np.imag(PROF_true)), np.max(np.imag(PROF_rec)))

    # True - Real
    ax1 = fig.add_subplot(2, 2, 1)
    im1 = ax1.imshow(np.real(PROF_true), extent=extent, origin='lower',
                     vmin=vmin_r, vmax=vmax_r, aspect='equal')
    plt.colorbar(im1, ax=ax1)
    ax1.set_xlabel(r'$x/\lambda_0$')
    ax1.set_ylabel(r'$y/\lambda_0$')
    ax1.set_title(r'Re[$\tau$]: actual')

    # True - Imaginary
    ax2 = fig.add_subplot(2, 2, 2)
    im2 = ax2.imshow(np.imag(PROF_true), extent=extent, origin='lower',
                     vmin=vmin_i, vmax=vmax_i, aspect='equal')
    plt.colorbar(im2, ax=ax2)
    ax2.set_xlabel(r'$x/\lambda_0$')
    ax2.set_ylabel(r'$y/\lambda_0$')
    ax2.set_title(r'Im[$\tau$]: actual')

    # Reconstructed - Real
    ax3 = fig.add_subplot(2, 2, 3)
    im3 = ax3.imshow(np.real(PROF_rec), extent=extent, origin='lower',
                     vmin=vmin_r, vmax=vmax_r, aspect='equal')
    plt.colorbar(im3, ax=ax3)
    ax3.set_xlabel(r'$x/\lambda_0$')
    ax3.set_ylabel(r'$y/\lambda_0$')
    ax3.set_title(r'Re[$\tau$]: reconstructed via BA')

    # Reconstructed - Imaginary
    ax4 = fig.add_subplot(2, 2, 4)
    im4 = ax4.imshow(np.imag(PROF_rec), extent=extent, origin='lower',
                     vmin=vmin_i, vmax=vmax_i, aspect='equal')
    plt.colorbar(im4, ax=ax4)
    ax4.set_xlabel(r'$x/\lambda_0$')
    ax4.set_ylabel(r'$y/\lambda_0$')
    ax4.set_title(r'Im[$\tau$]: reconstructed via BA')

    plt.tight_layout()
    return fig


def plot_normalized_reconstruction(
    xvec: np.ndarray,
    yvec: np.ndarray,
    PROF_true: np.ndarray,
    PROF_rec: np.ndarray,
    lambda0: float,
    fignum: int = 3
) -> plt.Figure:
    """
    Plot normalized reconstruction with true profile contour overlay.

    MATLAB equivalent (c2_Inversion_BORN.m Figure 3):
        imagesc(..., abs(PROF_rec_BORN)/max(max(abs(PROF_rec_BORN))))
        hold on, contour(..., real(PROF), 1, '--k', 'LineWidth', 2)

    Args:
        xvec: x-coordinates [m]
        yvec: y-coordinates [m]
        PROF_true: True contrast profile
        PROF_rec: Reconstructed contrast profile
        lambda0: Wavelength [m]
        fignum: Figure number

    Returns:
        matplotlib Figure object
    """
    fig = plt.figure(fignum, figsize=(8, 7))
    fig.clf()
    ax = fig.add_subplot(111)

    # Normalized coordinates
    xvec_norm = xvec / lambda0
    yvec_norm = yvec / lambda0
    extent = [xvec_norm[0], xvec_norm[-1], yvec_norm[0], yvec_norm[-1]]

    # Normalized reconstruction
    PROF_rec_norm = np.abs(PROF_rec) / np.max(np.abs(PROF_rec))

    im = ax.imshow(PROF_rec_norm, extent=extent, origin='lower', aspect='equal')
    plt.colorbar(im, ax=ax)

    # True profile contour
    X_norm, Y_norm = np.meshgrid(xvec_norm, yvec_norm)
    ax.contour(X_norm, Y_norm, np.real(PROF_true), levels=[1],
               colors='k', linestyles='--', linewidths=2)

    ax.set_xlabel(r'$x/\lambda_0$')
    ax.set_ylabel(r'$y/\lambda_0$')
    ax.set_title(r'Normalized Abs[$\tau$]: reconstructed via BA')

    plt.tight_layout()
    return fig


def plot_cross_sections(
    xvec: np.ndarray,
    yvec: np.ndarray,
    PROF_true: np.ndarray,
    PROF_rec: np.ndarray,
    cut_indices: Tuple[int, int],
    tolerance_band: Optional[Tuple[float, float]] = None,
    fignum: int = 3
) -> plt.Figure:
    """
    Plot cross-section comparisons (x-cut and y-cut).

    MATLAB equivalent (c2_Inversion_ExpData_BORN.m Figure 3):
        subplot(1,2,1), plot(xvec, real(PROF(Ny/2,:)+1), 'b')
        hold on, plot(xvec, real(PROF_rec_BORN(Ny/2,:)+1), '--r')

    Args:
        xvec: x-coordinates [m]
        yvec: y-coordinates [m]
        PROF_true: True contrast profile
        PROF_rec: Reconstructed contrast profile
        cut_indices: (y_index, x_index) for x-cut and y-cut
        tolerance_band: (lower, upper) values for tolerance lines
        fignum: Figure number

    Returns:
        matplotlib Figure object
    """
    fig = plt.figure(fignum, figsize=(12, 5))
    fig.clf()

    y_idx, x_idx = cut_indices
    Ny, Nx = PROF_true.shape

    # X-cut (along x at fixed y)
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(xvec, np.real(PROF_true[y_idx, :]) + 1, 'b-', linewidth=2, label='True')
    ax1.plot(xvec, np.real(PROF_rec[y_idx, :]) + 1, 'r--', linewidth=2, label='Reconstructed')
    if tolerance_band:
        ax1.axhline(y=tolerance_band[0], color='r', linestyle='-.', alpha=0.5)
        ax1.axhline(y=tolerance_band[1], color='r', linestyle='-.', alpha=0.5)
    ax1.set_xlabel('x [m]')
    ax1.set_ylabel(r'Re[$\varepsilon_r$]')
    ax1.set_title(f'X-cut at y={yvec[y_idx]*1000:.1f} mm')
    ax1.legend()
    ax1.grid(True)

    # Y-cut (along y at fixed x)
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(yvec, np.real(PROF_true[:, x_idx]) + 1, 'b-', linewidth=2, label='True')
    ax2.plot(yvec, np.real(PROF_rec[:, x_idx]) + 1, 'r--', linewidth=2, label='Reconstructed')
    if tolerance_band:
        ax2.axhline(y=tolerance_band[0], color='r', linestyle='-.', alpha=0.5)
        ax2.axhline(y=tolerance_band[1], color='r', linestyle='-.', alpha=0.5)
    ax2.set_xlabel('y [m]')
    ax2.set_ylabel(r'Re[$\varepsilon_r$]')
    ax2.set_title(f'Y-cut at x={xvec[x_idx]*1000:.1f} mm')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    return fig


def plot_field_animation(
    xvec: np.ndarray,
    yvec: np.ndarray,
    E_field: np.ndarray,
    lambda0: float,
    field_type: str = 'Incident',
    pause_time: float = 0.5
) -> None:
    """
    Animate field for each transmitter view.

    MATLAB equivalent (c1_Scenario.m Figure 4):
        for kv=1:Nv
            subplot(1,2,1), imagesc(..., abs(Einc_domain(:,:,kv)))
            subplot(1,2,2), imagesc(..., angle(Einc_domain(:,:,kv)))
            pause(0.5)
        end

    Args:
        xvec: x-coordinates [m]
        yvec: y-coordinates [m]
        E_field: Field array (Ny × Nx × Nv)
        lambda0: Wavelength [m]
        field_type: 'Incident' or 'Total'
        pause_time: Pause between frames in seconds
    """
    xvec_norm = xvec / lambda0
    yvec_norm = yvec / lambda0
    extent = [xvec_norm[0], xvec_norm[-1], yvec_norm[0], yvec_norm[-1]]

    Ny, Nx, Nv = E_field.shape

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for kv in range(Nv):
        ax1.clear()
        ax2.clear()

        # Amplitude
        im1 = ax1.imshow(np.abs(E_field[:, :, kv]), extent=extent,
                         origin='lower', aspect='equal')
        ax1.set_xlabel(r'$x/\lambda_0$')
        ax1.set_ylabel(r'$y/\lambda_0$')
        ax1.set_title(f'Amplitude of {field_type} Field [nv={kv+1}]')

        # Phase
        im2 = ax2.imshow(np.angle(E_field[:, :, kv]), extent=extent,
                         origin='lower', aspect='equal', cmap='hsv')
        ax2.set_xlabel(r'$x/\lambda_0$')
        ax2.set_ylabel(r'$y/\lambda_0$')
        ax2.set_title(f'Phase of {field_type} Field [nv={kv+1}]')

        plt.tight_layout()
        plt.pause(pause_time)

    plt.show()


def show():
    """Display all figures."""
    plt.show()
