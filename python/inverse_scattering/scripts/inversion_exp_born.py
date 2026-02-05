"""
Experimental data inversion script - Python port of c2_Inversion_ExpData_BORN.m

This script performs inverse scattering reconstruction on Fresnel Institute
experimental data using Born approximation and TSVD regularization.

MATLAB equivalent: c2_Inversion_ExpData_BORN.m

Usage:
    poetry run run-exp-inversion
    # or
    python -m inverse_scattering.scripts.inversion_exp_born
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional

from inverse_scattering.core.constants import EPSILON_0, MU_0, C
from inverse_scattering.core.utils import compute_wavenumber, nmse
from inverse_scattering.inverse.scattering_kernel import kernel_scattering_exp
from inverse_scattering.inverse.tsvd import (
    compute_svd,
    find_truncation_index,
    tsvd_solve,
    suggest_threshold
)
from inverse_scattering.data.mat_io import load_mat
from inverse_scattering.visualization.plots import (
    plot_singular_values,
    plot_reconstruction_comparison,
    plot_normalized_reconstruction,
    plot_cross_sections,
    show
)


def run_inversion_exp_born(
    scenario_file: str = 'DATA_scenario_exp_singletarget.mat',
    object_file: str = 'DATA_object_exp_singletarget.mat',
    data_dir: Optional[str] = None,
    threshold_db: Optional[float] = None,
    visualize: bool = True
) -> dict:
    """
    Run Born inversion on experimental data.

    MATLAB equivalent:
        c2_Inversion_ExpData_BORN.m

    Args:
        scenario_file: Scenario data file
        object_file: Object ground truth file
        data_dir: Directory containing data files
        threshold_db: Truncation threshold in dB (None = interactive)
        visualize: Whether to display plots

    Returns:
        Dictionary with inversion results
    """
    print("=" * 60)
    print("Experimental Data Inversion (Born Approximation)")
    print("=" * 60)

    # ========================================
    # Load scenario and object data
    # ========================================
    if data_dir is None:
        data_dir = '.'
    data_path = Path(data_dir)

    print(f"\nLoading: {scenario_file}")
    try:
        scenario_data = load_mat(data_path / scenario_file)
    except FileNotFoundError:
        print(f"File not found. Creating placeholder data for demonstration.")
        # Create placeholder data
        scenario_data = _create_placeholder_scenario()

    print(f"Loading: {object_file}")
    try:
        object_data = load_mat(data_path / object_file)
    except FileNotFoundError:
        print(f"File not found. Creating placeholder object data.")
        object_data = _create_placeholder_object(scenario_data)

    # Extract variables
    Escat = scenario_data['Escat']
    Einc_domain = scenario_data['Einc_domain']
    freq = float(scenario_data['freq'])
    lx = float(scenario_data['lx'])
    ly = float(scenario_data['ly'])
    Nx = int(scenario_data['Nx'])
    Ny = int(scenario_data['Ny'])
    eb = float(scenario_data['eb'])
    sb = float(scenario_data['sb'])
    Rm = float(scenario_data['Rm'])
    xvec = scenario_data['xvec']
    yvec = scenario_data['yvec']

    # Get wavelength
    lambda0 = C / freq

    # Ground truth profile
    PROF = object_data['PROF']

    # Determine dataset type
    is_two_targets = 'x0_l' in object_data

    Nm = Escat.shape[0]
    Nv = Escat.shape[1] if Escat.ndim > 1 else 1

    print(f"\nScenario parameters:")
    print(f"  freq = {freq/1e9:.1f} GHz, λ = {lambda0*100:.2f} cm")
    print(f"  DoI: {lx*100:.1f} x {ly*100:.1f} cm, grid: {Nx}x{Ny}")
    print(f"  Nm = {Nm}, Nv = {Nv}")

    # ========================================
    # Definition of useful parameters
    # ========================================
    # MATLAB: e0=8.85e-12; m0=4*pi*1e-7;
    e0 = EPSILON_0
    m0 = MU_0

    # MATLAB: eb_eq = eb - 1i*(sb/(e0*2*pi*freq))
    eb_eq = eb - 1j * (sb / (e0 * 2 * np.pi * freq))

    # MATLAB: kb = 2*pi*freq*sqrt(e0*m0*eb_eq)
    kb = compute_wavenumber(freq, eb, sb)
    print(f"  kb = {kb:.4f} rad/m")

    # ========================================
    # Born approximation: E_tot ≈ E_inc
    # ========================================
    # MATLAB: Etot_approx_BORN = Einc_domain
    Etot_approx_BORN = Einc_domain
    data_BORN = Escat

    # ========================================
    # Build scattering kernel
    # ========================================
    print("\nBuilding scattering kernel...")
    # MATLAB: S_BORN = kernel_scattering_exp(Etot_approx_BORN, Nx, Ny, lx, ly, 1, eb, sb, freq, Nm, Rm)
    S_BORN = kernel_scattering_exp(
        Etot_approx_BORN, Nx, Ny, lx, ly, 1, eb, sb, freq, Nm, Rm
    )
    print(f"  S_BORN shape: {S_BORN.shape}")

    # ========================================
    # Compute SVD
    # ========================================
    print("\nComputing SVD (economy)...")
    # MATLAB: [U, S, V] = svd(S_BORN, 'econ')
    U, s, Vh = compute_svd(S_BORN, full_matrices=False)
    print(f"  Singular values: {len(s)}")
    print(f"  σ_max = {s[0]:.6e}, σ_min = {s[-1]:.6e}")

    # Normalized singular values
    norm_sing_val = np.abs(s) / np.abs(s[0])
    sv_db = 20 * np.log10(norm_sing_val + 1e-15)

    # ========================================
    # Truncation threshold selection
    # ========================================
    if visualize:
        # Plot singular values
        fig1 = plot_singular_values(s, fignum=1)
        plt.draw()
        plt.pause(0.5)

    if threshold_db is None:
        # Interactive threshold selection
        print("\n" + "=" * 40)
        print("SINGULAR VALUE SPECTRUM")
        print("=" * 40)
        print(f"σ_1 = {s[0]:.4e} (max)")
        for i, level in enumerate([-10, -20, -30, -40, -50]):
            idx = find_truncation_index(s, level)
            print(f"  @ {level} dB: index = {idx}")

        threshold_db = float(input("\nTruncation threshold [dB]: "))
    else:
        print(f"\nUsing threshold: {threshold_db} dB")

    # Find truncation index
    # MATLAB: [~, Nt] = min(abs(20*log10(norm_sing_val) - threshold_dB))
    Nt = find_truncation_index(s, threshold_db)
    print(f"Truncation index: {Nt}")

    # Update plot with threshold
    if visualize:
        fig1 = plot_singular_values(s, threshold_db, Nt, fignum=1)

    # ========================================
    # TSVD reconstruction
    # ========================================
    print("\nPerforming TSVD reconstruction...")
    # MATLAB: PROF_rec_BORN = TSVD_solver(U, S, V, Nt, data_BORN, Nx, Ny)
    PROF_rec_BORN = tsvd_solve(U, s, Vh, Nt, data_BORN, Nx, Ny)

    # ========================================
    # Results
    # ========================================
    # Compute NMSE (if we have ground truth)
    nmse_value = nmse(PROF, PROF_rec_BORN)
    print(f"\nNMSE = {nmse_value:.6f}")

    # ========================================
    # Visualization
    # ========================================
    if visualize:
        # Figure 2: Reconstruction panels
        # MATLAB: subplot(1,3,1), imagesc(..., real(PROF_rec_BORN))
        fig2 = plt.figure(2, figsize=(15, 5))
        fig2.clf()

        xvec_norm = xvec / lambda0
        yvec_norm = yvec / lambda0
        extent = [xvec_norm[0], xvec_norm[-1], yvec_norm[0], yvec_norm[-1]]

        ax1 = fig2.add_subplot(1, 3, 1)
        im1 = ax1.imshow(np.real(PROF_rec_BORN), extent=extent, origin='lower', aspect='equal')
        plt.colorbar(im1, ax=ax1)
        ax1.set_xlabel(r'$x/\lambda_0$')
        ax1.set_ylabel(r'$y/\lambda_0$')
        ax1.set_title(r'Re[$\tau$]: reconstructed via BA')

        ax2 = fig2.add_subplot(1, 3, 2)
        im2 = ax2.imshow(np.imag(PROF_rec_BORN), extent=extent, origin='lower', aspect='equal')
        plt.colorbar(im2, ax=ax2)
        ax2.set_xlabel(r'$x/\lambda_0$')
        ax2.set_ylabel(r'$y/\lambda_0$')
        ax2.set_title(r'Im[$\tau$]: reconstructed via BA')

        ax3 = fig2.add_subplot(1, 3, 3)
        PROF_norm = np.abs(PROF_rec_BORN) / np.max(np.abs(PROF_rec_BORN))
        im3 = ax3.imshow(PROF_norm, extent=extent, origin='lower', aspect='equal')
        plt.colorbar(im3, ax=ax3)
        # Add ground truth contour
        X_norm, Y_norm = np.meshgrid(xvec_norm, yvec_norm)
        ax3.contour(X_norm, Y_norm, np.real(PROF), levels=[1],
                   colors='k', linestyles='--', linewidths=2)
        ax3.set_xlabel(r'$x/\lambda_0$')
        ax3.set_ylabel(r'$y/\lambda_0$')
        ax3.set_title(r'Normalized Abs[$\tau$]: reconstructed via BA')

        plt.tight_layout()

        # Figure 3: Cross-sections with tolerance bands
        fig3 = plt.figure(3, figsize=(12, 5))
        fig3.clf()

        if not is_two_targets:
            # Single target: x-cut and y-cut
            y0 = float(object_data.get('y0', 0))
            x0 = float(object_data.get('x0', 0.025))

            # Find closest indices
            y_idx = np.argmin(np.abs(yvec - y0))
            x_idx = np.argmin(np.abs(xvec - x0))

            ax1 = fig3.add_subplot(1, 2, 1)
            # MATLAB: plot(xvec, real(PROF(Ny/2,:)+1), 'b')
            ax1.plot(xvec, np.real(PROF[y_idx, :]) + 1, 'b-', linewidth=2, label='True')
            ax1.plot(xvec, np.real(PROF_rec_BORN[y_idx, :]) + 1, 'r--', linewidth=2, label='Reconstructed')
            # Tolerance bands: eps_r = 2.7 to 3.3
            ax1.axhline(y=3.3, color='r', linestyle='-.', alpha=0.5)
            ax1.axhline(y=2.7, color='r', linestyle='-.', alpha=0.5)
            ax1.set_xlabel('x [m]')
            ax1.set_ylabel(r'Re[$\varepsilon_r$]')
            ax1.set_title(f'X-cut @ y={y0} m\nBA reconstruction')
            ax1.set_xlim([xvec[0], xvec[-1]])
            ax1.set_ylim([0.9, 3.4])
            ax1.legend()
            ax1.grid(True)

            ax2 = fig3.add_subplot(1, 2, 2)
            ax2.plot(yvec, np.real(PROF[:, x_idx]) + 1, 'b-', linewidth=2, label='True')
            ax2.plot(yvec, np.real(PROF_rec_BORN[:, x_idx]) + 1, 'r--', linewidth=2, label='Reconstructed')
            ax2.axhline(y=3.3, color='r', linestyle='-.', alpha=0.5)
            ax2.axhline(y=2.7, color='r', linestyle='-.', alpha=0.5)
            ax2.set_xlabel('y [m]')
            ax2.set_ylabel(r'Re[$\varepsilon_r$]')
            ax2.set_title(f'Y-cut @ x={x0} m\nBA reconstruction')
            ax2.set_xlim([yvec[0], yvec[-1]])
            ax2.set_ylim([0.9, 3.4])
            ax2.legend()
            ax2.grid(True)

        else:
            # Two targets: x-cut through both
            y0_l = float(object_data.get('y0_l', 0.015))
            y_idx = np.argmin(np.abs(yvec - y0_l))

            ax = fig3.add_subplot(1, 1, 1)
            ax.plot(xvec, np.real(PROF[y_idx, :]) + 1, 'b-', linewidth=2, label='True')
            ax.plot(xvec, np.real(PROF_rec_BORN[y_idx, :]) + 1, 'r--', linewidth=2, label='Reconstructed')
            ax.axhline(y=3.3, color='r', linestyle='-.', alpha=0.5)
            ax.axhline(y=2.7, color='r', linestyle='-.', alpha=0.5)
            ax.set_xlabel('x [m]')
            ax.set_ylabel(r'Re[$\varepsilon_r$]')
            ax.set_title(f'X-cut @ y={y0_l} m\nBA reconstruction')
            ax.set_xlim([xvec[0], xvec[-1]])
            ax.set_ylim([0.9, 3.4])
            ax.legend()
            ax.grid(True)

        plt.tight_layout()
        show()

    # Return results
    return {
        'PROF_rec_BORN': PROF_rec_BORN,
        'NMSE': nmse_value,
        'threshold_db': threshold_db,
        'Nt': Nt,
        'U': U,
        's': s,
        'Vh': Vh,
        'S_BORN': S_BORN,
    }


def _create_placeholder_scenario():
    """Create placeholder scenario data for demonstration."""
    from inverse_scattering.core.utils import create_grid
    from inverse_scattering.forward.incident_field import (
        setup_transmitters, compute_incident_field_all_views
    )

    freq = 4e9
    lx = ly = 0.15
    Nx = Ny = 64
    eb, sb = 1.0, 0.0
    Rm, Rv = 0.76135, 0.72135
    Nm, Nv = 49, 36

    X, Y, xvec, yvec, dx, dy = create_grid(lx, ly, Nx, Ny)
    k = compute_wavenumber(freq, eb, sb)
    tx_pos = setup_transmitters(Nv, Rv)
    Einc_domain = compute_incident_field_all_views(X, Y, k, tx_pos, 'line')
    Escat = np.zeros((Nm, Nv), dtype=complex)

    return {
        'Escat': Escat,
        'Einc_domain': Einc_domain,
        'freq': freq,
        'lambda0': C / freq,
        'lx': lx, 'ly': ly,
        'Nx': Nx, 'Ny': Ny,
        'eb': eb, 'sb': sb,
        'Rm': Rm, 'Rv': Rv,
        'xvec': xvec, 'yvec': yvec,
    }


def _create_placeholder_object(scenario_data):
    """Create placeholder object data."""
    from inverse_scattering.core.utils import create_grid
    from inverse_scattering.forward.profiles import create_fresnel_single_target

    lx = scenario_data['lx']
    ly = scenario_data['ly']
    Nx = int(scenario_data['Nx'])
    Ny = int(scenario_data['Ny'])

    X, Y, _, _, _, _ = create_grid(lx, ly, Nx, Ny)
    PROF, params = create_fresnel_single_target(X, Y)

    return {'PROF': PROF, **params}


def main():
    """Main entry point for CLI."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Born inversion on Fresnel experimental data'
    )
    parser.add_argument(
        '--scenario', '-s',
        default='DATA_scenario_exp_singletarget.mat',
        help='Scenario data file'
    )
    parser.add_argument(
        '--object', '-o',
        default='DATA_object_exp_singletarget.mat',
        help='Object ground truth file'
    )
    parser.add_argument(
        '--data-dir', '-d',
        default='.',
        help='Directory containing data files'
    )
    parser.add_argument(
        '--threshold', '-t',
        type=float,
        default=None,
        help='Truncation threshold in dB (e.g., -25)'
    )
    parser.add_argument(
        '--no-visualize',
        action='store_true',
        help='Disable visualization'
    )

    args = parser.parse_args()

    run_inversion_exp_born(
        scenario_file=args.scenario,
        object_file=args.object,
        data_dir=args.data_dir,
        threshold_db=args.threshold,
        visualize=not args.no_visualize
    )


if __name__ == '__main__':
    main()
