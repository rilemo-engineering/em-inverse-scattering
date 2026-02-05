"""
Born inversion script for simulated data - Python port of c2_Inversion_BORN.m

This script performs linear inversion using:
- Born approximation (E_tot ≈ E_inc)
- Truncated Singular Value Decomposition (TSVD) regularization

MATLAB equivalent: c2_Inversion_BORN.m

Usage:
    poetry run run-inversion-born
    # or
    python -m inverse_scattering.scripts.inversion_born
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Tuple

from inverse_scattering.core.constants import EPSILON_0, MU_0
from inverse_scattering.core.utils import nmse
from inverse_scattering.data.mat_io import load_mat, save_mat
from inverse_scattering.utils.noise import awgn
from inverse_scattering.inverse.scattering_kernel import kernel_scattering
from inverse_scattering.inverse.tsvd import (
    compute_svd,
    find_truncation_index,
    tsvd_solve
)
from inverse_scattering.visualization.plots import show


def run_inversion_born(
    data_file: Optional[str] = None,
    output_dir: Optional[str] = None,
    snr_db: float = 30.0,
    threshold_db: Optional[float] = None,
    aspect_limited: bool = False,
    visualize: bool = True,
    seed: int = 345,
    verbose: bool = True
) -> dict:
    """
    Run Born inversion on simulated data.

    MATLAB equivalent:
        c2_Inversion_BORN.m

    This function:
    1. Loads scenario data from DATA_scenario.mat
    2. Adds AWGN noise to scattered field
    3. Builds Born scattering kernel (using E_tot ≈ E_inc)
    4. Performs SVD of the kernel
    5. Reconstructs profile via TSVD

    Args:
        data_file: Path to DATA_scenario.mat/.npz (default: look in current dir)
        output_dir: Directory for output files
        snr_db: Signal-to-noise ratio [dB] (default: 30)
        threshold_db: Truncation threshold [dB] (default: interactive)
        aspect_limited: If True, use aspect-limited configuration
        visualize: Whether to display plots
        seed: Random seed for reproducible noise
        verbose: Print progress

    Returns:
        Dictionary with reconstruction results
    """
    if verbose:
        print("=" * 60)
        print("Born Inversion (c2_Inversion_BORN.m port)")
        print("=" * 60)

    # ========================================
    # Load scenario data
    # ========================================
    # MATLAB: load DATA_scenario.mat
    if data_file is None:
        # Try to find data file
        for ext in ['.npz', '.mat']:
            p = Path(f'DATA_scenario{ext}')
            if p.exists():
                data_file = str(p)
                break
        else:
            raise FileNotFoundError(
                "DATA_scenario.mat or DATA_scenario.npz not found. "
                "Run scenario.py first to generate the data."
            )

    if verbose:
        print(f"\nLoading: {data_file}")

    if data_file.endswith('.npz'):
        # Note: numpy's npz files are safe for scientific computing use
        data = dict(np.load(data_file, allow_pickle=True))
        # Convert 0-d arrays to scalars
        for key in data:
            if isinstance(data[key], np.ndarray) and data[key].ndim == 0:
                data[key] = data[key].item()
    else:
        data = load_mat(data_file)

    # Extract variables
    Escat_original = data['Escat']
    PROF = data['PROF']
    Einc_domain = data['Einc_domain']
    freq = float(data['freq'])
    lx = float(data['lx'])
    ly = float(data['ly'])
    eb = float(data['eb'])
    sb = float(data['sb'])
    Nx = int(data['Nx'])
    Ny = int(data['Ny'])
    Rm = float(data['Rm'])
    Nm = int(data['Nm'])
    Nv = int(data['Nv'])
    lambda0 = float(data['lambda0'])
    xvec = data['xvec']
    yvec = data['yvec']

    if verbose:
        print(f"  Grid: {Nx} x {Ny}")
        print(f"  Nm: {Nm}, Nv: {Nv}")
        print(f"  freq: {freq/1e6:.1f} MHz")

    # ========================================
    # Define useful parameters
    # ========================================
    # MATLAB: e0=8.85e-12; m0=4*pi*1e-7;
    e0 = EPSILON_0
    m0 = MU_0

    # MATLAB: eb_eq=eb-1i*(sb/(e0*2*pi*freq));
    omega = 2 * np.pi * freq
    eb_eq = eb - 1j * (sb / (e0 * omega))

    # MATLAB: kb=2*pi*freq*sqrt(e0*m0*eb_eq);
    kb = 2 * np.pi * freq * np.sqrt(e0 * m0 * eb_eq)

    if verbose:
        print(f"\n  Background wavenumber kb = {kb:.4f}")

    # ========================================
    # Add white Gaussian noise on scattered field data
    # ========================================
    # MATLAB: SNR=30; Escat=awgn(Escat,SNR,'measured',345);
    if verbose:
        print(f"\nAdding noise: SNR = {snr_db} dB")

    Escat = awgn(Escat_original, snr_db, signal_power='measured', seed=seed)

    # ========================================
    # Linear inversion via TSVD and Born approximation
    # ========================================
    # Born approximation: E_tot ≈ E_inc
    # MATLAB: Etot_approx_BORN=Einc_domain;
    Etot_approx_BORN = Einc_domain
    data_BORN = Escat

    if verbose:
        print("\nBuilding Born scattering kernel...")

    # MATLAB: S_BORN=kernel_scattering(Etot_approx_BORN,Nx,Ny,lx,ly,1,eb,sb,freq,Nm,Rm);
    S_BORN = kernel_scattering(
        Etot_approx_BORN, Nx, Ny, lx, ly, 1, eb, sb, freq, Nm, Rm
    )

    if verbose:
        print(f"  S_BORN shape: {S_BORN.shape}")

    if not aspect_limited:
        # ========================================
        # Full-aspect reconstruction
        # ========================================
        # MATLAB: [U,S,V]=svd(S_BORN);
        if verbose:
            print("\nComputing SVD...")

        U, s, Vh = compute_svd(S_BORN)

        # MATLAB: S1=diag(S); norm_sing_val=abs(S1)./(abs(S1(1)));
        norm_sing_val = np.abs(s) / np.abs(s[0])

        if verbose:
            print(f"  Number of singular values: {len(s)}")
            print(f"  Condition number: {s[0]/s[-1]:.2e}")

        # ========================================
        # Plot singular values
        # ========================================
        if visualize:
            fig1, ax1 = plt.subplots(1, 1, figsize=(10, 6), num=1)
            fig1.clf()
            ax1 = fig1.add_subplot(111)

            # MATLAB: plot(20*log10(norm_sing_val),'b','linewidth',2)
            ax1.plot(20 * np.log10(norm_sing_val), 'b-', linewidth=2)
            ax1.set_xlabel('n')
            ax1.set_ylabel('dB')
            ax1.set_title('Normalized Singular Values [dB]')
            ax1.set_ylim([-80, 1])
            ax1.grid(True)

            plt.tight_layout()
            plt.draw()
            plt.pause(0.1)

        # ========================================
        # Determine truncation threshold
        # ========================================
        if threshold_db is None:
            if visualize:
                show(block=False)
                threshold_db = float(input('\nTruncation threshold [dB]: '))
            else:
                # Default threshold
                threshold_db = -30.0
                if verbose:
                    print(f"\nUsing default threshold: {threshold_db} dB")

        # MATLAB: [~,Nt]=min(abs(20*log10(norm_sing_val)-treashold_dB));
        Nt = find_truncation_index(s, threshold_db)

        if verbose:
            print(f"\nTruncation index: {Nt}")

        # Update plot with truncation line
        if visualize:
            ax1 = plt.figure(1).gca()
            # MATLAB: hold on,plot(ones(1,100)*Nt,linspace(-100,treashold_dB,100),'--r'...)
            ax1.axvline(x=Nt, color='r', linestyle='--', linewidth=1.5)
            ax1.axhline(y=threshold_db, color='r', linestyle='--', linewidth=1.5,
                       label=f'Truncation index: {Nt} @ {threshold_db} dB')
            ax1.legend()
            plt.draw()
            plt.pause(0.1)

        # ========================================
        # TSVD reconstruction
        # ========================================
        # MATLAB: PROF_rec_BORN=TSVD_solver(U,S,V,Nt,data_BORN,Nx,Ny);
        if verbose:
            print("\nPerforming TSVD reconstruction...")

        PROF_rec_BORN = tsvd_solve(U, s, Vh, Nt, data_BORN, Nx, Ny)

        if verbose:
            print(f"  Reconstructed profile shape: {PROF_rec_BORN.shape}")

        # ========================================
        # Compute NMSE
        # ========================================
        # MATLAB: NMSE_BORN=sum(sum(abs(PROF-PROF_rec_BORN).^2))/sum(sum(abs(PROF).^2))
        NMSE_BORN = nmse(PROF, PROF_rec_BORN)

        if verbose:
            print(f"\nNMSE (Born): {NMSE_BORN:.6f} ({10*np.log10(NMSE_BORN):.2f} dB)")

        # ========================================
        # Visualization
        # ========================================
        if visualize:
            # Use MATLAB's default 'parula' colormap (blue to yellow)
            # Parula colormap data (64 colors from MATLAB)
            from matplotlib.colors import LinearSegmentedColormap
            parula_data = [
                [0.2422, 0.1504, 0.6603], [0.2504, 0.1650, 0.7076], [0.2578, 0.1818, 0.7511],
                [0.2647, 0.1978, 0.7952], [0.2706, 0.2147, 0.8364], [0.2751, 0.2342, 0.8710],
                [0.2783, 0.2559, 0.8991], [0.2803, 0.2782, 0.9221], [0.2813, 0.3006, 0.9414],
                [0.2810, 0.3228, 0.9579], [0.2795, 0.3447, 0.9717], [0.2760, 0.3667, 0.9829],
                [0.2699, 0.3892, 0.9906], [0.2602, 0.4123, 0.9952], [0.2440, 0.4358, 0.9988],
                [0.2206, 0.4603, 0.9973], [0.1963, 0.4847, 0.9892], [0.1834, 0.5074, 0.9798],
                [0.1786, 0.5289, 0.9682], [0.1764, 0.5499, 0.9520], [0.1687, 0.5703, 0.9359],
                [0.1540, 0.5902, 0.9218], [0.1460, 0.6091, 0.9079], [0.1380, 0.6276, 0.8973],
                [0.1248, 0.6459, 0.8883], [0.1113, 0.6635, 0.8763], [0.0952, 0.6798, 0.8598],
                [0.0689, 0.6948, 0.8394], [0.0297, 0.7082, 0.8163], [0.0036, 0.7203, 0.7917],
                [0.0067, 0.7312, 0.7660], [0.0433, 0.7411, 0.7394], [0.0964, 0.7500, 0.7120],
                [0.1408, 0.7584, 0.6842], [0.1717, 0.7670, 0.6554], [0.1938, 0.7758, 0.6251],
                [0.2161, 0.7843, 0.5923], [0.2470, 0.7918, 0.5567], [0.2906, 0.7973, 0.5188],
                [0.3406, 0.8008, 0.4789], [0.3909, 0.8029, 0.4354], [0.4456, 0.8024, 0.3909],
                [0.5044, 0.7993, 0.3480], [0.5616, 0.7942, 0.3045], [0.6174, 0.7876, 0.2612],
                [0.6720, 0.7793, 0.2227], [0.7242, 0.7698, 0.1910], [0.7738, 0.7598, 0.1646],
                [0.8203, 0.7498, 0.1535], [0.8634, 0.7406, 0.1596], [0.9035, 0.7330, 0.1774],
                [0.9393, 0.7288, 0.2100], [0.9728, 0.7298, 0.2394], [0.9956, 0.7434, 0.2371],
                [0.9970, 0.7659, 0.2199], [0.9952, 0.7893, 0.2028], [0.9892, 0.8136, 0.1885],
                [0.9786, 0.8386, 0.1766], [0.9676, 0.8639, 0.1643], [0.9610, 0.8890, 0.1537],
                [0.9597, 0.9135, 0.1423], [0.9628, 0.9373, 0.1265], [0.9691, 0.9606, 0.1064],
                [0.9769, 0.9839, 0.0805]
            ]
            parula_cmap = LinearSegmentedColormap.from_list('parula', parula_data)

            # Color scale limits
            mm_r = min(np.min(np.real(PROF)), np.min(np.real(PROF_rec_BORN)))
            mm_i = min(np.min(np.imag(PROF)), np.min(np.imag(PROF_rec_BORN)))
            MM_r = max(np.max(np.real(PROF)), np.max(np.real(PROF_rec_BORN)))
            MM_i = max(np.max(np.imag(PROF)), np.max(np.imag(PROF_rec_BORN)))

            # Figure 2: Comparison
            fig2, axes = plt.subplots(2, 2, figsize=(12, 10), num=2)
            fig2.clf()
            axes = fig2.subplots(2, 2)

            # Actual Re[τ]
            im1 = axes[0, 0].imshow(np.real(PROF),
                                    extent=[xvec[0]/lambda0, xvec[-1]/lambda0,
                                           yvec[0]/lambda0, yvec[-1]/lambda0],
                                    origin='lower', aspect='equal',
                                    vmin=mm_r, vmax=MM_r, cmap=parula_cmap)
            plt.colorbar(im1, ax=axes[0, 0])
            axes[0, 0].set_xlabel(r'x/$\lambda_0$')
            axes[0, 0].set_ylabel(r'y/$\lambda_0$')
            axes[0, 0].set_title(r'Re[$\tau$]: actual')

            # Actual Im[τ]
            im2 = axes[0, 1].imshow(np.imag(PROF),
                                    extent=[xvec[0]/lambda0, xvec[-1]/lambda0,
                                           yvec[0]/lambda0, yvec[-1]/lambda0],
                                    origin='lower', aspect='equal',
                                    vmin=mm_i, vmax=MM_i, cmap=parula_cmap)
            plt.colorbar(im2, ax=axes[0, 1])
            axes[0, 1].set_xlabel(r'x/$\lambda_0$')
            axes[0, 1].set_ylabel(r'y/$\lambda_0$')
            axes[0, 1].set_title(r'Im[$\tau$]: actual')

            # Reconstructed Re[τ]
            im3 = axes[1, 0].imshow(np.real(PROF_rec_BORN),
                                    extent=[xvec[0]/lambda0, xvec[-1]/lambda0,
                                           yvec[0]/lambda0, yvec[-1]/lambda0],
                                    origin='lower', aspect='equal',
                                    vmin=mm_r, vmax=MM_r, cmap=parula_cmap)
            plt.colorbar(im3, ax=axes[1, 0])
            axes[1, 0].set_xlabel(r'x/$\lambda_0$')
            axes[1, 0].set_ylabel(r'y/$\lambda_0$')
            axes[1, 0].set_title(r'Re[$\tau$]: reconstructed via BA')

            # Reconstructed Im[τ]
            im4 = axes[1, 1].imshow(np.imag(PROF_rec_BORN),
                                    extent=[xvec[0]/lambda0, xvec[-1]/lambda0,
                                           yvec[0]/lambda0, yvec[-1]/lambda0],
                                    origin='lower', aspect='equal',
                                    vmin=mm_i, vmax=MM_i, cmap=parula_cmap)
            plt.colorbar(im4, ax=axes[1, 1])
            axes[1, 1].set_xlabel(r'x/$\lambda_0$')
            axes[1, 1].set_ylabel(r'y/$\lambda_0$')
            axes[1, 1].set_title(r'Im[$\tau$]: reconstructed via BA')

            fig2.suptitle(f'Born Inversion - NMSE = {NMSE_BORN:.4f}')
            plt.tight_layout()

            # Figure 3: Normalized reconstruction with contour
            fig3, ax3 = plt.subplots(1, 1, figsize=(8, 8), num=3)
            fig3.clf()
            ax3 = fig3.add_subplot(111)

            # MATLAB: imagesc(xvec/lambda0,yvec/lambda0,abs(PROF_rec_BORN)/max(max(abs(PROF_rec_BORN))))
            PROF_rec_norm = np.abs(PROF_rec_BORN) / np.max(np.abs(PROF_rec_BORN))
            im5 = ax3.imshow(PROF_rec_norm,
                            extent=[xvec[0]/lambda0, xvec[-1]/lambda0,
                                   yvec[0]/lambda0, yvec[-1]/lambda0],
                            origin='lower', aspect='equal', cmap=parula_cmap)
            plt.colorbar(im5, ax=ax3)

            # MATLAB: hold on, contour(xvec/lambda0,yvec/lambda0,real(PROF),1,'--k','LineWidth',2)
            X_norm = np.meshgrid(xvec/lambda0, yvec/lambda0)[0]
            Y_norm = np.meshgrid(xvec/lambda0, yvec/lambda0)[1]
            ax3.contour(X_norm, Y_norm, np.real(PROF), levels=[0.5*np.max(np.real(PROF))],
                       colors='k', linestyles='--', linewidths=2)

            ax3.set_xlabel(r'x/$\lambda_0$')
            ax3.set_ylabel(r'y/$\lambda_0$')
            ax3.set_title(r'Normalized |$\tau$|: reconstructed via BA')

            plt.tight_layout()
            show()

        results = {
            'PROF_rec_BORN': PROF_rec_BORN,
            'NMSE_BORN': NMSE_BORN,
            'Nt': Nt,
            'threshold_db': threshold_db,
            'U': U,
            's': s,
            'Vh': Vh,
            'norm_sing_val': norm_sing_val,
        }

    else:
        # ========================================
        # Aspect-limited reconstruction
        # ========================================
        if verbose:
            print("\n" + "=" * 40)
            print("ASPECT LIMITATION MODE")
            print("=" * 40)

        # MATLAB: mask1=zeros(Nm,Nv); mask1(2:10,2:10)=1;
        mask1 = np.zeros((Nm, Nv))
        mask1[1:10, 1:10] = 1  # 0-indexed: 1:10 corresponds to MATLAB 2:10

        # MATLAB: data_BORN_AL_1=data_BORN.*mask1;
        data_BORN_AL_1 = data_BORN * mask1

        # MATLAB: S_BORN_AL_1=S_BORN.*reshape(repmat(mask1,[1 1 Nx*Ny]),Nm*Nv,Nx*Ny);
        # Use Fortran order to match S_BORN row ordering (all receivers for view 0, then view 1, etc.)
        mask_flat = mask1.ravel(order='F')
        S_BORN_AL_1 = S_BORN * mask_flat[:, np.newaxis]

        if visualize:
            # Figure 4: Data matrices comparison
            fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(12, 5), num=4)
            fig4.clf()
            ax4a = fig4.add_subplot(121)
            ax4b = fig4.add_subplot(122)

            im4a = ax4a.imshow(np.abs(data_BORN), aspect='equal', origin='lower',
                              extent=[0.5, Nv+0.5, 0.5, Nm+0.5])
            plt.colorbar(im4a, ax=ax4a)
            ax4a.set_xlabel('Tx')
            ax4a.set_ylabel('Rx')
            ax4a.set_title('Full-aspect\ndata matrix')

            im4b = ax4b.imshow(np.abs(data_BORN_AL_1), aspect='equal', origin='lower',
                              extent=[0.5, Nv+0.5, 0.5, Nm+0.5])
            plt.colorbar(im4b, ax=ax4b)
            ax4b.set_xlabel('Tx')
            ax4b.set_ylabel('Rx')
            ax4b.set_title('Aspect-limited\ndata matrix, upper arc')

            plt.tight_layout()
            plt.draw()
            plt.pause(0.1)

        # SVD of aspect-limited kernel
        if verbose:
            print("\nComputing SVD for aspect-limited kernel...")

        U1, s1, Vh1 = compute_svd(S_BORN_AL_1)
        norm_sing_val1 = np.abs(s1) / np.abs(s1[0])

        if visualize:
            # Figure 5: Singular values
            fig5, ax5 = plt.subplots(1, 1, figsize=(10, 6), num=5)
            fig5.clf()
            ax5 = fig5.add_subplot(111)

            ax5.plot(20 * np.log10(norm_sing_val1), 'b-', linewidth=2)
            ax5.set_xlabel('n')
            ax5.set_ylabel('dB')
            ax5.set_title('Normalized Singular Values [dB] - Aspect Limited')
            ax5.set_ylim([-80, 1])
            ax5.grid(True)

            plt.tight_layout()
            plt.draw()
            plt.pause(0.1)

        if threshold_db is None:
            if visualize:
                show(block=False)
                threshold_db = float(input('\nTruncation threshold [dB]: '))
            else:
                threshold_db = -30.0

        Nt1 = find_truncation_index(s1, threshold_db)

        if verbose:
            print(f"\nTruncation index (AL): {Nt1}")

        if visualize:
            ax5 = plt.figure(5).gca()
            ax5.axvline(x=Nt1, color='r', linestyle='--', linewidth=1.5)
            ax5.axhline(y=threshold_db, color='r', linestyle='--', linewidth=1.5,
                       label=f'Truncation index: {Nt1} @ {threshold_db} dB')
            ax5.legend()
            plt.draw()
            plt.pause(0.1)

        # TSVD reconstruction
        PROF_rec_BORN_AL_1 = tsvd_solve(U1, s1, Vh1, Nt1, data_BORN_AL_1, Nx, Ny)

        # NMSE
        NMSE_BORN_AL_1 = nmse(PROF, PROF_rec_BORN_AL_1)

        if verbose:
            print(f"\nNMSE (Born, Aspect Limited): {NMSE_BORN_AL_1:.6f}")

        if visualize:
            # Figure 6: Normalized reconstruction
            fig6, ax6 = plt.subplots(1, 1, figsize=(8, 8), num=6)
            fig6.clf()
            ax6 = fig6.add_subplot(111)

            PROF_rec_norm = np.abs(PROF_rec_BORN_AL_1) / np.max(np.abs(PROF_rec_BORN_AL_1))
            im6 = ax6.imshow(PROF_rec_norm,
                            extent=[xvec[0]/lambda0, xvec[-1]/lambda0,
                                   yvec[0]/lambda0, yvec[-1]/lambda0],
                            origin='lower', aspect='equal', cmap='viridis')  # Use viridis as fallback
            plt.colorbar(im6, ax=ax6)

            X_norm = np.meshgrid(xvec/lambda0, yvec/lambda0)[0]
            Y_norm = np.meshgrid(xvec/lambda0, yvec/lambda0)[1]
            ax6.contour(X_norm, Y_norm, np.real(PROF), levels=[0.5*np.max(np.real(PROF))],
                       colors='k', linestyles='--', linewidths=2)

            ax6.set_xlabel(r'x/$\lambda_0$')
            ax6.set_ylabel(r'y/$\lambda_0$')
            ax6.set_title(r'Normalized |$\tau$|: reconstructed via BA and Aspect Lim.')

            plt.tight_layout()
            show()

        results = {
            'PROF_rec_BORN_AL': PROF_rec_BORN_AL_1,
            'NMSE_BORN_AL': NMSE_BORN_AL_1,
            'Nt': Nt1,
            'threshold_db': threshold_db,
            'U': U1,
            's': s1,
            'Vh': Vh1,
            'mask': mask1,
        }

    # ========================================
    # Save results
    # ========================================
    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        save_mat(output_path / 'DATA_inversion_born.mat', results)
        np.savez(output_path / 'DATA_inversion_born.npz', **results)

        if verbose:
            print(f"\nSaved: DATA_inversion_born.mat, DATA_inversion_born.npz")

    return results


def main():
    """Main entry point for CLI."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Run Born inversion on simulated data (c2_Inversion_BORN.m port)'
    )
    parser.add_argument(
        '--data-file', '-d',
        default=None,
        help='Path to DATA_scenario.mat/.npz'
    )
    parser.add_argument(
        '--output-dir', '-o',
        default=None,
        help='Output directory for results'
    )
    parser.add_argument(
        '--snr',
        type=float,
        default=30.0,
        help='Signal-to-noise ratio [dB]'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=None,
        help='Truncation threshold [dB] (default: interactive)'
    )
    parser.add_argument(
        '--aspect-limited', '-al',
        action='store_true',
        help='Use aspect-limited configuration'
    )
    parser.add_argument(
        '--no-visualize',
        action='store_true',
        help='Disable visualization'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=345,
        help='Random seed for noise'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress progress output'
    )

    args = parser.parse_args()

    run_inversion_born(
        data_file=args.data_file,
        output_dir=args.output_dir,
        snr_db=args.snr,
        threshold_db=args.threshold,
        aspect_limited=args.aspect_limited,
        visualize=not args.no_visualize,
        seed=args.seed,
        verbose=not args.quiet
    )


if __name__ == '__main__':
    main()
