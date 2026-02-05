"""
Experimental data scenario script - Python port of c1_Scenario_ExpData.m

This script loads Fresnel Institute experimental data and sets up the
scenario for inversion exercises.

MATLAB equivalent: c1_Scenario_ExpData.m

Usage:
    poetry run run-exp-scenario
    # or
    python -m inverse_scattering.scripts.scenario_exp
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional

from inverse_scattering.core.constants import C
from inverse_scattering.core.utils import create_grid, compute_wavenumber
from inverse_scattering.forward.profiles import (
    create_fresnel_single_target,
    create_fresnel_two_targets
)
from inverse_scattering.forward.incident_field import (
    setup_transmitters,
    compute_incident_field_all_views
)
from inverse_scattering.data.fresnel_loader import (
    load_fresnel_data,
    get_fresnel_parameters
)
from inverse_scattering.data.mat_io import save_mat
from inverse_scattering.visualization.plots import (
    plot_profile,
    plot_cross_sections,
    show
)


def run_scenario_exp(
    dataset: str = 'dielTM_dec8f.txt',
    data_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    visualize: bool = True
) -> dict:
    """
    Run experimental data scenario setup.

    MATLAB equivalent:
        c1_Scenario_ExpData.m

    Args:
        dataset: Dataset filename ('dielTM_dec8f.txt' or 'twodielTM_8f.txt')
        data_dir: Directory containing the Fresnel data files
        output_dir: Directory for output files (default: current directory)
        visualize: Whether to display plots

    Returns:
        Dictionary with scenario data
    """
    print("=" * 60)
    print("Experimental Data Scenario Setup")
    print("=" * 60)

    # ========================================
    # Parameters (matching MATLAB exactly)
    # ========================================
    # MATLAB: eb = 1; sb = 0;
    eb = 1.0  # Background: free space
    sb = 0.0  # No background conductivity

    # MATLAB: freq = 4*1e9;
    freq = 4e9  # 4 GHz

    # MATLAB: lambda0 = 3*1e8/freq;
    lambda0 = C / freq  # 7.5 cm

    # MATLAB: lx = 0.15; ly = lx;
    lx = 0.15  # 15 cm DoI side
    ly = lx

    # MATLAB: Nx = 64; Ny = 64;
    Nx = 64
    Ny = 64

    # MATLAB: Rv = 0.72135; Rm = 0.76135;
    Rv = 0.72135  # Transmitter radius [m]
    Rm = 0.76135  # Receiver radius [m]

    print(f"\nParameters:")
    print(f"  Frequency: {freq/1e9:.1f} GHz")
    print(f"  Wavelength: {lambda0*100:.2f} cm")
    print(f"  DoI size: {lx*100:.1f} x {ly*100:.1f} cm")
    print(f"  Grid: {Nx} x {Ny}")
    print(f"  Rv = {Rv*100:.3f} cm, Rm = {Rm*100:.3f} cm")

    # ========================================
    # Load Fresnel data
    # ========================================
    print(f"\nLoading dataset: {dataset}")

    if data_dir is None:
        # Try to find data file in standard locations
        # Project root relative paths (after folder reorganization)
        project_root = Path(__file__).parent.parent.parent.parent
        possible_paths = [
            Path(dataset),
            project_root / 'matlab' / 'experimental' / 'scenario' / dataset,
            Path('../matlab/experimental/scenario') / dataset,
            Path('../../matlab/experimental/scenario') / dataset,
        ]
        for p in possible_paths:
            if p.exists():
                data_path = p
                break
        else:
            print(f"Warning: Data file not found. Using placeholder data.")
            data_path = None
    else:
        data_path = Path(data_dir) / dataset

    # Load data or create placeholder
    if data_path is not None and data_path.exists():
        Escat, Einc_domain_raw = load_fresnel_data(str(data_path), freq, Nx, Ny)
        print(f"  Loaded: Escat shape = {Escat.shape}")
    else:
        # Create placeholder for demonstration
        print("  Creating placeholder data (file not found)")
        Nm_default = 49
        Nv_default = 36
        Escat = np.zeros((Nm_default, Nv_default), dtype=complex)
        Einc_domain_raw = np.zeros((Ny, Nx, Nv_default), dtype=complex)

    # MATLAB: Nm = size(Escat, 1); Nv = size(Escat, 2);
    Nm = Escat.shape[0]
    Nv = Escat.shape[1]
    print(f"  Nm (receivers): {Nm}")
    print(f"  Nv (transmitters): {Nv}")

    # ========================================
    # Create grid
    # ========================================
    # MATLAB: dx = lx/Nx; dy = ly/Ny;
    #         xvec = -lx/2+dx/2:dx:lx/2-dx/2; etc.
    X, Y, xvec, yvec, dx, dy = create_grid(lx, ly, Nx, Ny)

    # ========================================
    # Compute incident field on DoI
    # ========================================
    # For experimental data, we compute E_inc from transmitter positions
    k = compute_wavenumber(freq, eb, sb)
    tx_positions = setup_transmitters(Nv, Rv)
    Einc_domain = compute_incident_field_all_views(X, Y, k, tx_positions, 'line')
    print(f"  Einc_domain shape: {Einc_domain.shape}")

    # ========================================
    # Create ground truth profile
    # ========================================
    if 'twodiel' in dataset.lower():
        print("\nCreating two-target ground truth profile...")
        PROF, obj_params = create_fresnel_two_targets(X, Y)
        scenario_file = 'DATA_scenario_exp_twotargets'
        object_file = 'DATA_object_exp_twotargets'
    else:
        print("\nCreating single-target ground truth profile...")
        PROF, obj_params = create_fresnel_single_target(X, Y)
        scenario_file = 'DATA_scenario_exp_singletarget'
        object_file = 'DATA_object_exp_singletarget'

    print(f"  Object params: {obj_params}")
    print(f"  Max contrast: {np.max(np.real(PROF))}")

    # ========================================
    # Save scenario data
    # ========================================
    if output_dir is None:
        output_dir = '.'
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save scenario data (MATLAB: save DATA_scenario_exp_*.mat)
    scenario_data = {
        'Escat': Escat,
        'Einc_domain': Einc_domain,
        'freq': freq,
        'lambda0': lambda0,
        'lx': lx,
        'ly': ly,
        'Nx': Nx,
        'Ny': Ny,
        'eb': eb,
        'sb': sb,
        'Rv': Rv,
        'Rm': Rm,
        'Nm': Nm,
        'Nv': Nv,
        'xvec': xvec,
        'yvec': yvec,
        'X': X,
        'Y': Y,
        'dx': dx,
        'dy': dy,
        'dataset': dataset,
    }

    # Save as .mat for MATLAB compatibility
    save_mat(output_path / f'{scenario_file}.mat', scenario_data)
    print(f"\nSaved: {scenario_file}.mat")

    # Save as .npz for Python
    np.savez(output_path / f'{scenario_file}.npz', **scenario_data)
    print(f"Saved: {scenario_file}.npz")

    # Save object data
    object_data = {**obj_params, 'PROF': PROF}
    save_mat(output_path / f'{object_file}.mat', object_data)
    print(f"Saved: {object_file}.mat")
    np.savez(output_path / f'{object_file}.npz', **object_data)
    print(f"Saved: {object_file}.npz")

    # ========================================
    # Visualization
    # ========================================
    if visualize:
        print("\nGenerating plots...")

        # Figure 1: Profile (Re and Im)
        # MATLAB: figure(1), subplot(1,2,1), imagesc(..., real(PROF))
        fig1 = plot_profile(xvec, yvec, PROF, lambda0, fignum=1)
        if 'twodiel' in dataset.lower():
            fig1.suptitle('Fresnel Two Diel Target')
        else:
            fig1.suptitle('Fresnel Single Diel Target')

        # Figure 2: Cross-sections
        # MATLAB: figure(2), subplot(1,2,1), plot(xvec, real(PROF(Ny/2,:)))
        if 'twodiel' not in dataset.lower():
            # For single target, use center cuts
            y_cut_idx = Ny // 2  # MATLAB: Ny/2 (1-indexed becomes Ny//2 in 0-indexed)
            x_cut_idx = 43 - 1   # MATLAB: 43 (convert to 0-indexed)

            fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), num=2)
            fig2.clf()
            fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), num=2)

            # X-cut at y = y0 (center)
            ax1.plot(xvec, np.real(PROF[y_cut_idx, :]), 'b-', linewidth=2)
            ax1.set_xlabel('x [m]')
            ax1.set_title(f'Re[τ]: x-cut @ y={obj_params["y0"]}')
            ax1.grid(True)
            ax1.set_aspect('auto')

            # Y-cut at x = x0
            ax2.plot(yvec, np.real(PROF[:, x_cut_idx]), 'b-', linewidth=2)
            ax2.set_xlabel('y [m]')
            ax2.set_title(f'Re[τ]: y-cut @ x={obj_params["x0"]} m')
            ax2.grid(True)
            ax2.set_aspect('auto')

            plt.tight_layout()
        else:
            # For two targets, show x-cut through both
            y_cut_idx = 39 - 1  # MATLAB: 39

            fig2 = plt.figure(2, figsize=(8, 5))
            fig2.clf()
            ax = fig2.add_subplot(111)
            ax.plot(xvec, np.real(PROF[y_cut_idx, :]), 'b-', linewidth=2)
            ax.set_xlabel('x [m]')
            ax.set_title(f'Re[τ]: x-cut @ y={obj_params["y0_l"]}')
            ax.grid(True)
            plt.tight_layout()

        show()

    # Return all data
    return {
        **scenario_data,
        'PROF': PROF,
        **obj_params
    }


def main():
    """Main entry point for CLI."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Load Fresnel experimental data and set up scenario'
    )
    parser.add_argument(
        '--dataset', '-d',
        default='dielTM_dec8f.txt',
        choices=['dielTM_dec8f.txt', 'twodielTM_8f.txt'],
        help='Fresnel dataset to load'
    )
    parser.add_argument(
        '--data-dir',
        default=None,
        help='Directory containing data files'
    )
    parser.add_argument(
        '--output-dir', '-o',
        default='.',
        help='Output directory for saved files'
    )
    parser.add_argument(
        '--no-visualize',
        action='store_true',
        help='Disable visualization'
    )

    args = parser.parse_args()

    run_scenario_exp(
        dataset=args.dataset,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        visualize=not args.no_visualize
    )


if __name__ == '__main__':
    main()
