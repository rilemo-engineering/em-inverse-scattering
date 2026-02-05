"""
Simulated scenario script - Python port of c1_Scenario.m

This script sets up the forward scattering problem and saves all data
for subsequent inversion exercises.

MATLAB equivalent: c1_Scenario.m

Usage:
    poetry run run-scenario
    # or
    python -m inverse_scattering.scripts.scenario
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional

from inverse_scattering.core.constants import C
from inverse_scattering.core.utils import create_grid
from inverse_scattering.forward.forward_solver import forward_solver
from inverse_scattering.data.mat_io import save_mat
from inverse_scattering.visualization.plots import show


def run_scenario(
    n_iter: int = 1000,
    output_dir: Optional[str] = None,
    visualize: bool = True,
    # Optional scenario parameters
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
    # Target parameters
    target_epsilon_r: float = 2.0,
    target_sigma: float = 0.0,
    target_radius: Optional[float] = None,
    target_center: tuple = (0.0, 0.0),
    verbose: bool = True
) -> dict:
    """
    Run simulated scenario setup.

    MATLAB equivalent:
        c1_Scenario.m

    This function:
    1. Calls forward_solver to compute scattered fields
    2. Optionally visualizes the scenario
    3. Saves data to DATA_scenario.mat and DATA_scenario.npz

    Args:
        n_iter: Maximum CGFFT iterations
        output_dir: Directory for output files (default: current directory)
        visualize: Whether to display plots
        freq: Frequency [Hz] (default: 300 MHz)
        lx, ly: DoI dimensions [m]
        Nx, Ny: Grid dimensions
        eb, sb: Background permittivity and conductivity
        Rm, Rv: Measurement and transmitter radii [m]
        Nm, Nv: Number of receivers and transmitters
        target_epsilon_r: Target relative permittivity
        target_sigma: Target conductivity [S/m]
        target_radius: Target radius [m] (default: λ/4)
        target_center: (x0, y0) target center [m]
        verbose: Print progress

    Returns:
        Dictionary with all scenario data
    """
    # BEFORE RUNNING THE CODE, TAKE NOTE ABOUT USEFUL DATA INFORMATION
    # - freq: working frequency [MHz]
    # - eb: permittivity of the background
    # - sb: conductivity of the background
    # - lx: x-dimension of the Domain of Investigation [m]
    # - ly: y-dimension of the Domain of Investigation [m]
    # - Rm: radius of the measurement circular surface [m]
    # - Nm: number of measurement point, i.e., DoF=2*beta*a
    #       In this case, a=sqrt(2)*lx/2.
    #       NB. Round up the DoF number
    # - Nv: number of illumination directions. Usually, Nv=Nm.
    # - ex: higher permittivity in your profile
    # - sx: higher conductivity in your profile
    # - targets shape, dimension and position

    if verbose:
        print("=" * 60)
        print("Simulated Scenario Setup (c1_Scenario.m port)")
        print("=" * 60)

    # ========================================
    # Run forward solver
    # ========================================
    # MATLAB: [Escat, PROF, Einc_domain, Etot_domain, freq, lx, ly, eb, sb, Nx, Ny, Rm, DOF]=forward_solver(n_iter);
    result = forward_solver(
        n_iter=n_iter,
        freq=freq,
        lx=lx,
        ly=ly,
        Nx=Nx,
        Ny=Ny,
        eb=eb,
        sb=sb,
        Rm=Rm,
        Rv=Rv,
        Nm=Nm,
        Nv=Nv,
        target_epsilon_r=target_epsilon_r,
        target_sigma=target_sigma,
        target_radius=target_radius,
        target_center=target_center,
        verbose=verbose
    )

    # OBSERVING FORWARD SOLVER OUTPUTS
    # - nx: number of discretization cells for x-dimension
    # - ny: number of discretization cells for y-dimension
    # - Escat: multiview-multistatic data matrix. Dimension: Nm x Nv
    # - PROF: contrast profile. Dimension: Ny x Nx
    # - Einc_domain: incident field on the RoI. Dimension: Ny x Nx x Nv
    # - Etot_domain: actual total field on the RoI. Dimension: Ny x Nx x Nv

    # Derived parameters
    # MATLAB: lambda0=3*1e8/freq;
    lambda0 = result.lambda0

    # MATLAB: Nm=size(Escat,1); Nv=size(Einc_domain,3);
    Nm = result.Escat.shape[0]
    Nv = result.Einc_domain.shape[2]

    # MATLAB: dx=lx/Nx; dy=ly/Ny;
    dx = result.dx
    dy = result.dy

    # MATLAB: xvec=-lx/2+dx/2:dx:lx/2-dx/2;
    xvec = result.xvec
    yvec = result.yvec

    # MATLAB: [X,Y]=meshgrid(xvec,yvec);
    X = result.X
    Y = result.Y

    # MATLAB: meas_pos_theta=linspace(0,2*pi-2*pi/Nm,Nm);
    meas_pos_theta = np.linspace(0, 2*np.pi - 2*np.pi/Nm, Nm)

    if verbose:
        print(f"\nDerived parameters:")
        print(f"  lambda0 = {lambda0:.4f} m")
        print(f"  Nm = {Nm}, Nv = {Nv}")
        print(f"  dx = {dx:.6f} m, dy = {dy:.6f} m")

    # ========================================
    # Visualization
    # ========================================
    if visualize:
        if verbose:
            print("\nGenerating plots...")

        # Figure 1: Simulated Scenario
        # MATLAB: figure(1),clf,set(gcf,'color','w'),hold on,box on,grid on
        fig1, ax1 = plt.subplots(1, 1, figsize=(8, 8), num=1)
        fig1.clf()
        ax1 = fig1.add_subplot(111)

        # MATLAB: imagesc(xvec,yvec,abs(PROF)),colormap(flipud(gray))
        im = ax1.imshow(np.abs(result.PROF), extent=[xvec[0], xvec[-1], yvec[0], yvec[-1]],
                        origin='lower', cmap='gray_r', aspect='equal')

        # Plot DoI boundary
        # MATLAB: plot(xvec,ones(1,Ny)*lx/2,'k'), etc.
        lx = result.lx
        ly = result.ly
        ax1.plot([xvec[0], xvec[-1]], [ly/2, ly/2], 'k-', linewidth=1.5, label='RoI')
        ax1.plot([xvec[0], xvec[-1]], [-ly/2, -ly/2], 'k-', linewidth=1.5)
        ax1.plot([lx/2, lx/2], [yvec[0], yvec[-1]], 'k-', linewidth=1.5)
        ax1.plot([-lx/2, -lx/2], [yvec[0], yvec[-1]], 'k-', linewidth=1.5)

        # Plot measurement circle
        # MATLAB: plot(Rm*cos(linspace(0,2*pi,100)),Rm*sin(linspace(0,2*pi,100)),'--k')
        theta_circle = np.linspace(0, 2*np.pi, 100)
        ax1.plot(result.Rm * np.cos(theta_circle), result.Rm * np.sin(theta_circle),
                '--k', linewidth=1.5, label='Measurement Surface')

        # Plot measurement points
        # MATLAB: plot(Rm*cos(meas_pos_theta),Rm*sin(meas_pos_theta),'.r','markersize',20)
        ax1.plot(result.Rm * np.cos(meas_pos_theta), result.Rm * np.sin(meas_pos_theta),
                'r.', markersize=15, label='Measurement Points')

        ax1.set_xlabel('x [m]')
        ax1.set_ylabel('y [m]')
        ax1.set_title('Simulated Scenario')
        ax1.legend(loc='best')
        ax1.grid(True)
        ax1.set_aspect('equal')

        # Figure 2: Profile (Re and Im)
        # MATLAB: figure(2),clf,set(gcf,'color','w')
        fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(12, 5), num=2)
        fig2.clf()
        ax2a = fig2.add_subplot(121)
        ax2b = fig2.add_subplot(122)

        # MATLAB: subplot(1,2,1),imagesc(xvec/lambda0,yvec/lambda0,real(PROF)),colorbar
        im1 = ax2a.imshow(np.real(result.PROF),
                          extent=[xvec[0]/lambda0, xvec[-1]/lambda0,
                                  yvec[0]/lambda0, yvec[-1]/lambda0],
                          origin='lower', aspect='equal')
        plt.colorbar(im1, ax=ax2a)
        ax2a.set_xlabel(r'x/$\lambda_0$')
        ax2a.set_ylabel(r'y/$\lambda_0$')
        ax2a.set_title(r'Re[$\tau$]')

        # MATLAB: subplot(1,2,2),imagesc(xvec/lambda0,yvec/lambda0,imag(PROF)),colorbar
        im2 = ax2b.imshow(np.imag(result.PROF),
                          extent=[xvec[0]/lambda0, xvec[-1]/lambda0,
                                  yvec[0]/lambda0, yvec[-1]/lambda0],
                          origin='lower', aspect='equal')
        plt.colorbar(im2, ax=ax2b)
        ax2b.set_xlabel(r'x/$\lambda_0$')
        ax2b.set_ylabel(r'y/$\lambda_0$')
        ax2b.set_title(r'Im[$\tau$]')

        plt.tight_layout()

        # Figure 3: MVMS Data Matrix
        # MATLAB: figure(3),clf,set(gcf,'color','w')
        fig3, ax3 = plt.subplots(1, 1, figsize=(8, 6), num=3)
        fig3.clf()
        ax3 = fig3.add_subplot(111)

        # MATLAB: imagesc(1:1:Nv,1:1:Nm,abs(Escat)),colorbar
        im3 = ax3.imshow(np.abs(result.Escat), aspect='auto', origin='lower',
                         extent=[0.5, Nv+0.5, 0.5, Nm+0.5])
        plt.colorbar(im3, ax=ax3)
        ax3.set_xlabel('nv')
        ax3.set_ylabel('nm')
        ax3.set_title('MVMS Data Matrix\n(amplitude)')

        # Figure 4: Incident Field animation (show first view only for non-blocking)
        # MATLAB: figure(4), for kv=1:Nv, subplot... etc.
        fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(12, 5), num=4)
        fig4.clf()
        ax4a = fig4.add_subplot(121)
        ax4b = fig4.add_subplot(122)

        kv = 0  # Show first view
        im4a = ax4a.imshow(np.abs(result.Einc_domain[:, :, kv]),
                           extent=[xvec[0]/lambda0, xvec[-1]/lambda0,
                                   yvec[0]/lambda0, yvec[-1]/lambda0],
                           origin='lower', aspect='equal')
        plt.colorbar(im4a, ax=ax4a)
        ax4a.set_xlabel(r'x/$\lambda_0$')
        ax4a.set_ylabel(r'y/$\lambda_0$')
        ax4a.set_title(f'Amplitude of Incident Field\n[nv={kv+1}]')

        im4b = ax4b.imshow(np.angle(result.Einc_domain[:, :, kv]),
                           extent=[xvec[0]/lambda0, xvec[-1]/lambda0,
                                   yvec[0]/lambda0, yvec[-1]/lambda0],
                           origin='lower', aspect='equal', cmap='hsv')
        plt.colorbar(im4b, ax=ax4b)
        ax4b.set_xlabel(r'x/$\lambda_0$')
        ax4b.set_ylabel(r'y/$\lambda_0$')
        ax4b.set_title(f'Phase of Incident Field\n[nv={kv+1}]')

        plt.tight_layout()

        show()

    # ========================================
    # Save data
    # ========================================
    if output_dir is None:
        output_dir = '.'
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Prepare data dictionary (matching MATLAB save DATA_scenario.mat)
    scenario_data = {
        'Escat': result.Escat,
        'PROF': result.PROF,
        'Einc_domain': result.Einc_domain,
        'Etot_domain': result.Etot_domain,
        'freq': result.freq,
        'lx': result.lx,
        'ly': result.ly,
        'eb': result.eb,
        'sb': result.sb,
        'Nx': result.Nx,
        'Ny': result.Ny,
        'Rm': result.Rm,
        'Rv': result.Rv,
        'DOF': result.DOF,
        'Nm': Nm,
        'Nv': Nv,
        'lambda0': lambda0,
        'dx': dx,
        'dy': dy,
        'xvec': xvec,
        'yvec': yvec,
        'X': X,
        'Y': Y,
        'meas_pos_theta': meas_pos_theta,
        'n_iter': n_iter,
    }

    # MATLAB: save DATA_scenario.mat
    save_mat(output_path / 'DATA_scenario.mat', scenario_data)
    if verbose:
        print(f"\nSaved: DATA_scenario.mat")

    # Save as .npz for Python
    np.savez(output_path / 'DATA_scenario.npz', **scenario_data)
    if verbose:
        print(f"Saved: DATA_scenario.npz")

    return scenario_data


def main():
    """Main entry point for CLI."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Run forward scattering scenario (c1_Scenario.m port)'
    )
    parser.add_argument(
        '--n-iter', '-n',
        type=int,
        default=1000,
        help='Maximum CGFFT iterations'
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
    parser.add_argument(
        '--freq',
        type=float,
        default=None,
        help='Frequency [Hz] (default: 300 MHz)'
    )
    parser.add_argument(
        '--lx',
        type=float,
        default=None,
        help='DoI x-dimension [m] (default: 1.0)'
    )
    parser.add_argument(
        '--Nx',
        type=int,
        default=None,
        help='Number of x cells (default: 32)'
    )
    parser.add_argument(
        '--target-eps',
        type=float,
        default=2.0,
        help='Target relative permittivity'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress progress output'
    )

    args = parser.parse_args()

    run_scenario(
        n_iter=args.n_iter,
        output_dir=args.output_dir,
        visualize=not args.no_visualize,
        freq=args.freq,
        lx=args.lx,
        Nx=args.Nx,
        target_epsilon_r=args.target_eps,
        verbose=not args.quiet
    )


if __name__ == '__main__':
    main()
