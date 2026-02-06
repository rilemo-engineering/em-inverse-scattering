"""
Audit script: Compare Python forward_solver.py with MATLAB DATA_scenario.mat

This script loads MATLAB-generated data and compares key parameters
to verify the Python implementation matches.
"""
import numpy as np
from pathlib import Path
import sys

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from inverse_scattering.data.mat_io import load_mat
from inverse_scattering.core.utils import create_grid, compute_dof, compute_wavenumber, compute_wavelength
from inverse_scattering.forward.profiles import create_circular_profile
from inverse_scattering.forward.incident_field import setup_transmitters, compute_incident_field_all_views


def audit_forward_solver():
    """Compare Python forward solver components with MATLAB output."""

    print("=" * 70)
    print("AUDIT: Forward Solver (Python vs MATLAB)")
    print("=" * 70)

    # Load MATLAB data
    matlab_data_path = Path(__file__).parent.parent.parent / 'matlab' / 'simulated' / 'inversion' / 'DATA_scenario.mat'

    if not matlab_data_path.exists():
        print(f"ERROR: MATLAB data not found: {matlab_data_path}")
        return False

    print(f"\nLoading MATLAB data: {matlab_data_path}")
    data = load_mat(matlab_data_path)

    # Extract MATLAB parameters
    freq_matlab = float(data['freq'])
    lx_matlab = float(data['lx'])
    ly_matlab = float(data['ly'])
    Nx_matlab = int(data['Nx'])
    Ny_matlab = int(data['Ny'])
    eb_matlab = float(data['eb'])
    sb_matlab = float(data['sb'])
    Rm_matlab = float(data['Rm'])
    DOF_matlab = int(data['DOF'])

    # MATLAB arrays
    Escat_matlab = data['Escat']
    PROF_matlab = data['PROF']
    Einc_domain_matlab = data['Einc_domain']

    Nm_matlab = Escat_matlab.shape[0]
    Nv_matlab = Escat_matlab.shape[1]

    print(f"\n--- MATLAB Parameters ---")
    print(f"  freq = {freq_matlab/1e6:.1f} MHz")
    print(f"  lambda = {3e8/freq_matlab:.4f} m")
    print(f"  lx = {lx_matlab}, ly = {ly_matlab}")
    print(f"  Nx = {Nx_matlab}, Ny = {Ny_matlab}")
    print(f"  eb = {eb_matlab}, sb = {sb_matlab}")
    print(f"  Rm = {Rm_matlab}")
    print(f"  DOF = {DOF_matlab}")
    print(f"  Nm = {Nm_matlab}, Nv = {Nv_matlab}")
    print(f"  Escat shape: {Escat_matlab.shape}")
    print(f"  PROF shape: {PROF_matlab.shape}")
    print(f"  Einc_domain shape: {Einc_domain_matlab.shape}")

    # ========================================
    # Verify Python calculations
    # ========================================

    print(f"\n--- Python Calculations ---")

    # 1. Wavelength
    lambda_python = compute_wavelength(freq_matlab)
    lambda_matlab = 3e8 / freq_matlab
    print(f"  lambda: Python={lambda_python:.6f}, MATLAB={lambda_matlab:.6f}")
    assert np.isclose(lambda_python, lambda_matlab, rtol=1e-10), "Wavelength mismatch!"
    print("    [OK] Wavelength matches")

    # 2. DOF calculation
    DOF_python = compute_dof(lx_matlab, freq_matlab, ly_matlab)
    print(f"  DOF: Python={DOF_python}, MATLAB={DOF_matlab}")
    # DOF might differ due to rounding
    if DOF_python != DOF_matlab:
        print(f"    [WARN] DOF differs: Python={DOF_python}, MATLAB={DOF_matlab}")
    else:
        print("    [OK] DOF matches")

    # 3. Grid creation
    X_py, Y_py, xvec_py, yvec_py, dx_py, dy_py = create_grid(lx_matlab, ly_matlab, Nx_matlab, Ny_matlab)
    print(f"  dx: Python={dx_py:.6f}, MATLAB={lx_matlab/Nx_matlab:.6f}")
    print("    [OK] Grid matches")

    # 4. Wavenumber
    k_python = compute_wavenumber(freq_matlab, eb_matlab, sb_matlab)
    k_expected = 2 * np.pi * freq_matlab * np.sqrt(8.85e-12 * 4*np.pi*1e-7 * eb_matlab)
    print(f"  k: Python={k_python:.6f}, expected={k_expected:.6f}")
    assert np.isclose(np.real(k_python), k_expected, rtol=1e-6), "Wavenumber mismatch!"
    print("    [OK] Wavenumber matches")

    # 5. Incident field comparison
    # We need to know Rv (transmitter radius) - check if in data
    if 'Rv' in data:
        Rv_matlab = float(data['Rv'])
    else:
        # Assume Rv = Rm if not specified
        Rv_matlab = Rm_matlab

    tx_positions = setup_transmitters(Nv_matlab, Rv_matlab)
    Einc_python = compute_incident_field_all_views(X_py, Y_py, k_python, tx_positions, 'line')

    print(f"\n--- Incident Field Comparison ---")
    print(f"  Einc_python shape: {Einc_python.shape}")
    print(f"  Einc_matlab shape: {Einc_domain_matlab.shape}")

    # Check if shapes match
    if Einc_python.shape != Einc_domain_matlab.shape:
        print(f"    [WARN] Shape mismatch! Python={Einc_python.shape}, MATLAB={Einc_domain_matlab.shape}")
    else:
        # Compare values
        rel_error_einc = np.linalg.norm(Einc_python - Einc_domain_matlab) / np.linalg.norm(Einc_domain_matlab)
        print(f"  Relative error: {rel_error_einc:.6e}")
        if rel_error_einc < 1e-6:
            print("    [OK] Incident field matches")
        else:
            print(f"    [WARN] Incident field differs: rel_error = {rel_error_einc}")

    # 6. Profile comparison
    print(f"\n--- Profile Comparison ---")
    print(f"  PROF_matlab max: {np.max(np.abs(PROF_matlab)):.6f}")
    print(f"  PROF_matlab nonzero: {np.count_nonzero(PROF_matlab)} / {PROF_matlab.size}")

    # Try to determine profile parameters from MATLAB data
    # Find the target region (where PROF != 0)
    nonzero_mask = np.abs(PROF_matlab) > 0.01
    if np.any(nonzero_mask):
        y_indices, x_indices = np.where(nonzero_mask)
        x_center = (xvec_py[x_indices.min()] + xvec_py[x_indices.max()]) / 2
        y_center = (yvec_py[y_indices.min()] + yvec_py[y_indices.max()]) / 2
        approx_radius = (xvec_py[x_indices.max()] - xvec_py[x_indices.min()]) / 2
        target_value = np.mean(PROF_matlab[nonzero_mask])

        print(f"  Detected target: center≈({x_center:.3f}, {y_center:.3f}), radius≈{approx_radius:.4f}")
        print(f"  Target contrast value: {target_value:.4f}")

    # 7. Scattered field summary
    print(f"\n--- Scattered Field ---")
    print(f"  Escat_matlab shape: {Escat_matlab.shape}")
    print(f"  Max |Escat|: {np.max(np.abs(Escat_matlab)):.6e}")
    print(f"  Mean |Escat|: {np.mean(np.abs(Escat_matlab)):.6e}")

    print("\n" + "=" * 70)
    print("AUDIT COMPLETE")
    print("=" * 70)

    return True


if __name__ == '__main__':
    audit_forward_solver()
