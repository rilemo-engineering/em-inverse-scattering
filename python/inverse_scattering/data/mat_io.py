"""
MATLAB .mat file I/O compatibility.

This module provides functions to read and write MATLAB .mat files,
ensuring compatibility with the existing MATLAB data files.

Uses scipy.io for .mat file handling (supports v4, v6, v7 to 7.2 formats).

MATLAB equivalent: Direct .mat file operations (load, save)
"""

import numpy as np
from scipy import io as sio
from pathlib import Path
from typing import Dict, Any, Optional, Union


def load_mat(
    filepath: Union[str, Path],
    squeeze_me: bool = True,
    struct_as_record: bool = False
) -> Dict[str, Any]:
    """
    Load a MATLAB .mat file.

    MATLAB equivalent:
        load DATA_scenario.mat

    Args:
        filepath: Path to .mat file
        squeeze_me: Remove singleton dimensions (default: True)
        struct_as_record: If True, load MATLAB structs as numpy record arrays

    Returns:
        Dictionary with variable names as keys

    Note:
        MATLAB uses column-major (Fortran) order while NumPy uses row-major
        (C) order by default. scipy.io.loadmat handles this automatically,
        but be aware when comparing array shapes.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"MAT file not found: {filepath}")

    data = sio.loadmat(
        str(filepath),
        squeeze_me=squeeze_me,
        struct_as_record=struct_as_record
    )

    # Remove MATLAB metadata keys
    data = {k: v for k, v in data.items() if not k.startswith('__')}

    return data


def save_mat(
    filepath: Union[str, Path],
    data: Dict[str, Any],
    oned_as: str = 'column'
) -> None:
    """
    Save data to a MATLAB .mat file.

    MATLAB equivalent:
        save DATA_scenario.mat

    Args:
        filepath: Path for output .mat file
        data: Dictionary with variable names as keys
        oned_as: Treatment of 1D arrays ('row' or 'column', default 'column')

    Note:
        By default, 1D arrays are saved as column vectors to match MATLAB
        convention where vectors are typically column vectors.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    sio.savemat(str(filepath), data, oned_as=oned_as)


def load_scenario_data(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Load scenario data with standardized variable names.

    Expected variables from DATA_scenario.mat:
        - Escat: Scattered field matrix (Nm × Nv)
        - PROF: Contrast profile (Ny × Nx)
        - Einc_domain: Incident field on DoI (Ny × Nx × Nv)
        - Etot_domain: Total field on DoI (Ny × Nx × Nv)
        - freq: Frequency [Hz]
        - lx, ly: DoI dimensions [m]
        - Nx, Ny: Grid points
        - eb, sb: Background permittivity and conductivity
        - Rm: Measurement radius [m]
        - DOF: Degrees of freedom

    Args:
        filepath: Path to scenario .mat file

    Returns:
        Dictionary with scenario data
    """
    data = load_mat(filepath)

    # Standardize variable names (handle case variations)
    standardized = {}

    # Direct mappings
    direct_keys = [
        'Escat', 'PROF', 'Einc_domain', 'Etot_domain',
        'freq', 'lx', 'ly', 'Nx', 'Ny', 'eb', 'sb', 'Rm', 'DOF',
        'xvec', 'yvec', 'lambda0', 'X', 'Y', 'dx', 'dy',
        'Nm', 'Nv', 'meas_pos_theta'
    ]

    for key in direct_keys:
        if key in data:
            standardized[key] = data[key]
        # Try lowercase
        elif key.lower() in data:
            standardized[key] = data[key.lower()]

    return standardized


def load_experimental_scenario(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Load experimental scenario data.

    Expected variables from DATA_scenario_exp_*.mat:
        - Escat: Scattered field matrix (Nm × Nv)
        - Einc_domain: Incident field on DoI (Ny × Nx × Nv)
        - freq: Frequency [Hz]
        - lx, ly: DoI dimensions [m]
        - Nx, Ny: Grid points
        - eb, sb: Background properties
        - Rm, Rv: Measurement and transmitter radii [m]
        - xvec, yvec: Coordinate vectors
        - X, Y: Meshgrids

    Args:
        filepath: Path to experimental scenario .mat file

    Returns:
        Dictionary with experimental scenario data
    """
    return load_scenario_data(filepath)


def load_object_data(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Load object specification data.

    Expected variables from DATA_object_exp_*.mat:
        For single target:
            - r0: Cylinder radius
            - x0, y0: Cylinder center
            - PROF: Ground truth profile

        For two targets:
            - r0: Common radius
            - x0_l, y0_l: Left cylinder center
            - x0_r, y0_r: Right cylinder center
            - PROF: Ground truth profile

    Args:
        filepath: Path to object .mat file

    Returns:
        Dictionary with object data
    """
    return load_mat(filepath)


def save_scenario_data(
    filepath: Union[str, Path],
    Escat: np.ndarray,
    PROF: np.ndarray,
    Einc_domain: np.ndarray,
    Etot_domain: np.ndarray,
    freq: float,
    lx: float,
    ly: float,
    Nx: int,
    Ny: int,
    eb: float,
    sb: float,
    Rm: float,
    DOF: int,
    **kwargs
) -> None:
    """
    Save scenario data in MATLAB-compatible format.

    MATLAB equivalent:
        save DATA_scenario.mat

    Args:
        filepath: Output path
        Escat: Scattered field matrix
        PROF: Contrast profile
        Einc_domain: Incident field on DoI
        Etot_domain: Total field on DoI
        freq: Frequency [Hz]
        lx, ly: DoI dimensions [m]
        Nx, Ny: Grid points
        eb, sb: Background properties
        Rm: Measurement radius [m]
        DOF: Degrees of freedom
        **kwargs: Additional variables to save
    """
    data = {
        'Escat': Escat,
        'PROF': PROF,
        'Einc_domain': Einc_domain,
        'Etot_domain': Etot_domain,
        'freq': freq,
        'lx': lx,
        'ly': ly,
        'Nx': Nx,
        'Ny': Ny,
        'eb': eb,
        'sb': sb,
        'Rm': Rm,
        'DOF': DOF,
    }
    data.update(kwargs)

    save_mat(filepath, data)
