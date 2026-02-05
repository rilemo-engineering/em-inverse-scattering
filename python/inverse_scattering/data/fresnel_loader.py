"""
Fresnel Institute experimental data loader.

Loads scattering data from the Fresnel Institute benchmark datasets.
The data format is a text file with columns:
    [Tx_index, Rx_index, Freq_index, Re_scat, Im_scat, Re_inc, Im_inc]

MATLAB equivalent: load_data_fr2001.p

Reference:
    Geffrin et al., "Free space experimental scattering database,"
    Inverse Problems, 2005
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Union


# Fresnel dataset frequency mapping (8 frequencies version)
# Index 1 corresponds to 2 GHz, index 8 to 16 GHz
FRESNEL_FREQUENCIES_8F = {
    1: 2e9,   # 2 GHz
    2: 3e9,   # 3 GHz
    3: 4e9,   # 4 GHz  <- Default used in exercises
    4: 5e9,   # 5 GHz
    5: 6e9,   # 6 GHz
    6: 7e9,   # 7 GHz
    7: 8e9,   # 8 GHz
    8: 10e9,  # 10 GHz (or 16 GHz depending on dataset)
}


def load_fresnel_data(
    filepath: Union[str, Path],
    freq: float,
    nx: int,
    ny: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load Fresnel Institute experimental data.

    MATLAB equivalent:
        [Escat, Einc_domain] = load_data_fr2001(freq, dataset, Nx, Ny)

    Data file format (whitespace separated):
        Tx_idx  Rx_idx  Freq_idx  Re_scat  Im_scat  Re_inc  Im_inc

    Args:
        filepath: Path to the data file (e.g., 'dielTM_dec8f.txt')
        freq: Desired frequency in Hz (e.g., 4e9 for 4 GHz)
        nx: Number of x grid points (for Einc_domain shape)
        ny: Number of y grid points (for Einc_domain shape)

    Returns:
        Tuple of (Escat, Einc_domain) where:
            Escat: Scattered field matrix (Nm × Nv)
            Einc_domain: Incident field (Ny × Nx × Nv) - Note: simplified here

    Note:
        The original MATLAB function computes Einc_domain on the grid.
        This simplified version returns a placeholder; the actual incident
        field computation is done in the inversion code.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Fresnel data file not found: {filepath}")

    # Determine frequency index
    freq_idx = _frequency_to_index(freq)

    # Load raw data
    raw_data = np.loadtxt(filepath)

    # Data columns: Tx, Rx, Freq, Re_scat, Im_scat, Re_inc, Im_inc
    tx_idx = raw_data[:, 0].astype(int)
    rx_idx = raw_data[:, 1].astype(int)
    freq_col = raw_data[:, 2].astype(int)
    re_scat = raw_data[:, 3]
    im_scat = raw_data[:, 4]
    re_inc = raw_data[:, 5]
    im_inc = raw_data[:, 6]

    # Filter for desired frequency
    mask = freq_col == freq_idx
    tx_idx = tx_idx[mask]
    rx_idx = rx_idx[mask]
    re_scat = re_scat[mask]
    im_scat = im_scat[mask]
    re_inc = re_inc[mask]
    im_inc = im_inc[mask]

    # Determine dimensions
    n_tx = len(np.unique(tx_idx))
    n_rx = len(np.unique(rx_idx))

    # Map indices to 0-based
    tx_map = {old: new for new, old in enumerate(sorted(np.unique(tx_idx)))}
    rx_map = {old: new for new, old in enumerate(sorted(np.unique(rx_idx)))}

    # Build Escat matrix (Nm × Nv) = (n_rx × n_tx)
    Escat = np.zeros((n_rx, n_tx), dtype=complex)
    Einc_raw = np.zeros((n_rx, n_tx), dtype=complex)

    for i in range(len(tx_idx)):
        tx_i = tx_map[tx_idx[i]]
        rx_i = rx_map[rx_idx[i]]
        Escat[rx_i, tx_i] = re_scat[i] + 1j * im_scat[i]
        Einc_raw[rx_i, tx_i] = re_inc[i] + 1j * im_inc[i]

    # For Einc_domain, we need to compute incident field on the DoI grid
    # This is typically done using the incident field computation functions
    # For now, return a placeholder that will be computed in the scenario script
    # The actual Einc_domain should be computed using compute_incident_field_all_views

    # Create a simplified Einc_domain placeholder
    # In practice, this is computed from the transmitter positions
    Nv = n_tx
    Einc_domain = np.zeros((ny, nx, Nv), dtype=complex)

    # Store Einc_raw for reference (incident field at receiver positions)
    # This can be used for calibration purposes

    return Escat, Einc_domain


def _frequency_to_index(freq: float, tolerance: float = 0.1e9) -> int:
    """
    Convert frequency in Hz to Fresnel data file index.

    Args:
        freq: Frequency in Hz
        tolerance: Tolerance for frequency matching

    Returns:
        Frequency index (1-based as in data file)

    Raises:
        ValueError: If frequency not found in lookup table
    """
    for idx, f in FRESNEL_FREQUENCIES_8F.items():
        if abs(freq - f) < tolerance:
            return idx

    available = [f/1e9 for f in FRESNEL_FREQUENCIES_8F.values()]
    raise ValueError(
        f"Frequency {freq/1e9:.1f} GHz not in dataset. "
        f"Available: {available} GHz"
    )


def get_fresnel_parameters() -> dict:
    """
    Get standard Fresnel experiment parameters.

    These are the parameters used in the exercises:
        - freq: 4 GHz
        - DoI: 15 cm × 15 cm
        - Grid: 64 × 64
        - Rv: 72.135 cm (transmitter radius)
        - Rm: 76.135 cm (receiver radius)
        - Nm: 49 receivers
        - Nv: 36 transmitters

    Returns:
        Dictionary of standard Fresnel parameters
    """
    return {
        'freq': 4e9,           # 4 GHz
        'lambda0': 0.075,      # 7.5 cm
        'lx': 0.15,            # 15 cm
        'ly': 0.15,            # 15 cm
        'Nx': 64,
        'Ny': 64,
        'Rv': 0.72135,         # Transmitter radius [m]
        'Rm': 0.76135,         # Receiver radius [m]
        'eb': 1.0,             # Free space background
        'sb': 0.0,             # No background conductivity
        # Note: Nm and Nv depend on the specific dataset
    }


def load_data_fr2001(
    freq: float,
    dataset: str,
    Nx: int,
    Ny: int,
    data_dir: Optional[Union[str, Path]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    MATLAB-compatible interface for loading Fresnel data.

    MATLAB equivalent:
        [Escat, Einc_domain] = load_data_fr2001(freq, dataset, Nx, Ny)

    Args:
        freq: Frequency in Hz (e.g., 4e9)
        dataset: Dataset filename (e.g., 'dielTM_dec8f.txt')
        Nx: Number of x grid points
        Ny: Number of y grid points
        data_dir: Directory containing data file (optional)

    Returns:
        Tuple of (Escat, Einc_domain)
    """
    if data_dir is not None:
        filepath = Path(data_dir) / dataset
    else:
        filepath = Path(dataset)

    return load_fresnel_data(filepath, freq, Nx, Ny)
