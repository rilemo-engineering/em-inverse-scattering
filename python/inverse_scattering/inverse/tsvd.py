"""
Truncated Singular Value Decomposition (TSVD) solver for inverse scattering.

TSVD is a regularization technique that addresses the ill-posed nature of
the inverse scattering problem by discarding small singular values that
are dominated by noise.

Theory:
    Given the linearized problem: E_scat = S * τ

    SVD of S: S = U * Σ * V^T

    Naive inversion: τ = V * Σ^(-1) * U^T * E_scat

    This amplifies noise for small σ_i. TSVD truncates:

    τ_TSVD = Σ(i=1 to k) (u_i^T * E_scat / σ_i) * v_i

    where k is chosen based on noise level (truncation threshold).

MATLAB equivalent: TSVD_solver.p

Reference:
    - Theory document: 01_THEORY_AND_CONCEPTS.md, Section 5.3
    - Quick reference: 05_QUICK_REFERENCE.md
"""

import numpy as np
from typing import Tuple, Optional


def compute_svd(
    S: np.ndarray,
    full_matrices: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute SVD of the scattering operator matrix.

    S = U * diag(s) * V^H

    MATLAB equivalent:
        [U, S, V] = svd(S_BORN)
        [U, S, V] = svd(S_BORN, 'econ')  % for experimental data

    Args:
        S: Scattering operator matrix (Nm*Nv × Nx*Ny)
        full_matrices: If False, compute economy SVD (default)

    Returns:
        Tuple of (U, s, Vh) where:
            U: Left singular vectors (Nm*Nv × k)
            s: Singular values (k,) - 1D array, not diagonal matrix
            Vh: Right singular vectors conjugate transpose (k × Nx*Ny)

    Note:
        numpy.linalg.svd returns Vh (V hermitian), while MATLAB's svd
        returns V. So: V_matlab = Vh.T.conj() in Python.
    """
    U, s, Vh = np.linalg.svd(S, full_matrices=full_matrices)
    return U, s, Vh


def find_truncation_index(
    singular_values: np.ndarray,
    threshold_db: float
) -> int:
    """
    Find the truncation index for a given threshold in dB.

    The threshold is relative to the largest singular value:
        threshold_dB = 20 * log10(σ_k / σ_1)

    MATLAB equivalent:
        norm_sing_val = abs(S1) ./ abs(S1(1));
        [~, Nt] = min(abs(20*log10(norm_sing_val) - threshold_dB));

    Args:
        singular_values: Array of singular values (sorted descending)
        threshold_db: Threshold in dB (negative value, e.g., -25)

    Returns:
        Truncation index k (1-indexed for MATLAB compatibility)

    Example:
        >>> s = np.array([1.0, 0.5, 0.1, 0.01, 0.001])
        >>> find_truncation_index(s, -20)  # 20 dB below max
        3  # Keeps first 3 singular values
    """
    # Normalized singular values
    norm_sv = np.abs(singular_values) / np.abs(singular_values[0])

    # Convert to dB
    sv_db = 20 * np.log10(norm_sv + 1e-15)  # Add small value to avoid log(0)

    # Find index closest to threshold
    # MATLAB: [~, Nt] = min(abs(20*log10(norm_sing_val) - threshold_dB))
    idx = np.argmin(np.abs(sv_db - threshold_db))

    # Return 1-indexed for MATLAB compatibility
    return idx + 1


def tsvd_solve(
    U: np.ndarray,
    s: np.ndarray,
    Vh: np.ndarray,
    truncation_index: int,
    data: np.ndarray,
    nx: int,
    ny: int
) -> np.ndarray:
    """
    Solve the inverse problem using TSVD.

    τ_TSVD = Σ(i=0 to k-1) (u_i^H * E_scat / σ_i) * v_i

    MATLAB equivalent (TSVD_solver.p):
        PROF_rec_BORN = TSVD_solver(U, S, V, Nt, data_BORN, Nx, Ny)

    Args:
        U: Left singular vectors from SVD (Nm*Nv × k)
        s: Singular values (k,) - 1D array
        Vh: Right singular vectors (conjugate transpose) (k × Nx*Ny)
        truncation_index: Number of singular values to keep (1-indexed)
        data: Scattered field data (Nm × Nv) or flattened (Nm*Nv,)
        nx: Number of grid points in x
        ny: Number of grid points in y

    Returns:
        Reconstructed contrast profile (ny × nx)

    Note:
        The MATLAB code passes S as diag(s), but internally uses diag(S).
        We accept s as 1D array directly.
    """
    # Flatten data if needed - use Fortran order to match MATLAB's column-major convention
    # MATLAB: data(:) vectorizes column-by-column
    data_flat = np.ravel(data, order='F')

    # Number of singular values to use (convert to 0-indexed)
    k = truncation_index

    # TSVD solution: τ = Σ (u_i^H * data / σ_i) * v_i
    # Using matrix form: τ = V[:k,:].T @ (diag(1/s[:k]) @ U[:,:k].T.conj() @ data)
    #
    # Note: For ill-conditioned matrices (typical in inverse scattering), numpy 2.0
    # may emit overflow warnings during matmul. These are expected and don't affect
    # the TSVD result since we truncate small singular values. Suppress them.

    with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
        # Coefficient vector: c_i = u_i^H * data / σ_i
        # Vectorized: U[:, :k].T.conj() @ data_flat gives all u_i^H * data at once
        ui_H_data = U[:, :k].T.conj() @ data_flat  # (k,)

        # Divide by singular values
        coefficients = ui_H_data / s[:k]

        # Reconstruct: τ = Σ c_i * v_i = Vh[:k,:].T.conj() @ coefficients
        tau_flat = Vh[:k, :].T.conj() @ coefficients

    # Reshape to (ny, nx) - use Fortran order to match MATLAB's reshape
    # MATLAB: reshape(tau_flat, Ny, Nx) fills column-by-column
    tau_rec = tau_flat.reshape((ny, nx), order='F')

    return tau_rec


def tsvd_solver_matlab_interface(
    U: np.ndarray,
    S: np.ndarray,
    V: np.ndarray,
    Nt: int,
    data: np.ndarray,
    Nx: int,
    Ny: int
) -> np.ndarray:
    """
    MATLAB-compatible interface for TSVD solver.

    This matches the exact calling convention of TSVD_solver.p:
        PROF_rec_BORN = TSVD_solver(U, S, V, Nt, data_BORN, Nx, Ny)

    MATLAB convention:
        - S is a diagonal matrix (or full SVD output)
        - V is V (not V^H)

    Args:
        U: Left singular vectors (from MATLAB svd)
        S: Singular value diagonal matrix or full S matrix (from MATLAB svd)
        V: Right singular vectors V (not V^H) (from MATLAB svd)
        Nt: Truncation index (1-indexed)
        data: Scattered field data matrix (Nm × Nv)
        Nx: Number of grid points in x
        Ny: Number of grid points in y

    Returns:
        Reconstructed contrast profile (Ny × Nx)
    """
    # Extract singular values from diagonal
    if S.ndim == 2:
        s = np.diag(S)
    else:
        s = S

    # Convert V to Vh (MATLAB V = numpy Vh.T.conj())
    Vh = V.T.conj()

    return tsvd_solve(U, s, Vh, Nt, data, Nx, Ny)


def suggest_threshold(snr_db: float) -> float:
    """
    Suggest a truncation threshold based on SNR.

    Rule of thumb from documentation:
        threshold [dB] ≈ -(SNR - 5)

    MATLAB documentation:
        SNR = 40 dB → threshold = -35 dB
        SNR = 30 dB → threshold = -25 dB
        SNR = 20 dB → threshold = -15 dB
        SNR = 10 dB → threshold = -5 dB

    Args:
        snr_db: Signal-to-noise ratio in dB

    Returns:
        Suggested truncation threshold in dB
    """
    return -(snr_db - 5)
