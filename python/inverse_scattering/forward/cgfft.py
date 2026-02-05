"""
Conjugate Gradient FFT (CGFFT) solver for forward scattering.

The CGFFT method solves the Lippmann-Schwinger equation iteratively:

    E_tot = E_inc + A @ (τ * E_tot)

Rearranged as:
    (I - D_τ @ A) @ E_tot = E_inc

where D_τ is a diagonal matrix with τ on the diagonal.

The FFT acceleration exploits the Toeplitz structure of A when the
grid is uniform, reducing each matrix-vector product from O(N²) to O(N log N).

MATLAB equivalent: CGFFT.p

Reference:
    P. Zwamborn and P.M. van den Berg, "The three-dimensional weak form
    of the conjugate gradient FFT method for solving scattering problems,"
    IEEE Trans. Microwave Theory Tech., 1992.
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass

from inverse_scattering.forward.internal_operator import (
    build_toeplitz_green,
    fft_green,
    apply_operator_fft
)


@dataclass
class CGFFTResult:
    """Result of CGFFT solver."""
    E_tot: np.ndarray  # Total field (Ny, Nx)
    converged: bool
    n_iterations: int
    residual_history: list
    final_residual: float


def cgfft_solve(
    E_inc: np.ndarray,
    tau: np.ndarray,
    k: complex,
    dx: float,
    dy: float,
    max_iter: int = 1000,
    tol: float = 1e-6,
    verbose: bool = False
) -> CGFFTResult:
    """
    Solve for total field using Conjugate Gradient FFT method.

    Solves: E_tot = E_inc + k² ∫∫ G(r,r') τ(r') E_tot(r') dr'

    In matrix form: (I - D_τ @ A) @ E_tot = E_inc

    MATLAB equivalent: CGFFT.p

    Args:
        E_inc: Incident field (Ny × Nx)
        tau: Contrast function (Ny × Nx)
        k: Wavenumber (complex)
        dx: Cell size in x [m]
        dy: Cell size in y [m]
        max_iter: Maximum number of CG iterations
        tol: Convergence tolerance (relative residual)
        verbose: Print convergence info

    Returns:
        CGFFTResult with total field and convergence info
    """
    Ny, Nx = E_inc.shape

    # Build Toeplitz Green's function and its FFT
    G_toeplitz = build_toeplitz_green(Nx, Ny, k, dx, dy)
    G_fft = fft_green(G_toeplitz)

    # Initial guess: E_tot = E_inc
    E_tot = E_inc.copy()

    # Residual: r = E_inc - (I - D_τ @ A) @ E_tot = E_inc - E_tot + A @ (τ * E_tot)
    # Actually: r = b - A @ x where A = (I - D_τ @ G_op) and b = E_inc
    # So: r = E_inc - E_tot + G_op @ (τ * E_tot)

    def apply_system_operator(E: np.ndarray) -> np.ndarray:
        """Apply (I - D_τ @ G) to E."""
        tau_E = tau * E
        G_tau_E = apply_operator_fft(tau_E, G_fft, Nx, Ny)
        return E - G_tau_E

    # Initial residual
    r = E_inc - apply_system_operator(E_tot)
    p = r.copy()
    r_norm_sq = np.sum(np.abs(r)**2)
    r0_norm = np.sqrt(r_norm_sq)

    if verbose:
        print(f"CGFFT: Initial residual = {r0_norm:.6e}")

    residual_history = [r0_norm]

    for iteration in range(max_iter):
        # Ap = (I - D_τ @ G) @ p
        Ap = apply_system_operator(p)

        # Step size
        pAp = np.sum(np.conj(p) * Ap)
        alpha = r_norm_sq / pAp

        # Update solution
        E_tot = E_tot + alpha * p

        # Update residual
        r = r - alpha * Ap
        r_norm_sq_new = np.sum(np.abs(r)**2)
        r_norm = np.sqrt(r_norm_sq_new)

        residual_history.append(r_norm)

        # Check convergence
        relative_residual = r_norm / r0_norm
        if verbose and (iteration + 1) % 100 == 0:
            print(f"  Iteration {iteration+1}: residual = {relative_residual:.6e}")

        if relative_residual < tol:
            if verbose:
                print(f"CGFFT: Converged at iteration {iteration+1}")
            return CGFFTResult(
                E_tot=E_tot,
                converged=True,
                n_iterations=iteration + 1,
                residual_history=residual_history,
                final_residual=relative_residual
            )

        # Update search direction
        beta = r_norm_sq_new / r_norm_sq
        p = r + beta * p
        r_norm_sq = r_norm_sq_new

    if verbose:
        print(f"CGFFT: Did not converge after {max_iter} iterations")

    return CGFFTResult(
        E_tot=E_tot,
        converged=False,
        n_iterations=max_iter,
        residual_history=residual_history,
        final_residual=np.sqrt(r_norm_sq) / r0_norm
    )


def cgfft_solve_all_views(
    Einc_domain: np.ndarray,
    tau: np.ndarray,
    k: complex,
    dx: float,
    dy: float,
    max_iter: int = 1000,
    tol: float = 1e-6,
    verbose: bool = False
) -> np.ndarray:
    """
    Solve for total field for all transmitter views.

    Args:
        Einc_domain: Incident fields (Ny × Nx × Nv)
        tau: Contrast function (Ny × Nx)
        k: Wavenumber
        dx, dy: Cell sizes
        max_iter: Maximum CG iterations
        tol: Convergence tolerance
        verbose: Print progress

    Returns:
        Etot_domain: Total fields (Ny × Nx × Nv)
    """
    Ny, Nx, Nv = Einc_domain.shape
    Etot_domain = np.zeros_like(Einc_domain)

    # Build Toeplitz Green's function once (reused for all views)
    G_toeplitz = build_toeplitz_green(Nx, Ny, k, dx, dy)
    G_fft = fft_green(G_toeplitz)

    def apply_system_operator(E: np.ndarray) -> np.ndarray:
        tau_E = tau * E
        G_tau_E = apply_operator_fft(tau_E, G_fft, Nx, Ny)
        return E - G_tau_E

    for v in range(Nv):
        if verbose:
            print(f"\nSolving view {v+1}/{Nv}")

        E_inc = Einc_domain[:, :, v]
        E_tot = E_inc.copy()

        # CG iteration
        r = E_inc - apply_system_operator(E_tot)
        p = r.copy()
        r_norm_sq = np.sum(np.abs(r)**2)
        r0_norm = np.sqrt(r_norm_sq)

        for iteration in range(max_iter):
            Ap = apply_system_operator(p)
            pAp = np.sum(np.conj(p) * Ap)
            alpha = r_norm_sq / pAp

            E_tot = E_tot + alpha * p
            r = r - alpha * Ap
            r_norm_sq_new = np.sum(np.abs(r)**2)

            if np.sqrt(r_norm_sq_new) / r0_norm < tol:
                break

            beta = r_norm_sq_new / r_norm_sq
            p = r + beta * p
            r_norm_sq = r_norm_sq_new

        Etot_domain[:, :, v] = E_tot

        if verbose:
            print(f"  View {v+1}: {iteration+1} iterations")

    return Etot_domain


def bicgstab_solve(
    E_inc: np.ndarray,
    tau: np.ndarray,
    k: complex,
    dx: float,
    dy: float,
    max_iter: int = 1000,
    tol: float = 1e-6,
    verbose: bool = False
) -> CGFFTResult:
    """
    Solve using BiCGSTAB (Bi-Conjugate Gradient Stabilized).

    BiCGSTAB can be more stable than standard CG for non-symmetric systems.

    Args:
        Same as cgfft_solve

    Returns:
        CGFFTResult with total field and convergence info
    """
    Ny, Nx = E_inc.shape

    # Build operator
    G_toeplitz = build_toeplitz_green(Nx, Ny, k, dx, dy)
    G_fft = fft_green(G_toeplitz)

    def apply_A(E: np.ndarray) -> np.ndarray:
        tau_E = tau * E
        G_tau_E = apply_operator_fft(tau_E, G_fft, Nx, Ny)
        return E - G_tau_E

    # Initial guess
    x = E_inc.copy()
    b = E_inc

    # Initial residual
    r = b - apply_A(x)
    r_tilde = r.copy()  # Shadow residual

    rho = np.sum(np.conj(r_tilde) * r)
    p = r.copy()

    r0_norm = np.sqrt(np.sum(np.abs(r)**2))
    residual_history = [r0_norm]

    for iteration in range(max_iter):
        # Ap
        v = apply_A(p)

        alpha = rho / np.sum(np.conj(r_tilde) * v)

        s = r - alpha * v
        s_norm = np.sqrt(np.sum(np.abs(s)**2))

        if s_norm / r0_norm < tol:
            x = x + alpha * p
            return CGFFTResult(
                E_tot=x,
                converged=True,
                n_iterations=iteration + 1,
                residual_history=residual_history,
                final_residual=s_norm / r0_norm
            )

        # As
        t = apply_A(s)

        omega = np.sum(np.conj(t) * s) / np.sum(np.conj(t) * t)

        x = x + alpha * p + omega * s
        r = s - omega * t

        r_norm = np.sqrt(np.sum(np.abs(r)**2))
        residual_history.append(r_norm)

        if r_norm / r0_norm < tol:
            return CGFFTResult(
                E_tot=x,
                converged=True,
                n_iterations=iteration + 1,
                residual_history=residual_history,
                final_residual=r_norm / r0_norm
            )

        rho_new = np.sum(np.conj(r_tilde) * r)
        beta = (rho_new / rho) * (alpha / omega)
        p = r + beta * (p - omega * v)
        rho = rho_new

    return CGFFTResult(
        E_tot=x,
        converged=False,
        n_iterations=max_iter,
        residual_history=residual_history,
        final_residual=r_norm / r0_norm
    )
