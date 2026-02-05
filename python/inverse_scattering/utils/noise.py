"""
Noise generation utilities for inverse scattering simulations.

This module provides functions to add noise to signals, primarily for
simulating realistic measurement conditions in inverse scattering.

MATLAB equivalent: awgn() function from Communications Toolbox
"""

import numpy as np
from typing import Optional, Union


def awgn(
    signal: np.ndarray,
    snr_db: float,
    signal_power: str = 'measured',
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Add White Gaussian Noise to a signal at specified SNR.

    MATLAB equivalent:
        Escat = awgn(Escat, SNR, 'measured', 345)

    SNR is defined as:
        SNR = 10 * log10(P_signal / P_noise)

    So:
        P_noise = P_signal / 10^(SNR/10)

    Args:
        signal: Input signal (can be complex)
        snr_db: Signal-to-noise ratio in dB
        signal_power: 'measured' to compute from signal, or float value
        seed: Random seed for reproducibility (MATLAB uses this for repeatable noise)

    Returns:
        Noisy signal with same shape as input

    Example:
        >>> Escat_noisy = awgn(Escat, 30, 'measured', 345)  # 30 dB SNR
    """
    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)

    # Compute signal power
    if signal_power == 'measured':
        # MATLAB 'measured': compute average power from signal
        sig_power = np.mean(np.abs(signal.ravel())**2)
    else:
        sig_power = float(signal_power)

    # Compute noise power from SNR
    # SNR_dB = 10*log10(P_sig/P_noise) → P_noise = P_sig / 10^(SNR/10)
    noise_power = sig_power / (10 ** (snr_db / 10))

    # Generate complex Gaussian noise if signal is complex
    if np.iscomplexobj(signal):
        # For complex signals, split power equally between real and imaginary
        # Variance of each component = noise_power / 2
        std_dev = np.sqrt(noise_power / 2)
        noise = std_dev * (np.random.randn(*signal.shape) +
                          1j * np.random.randn(*signal.shape))
    else:
        # For real signals
        std_dev = np.sqrt(noise_power)
        noise = std_dev * np.random.randn(*signal.shape)

    return signal + noise


def estimate_snr(
    noisy_signal: np.ndarray,
    clean_signal: np.ndarray
) -> float:
    """
    Estimate SNR between noisy and clean signals.

    SNR = 10 * log10(P_signal / P_noise)

    where P_noise = mean(|noisy - clean|^2)

    Args:
        noisy_signal: Signal with noise
        clean_signal: Original clean signal

    Returns:
        Estimated SNR in dB
    """
    signal_power = np.mean(np.abs(clean_signal.ravel())**2)
    noise_power = np.mean(np.abs((noisy_signal - clean_signal).ravel())**2)

    if noise_power < 1e-15:
        return np.inf

    return 10 * np.log10(signal_power / noise_power)


def add_noise_snr(
    signal: np.ndarray,
    snr_db: float,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Simplified interface for adding noise at specified SNR.

    Equivalent to awgn(signal, snr_db, 'measured', seed).

    Args:
        signal: Input signal
        snr_db: Desired SNR in dB
        seed: Random seed

    Returns:
        Noisy signal
    """
    return awgn(signal, snr_db, 'measured', seed)
