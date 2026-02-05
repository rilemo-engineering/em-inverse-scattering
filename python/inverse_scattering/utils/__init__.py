"""
Utilities module: Noise generation and helper functions.

This module provides utility functions:
- AWGN noise addition (matching MATLAB's awgn function)
"""

from inverse_scattering.utils.noise import awgn

__all__ = [
    "awgn",
]
