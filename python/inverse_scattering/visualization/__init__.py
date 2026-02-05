"""
Visualization module: Plotting functions matching MATLAB figures.

This module provides matplotlib-based plotting functions that recreate
the MATLAB visualization from the original exercises.
"""

from inverse_scattering.visualization.plots import (
    plot_profile,
    plot_scenario,
    plot_singular_values,
    plot_reconstruction_comparison,
    plot_cross_sections,
    show,
)

__all__ = [
    "plot_profile",
    "plot_scenario",
    "plot_singular_values",
    "plot_reconstruction_comparison",
    "plot_cross_sections",
    "show",
]
