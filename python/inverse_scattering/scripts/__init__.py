"""
Scripts module: Main executable scripts for running exercises.

This module contains the main scripts that replicate the MATLAB exercises:

Simulated data:
- scenario.py: Port of c1_Scenario.m (forward problem)
- inversion_born.py: Port of c2_Inversion_BORN.m (Born inversion)

Experimental data:
- scenario_exp.py: Port of c1_Scenario_ExpData.m (Fresnel data setup)
- inversion_exp_born.py: Port of c2_Inversion_ExpData_BORN.m (Born inversion on exp data)

CLI entry points (via poetry run):
- run-scenario: Run simulated scenario
- run-inversion: Run Born inversion on simulated data
- run-exp-scenario: Run experimental data scenario setup
- run-exp-inversion: Run Born inversion on experimental data
"""

from inverse_scattering.scripts.scenario import run_scenario
from inverse_scattering.scripts.inversion_born import run_inversion_born
from inverse_scattering.scripts.scenario_exp import run_scenario_exp
from inverse_scattering.scripts.inversion_exp_born import run_inversion_exp_born

__all__ = [
    "run_scenario",
    "run_inversion_born",
    "run_scenario_exp",
    "run_inversion_exp_born",
]
