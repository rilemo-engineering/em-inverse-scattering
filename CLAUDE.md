# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a MATLAB educational codebase for **electromagnetic inverse scattering** problems. It demonstrates solving the 2D TM (Transverse Magnetic) scalar wave scattering problem using the Born Approximation and TSVD (Truncated Singular Value Decomposition) regularization.

## Repository Structure

```
matlab/
├── simulated/
│   ├── scenario/         # Forward problem: generate synthetic scattered field data
│   │   ├── c1_Scenario.m # Main script - configures and runs forward solver
│   │   └── *.p           # Protected functions (forward_solver, CGFFT, etc.)
│   └── inversion/        # Inverse problem: reconstruct profiles from simulated data
│       ├── c2_Inversion_BORN.m   # Main script - TSVD inversion with Born approximation
│       └── *.p           # Protected functions (kernel_scattering, TSVD_solver)
│
└── experimental/
    ├── scenario/         # Load Fresnel experimental dataset
    │   ├── c1_Scenario_ExpData.m
    │   └── *.txt         # Raw experimental data files
    └── inversion/        # Inversion on experimental data
        └── c2_Inversion_ExpData_BORN.m

python/                   # Python port (see python/README.md for details)
docs/                     # Theory documentation
```

## Workflow

### Simulated Data Pipeline
1. **Run** `matlab/simulated/scenario/c1_Scenario.m` to generate forward scattering data
   - Configures investigation domain, frequencies, background properties
   - Calls `forward_solver.p` (iterative CGFFT-based solver)
   - Outputs `DATA_scenario.mat` with: `Escat`, `PROF`, `Einc_domain`, `Etot_domain`

2. **Run** `matlab/simulated/inversion/c2_Inversion_BORN.m` to reconstruct profiles
   - Loads `DATA_scenario.mat`
   - Prompts for SVD truncation threshold (dB)
   - Computes reconstruction via Born Approximation + TSVD

### Experimental Data Pipeline
1. **Run** `matlab/experimental/scenario/c1_Scenario_ExpData.m` to load Fresnel Institute datasets
2. **Run** `matlab/experimental/inversion/c2_Inversion_ExpData_BORN.m` for inversion

## Key Physical Parameters

| Parameter | Description |
|-----------|-------------|
| `freq` | Working frequency [Hz] |
| `eb`, `sb` | Background permittivity and conductivity |
| `lx`, `ly` | Investigation domain dimensions [m] |
| `Nx`, `Ny` | Discretization grid size |
| `Rm` | Measurement surface radius [m] |
| `Nm`, `Nv` | Number of receivers/transmitters |
| `PROF` | Contrast function τ = ε_r - ε_b (complex) |

## Key Variables in Data Files

- `Escat` - Scattered field data matrix (Nm × Nv)
- `Einc_domain` - Incident field on investigation domain (Ny × Nx × Nv)
- `Etot_domain` - Total field on investigation domain (Ny × Nx × Nv)
- `PROF` - Object contrast profile (Ny × Nx)

## Important Notes

- The `.p` files are protected MATLAB functions (compiled) - cannot be edited
- Inversion scripts are interactive: they prompt for truncation threshold
- SNR for noise addition can be adjusted in `c2_Inversion_BORN.m` (default: 30 dB)
- Aspect-limited configurations available by setting `AL=1`
- Experimental datasets: `dielTM_dec8f.txt` (single cylinder), `twodielTM_8f.txt` (two cylinders)

## Python Implementation

A complete Python port of the inverse scattering code is available in the `python/` directory.

### Setup (Apple Silicon)

```bash
# Install miniforge (required for Accelerate framework)
brew install miniforge
conda init "$(basename "${SHELL}")"
# Restart terminal, then:

# Create optimized environment
conda create -n inverse_accel python=3.9 "libblas=*=*accelerate" numpy scipy matplotlib h5py pytest -y
conda activate inverse_accel
cd python && pip install -e . --no-deps
```

**Why this matters:** Default pip numpy uses OpenBLAS (~32s for SVD). Conda with Accelerate uses Apple's optimized framework (~0.14s for SVD) - **~230x faster**, matching MATLAB performance.

### Running the Python Inversion

```bash
conda activate inverse_accel
cd python
python -c "
from inverse_scattering.scripts.inversion_born import run_inversion_born
run_inversion_born(data_file='../matlab/simulated/inversion/DATA_scenario_square.mat',
                   snr_db=30, threshold_db=-25, visualize=True)
"
```

### Python-MATLAB Correspondence

| MATLAB | Python | Notes |
|--------|--------|-------|
| `kernel_scattering.p` | `inverse/scattering_kernel.py` | Uses H₀⁽²⁾ Green's function |
| `TSVD_solver.p` | `inverse/tsvd.py` | Full SVD then truncate |
| `awgn()` | `utils/noise.py` | Same SNR formula |
| `e0, m0` | `core/constants.py` | 8.85e-12, 4π×10⁻⁷ |

### Validated Results

NMSE comparison (square target, SNR=40dB, threshold=-25dB):
- **MATLAB**: 0.1542
- **Python**: 0.1546
- **Match**: ✓ (difference due to different RNG for noise)
