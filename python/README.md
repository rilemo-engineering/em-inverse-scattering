# Inverse Scattering - Python Port

> **Disclaimer:** This Python port and its documentation were 100% vibecoded (AI-generated using Claude).

A Python implementation of 2D TM electromagnetic inverse scattering exercises using Born approximation and TSVD regularization.

This is a **1:1 port** of the original MATLAB exercises, maintaining the same algorithms, parameters, and workflow.

## Installation

### Recommended: Conda with Accelerate (Apple Silicon)

For optimal performance on Apple Silicon Macs (~125x faster than pip):

```bash
# Install miniforge if not already installed
brew install miniforge
conda init "$(basename "${SHELL}")"
# Restart terminal, then:

# Create environment with Apple Accelerate framework
conda create -n inverse_accel python=3.9 "libblas=*=*accelerate" numpy scipy matplotlib h5py pytest -y
conda activate inverse_accel
cd python
pip install -e . --no-deps
pip install pytest pytest-cov toml
```

### Alternative: Poetry (slower on Apple Silicon)

```bash
cd python
poetry install
```

**Warning:** Poetry/pip uses OpenBLAS which is ~230x slower for SVD on Apple Silicon.

## Quick Start

### Simulated Data Workflow

1. **Generate Forward Problem Data (Scenario)**
   ```bash
   # With conda (recommended)
   python -m inverse_scattering.scripts.scenario

   # Or with poetry
   poetry run run-scenario
   ```
   This generates synthetic scattered field data for a defined object profile.

2. **Run Inverse Problem (Reconstruction)**
   ```bash
   # With conda (recommended)
   python -m inverse_scattering.scripts.inversion_born

   # Or with poetry
   poetry run run-inversion
   ```
   This reconstructs the object profile from scattered field data using Born approximation + TSVD.

### Experimental Data Workflow

1. **Load Fresnel Experimental Data**
   ```bash
   python -m inverse_scattering.scripts.scenario_exp
   ```

2. **Run Inversion on Experimental Data**
   ```bash
   python -m inverse_scattering.scripts.inversion_exp_born
   ```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test module
pytest tests/integration/test_matlab_scenarios.py -v
```

## Package Structure

```
inverse_scattering/
├── core/               # Physical constants, utilities, Green's functions
│   ├── constants.py    # e0, m0, c
│   ├── utils.py        # Grid generation, wavelength calculations
│   └── greens_function.py  # 2D Green's function (Hankel)
├── forward/            # Forward problem
│   ├── profiles.py     # Object profile generation (Profili.p)
│   ├── incident_field.py   # E_inc computation
│   ├── internal_operator.py # Internal field operator (ainterno.p)
│   ├── cgfft.py        # CG-FFT solver (CGFFT.p)
│   └── solver.py       # Main forward solver (forward_solver.p)
├── inverse/            # Inverse problem
│   ├── scattering_kernel.py  # Born scattering operator (kernel_scattering.p)
│   └── tsvd.py         # TSVD reconstruction (TSVD_solver.p)
├── data/               # Data handling
│   ├── fresnel_loader.py   # Fresnel .txt data (load_data_fr2001.p)
│   └── mat_io.py       # MATLAB .mat file compatibility
├── utils/              # Utilities
│   └── noise.py        # AWGN noise addition
├── visualization/      # Plotting
│   └── plots.py        # MATLAB-equivalent figures
└── scripts/            # Main executables
    ├── scenario.py     # c1_Scenario.m port
    ├── inversion_born.py   # c2_Inversion_BORN.m port
    ├── scenario_exp.py     # c1_Scenario_ExpData.m port
    └── inversion_exp_born.py # c2_Inversion_ExpData_BORN.m port
```

## Key Concepts

### Forward Problem
Given an object profile τ(x,y), compute the scattered field E_scat at measurement points.

### Inverse Problem
Given scattered field measurements E_scat, reconstruct the object profile τ(x,y).

### Born Approximation
Linearizes the inverse problem by assuming weak scattering: E_tot ≈ E_inc

### TSVD Regularization
Truncated Singular Value Decomposition controls the ill-posedness by discarding small singular values below a threshold.

## Mathematical Background

See the [`docs/`](../docs/) folder for detailed theory:
- [`01_THEORY_AND_CONCEPTS.md`](../docs/01_THEORY_AND_CONCEPTS.md) - Electromagnetic theory
- [`02_FORWARD_PROBLEM_GUIDE.md`](../docs/02_FORWARD_PROBLEM_GUIDE.md) - Forward problem exercises
- [`03_INVERSE_PROBLEM_GUIDE.md`](../docs/03_INVERSE_PROBLEM_GUIDE.md) - Inverse problem exercises
- [`04_EXPERIMENTAL_DATA_GUIDE.md`](../docs/04_EXPERIMENTAL_DATA_GUIDE.md) - Fresnel data exercises
- [`05_QUICK_REFERENCE.md`](../docs/05_QUICK_REFERENCE.md) - Formula reference card

## Compatibility

This Python port maintains full compatibility with the MATLAB implementation:
- Can read/write .mat files using scipy.io
- Produces equivalent numerical results (within floating-point tolerance)
- Uses the same parameter naming conventions

## Additional Documentation

- [`VALIDATION.md`](VALIDATION.md) - Detailed comparison with MATLAB results
- [`PERFORMANCE.md`](PERFORMANCE.md) - Performance benchmarks and optimization notes
