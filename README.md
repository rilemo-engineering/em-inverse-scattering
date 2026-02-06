# Electromagnetic Inverse Scattering

> The Python port and scientific documentation in this repository were 100% vibecoded.

Educational materials for solving 2D TM (Transverse Magnetic) electromagnetic inverse scattering problems using the Born approximation and TSVD (Truncated Singular Value Decomposition) regularization.

## Contents

This repository contains:

- **MATLAB reference code** - Original exercises with protected functions (.p files)
- **Python port** - Complete implementation of the inverse scattering algorithms
- **Documentation** - Theory guides and quick reference materials

## Repository Structure

```
.
├── matlab/                 # MATLAB reference code
│   ├── simulated/          # Simulated data exercises
│   │   ├── scenario/       # Forward problem (c1_Scenario.m)
│   │   └── inversion/      # Inverse problem (c2_Inversion_BORN.m)
│   └── experimental/       # Fresnel Institute data exercises
│       ├── scenario/       # Data loading (c1_Scenario_ExpData.m)
│       └── inversion/      # Experimental inversion
│
├── python/                 # Python implementation
│   └── README.md           # Setup and usage instructions
│
└── docs/                   # Theory documentation
    ├── 01_THEORY_AND_CONCEPTS.md
    ├── 02_FORWARD_PROBLEM_GUIDE.md
    ├── 03_INVERSE_PROBLEM_GUIDE.md
    ├── 04_EXPERIMENTAL_DATA_GUIDE.md
    └── 05_QUICK_REFERENCE.md
```

## Quick Start

### Python (Recommended)

See [python/README.md](python/README.md) for detailed setup instructions.

```bash
# Quick setup (Apple Silicon)
conda create -n inverse_accel python=3.9 "libblas=*=*accelerate" numpy scipy matplotlib h5py pytest -y
conda activate inverse_accel
cd python && pip install -e . --no-deps

# Run inversion
python -c "
from inverse_scattering.scripts.inversion_born import run_inversion_born
run_inversion_born(data_file='../matlab/simulated/inversion/DATA_scenario_square.mat',
                   snr_db=30, threshold_db=-25, visualize=True)
"
```

### MATLAB

1. Open MATLAB and navigate to `matlab/simulated/scenario/`
2. Run `c1_Scenario.m` to generate forward data (`DATA_scenario.mat`)
3. Copy the generated file to the inversion folder: `matlab/simulated/inversion/`
4. Navigate to `matlab/simulated/inversion/`
5. Run `c2_Inversion_BORN.m` to perform reconstruction

**Note:** Pre-generated data files (`DATA_scenario.mat`, `DATA_scenario_square.mat`, etc.) are already included in `matlab/simulated/inversion/`, so you can skip steps 1-3 and run the inversion directly.

## Key Concepts

| Term | Description |
|------|-------------|
| Forward Problem | Given object profile τ(x,y), compute scattered field E_scat |
| Inverse Problem | Given E_scat measurements, reconstruct τ(x,y) |
| Born Approximation | Linearization assuming E_tot ≈ E_inc (weak scattering) |
| TSVD | Regularization by truncating small singular values |
| DoI | Domain of Investigation - region being imaged |

## Physical Parameters

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| `freq` | Working frequency | 300 MHz - 4 GHz |
| `eb`, `sb` | Background permittivity/conductivity | 1.0, 0.0 (free space) |
| `lx`, `ly` | DoI dimensions | 0.1 - 0.15 m |
| `Nx`, `Ny` | Grid discretization | 32 - 64 |
| `Rm` | Measurement radius | > DoI size |
| `Nm`, `Nv` | Receivers / transmitters | 12 - 72 |

## Documentation

- **[Theory Guide](docs/01_THEORY_AND_CONCEPTS.md)** - Electromagnetic scattering fundamentals
- **[Forward Problem](docs/02_FORWARD_PROBLEM_GUIDE.md)** - Computing scattered fields
- **[Inverse Problem](docs/03_INVERSE_PROBLEM_GUIDE.md)** - Profile reconstruction
- **[Experimental Data](docs/04_EXPERIMENTAL_DATA_GUIDE.md)** - Fresnel Institute datasets
- **[Quick Reference](docs/05_QUICK_REFERENCE.md)** - Formula card

## Validation

The Python implementation produces results matching MATLAB within floating-point tolerance:

| Dataset | SNR | Threshold | MATLAB NMSE | Python NMSE |
|---------|-----|-----------|-------------|-------------|
| square | 40 dB | -25 dB | 0.1542 | 0.1546 |
| square | 30 dB | -15 dB | ~0.156 | 0.1557 |
| noweak | 30 dB | -15 dB | ~0.2-0.3 | Similar |

See [python/VALIDATION.md](python/VALIDATION.md) for detailed comparison.

## License

Educational use.

