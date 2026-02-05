# Performance Optimization for Apple Silicon

## Problem

The default pip-installed NumPy uses OpenBLAS, which is **not optimized for Apple Silicon**. This causes SVD operations to be extremely slow (~30+ seconds for a 400×1024 matrix).

## Solution

Use conda with the **Apple Accelerate framework** instead of OpenBLAS. This provides ~100-200x speedup for linear algebra operations.

### Performance Comparison

| Operation | OpenBLAS (pip) | Accelerate (conda) | Speedup |
|-----------|----------------|---------------------|---------|
| SVD (400×1024 complex) | 32.5s | 0.14s | **~230x** |
| Born Inversion (total) | 32.6s | 0.26s | **~125x** |

## Setup Instructions

### Option 1: Quick Setup (Recommended)

```bash
# Install miniforge (if not already installed)
brew install miniforge

# Initialize conda (restart terminal after this)
conda init "$(basename "${SHELL}")"

# Create environment with Accelerate
conda create -n inverse_accel python=3.9 "libblas=*=*accelerate" numpy scipy matplotlib h5py pytest -y

# Activate and install project
conda activate inverse_accel
pip install -e . --no-deps
pip install pytest pytest-cov toml
```

### Option 2: Using the Setup Script

```bash
chmod +x setup_accelerate.sh
./setup_accelerate.sh
```

## Usage

Always activate the conda environment before running:

```bash
conda activate inverse_accel
python -m inverse_scattering.scripts.inversion_born
```

## Verification

Check that NumPy is using Accelerate (not OpenBLAS):

```python
import numpy as np
np.show_config()
```

Look for `"blas": {"name": "blas", ...}` (NOT `"openblas64"`).

Quick benchmark:
```python
import numpy as np
import time

A = np.random.randn(400, 1024) + 1j * np.random.randn(400, 1024)
t0 = time.perf_counter()
np.linalg.svd(A, full_matrices=False)
print(f"SVD time: {time.perf_counter()-t0:.4f}s")  # Should be < 0.2s
```

## Why This Works

- **MATLAB uses Apple's Accelerate** framework for linear algebra, which is highly optimized for Apple Silicon
- **pip-installed NumPy/SciPy** use OpenBLAS, which has poor ARM64/Apple Silicon optimization
- **conda-forge packages** can be built with Accelerate on macOS, matching MATLAB's performance

## Troubleshooting

### SVD still slow after conda install?

pip may have overwritten the conda numpy. Fix with:
```bash
conda install -y --force-reinstall "libblas=*=*accelerate" numpy
```

### Import errors after activating conda?

Reinstall the project:
```bash
pip install -e . --no-deps
```
