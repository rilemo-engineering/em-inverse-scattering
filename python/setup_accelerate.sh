#!/bin/bash
# Setup script for optimized numpy/scipy with Apple Accelerate framework
#
# This fixes the slow SVD issue on Apple Silicon by using Accelerate instead of OpenBLAS.
# MATLAB uses Accelerate, so this ensures 1:1 performance parity.
#
# Usage:
#   chmod +x setup_accelerate.sh
#   ./setup_accelerate.sh

set -e

echo "============================================================"
echo "Setting up Accelerate-backed Python environment"
echo "============================================================"

# Check for Homebrew
if ! command -v brew &> /dev/null; then
    echo "Error: Homebrew not found. Install from https://brew.sh"
    exit 1
fi

# Install miniforge if not present
if ! command -v conda &> /dev/null; then
    echo ""
    echo "Installing miniforge via Homebrew..."
    brew install miniforge

    # Initialize conda
    echo "Initializing conda..."
    conda init "$(basename "${SHELL}")"

    echo ""
    echo "IMPORTANT: Please restart your terminal and run this script again."
    exit 0
fi

# Create environment with Accelerate-backed packages
ENV_NAME="inverse_scattering_accel"

echo ""
echo "Creating conda environment: $ENV_NAME"
echo "This will install numpy/scipy built with Apple Accelerate framework."

# Create environment with conda-forge packages (use libblas=*=*accelerate)
conda create -n $ENV_NAME -y python=3.9 \
    "libblas=*=*accelerate" \
    numpy scipy matplotlib \
    h5py pytest pytest-cov

echo ""
echo "Installing additional packages..."
conda activate $ENV_NAME

# Install remaining dependencies via pip
pip install scipy-io toml

echo ""
echo "============================================================"
echo "Setup complete!"
echo "============================================================"
echo ""
echo "To use the optimized environment:"
echo "  conda activate $ENV_NAME"
echo "  cd $(pwd)"
echo "  pip install -e ."
echo ""
echo "To verify Accelerate is being used:"
echo "  python -c \"import numpy; numpy.show_config()\" | grep -A5 blas"
echo ""
echo "Expected output should show 'accelerate' instead of 'openblas'"
