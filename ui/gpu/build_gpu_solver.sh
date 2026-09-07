#!/usr/bin/env bash
# Run this ON THE GPU LAPTOP, in the same directory as gpu_solver.cu.
# Builds the CUDA PDHG solver binary the frontend shells out to.
set -euo pipefail

if ! command -v nvcc >/dev/null 2>&1; then
    echo "ERROR: nvcc not found. Install the CUDA Toolkit first" \
         "(https://developer.nvidia.com/cuda-downloads) and make sure" \
         "nvcc is on PATH." >&2
    exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "WARNING: nvidia-smi not found -- no GPU driver detected." \
         "Compilation may still succeed but the binary will fail at" \
         "runtime with a CUDA device error." >&2
fi

echo "Compiling gpu_solver.cu ..."
nvcc -O3 gpu_solver.cu -o gpu_solver -lcublas -lcusparse

echo "Build OK -> ./gpu_solver"
echo
echo "Smoke test:"
nvidia-smi -L 2>/dev/null || true
