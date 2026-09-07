#!/bin/bash
# Source from any directory. Override these paths before sourcing on another account.
export PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export G05_OUTPUT_DIR="${G05_OUTPUT_DIR:-$PROJECT_ROOT/runs}"
export HF_HOME="${HF_HOME:-/public/node03/users/panbk/data/hf-cache}"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export LIBERO_DATA_ROOT="${LIBERO_DATA_ROOT:-/public/node03/users/panbk/data/FasterWAM/data/libero_mujoco3.3.2}"
export LIBERO_SOURCE_ROOT="${LIBERO_SOURCE_ROOT:-/public/node03/users/panbk/data/third_party/LIBERO}"
export LIBERO_CONFIG_PATH="$PROJECT_ROOT/experiments/libero"
export PYTHONPATH="$PROJECT_ROOT:$LIBERO_SOURCE_ROOT:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
# Verified on this cluster; model inference still uses CUDA independently.
# Override to egl only after validating the node's NVIDIA EGL vendor libraries.
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source "$PROJECT_ROOT/.venv/bin/activate"
