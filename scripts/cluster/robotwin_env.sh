#!/bin/bash
export PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export HF_HOME=/public/node03/users/panbk/data/hf-cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export HYDRA_FULL_ERROR=1
export G05_OUTPUT_DIR="$PROJECT_ROOT/runs"
export ROBOTWIN_ROOT="$PROJECT_ROOT/third_party/RoboTwin"
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/src:${PYTHONPATH:-}"
source "$PROJECT_ROOT/scripts/cluster/graphics.sh"
export VK_ICD_FILENAMES="$G05_NVIDIA_GRAPHICS_ROOT/usr/share/vulkan/icd.d/nvidia_icd.json"
export VK_DRIVER_FILES="$VK_ICD_FILENAMES"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source "$PROJECT_ROOT/.venv-robotwin/bin/activate"
