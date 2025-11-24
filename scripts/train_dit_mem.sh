#!/bin/bash
# Multi-GPU training script for DiT-Mem with 3D FFT dual attention resampler
# Uses Accelerate for distributed training across multiple GPUs

# Set project root and Python path
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
export PATH="$CONDA_PREFIX/bin:$PATH"

# GPU Configuration
# Set CUDA_VISIBLE_DEVICES to specify which GPUs to use (comma-separated, no spaces)
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Count number of available GPUs
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

# Get Python executable path
PYTHON_PATH=$(which python)

# WandB Configuration
# Set WANDB_API_KEY and WANDB_ENTITY environment variables or use defaults
# These can be set in your environment or .bashrc instead of hardcoding here
if [ -z "$WANDB_API_KEY" ]; then
    export WANDB_API_KEY="your_wandb_api_key_here"
fi
if [ -z "$WANDB_ENTITY" ]; then
    export WANDB_ENTITY="your_wandb_entity_here"
fi

# Launch distributed training with Accelerate
# --num_processes: number of GPUs to use
# --mixed_precision: bf16 for A100/H100, fp16 for older GPUs
# --multi_gpu: enable multi-GPU training
cd "$PROJECT_ROOT"
$PYTHON_PATH -m accelerate.commands.launch \
    --num_processes=$NUM_GPUS \
    --mixed_precision=bf16 \
    --multi_gpu \
    training/train_dit_mem.py \
    --config config/train_dit_mem.yaml \
    "$@"
