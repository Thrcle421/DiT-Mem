#!/bin/bash
# VBench evaluation script
# Generates videos from txt files in prompts/ subdirectory
# Each prompt generates 5 videos (total 50 videos per category)

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
export PATH="$CONDA_PREFIX/bin:$PATH"

# ============================================================================
# CONFIGURATION - PLEASE MODIFY THESE PATHS
# ============================================================================

# Model paths
BASE_MODEL="models/Wan2.1-T2V-1.3B"
CHECKPOINT_PATH="checkpoint/DiT-Mem-1.3B.safetensors"

# Retrieval system paths
RETRIEVAL_INDEX="memory_index/labels.index"
RETRIEVAL_ID_MAP="memory_index/metadata.json"
RETRIEVAL_VIDEO_DIR="videos"
LATENTS_DIR="latents"

# Output directory
OUTPUT_DIR="$PROJECT_ROOT/evaluation/vbench/outputs"

# Inference parameters
NUM_INFERENCE_STEPS=40
RETRIEVAL_K=5
FPS=16
VIDEO_QUALITY=5

# GPU configuration
GPU_ID=0

# Optional: Specify categories to process (leave empty to process all)
# Example: CATEGORIES="scene color human_action"
CATEGORIES=""

# ============================================================================
# END CONFIGURATION
# ============================================================================

echo "════════════════════════════════════════════════════════════════════════════════"
echo "VBench Evaluation"
echo "════════════════════════════════════════════════════════════════════════════════"
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Base model: $BASE_MODEL"
echo "Output directory: $OUTPUT_DIR"
echo "Inference steps: $NUM_INFERENCE_STEPS"
echo "Retrieval top-K: $RETRIEVAL_K"
echo "FPS: $FPS"
echo "Video quality: $VIDEO_QUALITY"
echo "GPU: $GPU_ID"
if [ -n "$CATEGORIES" ]; then
    echo "Categories: $CATEGORIES"
else
    echo "Categories: All"
fi
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""

# Validate input files
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "❌ Error: Checkpoint not found: $CHECKPOINT_PATH"
    echo "Please edit this script and set CHECKPOINT_PATH to your trained checkpoint"
    exit 1
fi

if [ ! -d "$BASE_MODEL" ]; then
    echo "❌ Error: Base model directory not found: $BASE_MODEL"
    echo "Please edit this script and set BASE_MODEL to your Wan2.1-T2V-1.3B model path"
    exit 1
fi

if [ ! -f "$RETRIEVAL_INDEX" ]; then
    echo "❌ Error: Retrieval index not found: $RETRIEVAL_INDEX"
    echo "Please edit this script and set RETRIEVAL_INDEX to your FAISS index path"
    exit 1
fi

if [ ! -f "$RETRIEVAL_ID_MAP" ]; then
    echo "❌ Error: Retrieval ID map not found: $RETRIEVAL_ID_MAP"
    echo "Please edit this script and set RETRIEVAL_ID_MAP to your metadata.json path"
    exit 1
fi

echo "✅ Input validation passed"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Build command
CMD="CUDA_VISIBLE_DEVICES=$GPU_ID python $PROJECT_ROOT/evaluation/vbench/generate_vbench_videos.py \
    --checkpoint \"$CHECKPOINT_PATH\" \
    --base_model \"$BASE_MODEL\" \
    --retrieval_index \"$RETRIEVAL_INDEX\" \
    --retrieval_id_map \"$RETRIEVAL_ID_MAP\" \
    --retrieval_video_dir \"$RETRIEVAL_VIDEO_DIR\" \
    --latents_dir \"$LATENTS_DIR\" \
    --retrieval_k $RETRIEVAL_K \
    --output_dir \"$OUTPUT_DIR\" \
    --num_inference_steps $NUM_INFERENCE_STEPS \
    --fps $FPS \
    --video_quality $VIDEO_QUALITY"

# Add categories if specified
if [ -n "$CATEGORIES" ]; then
    CMD="$CMD --categories $CATEGORIES"
fi

# Add any additional arguments passed to this script
CMD="$CMD $@"

# Run inference
echo "════════════════════════════════════════════════════════════════════════════════"
echo "Starting inference"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""

eval $CMD

EXIT_CODE=$?

echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Inference completed successfully"
    echo "════════════════════════════════════════════════════════════════════════════════"
    echo "Output directory: $OUTPUT_DIR"
    echo ""
    
    # Count videos per category
    if [ -d "$OUTPUT_DIR" ]; then
        echo "Videos per category:"
        for category_dir in "$OUTPUT_DIR"/*; do
            if [ -d "$category_dir" ]; then
                category_name=$(basename "$category_dir")
                video_count=$(ls -1 "$category_dir"/*.mp4 2>/dev/null | wc -l)
                echo "  $category_name: $video_count videos"
            fi
        done
    fi
    
    echo ""
    echo "Next steps:"
    echo "1. Submit the generated videos to VBench evaluation server"
    echo "2. Or run local evaluation using VBench toolkit"
    echo "════════════════════════════════════════════════════════════════════════════════"
    exit 0
else
    echo "❌ Inference failed"
    echo "════════════════════════════════════════════════════════════════════════════════"
    echo "Please check the error messages above"
    echo "════════════════════════════════════════════════════════════════════════════════"
    exit 1
fi

