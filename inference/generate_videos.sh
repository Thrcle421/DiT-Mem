#!/bin/bash
# Video generation inference script from CSV file.
# Generates videos using a trained checkpoint with retrieval-augmented generation.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
export PATH="$CONDA_PREFIX/bin:$PATH"

# Model and checkpoint paths
BASE_MODEL="model/Wan2.1-T2V-1.3B"
CHECKPOINT_PATH="checkpoint/DiT-Mem-1.3B.safetensors"
OUTPUT_DIR="$PROJECT_ROOT/output_videos/dit_mem"
CSV_FILE="$PROJECT_ROOT/data/inference.csv"

# Retrieval system paths
RETRIEVAL_INDEX="memory_index/labels.index"
RETRIEVAL_ID_MAP="memory_index/id_map.json"
RETRIEVAL_VIDEO_DIR="video"
LATENTS_DIR="latent"

# Inference parameters
NUM_INFERENCE_STEPS=40
RETRIEVAL_K=5
SEED=42
FPS=16
VIDEO_QUALITY=5
GPU_ID=0

echo "Video Generation Inference"
echo "=========================="
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Base model: $BASE_MODEL"
echo "CSV file: $CSV_FILE"
echo "Output directory: $OUTPUT_DIR"
echo "Inference steps: $NUM_INFERENCE_STEPS"
echo "Retrieval top-K: $RETRIEVAL_K"
echo "Seed: $SEED"
echo "FPS: $FPS"
echo "Video quality: $VIDEO_QUALITY"
echo "GPU: $GPU_ID"
echo "=========================="
echo ""

# Validate input files
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "Error: Checkpoint not found: $CHECKPOINT_PATH"
    exit 1
fi

if [ ! -d "$BASE_MODEL" ]; then
    echo "Error: Base model directory not found: $BASE_MODEL"
    exit 1
fi

if [ ! -f "$CSV_FILE" ]; then
    echo "Error: CSV file not found: $CSV_FILE"
    exit 1
fi

if [ ! -f "$RETRIEVAL_INDEX" ]; then
    echo "Error: Retrieval index not found: $RETRIEVAL_INDEX"
    exit 1
fi

if [ ! -f "$RETRIEVAL_ID_MAP" ]; then
    echo "Error: Retrieval ID map not found: $RETRIEVAL_ID_MAP"
    exit 1
fi

echo "Input validation passed"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run inference
echo "Starting inference..."
echo ""

CUDA_VISIBLE_DEVICES=$GPU_ID python "$PROJECT_ROOT/inference/generate_videos.py" \
    --csv_file "$CSV_FILE" \
    --checkpoint "$CHECKPOINT_PATH" \
    --base_model "$BASE_MODEL" \
    --retrieval_index "$RETRIEVAL_INDEX" \
    --retrieval_id_map "$RETRIEVAL_ID_MAP" \
    --retrieval_video_dir "$RETRIEVAL_VIDEO_DIR" \
    --latents_dir "$LATENTS_DIR" \
    --retrieval_k $RETRIEVAL_K \
    --output_dir "$OUTPUT_DIR" \
    --num_inference_steps $NUM_INFERENCE_STEPS \
    --seed $SEED \
    --fps $FPS \
    --video_quality $VIDEO_QUALITY \
    "$@"

EXIT_CODE=$?

echo ""
echo "=========================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "Inference completed successfully"
    echo "Output directory: $OUTPUT_DIR"
    VIDEO_COUNT=$(ls -1 "$OUTPUT_DIR"/*.mp4 2>/dev/null | wc -l)
    echo "Generated videos: $VIDEO_COUNT"
    exit 0
else
    echo "Inference failed with exit code: $EXIT_CODE"
    exit 1
fi

