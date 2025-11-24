#!/bin/bash

# Default parameters
SOURCE_DIR="video"
OUTPUT_DIR="latent"
MODEL_PATH="model/Wan2.1-T2V-1.3B"
CSV_FILE="data/example.csv"
GPUS="0,1,2,3,4,5,6,7"

python latent_processing/vae_latent_processing.py \
  --source_dir ${SOURCE_DIR} \
  --output_dir ${OUTPUT_DIR} \
  --model_path ${MODEL_PATH} \
  --csv_file ${CSV_FILE} \
  --gpus ${GPUS}
