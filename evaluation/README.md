# Evaluation

This directory contains evaluation scripts and prompts for two text-to-video benchmarks: **PhyGenBench** and **VBench**.

## 📂 Directory Structure

```
evaluation/
├── README.md                          # This file
├── phygenbench/                       # PhyGenBench evaluation
│   ├── prompts.json                   # 160 evaluation prompts
│   ├── generate_phygenbench_videos.py # Video generation script
│   └── run_phygenbench_evaluation.sh  # Bash wrapper script
└── vbench/                            # VBench evaluation
    ├── prompts/                       # Prompt files directory
    │   └── *.txt                      # 11 category prompt files (10 prompts each)
    ├── generate_vbench_videos.py      # Video generation script
    └── run_vbench_evaluation.sh       # Bash wrapper script
```

## 🛠️ Prerequisites

Before running evaluations, ensure you have the following:

1.  **Trained Checkpoint**: `checkpoint/DiT-Mem-1.3B.safetensors`
2.  **Base Model**: `models/Wan2.1-T2V-1.3B`
3.  **Retrieval System**:
    *   Index: `memory_index/labels.index`
    *   Metadata: `memory_index/metadata.json`
    *   Videos: `videos/` (Directory containing memory bank videos)
    *   Latents: `latents/` (Pre-computed VAE latents)

## 🚀 PhyGenBench

[PhyGenBench](https://phygenbench.github.io/) evaluates the model's understanding of physical laws.

### Usage

1.  **Configure**: Open `phygenbench/run_phygenbench_evaluation.sh` and verify the paths in the configuration section.
    ```bash
    # Example configuration in script
    CHECKPOINT_PATH="checkpoint/DiT-Mem-1.3B.safetensors"
    BASE_MODEL="models/Wan2.1-T2V-1.3B"
    # ...
    ```

2.  **Run**:
    ```bash
    bash evaluation/phygenbench/run_phygenbench_evaluation.sh
    ```

3.  **Output**:
    *   Videos will be saved to `evaluation/phygenbench/outputs/`.
    *   Format: `video_output_1.mp4` to `video_output_160.mp4`.

## 📊 VBench

[VBench](https://vchitect.github.io/VBench-project/) provides a comprehensive evaluation of video generation quality across multiple dimensions.

### Usage

1.  **Configure**: Open `vbench/run_vbench_evaluation.sh` and verify the paths.
    *   You can optionally filter categories by setting the `CATEGORIES` variable (e.g., `CATEGORIES="scene color"`).

2.  **Run**:
    ```bash
    bash evaluation/vbench/run_vbench_evaluation.sh
    ```

3.  **Output**:
    *   Videos will be saved to `evaluation/vbench/outputs/`.
    *   Structure: Subdirectories for each category (e.g., `human_action`, `scene`), each containing 50 videos (10 prompts × 5 samples).

## 📝 Notes

*   **GPU Usage**: The scripts default to using `GPU_ID=0`. You can change this in the scripts or override it by setting `CUDA_VISIBLE_DEVICES` when running the command.
*   **Parameters**: Default inference parameters (Steps=40, K=5, FPS=16) are set in the scripts but can be modified as needed.
