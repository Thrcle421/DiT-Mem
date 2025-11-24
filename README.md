# Learning Plug-and-play Memory for Guiding Video Diffusion Models

<!-- [![arXiv](https://img.shields.io/badge/arXiv-24xx.xxxxx-b31b1b.svg)](https://arxiv.org/abs/24xx.xxxxx) -->
[![Project Page](https://img.shields.io/badge/Project-Page-green)](https://thrcle421.github.io/DiT-Mem-Web/)
[![HuggingFace Model](https://img.shields.io/badge/🤗-HuggingFace-yellow)](https://huggingface.co/Thrcle/DiT-Mem-1.3B)
[![HuggingFace Dataset](https://img.shields.io/badge/🤗-Dataset-blue)](https://huggingface.co/datasets/Thrcle/DiT-Mem-Data)

This repository contains the official implementation of the paper **"Learning Plug-and-play Memory for Guiding Video Diffusion Models"**.

## 📰 News
- **[2025-11-24]** Code and paper released. We also release our training data, memory data, and DiT-Mem-1.3B training weights.

## Table of Contents
- [Introduction](#introduction)
- [Method](#method)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Model Weights](#model-weights)
- [Training](#training)
- [Inference](#inference)
- [Evaluation](#evaluation)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)

## Introduction
**DiT-Mem** is a plug-and-play memory module for DiT-based video diffusion models designed to inject rich world knowledge during generation. Instead of scaling model size or data, DiT-Mem retrieves a few relevant reference videos and encodes them into compact memory tokens using 3D CNNs, frequency-domain filtering (HPF/LPF), and lightweight attention. These tokens are then inserted into the DiT’s self-attention layers to guide generation without modifying the backbone.

Our method requires finetuning only a small memory encoder on 10K videos while keeping the diffusion model frozen. When applied to Wan2.1 and Wan2.2, DiT-Mem improves controllability, semantic reasoning, and physics consistency—often surpassing strong commercial systems—while remaining efficient and fully modular.

## Method
![Pipeline](figures/pipeline.jpg)

Our framework consists of three main steps:
1. **Retrieval**: Given a text prompt, we retrieve relevant reference videos from an external memory bank.
2. **Memory Encoding**: A lightweight encoder processes these videos using 3D CNNs for downsampling, frequency-domain filters (Low-Pass/High-Pass) for feature disentanglement, and self-attention for aggregation.
3. **Injection**: The resulting memory tokens are concatenated with the hidden states of the frozen DiT backbone during inference, providing guidance without altering the original model weights.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Thrcle421/DiT-Mem.git
   cd DiT-Mem
   ```

2. Create a conda environment and install dependencies:
   ```bash
   conda create -n dit_mem python=3.10
   conda activate dit_mem
   pip install -r requirements.txt
   ```

## Data Preparation

### Dataset
Our training and memory data are derived from [OpenVid-1M](https://github.com/NJU-PCALab/OpenVid-1M), specifically the [OpenVidHD-0.4M](https://huggingface.co/datasets/nkp37/OpenVid-1M/tree/main/OpenVidHD) subset.

- **Training Data**: We randomly selected 10k videos from OpenVidHD-0.4M, weighted by the volume of each part.
- **Memory Data**: We used the remaining videos from OpenVidHD-0.4M, excluding 100 videos reserved for benchmark testing.

### Download
1. **CSV Files**: Please download the corresponding CSV files from [HuggingFace Dataset](https://huggingface.co/datasets/Thrcle/DiT-Mem-Data) and place them in the `data/` directory.
2. **Video Data**: Download the full OpenVidHD-0.4M video dataset and place it in the `video/` directory.

```bash
# Example structure
data/
├── train.csv   # 10k training samples
└── memory.csv  # Memory bank videos

video/
└── ...         # Video files
```

### Retrieval Index
To build the retrieval index, follow these steps:

1. **Download Model**: Download the [Alibaba-NLP/gte-base-en-v1.5](https://huggingface.co/Alibaba-NLP/gte-base-en-v1.5) model and place it in `model/gte-base-en-v1.5`.
2. **Build Index**: Run the following command to generate `labels.index` and `id_map.json`:

   ```bash
   python memory_index/build_retrieve_index.py
   ```

   This will create:
   - `memory_index/labels.index`: FAISS index for retrieval.
   - `memory_index/id_map.json`: Mapping from IDs to video paths.

### Latent Pre-computation
To accelerate training and inference, we pre-compute VAE latents for all videos in the memory bank.

1. **Run Pre-computation**:
   ```bash
   bash latent_processing/vae_latent_processing.sh
   ```
   
   Ensure that `CSV_FILE` in the script points to your memory data CSV (e.g., `data/memory.csv`).
   The encoded latents will be saved in the `latent/` directory.

## Model Weights

1. **Base Model**: Download the Wan2.1-T2V-1.3B model and place it in `model/Wan2.1-T2V-1.3B`.
2. **DiT-Mem Checkpoint**: Download our trained checkpoint from [HuggingFace Model](https://huggingface.co/Thrcle/DiT-Mem-1.3B) and place it in `checkpoint/DiT-Mem-1.3B.safetensors`.

Structure:
```
DiT-Mem/
├── checkpoint/
│   └── DiT-Mem-1.3B.safetensors
├── model/
│   ├── Wan2.1-T2V-1.3B/
│   │   ├── config.json
│   │   └── ...
│   └── gte-base-en-v1.5/
│       ├── config.json
│       └── ...
```

## Training

To train the memory encoder, use the training script:

```bash
bash scripts/train_dit_mem.sh
```

Training config is located at `config/train_dit_mem.yaml`.

## Inference

To generate videos using DiT-Mem, run the provided script:

```bash
bash inference/generate_videos.sh
```

**Parameters in `inference/generate_videos.sh`:**
- `CHECKPOINT_PATH`: Path to the DiT-Mem checkpoint (prepared in [Model Weights](#model-weights)).
- `BASE_MODEL`: Path to the frozen base model (prepared in [Model Weights](#model-weights)).
- `CSV_FILE`: Input CSV containing prompts.
- `RETRIEVAL_K`: Number of reference videos to retrieve (default: 5).
- `NUM_INFERENCE_STEPS`: Number of denoising steps (default: 40).


## Evaluation

We provide scripts to evaluate DiT-Mem on two public benchmarks:

- **VBench**  
  - Script: `evaluation/vbench/run_vbench_evaluation.sh`  
  - Official project page: [VBench project page](https://vchitect.github.io/VBench-project/)

- **PhyGenBench**  
  - Script: `evaluation/phygenbench/run_phygenbench_evaluation.sh`  
  - Official project page: [PhyGenBench project page](https://phygenbench.github.io/)

For detailed instructions, please refer to [evaluation/README.md](evaluation/README.md).

## Citation

If you find this work useful, please cite our paper:

```bibtex
@article{ditmem2025,
  title={Learning Plug-and-play Memory for Guiding Video Diffusion Models},
  author={Author One and Author Two and Author Three},
  journal={arXiv preprint arXiv:24xx.xxxxx},
  year={2025}
}
```

## Acknowledgements
This codebase is built upon [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) and [Wan2.1](https://github.com/Wan-Video/Wan2.1). 
We also acknowledge the use of [OpenVid-1M](https://github.com/NJU-PCALab/OpenVid-1M) for training data and [gte-base-en-v1.5](https://huggingface.co/Alibaba-NLP/gte-base-en-v1.5) for the retrieval model.
We thank [VBench](https://github.com/Vchitect/VBench) and [PhyGenBench](https://github.com/PhyGenBench/PhyGenBench) for their evaluation benchmarks.
We thank the authors for their open-source contributions.
