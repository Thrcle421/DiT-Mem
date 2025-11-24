#!/usr/bin/env python3
"""
Video generation from CSV prompts using retrieval-augmented diffusion model.

Reads prompts from CSV file and generates videos using a trained DiT-Mem checkpoint
with retrieval-augmented generation (RAG). CSV should contain 'video' and 'caption' columns.
Output filenames are derived from the 'video' column.
"""
import argparse
import csv
import os
import sys
from pathlib import Path
import torch

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from source.data.video import save_video
from source.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
from source.models.wan_video_dit_mem import DiTMem
from source.utils.video_retrieval import VideoRetriever


def load_prompts_from_csv(csv_path: Path, output_dir: Path) -> list[dict]:
    """
    Load prompts from CSV file and filter out already generated videos.
    
    Args:
        csv_path: Path to CSV file containing 'video' and 'caption' columns
        output_dir: Output directory to check for existing videos
    
    Returns:
        List of dictionaries with 'prompt' and 'filename' keys
    """
    prompts = []
    
    with csv_path.open("r", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        for idx, row in enumerate(reader, start=1):
            caption = row.get("cation") or row.get("caption") or row.get("Caption")
            video_name = row.get("video") or row.get("Video") or row.get("video_path")
            
            if not caption:
                print(f"Warning: Row {idx} missing caption, skipping")
                continue
            
            if not video_name:
                print(f"Warning: Row {idx} missing video name, skipping")
                continue
            
            if not video_name.endswith('.mp4'):
                video_name = f"{video_name}.mp4"
            
            prompts.append({
                "prompt": caption,
                "filename": video_name,
                "original_idx": idx
            })
    
    # Filter out already generated videos
    pending_prompts = []
    for p in prompts:
        output_path = output_dir / p["filename"]
        if not output_path.exists():
            pending_prompts.append(p)
    
    total_pending = len(pending_prompts)
    print(f"Total prompts: {len(prompts)}, Already generated: {len(prompts) - total_pending}, Pending: {total_pending}")
    
    return pending_prompts


def load_base_model_with_checkpoint(
    checkpoint_path,
    base_model_path,
    retrieval_index_path,
    retrieval_id_map_path,
    retrieval_video_dir,
    latents_dir,
    device="cuda",
    retrieval_k=5,
    torch_dtype=torch.bfloat16,
):
    """
    Load base model, checkpoint, and initialize retrieval-augmented pipeline.
    
    Args:
        checkpoint_path: Path to trained checkpoint
        base_model_path: Path to base Wan2.1 model
        retrieval_index_path: Path to FAISS retrieval index
        retrieval_id_map_path: Path to video ID mapping file
        retrieval_video_dir: Directory containing retrieval videos
        latents_dir: Directory containing precomputed video latents
        device: Device to run inference on
        retrieval_k: Number of top-k videos to retrieve
        torch_dtype: Data type for model weights
    
    Returns:
        Initialized WanVideoPipeline with dit_mem and retrieval system
    """
    print("=" * 80)
    print("Loading inference pipeline")
    print("=" * 80)
    
    # Load base Wan2.1 model
    print("\n[1/4] Loading base Wan2.1 model")
    print(f"    Path: {base_model_path}")
    
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch_dtype,
        device=device,
        model_configs=[
            ModelConfig(
                path=f"{base_model_path}/diffusion_pytorch_model.safetensors", 
                offload_device="cpu"
            ),
            ModelConfig(
                path=f"{base_model_path}/models_t5_umt5-xxl-enc-bf16.pth", 
                offload_device="cpu"
            ),
            ModelConfig(
                path=f"{base_model_path}/Wan2.1_VAE.pth", 
                offload_device="cpu"
            ),
        ],
        tokenizer_config=ModelConfig(path=f"{base_model_path}/google/umt5-xxl"),
    )
    pipe.enable_vram_management()
    print("    Base model loaded")
    
    # Initialize DiTMem
    print("\n[2/4] Initializing DiTMem module")
    pipe.dit_mem = DiTMem(
        in_channels=16,
        target_seq_len=100,
        target_feat_dim=1536,
        fft_dropout=0.5
    ).to(device=device, dtype=torch_dtype)
    num_params = sum(p.numel() for p in pipe.dit_mem.parameters()) / 1e6
    print(f"    DiTMem initialized ({num_params:.2f}M parameters)")
    
    # Load checkpoint
    print("\n[3/4] Loading checkpoint")
    print(f"    Path: {checkpoint_path}")
    
    from safetensors.torch import load_file
    checkpoint = load_file(checkpoint_path)
    
    # Separate DiT and DiTMem weights
    dit_state_dict = {}
    dit_mem_state_dict = {}
    other_keys = []
    
    for key, value in checkpoint.items():
        if "dit_mem" in key.lower():
            if key.startswith("pipe.dit_mem."):
                clean_key = key[len("pipe.dit_mem."):]
            elif key.startswith("dit_mem."):
                clean_key = key[len("dit_mem."):]
            else:
                clean_key = key
            dit_mem_state_dict[clean_key] = value
        elif "dit" in key.lower() or "pipe.dit" in key.lower():
            # Skip dit_mem keys that were already processed
            if "dit_mem" not in key.lower():
                clean_key = key.replace("pipe.dit.", "").replace("dit.", "")
                dit_state_dict[clean_key] = value
        else:
            other_keys.append(key)
    
    print(f"    Checkpoint statistics:")
    print(f"      - DiT weights: {len(dit_state_dict)} parameters")
    print(f"      - DiTMem weights: {len(dit_mem_state_dict)} parameters")
    if other_keys:
        print(f"      - Unclassified weights: {len(other_keys)}")
    
    # Load DiTMem weights
    if dit_mem_state_dict:
        missing_keys, unexpected_keys = pipe.dit_mem.load_state_dict(
            dit_mem_state_dict, strict=False
        )
        
        critical_missing = []
        for k in missing_keys:
            if any(x in k for x in ['running_mean', 'running_var', 'num_batches_tracked']):
                continue
            if 'cnn.0.' in k or 'cnn.1.' in k:
                continue
            if 'complex_weight' in k:
                continue
            critical_missing.append(k)
        
        if critical_missing:
            print(f"    Warning: DiTMem critical missing keys: {len(critical_missing)}")
        if unexpected_keys:
            print(f"    Warning: DiTMem unexpected keys: {len(unexpected_keys)}")
        
        if not critical_missing and not unexpected_keys:
            print("    DiTMem weights loaded (perfect match)")
        else:
            print("    DiTMem weights loaded (partial match)")
    else:
        print("    Warning: No DiTMem weights found, using random initialization")
    
    # Load DiT weights
    if dit_state_dict:
        missing_keys, unexpected_keys = pipe.dit.load_state_dict(
            dit_state_dict, strict=False
        )
        
        if len(missing_keys) > 0 or len(unexpected_keys) > 0:
            print(f"    DiT weights partially updated:")
            print(f"      - Loaded: {len(dit_state_dict)} parameters")
            if len(missing_keys) > 0:
                print(f"      - Missing (kept pretrained): {len(missing_keys)}")
            if len(unexpected_keys) > 0:
                print(f"      - Unexpected (skipped): {len(unexpected_keys)}")
        print("    DiT weights loaded")
    else:
        print("    No DiT weights found, using pretrained weights")
    
    pipe.dit_mem.eval()
    
    # Initialize retrieval system
    print("\n[4/4] Initializing video retrieval system")
    print(f"    Index: {retrieval_index_path}")
    print(f"    ID map: {retrieval_id_map_path}")
    print(f"    Video directory: {retrieval_video_dir}")
    print(f"    Latents directory: {latents_dir}")
    print(f"    Top-K: {retrieval_k}")
    
    pipe.video_retriever = VideoRetriever(
        index_path=retrieval_index_path,
        id_map_path=retrieval_id_map_path,
        video_dir=retrieval_video_dir,
        latents_dir=latents_dir
    )
    pipe.retrieval_k = retrieval_k
    print("    Retrieval system initialized")
    
    print("\n" + "=" * 80)
    print("Inference pipeline loaded")
    print("=" * 80)
    
    return pipe


def run_inference(
    pipe,
    prompts,
    output_dir="./outputs",
    seed=42,
    num_inference_steps=40,
    fps=16,
    video_quality=5,
    skip_existing=True,
):
    """
    Run inference to generate videos from prompts.
    
    Args:
        pipe: Configured WanVideoPipeline
        prompts: List of dictionaries with 'prompt' and 'filename' keys
        output_dir: Output directory for generated videos
        seed: Random seed for reproducibility
        num_inference_steps: Number of diffusion steps
        fps: Video frame rate
        video_quality: Video quality (1-10)
        skip_existing: Whether to skip already generated videos
    """
    os.makedirs(output_dir, exist_ok=True)
    
    negative_prompt = (
        "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
        "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，"
        "画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，"
        "静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
    )
    
    print("\n" + "=" * 80)
    print("Starting video generation")
    print("=" * 80)
    print(f"Output directory: {output_dir}")
    print(f"Total prompts: {len(prompts)}")
    print(f"Inference steps: {num_inference_steps}")
    print(f"Seed: {seed}")
    print(f"Skip existing: {skip_existing}")
    
    success_count = 0
    skipped_count = 0
    failed_count = 0
    
    for i, test_case in enumerate(prompts, 1):
        print(f"\n{'─' * 80}")
        print(f"[{i}/{len(prompts)}] Generating: {test_case['filename']}")
        print(f"{'─' * 80}")
        print(f"Prompt: {test_case['prompt']}")
        
        output_path = os.path.join(output_dir, test_case['filename'])
        if skip_existing and os.path.exists(output_path):
            print(f"Skipping (already exists): {output_path}")
            skipped_count += 1
            continue
        
        try:
            video = pipe(
                prompt=test_case['prompt'],
                negative_prompt=negative_prompt,
                seed=seed,
                num_inference_steps=num_inference_steps,
                tiled=True,  # Enable tiled VAE to save memory
            )
            
            save_video(video, output_path, fps=fps, quality=video_quality)
            
            print(f"Success: {output_path}")
            success_count += 1
            
        except Exception as e:
            print(f"Failed: {str(e)}")
            import traceback
            traceback.print_exc()
            failed_count += 1
            continue
    
    print("\n" + "=" * 80)
    print("Inference completed")
    print("=" * 80)
    print(f"Success: {success_count}/{len(prompts)}")
    if skipped_count > 0:
        print(f"Skipped: {skipped_count}/{len(prompts)}")
    if failed_count > 0:
        print(f"Failed: {failed_count}/{len(prompts)}")
    print(f"Output directory: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Video generation inference script from CSV prompts"
    )
    
    parser.add_argument(
        "--csv_file",
        type=Path,
        required=True,
        help="CSV file path containing 'video' and 'caption' columns",
    )
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="results/step-1500.safetensors",
        help="Path to trained checkpoint"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="model/Wan2.1-T2V-1.3B",
        help="Path to base Wan2.1 model"
    )
    
    parser.add_argument(
        "--retrieval_index",
        type=str,
        default="memory_index/labels.index",
        help="Path to FAISS retrieval index"
    )
    parser.add_argument(
        "--retrieval_id_map",
        type=str,
        default="memory_index/id_map.json",
        help="Path to video ID mapping file"
    )
    parser.add_argument(
        "--retrieval_video_dir",
        type=str,
        default="video",
        help="Directory containing retrieval videos"
    )
    parser.add_argument(
        "--latents_dir",
        type=str,
        default="video_latents",
        help="Directory containing precomputed video latents"
    )
    parser.add_argument(
        "--retrieval_k",
        type=int,
        default=5,
        help="Number of top-k videos to retrieve (default: 5)"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output_videos/dit_mem",
        help="Output directory for generated videos"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=40,
        help="Number of diffusion steps (default: 40)"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=16,
        help="Video frame rate (default: 16)"
    )
    parser.add_argument(
        "--video_quality",
        type=int,
        default=5,
        help="Video quality 1-10 (default: 5)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run inference on (default: cuda)"
    )
    parser.add_argument(
        "--force_overwrite",
        action="store_true",
        help="Force overwrite existing files (default: skip existing)"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Validate input files
    if not args.csv_file.exists():
        print(f"Error: CSV file not found: {args.csv_file}")
        return
    
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        return
    
    if not os.path.exists(args.base_model):
        print(f"Error: Base model directory not found: {args.base_model}")
        return
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load prompts
    print(f"\nLoading prompts from CSV: {args.csv_file}")
    prompts = load_prompts_from_csv(args.csv_file, output_dir)
    print(f"Loaded {len(prompts)} prompts to process")
    
    if len(prompts) == 0:
        print("No prompts to process (all videos may already be generated)")
        return
    
    # Load model and checkpoint
    pipe = load_base_model_with_checkpoint(
        checkpoint_path=args.checkpoint,
        base_model_path=args.base_model,
        retrieval_index_path=args.retrieval_index,
        retrieval_id_map_path=args.retrieval_id_map,
        retrieval_video_dir=args.retrieval_video_dir,
        latents_dir=args.latents_dir,
        device=args.device,
        retrieval_k=args.retrieval_k,
        torch_dtype=torch.bfloat16,
    )
    
    # Run inference
    try:
        run_inference(
            pipe=pipe,
            prompts=prompts,
            output_dir=args.output_dir,
            seed=args.seed,
            num_inference_steps=args.num_inference_steps,
            fps=args.fps,
            video_quality=args.video_quality,
            skip_existing=not args.force_overwrite,
        )
    finally:
        # Clean up GPU memory
        del pipe
        torch.cuda.empty_cache()
    
    print("\nAll tasks completed")


if __name__ == "__main__":
    main()

