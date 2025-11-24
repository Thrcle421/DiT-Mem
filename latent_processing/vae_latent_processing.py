import os
import sys
import json
import csv
import torch
import torch.multiprocessing as mp
from pathlib import Path
from tqdm import tqdm
import argparse
from typing import List, Tuple, Dict, Set, Optional
import time
import numpy as np
from PIL import Image
from einops import repeat, reduce
import cv2
import concurrent.futures
from functools import partial
from queue import Empty

class OptimizedWanVAEEncoder:
    """Wan2.1 VAE encoder wrapper."""
    
    def __init__(self, model_path: str, device: str = 'cuda:0'):
        """
        Initialize VAE encoder.
        
        Args:
            model_path: Wan2.1 model directory path
            device: Device to load model
        """
        self.device = torch.device(device)
        self.model_path = model_path
        
        from vae import WanVAE, WanVAE_
        
        print(f"Loading VAE from {model_path} to {device}...")
        vae_checkpoint = os.path.join(model_path, "Wan2.1_VAE.pth")
        
        with torch.device('meta'):
            vae_model = WanVAE_(
                dim=96,
                z_dim=16,
                dim_mult=[1, 2, 4, 4],
                num_res_blocks=2,
                attn_scales=[],
                temperal_downsample=[False, True, True],
                dropout=0.0
            )
        
        self.vae = WanVAE(
            vae=vae_model,
            vae_pth=vae_checkpoint,
            dtype=torch.bfloat16,
            device=self.device
        )
        
        self.vae.load_weight()
        self.vae.model.eval()
        print(f"VAE loaded to {device}")
        
        self.height = 480
        self.width = 832
        self.frames = 81
        self.interval = 1
        self.torch_dtype = torch.bfloat16
        self.processed_videos: Set[str] = set()
        
    def load_processed_videos(self, output_dir: str):
        """
        Load processed video IDs to avoid reprocessing.
        
        Args:
            output_dir: Output directory
        """
        if not os.path.exists(output_dir):
            return
            
        for file_path in Path(output_dir).glob("*.pt"):
            self.processed_videos.add(file_path.stem)
        
        print(f"Loaded {len(self.processed_videos)} processed video IDs")
    
    def extract_frames_cv2(self, video_path: str) -> torch.Tensor:
        """
        Extract and preprocess video frames using OpenCV.
        
        Args:
            video_path: Video file path
            
        Returns:
            Preprocessed video tensor [C, T, H, W]
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Cannot open video: {video_path}")
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if total_frames >= self.frames:
                indices = np.linspace(0, total_frames - 1, self.frames, dtype=int)
            else:
                indices = list(range(total_frames)) + [total_frames - 1] * (self.frames - total_frames)
            
            frames = []
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret:
                    raise ValueError(f"Cannot read frame {idx}")
                
                frame = cv2.resize(frame, (self.width, self.height))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            
            cap.release()
            
            video_tensor = torch.stack([
                torch.from_numpy(frame).float().permute(2, 0, 1) for frame in frames
            ], dim=1)
            
            video_tensor = video_tensor / 127.5 - 1.0
            
            return video_tensor
            
        except Exception as e:
            print(f"Error extracting frames: {e}")
            raise
    
    def encode_video(self, video_path: str) -> Tuple[torch.Tensor, bool]:
        """
        Encode single video to VAE latent vector.
        
        Args:
            video_path: Video file path
            
        Returns:
            Tuple of (latent tensor, success flag)
        """
        try:
            video = self.extract_frames_cv2(video_path)
            video = video.to(dtype=torch.bfloat16, device=self.device).unsqueeze(0)
            
            with torch.no_grad():
                latent = self.vae.encode([video[0]])[0]
            
            latent = latent.cpu()
            
            return latent, True
            
        except Exception as e:
            print(f"Error encoding {video_path}: {e}")
            return None, False
    
    def cleanup(self):
        """Clean up GPU memory."""
        del self.vae
        torch.cuda.empty_cache()


def get_video_files(source_dir: str, csv_file: str = None) -> List[str]:
    """
    Get video files to process.
    
    Args:
        source_dir: Directory containing video files
        csv_file: Optional CSV file with video filenames in first column
        
    Returns:
        List of full video file paths
    """
    if csv_file is not None:
        print(f"Reading video list from CSV: {csv_file}")
        video_filenames = []
        
        with open(csv_file, 'r', encoding='utf-8', newline='') as f:
            reader = csv.reader(f)
            header = next(reader)
            
            for row in reader:
                if row:
                    video_filename = row[0]
                    video_filenames.append(video_filename)
        
        video_files = [os.path.join(source_dir, fname) for fname in video_filenames]
        existing_files = [f for f in video_files if os.path.exists(f)]
        missing_count = len(video_files) - len(existing_files)
        
        if missing_count > 0:
            print(f"Warning: {missing_count} videos from CSV not found in {source_dir}")
        
        return sorted(existing_files)
    else:
        video_files = sorted(Path(source_dir).glob("*.mp4"))
        return [str(f) for f in video_files]


def process_videos_on_gpu(
    gpu_id: int,
    task_queue: mp.Queue,
    stats_queue: mp.Queue,
    output_dir: str,
    model_path: str,
    total_videos: int
):
    """
    Process videos from task queue on specific GPU.
    
    Args:
        gpu_id: GPU device ID
        task_queue: Queue containing video paths to process
        stats_queue: Queue for reporting statistics
        output_dir: Output directory for latent vectors
        model_path: Wan2.1 model path
        total_videos: Total number of videos (for progress display)
    """
    device = f'cuda:{gpu_id}'
    print(f"GPU {gpu_id}: Starting, waiting for tasks...")
    
    encoder = OptimizedWanVAEEncoder(model_path, device=device)
    
    success_count = 0
    fail_count = 0
    skip_count = 0
    
    pbar = tqdm(
        total=0,
        desc=f"GPU {gpu_id}",
        position=gpu_id
    )
    
    while True:
        try:
            video_path = task_queue.get(timeout=1)
            
            if video_path is None:
                break
            
            video_id = Path(video_path).stem
            output_path = os.path.join(output_dir, f"{video_id}.pt")
            
            if os.path.exists(output_path):
                skip_count += 1
                pbar.update(1)
                pbar.set_postfix({
                    'success': success_count,
                    'skip': skip_count,
                    'fail': fail_count
                })
                continue
            
            latent, success = encoder.encode_video(video_path)
            
            if success and latent is not None:
                latent_bf16 = latent.to(dtype=torch.bfloat16)
                
                torch.save(
                    {
                        'latent': latent_bf16,
                        'video_id': video_id,
                        'shape': list(latent_bf16.shape),
                        'dtype': str(latent_bf16.dtype)
                    },
                    output_path
                )
                success_count += 1
            else:
                fail_count += 1
            
            pbar.update(1)
            pbar.set_postfix({
                'success': success_count,
                'skip': skip_count,
                'fail': fail_count
            })
            
        except Empty:
            continue
        except Exception as e:
            print(f"GPU {gpu_id} error processing video: {e}")
            fail_count += 1
            continue
    
    pbar.close()
    encoder.cleanup()
    stats_queue.put((gpu_id, success_count, skip_count, fail_count))
    print(f"GPU {gpu_id} finished: {success_count} success, {skip_count} skip, {fail_count} fail")


def main():
    parser = argparse.ArgumentParser(description='Optimized VAE pre-encoding script (load-balanced)')
    parser.add_argument(
        '--source_dir',
        type=str,
        default='video',
        help='Source directory containing videos'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='video_latents',
        help='Output directory for latent vectors'
    )
    parser.add_argument(
        '--model_path',
        type=str,
        default='model/Wan2.1-T2V-1.3B',
        help='Wan2.1 model directory path'
    )
    parser.add_argument(
        '--csv_file',
        type=str,
        default='data/example.csv',
        help='CSV file with video filenames in first column. If not specified, process all videos in source_dir'
    )
    parser.add_argument(
        '--gpus',
        type=str,
        default='0,1,2,3,4,5,6,7',
        help='Comma-separated list of GPU IDs to use'
    )
    
    args = parser.parse_args()
    
    gpu_ids = [int(x) for x in args.gpus.split(',')]
    print(f"Using GPUs: {gpu_ids}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.csv_file:
        print(f"Loading video list from CSV: {args.csv_file}")
    else:
        print("Scanning directory for video files...")
    
    video_files = get_video_files(args.source_dir, args.csv_file)
    total_videos = len(video_files)
    print(f"Found {total_videos} videos to process")
    
    unprocessed_videos = []
    for video_path in video_files:
        video_id = Path(video_path).stem
        output_path = os.path.join(args.output_dir, f"{video_id}.pt")
        if not os.path.exists(output_path):
            unprocessed_videos.append(video_path)
    
    print(f"Unprocessed: {len(unprocessed_videos)}")
    print(f"Already processed: {total_videos - len(unprocessed_videos)}")
    
    if len(unprocessed_videos) == 0:
        print("All videos already processed!")
        return
    
    mp.set_start_method('spawn', force=True)
    
    task_queue = mp.Queue()
    stats_queue = mp.Queue()
    
    print(f"\nAdding {len(unprocessed_videos)} tasks to queue...")
    for video_path in unprocessed_videos:
        task_queue.put(video_path)
    
    for _ in gpu_ids:
        task_queue.put(None)
    
    print("\nStarting parallel processing (dynamic load balancing)...")
    print(f"Number of GPUs: {len(gpu_ids)}\n")
    
    processes = []
    for gpu_id in gpu_ids:
        p = mp.Process(
            target=process_videos_on_gpu,
            args=(
                gpu_id,
                task_queue,
                stats_queue,
                args.output_dir,
                args.model_path,
                total_videos
            )
        )
        p.start()
        processes.append(p)
        time.sleep(2)
    
    for p in processes:
        p.join()
    
    print("\n" + "="*50)
    print("Processing completed!")
    print("="*50)
    
    total_success = 0
    total_skip = 0
    total_fail = 0
    
    while not stats_queue.empty():
        gpu_id, success, skip, fail = stats_queue.get()
        total_success += success
        total_skip += skip
        total_fail += fail
        print(f"GPU {gpu_id}: {success} success, {skip} skip, {fail} fail")
    
    print("="*50)
    print(f"Total: {total_success} success, {total_skip} skip, {total_fail} fail")
    print(f"Output directory: {args.output_dir}")
    
    latent_files = list(Path(args.output_dir).glob("*.pt"))
    print(f"Total latent vectors created: {len(latent_files)} / {total_videos}")
    print("="*50)


if __name__ == '__main__':
    main()
