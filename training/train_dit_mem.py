"""
Training script for DiT-Mem with pre-encoded VAE latents and video retrieval.
"""

import torch, os, sys, json, csv, time
from datetime import datetime
from pathlib import Path
from packaging import version

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from source import load_state_dict
from source.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
from source.trainers.utils import DiffusionTrainingModule, ModelLogger, safe_unwrap_model
from source.utils.video_retrieval import VideoRetriever
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from tqdm import tqdm
import argparse
import yaml
from functools import lru_cache
os.environ["TOKENIZERS_PARALLELISM"] = "false"

if torch.cuda.is_available():
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except AttributeError:
        pass
    torch.backends.cudnn.benchmark = True


@lru_cache(maxsize=256)
def load_latent_cached(latent_path: str):
    """
    Load latent file with LRU caching to avoid redundant disk reads.
    
    Args:
        latent_path: Path to .pt file
    
    Returns:
        Dictionary with 'latent' tensor and metadata
    """
    return torch.load(latent_path, map_location='cpu', weights_only=False)


class RetrievalLogger:
    """Logger for tracking video retrieval results during training."""
    
    def __init__(self, output_path, csv_filename="retrieval_log.csv"):
        """
        Initialize retrieval logger.
        
        Args:
            output_path: Directory to save logs
            csv_filename: Name of CSV log file
        """
        self.output_path = output_path
        self.csv_path = os.path.join(output_path, csv_filename)
        self.total_retrievals = 0
        self.total_with_gt = 0
        self.total_retrieved = 0
        self._write_count = 0
        self.csv_file = None
        self.csv_writer = None
        
        # Create output directory if it doesn't exist
        os.makedirs(output_path, exist_ok=True)
        
        # Initialize CSV file with headers
        self.csv_file = open(self.csv_path, 'w', newline='', encoding='utf-8')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            'prompt', 
            'gt_video_id',
            'retrieved_video_ids',
            'retrieved_types',
            'retrieved_ranks',
            'num_retrieved',
            'timestamp'
        ])
        self.csv_file.flush()
        
    
    def log_retrieval(self, prompt, gt_video_id, retrieved_videos):
        """
        Log a retrieval result.
        
        Args:
            prompt: Text prompt used for retrieval
            gt_video_id: Ground truth video ID
            retrieved_videos: List of dicts with keys: video_id, type, rank
        """
        # Extract video IDs, types, and ranks
        video_ids = [v['video_id'] for v in retrieved_videos]
        types = [v['type'] for v in retrieved_videos]
        ranks = [v['rank'] for v in retrieved_videos]
        
        # Write to CSV
        self.csv_writer.writerow([
            prompt,
            gt_video_id if gt_video_id else '',
            '|'.join(video_ids),
            '|'.join(types),
            '|'.join(map(str, ranks)),
            len(retrieved_videos),
            datetime.now().isoformat()
        ])
        
        self._write_count += 1

        # Flush every 10 records to ensure data is saved
        if self._write_count % 10 == 0:
            self.csv_file.flush()
        
        # Update running statistics without retaining the full history
        self.total_retrievals += 1
        if gt_video_id:
            self.total_with_gt += 1
        self.total_retrieved += len(retrieved_videos)
    
    def get_statistics(self):
        """Get retrieval statistics."""
        if self.total_retrievals == 0:
            return {}
        
        total_retrievals = self.total_retrievals
        with_gt = self.total_with_gt
        avg_num_retrieved = self.total_retrieved / total_retrievals if total_retrievals else 0.0
        
        return {
            'total_retrievals': total_retrievals,
            'with_gt': with_gt,
            'without_gt': total_retrievals - with_gt,
            'avg_num_retrieved': avg_num_retrieved
        }
    
    def close(self):
        """Close the CSV file."""
        if self.csv_file and not self.csv_file.closed:
            self.csv_file.flush()
            self.csv_file.close()
            stats = self.get_statistics()
            if stats:
                print(f"Retrieval log saved: {self.csv_path}")
                # print(f"  Total: {stats['total_retrievals']}, With GT: {stats['with_gt']}, Avg: {stats['avg_num_retrieved']:.2f}")
            
            self.csv_file = None
            self.csv_writer = None
    
    def __del__(self):
        """Ensure CSV is closed on deletion."""
        self.close()


class EnhancedModelLogger(ModelLogger):
    """Enhanced logger with WandB and CSV support."""
    
    def __init__(
        self, 
        output_path, 
        remove_prefix_in_ckpt=None, 
        state_dict_converter=lambda x:x,
        use_wandb=False,
        wandb_project="video-dit-training",
        wandb_name=None,
        wandb_config=None,
        grad_norm_interval=50,
        csv_flush_interval=10,
    ):
        super().__init__(output_path, remove_prefix_in_ckpt, state_dict_converter)
        self.use_wandb = use_wandb
        self.wandb_initialized = False
        self.grad_norm_interval = max(0, grad_norm_interval or 0)
        self.csv_flush_interval = csv_flush_interval
        
        rank_env = os.environ.get("RANK")
        try:
            self.is_local_main = rank_env is None or int(rank_env) == 0
        except ValueError:
            self.is_local_main = True
        
        # CSV logging (always enabled)
        self.csv_path = os.path.join(output_path, "training_log.csv")
        os.makedirs(output_path, exist_ok=True)
        
        # Persistent CSV file handle for better I/O performance
        self.csv_file = None
        self.csv_writer = None
        self.csv_write_count = 0
        
        # Initialize CSV file (main process only to avoid races)
        if self.is_local_main:
            # Create header if file doesn't exist
            file_exists = os.path.exists(self.csv_path)
            if not file_exists:
                with open(self.csv_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        'timestamp', 'epoch', 'step', 'loss', 'grad_norm', 'learning_rate', 'elapsed_time'
                    ])
            
            # Open persistent file handle for appending
            self.csv_file = open(self.csv_path, 'a', newline='', buffering=8192)  # 8KB buffer
            self.csv_writer = csv.writer(self.csv_file)
        
        self.start_time = time.time()
        self.current_epoch = 0
        
        # Store WandB config for lazy initialization
        self.wandb_project = wandb_project
        self.wandb_name = wandb_name
        self.wandb_config = wandb_config
        
    
    
    def init_wandb(self, accelerator):
        """Initialize WandB (call this after Accelerator is created, only in main process)."""
        if not self.use_wandb or self.wandb_initialized:
            return
        
        # Only initialize in main process
        if not accelerator.is_main_process:
            return
        
        try:
            import wandb
            
            # Auto-generate name if not provided
            if self.wandb_name is None:
                self.wandb_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            wandb.init(
                project=self.wandb_project,
                name=self.wandb_name,
                config=self.wandb_config or {},
                dir=self.output_path,
            )
            self.wandb_initialized = True
        except ImportError:
            print("WandB not installed, using CSV-only logging")
            self.use_wandb = False
        except Exception as e:
            print(f"WandB initialization failed: {e}, using CSV-only logging")
            self.use_wandb = False
    
    
    def log_metrics(self, metrics, step=None, epoch=None):
        """Log metrics to WandB and CSV with buffered I/O."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        elapsed_time = time.time() - self.start_time
        
        # Add common fields
        metrics_with_meta = {
            'timestamp': timestamp,
            'epoch': epoch if epoch is not None else self.current_epoch,
            'step': step if step is not None else self.num_steps,
            'elapsed_time': elapsed_time,
            **metrics
        }
        
        # Log to WandB
        if self.wandb_initialized:
            import wandb
            wandb.log(metrics, step=self.num_steps)
        
        # Log to CSV using persistent file handle (no repeated open/close)
        if self.csv_writer is not None:
            self.csv_writer.writerow([
                metrics_with_meta['timestamp'],
                metrics_with_meta['epoch'],
                metrics_with_meta['step'],
                metrics_with_meta.get('loss', ''),
                metrics_with_meta.get('grad_norm', ''),
                metrics_with_meta.get('learning_rate', ''),
                f"{elapsed_time:.2f}",
            ])
            self.csv_write_count += 1
            
            # Flush periodically (every N writes) instead of every step
            if self.csv_write_count % self.csv_flush_interval == 0:
                self.csv_file.flush()
    
    
    def on_step_end(self, accelerator, model, save_steps=None, loss=None, optimizer=None):
        """Enhanced step end with metric logging."""
        self.num_steps += 1
        
        # Log metrics
        if loss is not None and accelerator.is_main_process:
            metrics = {'loss': loss.item()}
            
            compute_grad_norm = (
                self.grad_norm_interval
                and accelerator.sync_gradients
                and (self.num_steps % self.grad_norm_interval == 0)
                and hasattr(model, 'parameters')
            )
            if compute_grad_norm:
                total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))
                metrics['grad_norm'] = total_norm.item()
            
            # Add learning rate if optimizer provided
            if optimizer is not None:
                metrics['learning_rate'] = optimizer.param_groups[0]['lr']
            
            self.log_metrics(metrics)
        
        # Save checkpoint
        if save_steps is not None and self.num_steps % save_steps == 0:
            self.save_model(accelerator, model, f"step-{self.num_steps}.safetensors")
    
    
    def on_epoch_end(self, accelerator, model, epoch_id):
        """Enhanced epoch end with metric logging."""
        self.current_epoch = epoch_id
        
        if accelerator.is_main_process:
            print(f"Epoch {epoch_id} completed: {self.num_steps} steps, {time.time() - self.start_time:.2f}s")
    
    def __del__(self):
        """Ensure CSV file is properly closed on destruction."""
        self.close_csv()
    
    def close_csv(self):
        """Close persistent CSV file handle."""
        if self.csv_file is not None and not self.csv_file.closed:
            self.csv_file.flush()
            self.csv_file.close()
            self.csv_file = None
            self.csv_writer = None
    
    
    def on_training_end(self, accelerator, model, save_steps=None):
        """Enhanced training end."""
        if accelerator.is_main_process:
            total_time = time.time() - self.start_time
            print(f"\nTraining completed: {self.num_steps} steps, {total_time/60:.2f}min")
            print(f"Logs saved to: {self.csv_path}\n")
        
        super().on_training_end(accelerator, model, save_steps)
        
        # Close WandB
        if self.wandb_initialized:
            import wandb
            wandb.finish()


class LatentDataset(torch.utils.data.Dataset):
    """Dataset for loading pre-encoded VAE latents from CSV metadata."""
    
    def __init__(self, csv_path, latents_dir, repeat=1, max_samples=None):
        """
        Args:
            csv_path: Path to CSV file with columns 'video_path' and 'caption'
            latents_dir: Directory containing .pt files with pre-encoded latents
            repeat: Number of times to repeat the dataset
            max_samples: Maximum number of samples to use (None = use all)
        """
        self.latents_dir = latents_dir
        self.repeat = repeat
        self.max_samples = max_samples
        
        # Read CSV file
        print(f"Loading metadata from {csv_path}...")
        import pandas as pd
        df = pd.read_csv(csv_path)
        
        if 'video_path' not in df.columns or 'caption' not in df.columns:
            raise ValueError(f"CSV must have 'video_path' and 'caption' columns. Found: {df.columns.tolist()}")
        
        # Build dataset: extract video_id from video_path and construct latent_path
        self.data = []
        missing_latents = []
        
        for idx, row in df.iterrows():
            video_path = row['video_path']
            caption = row['caption']
            
            # Extract video_id from path: /path/to/78687.mp4 -> 78687
            video_id = os.path.splitext(os.path.basename(video_path))[0]
            latent_path = os.path.join(latents_dir, f"{video_id}.pt")
            
            # Check if latent file exists
            if os.path.exists(latent_path):
                self.data.append({
                    'latent_path': latent_path,
                    'prompt': str(caption) if pd.notna(caption) else "",
                    'video_id': video_id
                })
            else:
                missing_latents.append(video_id)
        
        if missing_latents:
            if len(missing_latents) <= 10:
                print(f"Warning: Missing latents for video IDs: {missing_latents}")
            else:
                print(f"Warning: Missing latents for {len(missing_latents)} videos (showing first 10): {missing_latents[:10]}")
        
        if len(self.data) == 0:
            raise ValueError(f"No matching latent files found in {latents_dir} for CSV entries")
        
        if self.max_samples is not None and self.max_samples > 0 and len(self.data) > self.max_samples:
            self.data = self.data[:self.max_samples]
    
    def __len__(self):
        return len(self.data) * self.repeat
    
    def __getitem__(self, idx):
        """Load a pre-encoded latent file."""
        data_idx = idx % len(self.data)
        entry = self.data[data_idx]
        
        latent_data = load_latent_cached(entry['latent_path'])
        
        if not isinstance(latent_data, dict):
            raise ValueError(f"Expected dict from {entry['latent_path']}, got {type(latent_data)}")
        
        if 'latent' not in latent_data:
            raise ValueError(f"Missing 'latent' key in {entry['latent_path']}")
        
        # Clone to avoid sharing tensors from cache across training steps
        return {
            'latent': latent_data['latent'].clone(),
            'prompt': entry['prompt'],
            'video_id': entry['video_id']
        }
class WanTrainingWithRetrievalLatentsModule(DiffusionTrainingModule):
    """Training module that uses pre-encoded latents instead of raw videos."""
    
    def __init__(
        self,
        model_paths=None, model_id_with_origin_paths=None,
        trainable_models=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        max_timestep_boundary=1.0,
        min_timestep_boundary=0.0,
        # Retrieval-specific parameters
        enable_retrieval=False,
        retrieval_index_path=None,
        retrieval_id_map_path=None,
        retrieval_video_dir=None,
        retrieval_k=5,
        retrieval_latents_dir=None,  # Path to pre-encoded latents for retrieval
    ):
        super().__init__()
        # Load models
        model_configs = self.parse_model_configs(model_paths, model_id_with_origin_paths, enable_fp8_training=False)
        self.pipe = WanVideoPipeline.from_pretrained(torch_dtype=torch.bfloat16, device="cpu", model_configs=model_configs)
        
        if enable_retrieval:
            from source.models.wan_video_dit_mem import DiTMem
            self.pipe.dit_mem = DiTMem(
                in_channels=16,
                target_seq_len=100,
                target_feat_dim=1536,
                fft_dropout=0.5,
            ).to(device="cpu", dtype=torch.bfloat16)

            is_trainable = "dit_mem" in (trainable_models or "").lower()
            if not is_trainable:
                self.pipe.dit_mem.requires_grad_(False)
                self.pipe.dit_mem.eval()
        
        self.enable_retrieval = enable_retrieval
        self.retrieval_config = None
        
        if self.enable_retrieval:
            if not all([retrieval_index_path, retrieval_id_map_path, retrieval_video_dir]):
                raise ValueError("Retrieval enabled but missing required paths")
            
            self.retrieval_config = {
                'index_path': retrieval_index_path,
                'id_map_path': retrieval_id_map_path,
                'video_dir': retrieval_video_dir,
                'latents_dir': retrieval_latents_dir,
                'k': retrieval_k
            }
        
        # Training mode
        self.switch_pipe_to_training_mode(
            self.pipe, trainable_models,
            enable_fp8_training=False,
        )
        
        if enable_retrieval and "dit_mem" in (trainable_models or "").lower():
            if hasattr(self.pipe, 'dit_mem') and self.pipe.dit_mem is not None:
                self.pipe.dit_mem.train()
                self.pipe.dit_mem.requires_grad_(True)
        
        self._enable_torch_compile = False
        if hasattr(torch, 'compile') and version.parse(torch.__version__) >= version.parse("2.0.0"):
            self._enable_torch_compile = True
        
        # Store other configs
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.max_timestep_boundary = max_timestep_boundary
        self.min_timestep_boundary = min_timestep_boundary
        self.retrieval_k = retrieval_k
        self.retrieval_latents_dir = retrieval_latents_dir or retrieval_video_dir  # Default to video_dir if not provided
        
        self.retrieved_latents_path_cache = {}
        self.dit_mem_is_trainable = "dit_mem" in (trainable_models or "").lower()
        self.reference_tokens_cache = {} if not self.dit_mem_is_trainable else None
        self.training_step = 0
        self.training_seed = None
        self.use_deterministic_seed = False
        self.retrieval_logger = None
    
    

    def forward(self, data, inputs=None):
        if isinstance(data, dict) and isinstance(data.get('latent'), torch.Tensor):
            return self._forward_batch(data)
        if isinstance(data, list):
            batch = self._collate_from_list(data)
            return self._forward_batch(batch)
        return self._forward_single(data)

    def _collate_from_list(self, samples):
        latents = torch.stack([s['latent'] for s in samples], dim=0)
        prompts = [s['prompt'] for s in samples]
        video_ids = [s.get('video_id') for s in samples]
        return {'latent': latents, 'prompt': prompts, 'video_id': video_ids}

    def _forward_single(self, data):
        batch = {
            'latent': data['latent'].unsqueeze(0),
            'prompt': [data['prompt']],
            'video_id': [data.get('video_id')],
        }
        return self._forward_batch(batch)

    def _build_reference_tokens_batch(self, prompts, video_ids=None):
        """Build reference tokens from retrieved videos. Caches tokens when dit_mem is frozen."""
        if not self.enable_retrieval or self.pipe.video_retriever is None:
            return None
        
        # If dit_mem is frozen, check cache first
        if self.reference_tokens_cache is not None:
            # Try to get all tokens from cache
            cached_tokens = []
            prompts_needing_compute = []
            prompt_indices_needing_compute = []
            
            for idx, prompt in enumerate(prompts):
                current_video_id = video_ids[idx] if video_ids and idx < len(video_ids) else None
                cache_key = (prompt, current_video_id) if current_video_id else prompt
                if cache_key in self.reference_tokens_cache:
                    cached_tokens.append((idx, self.reference_tokens_cache[cache_key]))
                else:
                    prompts_needing_compute.append(prompt)
                    prompt_indices_needing_compute.append(idx)
            
            # If all prompts are cached, return immediately
            if not prompts_needing_compute and cached_tokens:
                # Reconstruct batch from cache
                seq_len = cached_tokens[0][1].shape[0]
                feat_dim = cached_tokens[0][1].shape[1]
                stacked = torch.zeros(len(prompts), seq_len, feat_dim, 
                                    device=self.pipe.device, dtype=self.pipe.torch_dtype)
                for idx, tokens in cached_tokens:
                    stacked[idx] = tokens.to(device=self.pipe.device, dtype=self.pipe.torch_dtype, non_blocking=True)
                return stacked

        self.pipe.load_models_to_device(["dit_mem"])
        tokens_per_sample = []

        for idx, prompt in enumerate(prompts):
            current_video_id = video_ids[idx] if video_ids and idx < len(video_ids) else None
            cache_key = (prompt, current_video_id) if current_video_id else prompt
            
            if self.reference_tokens_cache is not None and cache_key in self.reference_tokens_cache:
                cached_tokens = self.reference_tokens_cache[cache_key]
                tokens_per_sample.append(cached_tokens.to(device=self.pipe.device, non_blocking=True))
                continue
            
            if cache_key in self.retrieved_latents_path_cache:
                latent_paths = self.retrieved_latents_path_cache[cache_key]
            else:
                retrieved_paths = self.pipe.video_retriever.retrieve(prompt, k=self.retrieval_k)
                latent_paths = []
                retrieved_video_info = []
                
                if retrieved_paths:
                    rank = 1
                    for latent_path in retrieved_paths:
                        video_id = os.path.splitext(os.path.basename(latent_path))[0]
                        if os.path.exists(latent_path):
                            latent_paths.append(latent_path)
                            retrieved_video_info.append({
                                'video_id': video_id,
                                'type': 'retrieved',
                                'rank': rank
                            })
                            rank += 1
                elif not current_video_id:
                    print(f"Warning: No GT video_id and retrieval failed for prompt: {prompt[:50]}...")
                
                self.retrieved_latents_path_cache[cache_key] = latent_paths
                
                if hasattr(self, 'retrieval_logger') and self.retrieval_logger is not None:
                    self.retrieval_logger.log_retrieval(
                        prompt=prompt,
                        gt_video_id=current_video_id,
                        retrieved_videos=retrieved_video_info
                    )

            retrieved_latents_list = []
            expected_shape = None
            for latent_path in latent_paths:
                latent_data = load_latent_cached(latent_path)
                latent_tensor = latent_data.get('latent')
                if latent_tensor is None:
                    continue
                
                if expected_shape is None or latent_tensor.shape == expected_shape:
                    expected_shape = latent_tensor.shape
                    retrieved_latents_list.append(latent_tensor)

            if not retrieved_latents_list:
                tokens_per_sample.append(None)
                continue

            retrieved_latents = torch.stack(retrieved_latents_list, dim=0).unsqueeze(0)
            retrieved_latents = retrieved_latents.to(device=self.pipe.device, dtype=self.pipe.torch_dtype, non_blocking=True)
            tokens = self.pipe.dit_mem(retrieved_latents).squeeze(0)
            
            if self.reference_tokens_cache is not None:
                self.reference_tokens_cache[cache_key] = tokens.detach().cpu()
            
            tokens_per_sample.append(tokens)

        if not any(token is not None for token in tokens_per_sample):
            return None

        seq_len = next(token.shape[0] for token in tokens_per_sample if token is not None)
        feat_dim = next(token.shape[1] for token in tokens_per_sample if token is not None)
        stacked = torch.zeros(len(prompts), seq_len, feat_dim, device=self.pipe.device, dtype=self.pipe.torch_dtype)
        for idx, token in enumerate(tokens_per_sample):
            if token is not None:
                stacked[idx] = token.to(device=self.pipe.device, dtype=self.pipe.torch_dtype, non_blocking=True)
        return stacked

    def _forward_batch(self, data):
        latents = data['latent']
        prompts = data.get('prompt') or []
        video_ids = data.get('video_id') or []

        if latents.dim() != 5:
            raise ValueError("Expected batched latents with shape [B, C, T, H, W]")

        batch_size = latents.shape[0]
        C, T, H, W = latents.shape[1:]
        height = H * 8
        width = W * 8
        num_frames = (T - 1) * 4 + 1

        latents = latents.to(dtype=self.pipe.torch_dtype, device=self.pipe.device, non_blocking=True)

        if getattr(self, 'use_deterministic_seed', False) and self.training_seed is not None:
            base_seed = self.training_seed + self.training_step
            seeds = [base_seed + i for i in range(batch_size)]
        else:
            seeds = torch.randint(0, 2**32, (batch_size,), device='cpu').tolist()
        self.training_step += batch_size

        inputs_posi = {"prompt": prompts}
        inputs_nega = {}

        inputs_shared = {
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "cfg_scale": 1,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "cfg_merge": False,
            "vace_scale": 1,
            "max_timestep_boundary": self.max_timestep_boundary,
            "min_timestep_boundary": self.min_timestep_boundary,
            "enable_retrieval": self.enable_retrieval,
            "prompt": prompts,
            "seed": seeds,
            "vace_reference_image": None,
            "input_video": None,
            "input_latents": latents,
            "video_id": video_ids,
        }

        reference_tokens = self._build_reference_tokens_batch(prompts, video_ids)
        if reference_tokens is not None:
            inputs_shared['reference_tokens'] = reference_tokens

        noise_unit = None
        for unit in self.pipe.units:
            if 'NoiseInitializer' in unit.__class__.__name__:
                noise_unit = unit
                break
        if noise_unit is not None:
            noise_output = self.pipe.unit_runner(noise_unit, self.pipe, inputs_shared, inputs_posi, inputs_nega)
            inputs_shared.update(noise_output[0])

        for unit in self.pipe.units:
            if 'NoiseInitializer' in unit.__class__.__name__:
                continue
            if 'InputVideoEmbedder' in unit.__class__.__name__:
                continue
            if 'RetrievedVideoEmbedder' in unit.__class__.__name__:
                continue
            inputs_shared, inputs_posi, inputs_nega = self.pipe.unit_runner(unit, self.pipe, inputs_shared, inputs_posi, inputs_nega)

        inputs = {**inputs_shared, **inputs_posi}
        models = {name: getattr(self.pipe, name) for name in self.pipe.in_iteration_models}
        loss = self.pipe.training_loss(**models, **inputs)
        return loss

def launch_training_task(
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: EnhancedModelLogger,
    learning_rate: float = 1e-5,
    weight_decay: float = 1e-2,
    num_workers: int = 4,
    save_steps: int = None,
    num_epochs: int = 1,
    batch_size: int = 1,
    gradient_accumulation_steps: int = 1,
    find_unused_parameters: bool = False,
):
    """Training loop with logging and checkpointing."""
    optimizer_kwargs = {'lr': learning_rate, 'weight_decay': weight_decay}
    if torch.cuda.is_available():
        try:
            optimizer = torch.optim.AdamW(model.trainable_modules(), fused=True, **optimizer_kwargs)
        except (TypeError, RuntimeError):
            # Fallback for PyTorch < 2.0 or if fused not available
            optimizer = torch.optim.AdamW(model.trainable_modules(), **optimizer_kwargs)
    else:
        optimizer = torch.optim.AdamW(model.trainable_modules(), **optimizer_kwargs)
    
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision="bf16",
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=find_unused_parameters)],
    )
    
    num_processes = accelerator.num_processes
    if num_processes > 1 and num_workers > 0:
        adjusted_workers = max(1, (num_workers + num_processes - 1) // num_processes)
        num_workers = adjusted_workers
    
    use_pin_memory = accelerator.device.type == "cuda"
    per_device_batch_size = batch_size if batch_size and batch_size > 0 else 1
    def collate_fn(batch):
        latents = torch.stack([item['latent'] for item in batch], dim=0)
        prompts = [item['prompt'] for item in batch]
        video_ids = [item.get('video_id') for item in batch]
        return {
            'latent': latents,
            'prompt': prompts,
            'video_id': video_ids,
        }

    dataloader_kwargs = dict(
        dataset=dataset,
        batch_size=per_device_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        persistent_workers=True if num_workers > 0 else False,
    )
    if num_workers > 0:
        dataloader_kwargs["prefetch_factor"] = 2

    dataloader = torch.utils.data.DataLoader(**dataloader_kwargs)
    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)
    
    unwrapped_model = safe_unwrap_model(accelerator, model)
    
    if unwrapped_model.retrieval_config is not None:
        from source.utils.video_retrieval import VideoRetriever
        
        retriever_device = "cuda:0" if (accelerator.is_main_process and torch.cuda.is_available()) else "cpu"
        
        unwrapped_model.pipe.video_retriever = VideoRetriever(
            index_path=unwrapped_model.retrieval_config['index_path'],
            id_map_path=unwrapped_model.retrieval_config['id_map_path'],
            video_dir=unwrapped_model.retrieval_config['video_dir'],
            latents_dir=unwrapped_model.retrieval_config['latents_dir'],
            device=retriever_device
        )
        unwrapped_model.pipe.retrieval_k = unwrapped_model.retrieval_config['k']
        
    if unwrapped_model.enable_retrieval and accelerator.is_main_process:
        output_path = model_logger.output_path
        unwrapped_model.retrieval_logger = RetrievalLogger(
            output_path=output_path,
            csv_filename="retrieval_log.csv"
        )
    
    if hasattr(unwrapped_model, '_enable_torch_compile') and unwrapped_model._enable_torch_compile:
        try:
            unwrapped_model.pipe.dit = torch.compile(
                unwrapped_model.pipe.dit,
                mode="reduce-overhead",
                fullgraph=False,
                dynamic=False
            )
        except Exception as e:
            if accelerator.is_main_process:
                print(f"torch.compile() failed: {e}, continuing without compilation")
    
    # Initialize WandB in main process
    model_logger.init_wandb(accelerator)
    
    # Training loop
    for epoch_id in range(num_epochs):
        model_logger.current_epoch = epoch_id
        
        for data in tqdm(dataloader, desc=f"Epoch {epoch_id}"):
            with accelerator.accumulate(model):
                # Forward pass (data already contains latents)
                loss = model(data)
                
                # Backward pass
                accelerator.backward(loss)
                
                # Optimizer step
                optimizer.step()
                
                # LR scheduler step (only when gradients are synchronized)
                if accelerator.sync_gradients:
                    scheduler.step()
                
                model_logger.on_step_end(accelerator, model, save_steps, loss=loss, optimizer=optimizer)
                
                # Clear gradients
                optimizer.zero_grad()
        
        if save_steps is None:
            model_logger.on_epoch_end(accelerator, model, epoch_id)
    
    model_logger.on_training_end(accelerator, model, save_steps)
    
    # Close retrieval logger if it exists
    if hasattr(unwrapped_model, 'retrieval_logger') and unwrapped_model.retrieval_logger is not None:
        unwrapped_model.retrieval_logger.close()
    
    # Wait for all processes to finish before cleanup
    accelerator.wait_for_everyone()
    
    # Properly cleanup distributed process group to avoid NCCL warnings
    if accelerator.num_processes > 1 and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def parse_args():
    parser = argparse.ArgumentParser(description="Train with pre-encoded VAE latents")
    
    # Model paths
    parser.add_argument("--model_paths", type=str, default=None,
                        help="Path to pretrained model checkpoint (JSON list or comma separated).")
    parser.add_argument("--model_id_with_origin_paths", type=str, default=None,
                        help="Model ID with origin paths (advanced)")
    
    # Training settings
    parser.add_argument("--output_path", type=str, default=None,
                        help="Output directory for checkpoints and logs")
    parser.add_argument("--trainable_models", type=str, default="dit,dit_mem",
                        help="Comma-separated list of trainable models (dit,dit_mem,vae,etc)")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-2,
                        help="Weight decay")
    parser.add_argument("--num_epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--save_steps", type=int, default=500,
                        help="Save checkpoint every N steps")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of dataloader workers")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Per-device batch size for training")
    parser.add_argument("--find_unused_parameters", action="store_true",
                        help="Enable find_unused_parameters in DDP")
    parser.add_argument("--log_grad_norm_interval", type=int, default=50,
                        help="Log gradient norm every N steps (0 disables logging)")
    
    # Dataset settings
    parser.add_argument("--csv_path", type=str, default=None,
                        help="Path to CSV file with video_path and caption columns")
    parser.add_argument("--latents_dir", type=str, default=None,
                        help="Directory containing pre-encoded VAE latents (.pt files)")
    parser.add_argument("--dataset_repeat", type=int, default=1,
                        help="Number of times to repeat the dataset")
    parser.add_argument("--max_train_samples", type=int, default=None,
                        help="Limit dataset to the first N samples (after repeat). Use for debugging")
    
    # Gradient checkpointing
    parser.add_argument("--use_gradient_checkpointing_offload", action="store_true",
                        help="Enable gradient checkpointing with CPU offload")
    
    # Timestep boundaries
    parser.add_argument("--max_timestep_boundary", type=float, default=1.0,
                        help="Maximum timestep boundary")
    parser.add_argument("--min_timestep_boundary", type=float, default=0.0,
                        help="Minimum timestep boundary")
    
    # Retrieval settings
    parser.add_argument("--enable_retrieval", action="store_true",
                        help="Enable video retrieval augmentation")
    parser.add_argument("--retrieval_index_path", type=str, default=None,
                        help="Path to FAISS index file")
    parser.add_argument("--retrieval_id_map_path", type=str, default=None,
                        help="Path to ID mapping JSON file")
    parser.add_argument("--retrieval_video_dir", type=str, default=None,
                        help="Directory containing video files for retrieval")
    parser.add_argument("--retrieval_k", type=int, default=5,
                        help="Number of videos to retrieve (k)")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file to load training parameters from")
    
    # Logging settings
    parser.add_argument("--use_wandb", action="store_true",
                        help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="s2s-video-dit",
                        help="WandB project name")
    parser.add_argument("--wandb_name", type=str, default=None,
                        help="WandB run name (auto-generated if not provided)")
    
    # Reproducibility
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility. If None, training is non-deterministic.")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # Load YAML config (if provided) and create a simple getter that prefers YAML values
    cfg = {}
    if getattr(args, 'config', None):
        cfg_path = args.config
        if os.path.exists(cfg_path):
            with open(cfg_path, 'r') as f:
                cfg = yaml.safe_load(f) or {}
            print(f"Loaded config from {cfg_path}")
        else:
            print(f"Config file not found: {cfg_path}, using CLI args")

    def cfg_get(key, default=None):
        """Return value from config if present, otherwise fall back to CLI args or default."""
        # Support nested keys using dot notation in YAML (e.g., retrieval.index_path)
        if not cfg:
            return getattr(args, key, default)
        # direct key
        if key in cfg:
            return cfg[key]
        # try to find nested keys in top-level sections
        parts = key.split('.')
        cur = cfg
        for p in parts:
            if isinstance(cur, dict) and p in cur:
                cur = cur[p]
            else:
                cur = None
                break
        if cur is not None:
            return cur
        return getattr(args, key, default)

    def resolve_path(value):
        """Resolve paths relative to the project root while preserving absolute inputs."""
        if value is None:
            return None
        if isinstance(value, (list, tuple)):
            return [resolve_path(item) for item in value]
        value_str = str(value)
        if not value_str:
            return value_str
        candidate = Path(value_str)
        if candidate.is_absolute():
            return str(candidate)
        return str((PROJECT_ROOT / candidate).resolve())

    def parse_model_path_list(value):
        """Normalize model path input (string, JSON string, list) into a list of strings."""
        if value is None:
            return None
        if isinstance(value, (list, tuple)):
            return [str(v) for v in value]
        value_str = str(value).strip()
        if not value_str:
            return None
        try:
            loaded = json.loads(value_str)
            if isinstance(loaded, str):
                return [loaded]
            if isinstance(loaded, (list, tuple)):
                return [str(v) for v in loaded]
        except json.JSONDecodeError:
            pass
        return [p.strip() for p in value_str.split(",") if p.strip()]
    
    # Validate required parameters
    output_path_raw = cfg_get('output_path', args.output_path)
    if output_path_raw is None:
        raise ValueError("output_path must be specified either in YAML config or via --output_path CLI argument")
    output_path = resolve_path(output_path_raw)
    
    # Setup reproducibility
    seed = cfg_get('seed', args.seed)
    if seed is not None:
        import random
        import numpy as np
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        # For deterministic behavior (may impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(f"Random seed set to {seed}")
    else:
        print("No seed specified, training will be non-deterministic")
    
    # Resolve frequently used paths
    enable_retrieval = cfg_get('enable_retrieval', args.enable_retrieval)
    csv_path = resolve_path(cfg_get('csv_path', args.csv_path))
    latents_dir = resolve_path(cfg_get('latents_dir', args.latents_dir))
    
    # Validate required dataset paths
    if csv_path is None:
        raise ValueError("csv_path must be specified either in YAML config or via --csv_path CLI argument")
    if latents_dir is None:
        raise ValueError("latents_dir must be specified either in YAML config or via --latents_dir CLI argument")
    
    model_paths_cfg = parse_model_path_list(cfg_get('model_paths', args.model_paths))
    model_paths_resolved = resolve_path(model_paths_cfg) if model_paths_cfg else None
    model_id_with_origin_paths = cfg_get('model_id_with_origin_paths', args.model_id_with_origin_paths)
    if not model_paths_resolved and not model_id_with_origin_paths:
        raise ValueError("No model paths provided. Set `model_paths` in the config or CLI.")
    
    retrieval_index_path = resolve_path(cfg_get('retrieval_index_path', args.retrieval_index_path)) if enable_retrieval else None
    retrieval_id_map_path = resolve_path(cfg_get('retrieval_id_map_path', args.retrieval_id_map_path)) if enable_retrieval else None
    retrieval_video_dir = resolve_path(cfg_get('retrieval_video_dir', args.retrieval_video_dir)) if enable_retrieval else None
    
    # Validate retrieval paths if retrieval is enabled
    if enable_retrieval:
        if retrieval_index_path is None:
            raise ValueError("retrieval_index_path must be specified when enable_retrieval is True")
        if retrieval_id_map_path is None:
            raise ValueError("retrieval_id_map_path must be specified when enable_retrieval is True")
        if retrieval_video_dir is None:
            raise ValueError("retrieval_video_dir must be specified when enable_retrieval is True")

    # Save training config (merge YAML and CLI)
    os.makedirs(output_path, exist_ok=True)
    training_config = {
        "output_path": output_path,
        "csv_path": csv_path,
        "learning_rate": cfg_get('learning_rate', args.learning_rate),
        "weight_decay": cfg_get('weight_decay', args.weight_decay),
        "num_epochs": cfg_get('num_epochs', args.num_epochs),
        "gradient_accumulation_steps": cfg_get('gradient_accumulation_steps', args.gradient_accumulation_steps),
        "save_steps": cfg_get('save_steps', args.save_steps),
        "trainable_models": cfg_get('trainable_models', args.trainable_models),
        "latents_dir": latents_dir,
        "max_train_samples": cfg_get('max_train_samples', args.max_train_samples),
        "enable_retrieval": enable_retrieval,
        "model_paths": model_paths_resolved,
        "batch_size": cfg_get('batch_size', args.batch_size),
        "use_wandb": cfg_get('use_wandb', args.use_wandb),
        "log_grad_norm_interval": cfg_get('log_grad_norm_interval', args.log_grad_norm_interval),
    }

    if enable_retrieval:
        training_config.update({
            "retrieval_index_path": retrieval_index_path,
            "retrieval_id_map_path": retrieval_id_map_path,
            "retrieval_video_dir": retrieval_video_dir,
            "retrieval_k": cfg_get('retrieval_k', args.retrieval_k),
        })

    with open(os.path.join(output_path, "training_config.json"), "w") as f:
        json.dump(training_config, f, indent=2)
    
    dataset = LatentDataset(
        csv_path=csv_path,
        latents_dir=latents_dir,
        repeat=cfg_get('dataset_repeat', args.dataset_repeat),
        max_samples=cfg_get('max_train_samples', args.max_train_samples),
    )
    print(f"Dataset: {len(dataset)} samples")
    model = WanTrainingWithRetrievalLatentsModule(
        model_paths=model_paths_resolved,
        model_id_with_origin_paths=model_id_with_origin_paths,
        trainable_models=cfg_get('trainable_models', args.trainable_models),
        use_gradient_checkpointing_offload=cfg_get('use_gradient_checkpointing_offload', args.use_gradient_checkpointing_offload),
        max_timestep_boundary=cfg_get('max_timestep_boundary', args.max_timestep_boundary),
        min_timestep_boundary=cfg_get('min_timestep_boundary', args.min_timestep_boundary),
        enable_retrieval=enable_retrieval,
        retrieval_index_path=retrieval_index_path,
        retrieval_id_map_path=retrieval_id_map_path,
        retrieval_video_dir=retrieval_video_dir,
        retrieval_k=cfg_get('retrieval_k', args.retrieval_k),
        retrieval_latents_dir=latents_dir,  # Use same latents_dir
    )
    
    if seed is not None:
        model.use_deterministic_seed = True
        model.training_seed = seed
    else:
        model.use_deterministic_seed = False
    
    print("Model initialized")
    model_logger = EnhancedModelLogger(
        output_path,
        use_wandb=cfg_get('use_wandb', args.use_wandb),
        wandb_project=cfg_get('wandb_project', args.wandb_project),
        wandb_name=cfg_get('wandb_name', args.wandb_name),
        wandb_config=training_config,
        grad_norm_interval=cfg_get('log_grad_norm_interval', args.log_grad_norm_interval),
    )
    
    print("Starting training...\n")
    launch_training_task(
        dataset=dataset,
        model=model,
        model_logger=model_logger,
        learning_rate=cfg_get('learning_rate', args.learning_rate),
        weight_decay=cfg_get('weight_decay', args.weight_decay),
        num_workers=cfg_get('num_workers', args.num_workers),
        save_steps=cfg_get('save_steps', args.save_steps),
        num_epochs=cfg_get('num_epochs', args.num_epochs),
        batch_size=cfg_get('batch_size', args.batch_size),
        gradient_accumulation_steps=cfg_get('gradient_accumulation_steps', args.gradient_accumulation_steps),
        find_unused_parameters=cfg_get('find_unused_parameters', args.find_unused_parameters),
    )
    
    print("\nTraining completed!")
