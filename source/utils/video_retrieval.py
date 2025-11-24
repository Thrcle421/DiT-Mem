"""
Video Retrieval using FAISS index.
"""

import os
import json
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

try:
    import faiss  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    faiss = None

try:
    from transformers import AutoTokenizer, AutoModel  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError("`transformers` is required for video retrieval. Install via `pip install transformers`." ) from exc


class VideoRetriever:
    """
    Video retrieval system using FAISS for semantic search.
    
    Args:
        index_path: Path to the FAISS index file (.index)
        id_map_path: Path to JSON/pickle file mapping FAISS index IDs to video IDs/paths
        video_dir: Directory containing video files
        latents_dir: Optional directory containing pre-computed latent (.pt) files
        model_name: Name of the transformer model for text embedding
        device: Device to run the embedding model on (default: auto-detect CUDA/CPU)
    """

    VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov', '.webm', '.mkv', '.flv', '.wmv', '.gif')
    
    def __init__(
        self,
        index_path: str,
        id_map_path: str,
        video_dir: str,
        latents_dir: str = None,
        model_name: str = "Alibaba-NLP/gte-base-en-v1.5",
        device: str = None
    ):
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        if faiss is None:
            raise ImportError("faiss library is required for retrieval. Install via `pip install faiss-cpu`." )

        if not os.path.exists(index_path):
            raise FileNotFoundError(f"FAISS index not found: {index_path}")
        print(f"Loading FAISS index from: {index_path}")
        self.index = faiss.read_index(index_path)
        print(f"✓ Loaded FAISS index with {self.index.ntotal} vectors")
        
        if not os.path.exists(id_map_path):
            raise FileNotFoundError(f"ID map not found: {id_map_path}")
        
        self.metadata_list = None
        try:
            import pickle
            import pandas as pd
            with open(id_map_path, 'rb') as f:
                id_map_data = pickle.load(f)

            if isinstance(id_map_data, pd.DataFrame):
                candidate_cols = [col for col in ['video_path', 'path', 'video', 'video_id', 'id'] if col in id_map_data.columns]
                if not candidate_cols:
                    raise ValueError("ID map DataFrame is missing required identifier columns (video_path/path/video/video_id/id)")

                primary_series = id_map_data[candidate_cols].bfill(axis=1).iloc[:, 0]
                primary_series = primary_series.where(primary_series.notna(), None)

                self.id_map_df = id_map_data
                self.id_map_list = primary_series.tolist()
                self.id_map = None

                metadata_cols = [col for col in id_map_data.columns if col not in candidate_cols]
                if metadata_cols:
                    metadata_df = id_map_data[metadata_cols].where(id_map_data[metadata_cols].notna(), None)
                    self.metadata_list = metadata_df.to_dict('records')
                else:
                    self.metadata_list = None

                print(f"✓ Loaded ID map from pickle (DataFrame) with {len(id_map_data)} entries")
            elif isinstance(id_map_data, dict):
                max_idx = max(int(k) for k in id_map_data.keys())
                self.id_map_list = [None] * (max_idx + 1)
                for k, v in id_map_data.items():
                    self.id_map_list[int(k)] = v
                self.id_map_df = None
                self.id_map = None
                self.metadata_list = None
                print(f"✓ Loaded ID map from pickle (dict format) with {len(id_map_data)} entries")
            elif isinstance(id_map_data, list):
                self.id_map_list = id_map_data
                self.id_map_df = None
                self.id_map = None
                self.metadata_list = None
                print(f"✓ Loaded ID map from pickle (list format) with {len(id_map_data)} entries")
            elif hasattr(id_map_data, 'to_dict'):
                series_dict = id_map_data.to_dict()
                max_idx = max(int(k) for k in series_dict.keys())
                self.id_map_list = [None] * (max_idx + 1)
                for k, v in series_dict.items():
                    self.id_map_list[int(k)] = v
                self.id_map_df = None
                self.id_map = None
                self.metadata_list = None
                print(f"✓ Loaded ID map from pickle (Series) with {len(series_dict)} entries")
            else:
                raise ValueError(f"Pickle format not recognized: {type(id_map_data)}")
        except (pickle.UnpicklingError, UnicodeDecodeError):
            with open(id_map_path, 'r') as f:
                id_map_dict = json.load(f)
            if isinstance(id_map_dict, list):
                if id_map_dict and isinstance(id_map_dict[0], dict):
                    video_path_candidates = ['video_path', 'path', 'video', 'video_id', 'id']
                    video_key = None
                    for key in video_path_candidates:
                        if key in id_map_dict[0]:
                            video_key = key
                            break
                    
                    if video_key:
                        self.id_map_list = [item.get(video_key) for item in id_map_dict]
                        self.metadata_list = [
                            {k: v for k, v in item.items() if k != video_key}
                            for item in id_map_dict
                        ]
                        print(f"✓ Loaded ID map from JSON (list of dicts) with {len(id_map_dict)} entries")
                    else:
                        self.id_map_list = id_map_dict
                        self.metadata_list = None
                        print(f"✓ Loaded ID map from JSON (list of dicts) with {len(id_map_dict)} entries")
                else:
                    self.id_map_list = id_map_dict
                    self.metadata_list = None
                    print(f"✓ Loaded ID map from JSON (simple list) with {len(id_map_dict)} entries")
            else:
                max_idx = max(int(k) for k in id_map_dict.keys())
                self.id_map_list = [None] * (max_idx + 1)
                for k, v in id_map_dict.items():
                    self.id_map_list[int(k)] = v
                self.metadata_list = None
                print(f"✓ Loaded ID map from JSON (dict) with {len(id_map_dict)} entries")
            
            self.id_map_df = None
            self.id_map = None
        
        if not os.path.exists(video_dir):
            raise FileNotFoundError(f"Video directory not found: {video_dir}")
        self.video_dir = video_dir
        if latents_dir is None or not os.path.exists(latents_dir):
            print(f"Warning: Latents directory not found")
        self.latents_dir = latents_dir
        self.latent_lookup, self.video_lookup = self._index_video_directory(self.video_dir)
        self._path_cache = {}
        
        print(f"Loading embedding model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(self.device)
        self.model.eval()
        print(f"✓ Embedding model loaded on device: {self.device}")

    def _index_video_directory(self, video_dir: str):
        """
        Scan video directory to create lookup tables for latent (.pt) and video files.
        
        Args:
            video_dir: Directory to scan
            
        Returns:
            Tuple of (latent_lookup, video_lookup) dictionaries
        """
        latent_lookup = {}
        video_lookup = {}

        try:
            with os.scandir(video_dir) as entries:
                for entry in entries:
                    if not entry.is_file():
                        continue
                    base, ext = os.path.splitext(entry.name)
                    ext_lower = ext.lower()
                    if ext_lower == '.pt':
                        latent_lookup[base] = entry.path
                    elif ext_lower in self.VIDEO_EXTENSIONS:
                        video_lookup.setdefault(base, entry.path)
        except FileNotFoundError:
            pass

        return latent_lookup, video_lookup
    
    def retrieve_video_ids(self, query: str, k: int = 5, return_metadata: bool = False):
        """
        Retrieve k most similar video IDs based on text query.
        
        Args:
            query: Text description to search for
            k: Number of videos to retrieve
            return_metadata: If True, return (video_ids, metadata_list) tuple
            
        Returns:
            List of video IDs, or (video_ids, metadata_list) tuple if return_metadata=True
        """
        current_device = next(self.model.parameters()).device
        
        with torch.no_grad():
            batch_dict = self.tokenizer(
                [query], 
                max_length=512, 
                padding=True, 
                truncation=True, 
                return_tensors='pt'
            ).to(current_device)
            outputs = self.model(**batch_dict)
            query_embedding = outputs.last_hidden_state[:, 0]
            normalized_query_embedding = F.normalize(query_embedding, p=2, dim=1)
            query_embedding_np = normalized_query_embedding.cpu().numpy().astype('float32')
        
        distances, indices = self.index.search(query_embedding_np, k)
        
        video_ids = []
        metadata_list = []
        for idx in indices[0]:
            if idx < 0 or idx >= len(self.id_map_list):
                continue
            
            video_id = self.id_map_list[idx]
            if video_id is not None:
                video_ids.append(video_id)
                
                if return_metadata:
                    if self.metadata_list and idx < len(self.metadata_list):
                        metadata_list.append(self.metadata_list[idx] or {})
                    else:
                        metadata_list.append({})
        
        if return_metadata:
            return video_ids, metadata_list
        return video_ids
    
    def retrieve_video_ids_batch(self, queries: list[str], k: int = 5, return_metadata: bool = False):
        """
        Batch retrieve k most similar video IDs for multiple queries.
        
        Args:
            queries: List of text descriptions to search for
            k: Number of videos to retrieve per query
            return_metadata: If True, return (video_ids_list, metadata_lists) tuple
            
        Returns:
            List of video ID lists, or (video_ids_lists, metadata_lists) tuple if return_metadata=True
        """
        if not queries:
            return [] if not return_metadata else ([], [])
        
        current_device = next(self.model.parameters()).device
        
        with torch.no_grad():
            batch_dict = self.tokenizer(
                queries,
                max_length=512, 
                padding=True, 
                truncation=True, 
                return_tensors='pt'
            ).to(current_device)
            outputs = self.model(**batch_dict)
            query_embeddings = outputs.last_hidden_state[:, 0]
            normalized_query_embeddings = F.normalize(query_embeddings, p=2, dim=1)
            query_embeddings_np = normalized_query_embeddings.cpu().numpy().astype('float32')
        
        distances, indices = self.index.search(query_embeddings_np, k)
        
        all_video_ids = []
        all_metadata = [] if return_metadata else None
        
        for query_indices in indices:
            video_ids = []
            metadata_list = []
            
            for idx in query_indices:
                if idx < 0 or idx >= len(self.id_map_list):
                    continue
                
                video_id = self.id_map_list[idx]
                if video_id is not None:
                    video_ids.append(video_id)
                    
                    if return_metadata:
                        if self.metadata_list and idx < len(self.metadata_list):
                            metadata_list.append(self.metadata_list[idx] or {})
                        else:
                            metadata_list.append({})
            
            all_video_ids.append(video_ids)
            if return_metadata:
                all_metadata.append(metadata_list)
        
        if return_metadata:
            return all_video_ids, all_metadata
        return all_video_ids
    
    def get_video_paths(self, video_ids: List[str]) -> List[str]:
        """
        Convert video IDs to full file paths (prefers .pt latent files if available).
        
        Args:
            video_ids: List of video IDs
            
        Returns:
            List of full file paths
        """
        video_paths = []
        for vid in video_ids:
            base_name = os.path.splitext(os.path.basename(vid))[0]
            latent_path = os.path.join(self.latents_dir, f"{base_name}.pt")
            if os.path.exists(latent_path):
                video_paths.append(latent_path)
                continue
            else:
                video_path = os.path.join(self.video_dir, vid)
                if os.path.exists(video_path):
                    video_paths.append(video_path)
                else:
                    print(f"Warning: Video file not found: {video_path}")
        
        return video_paths

    def _resolve_video_path(self, vid: str, prefer_latents: bool):
        """
        Resolve video identifier to absolute path.
        
        Args:
            vid: Video identifier
            prefer_latents: If True, prefer .pt latent files
            
        Returns:
            Resolved file path, or None if not found
        """
        cache_key = (vid, prefer_latents)
        if cache_key in self._path_cache:
            return self._path_cache[cache_key]

        resolved_path = None

        if not vid:
            self._path_cache[cache_key] = resolved_path
            return resolved_path

        if os.path.isabs(vid):
            if prefer_latents and not vid.endswith('.pt'):
                base = os.path.splitext(os.path.basename(vid))[0]
                resolved_path = self.latent_lookup.get(base)
            if resolved_path is None and os.path.exists(vid):
                resolved_path = vid
        else:
            base_name, ext = os.path.splitext(vid)
            ext_lower = ext.lower()

            if ext:
                identifier = base_name or vid

                if prefer_latents and ext_lower in self.VIDEO_EXTENSIONS:
                    resolved_path = self.latent_lookup.get(identifier)

                if resolved_path is None:
                    candidate = os.path.join(self.video_dir, vid)
                    if os.path.exists(candidate):
                        resolved_path = candidate
                    elif identifier in self.video_lookup:
                        resolved_path = self.video_lookup[identifier]

                if resolved_path is None and identifier in self.latent_lookup:
                    resolved_path = self.latent_lookup[identifier]
            else:
                identifier = vid

                if prefer_latents:
                    resolved_path = self.latent_lookup.get(identifier)

                if resolved_path is None:
                    if identifier in self.video_lookup:
                        resolved_path = self.video_lookup[identifier]
                    else:
                        for ext_candidate in self.VIDEO_EXTENSIONS:
                            candidate = os.path.join(self.video_dir, f"{identifier}{ext_candidate}")
                            if os.path.exists(candidate):
                                self.video_lookup.setdefault(identifier, candidate)
                                resolved_path = candidate
                                break

                if resolved_path is None:
                    latent_candidate = os.path.join(self.video_dir, f"{identifier}.pt")
                    if os.path.exists(latent_candidate):
                        self.latent_lookup.setdefault(identifier, latent_candidate)
                        resolved_path = latent_candidate

            if resolved_path is None:
                identifier = base_name if ext else vid
                if identifier in self.latent_lookup:
                    resolved_path = self.latent_lookup[identifier]
                elif identifier in self.video_lookup:
                    resolved_path = self.video_lookup[identifier]

        if resolved_path is None:
            print(f"Warning: Video file not found for ID: {vid}")

        self._path_cache[cache_key] = resolved_path
        return resolved_path
    
    def retrieve_with_metadata(self, query: str, k: int = 5):
        """
        Retrieve k most similar videos with metadata.
        
        Args:
            query: Text query string
            k: Number of videos to retrieve
            
        Returns:
            Tuple of (video_paths, metadata_list)
        """
        video_ids, metadata_list = self.retrieve_video_ids(query, k=k, return_metadata=True)
        video_paths = self.get_video_paths(video_ids)
        
        return video_paths, metadata_list

    def retrieve(self, query: str, k: int = 5):
        """
        Retrieve k most similar videos based on text query.
        
        Args:
            query: Text query string
            k: Number of videos to retrieve
            
        Returns:
            List of video file paths (prefers .pt latent files if available)
        """
        video_ids = self.retrieve_video_ids(query, k=k)       
        video_paths = self.get_video_paths(video_ids)
        
        return video_paths
    
