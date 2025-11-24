"""
Dual Resampler with 3D Frequency Domain Filtering (High-pass + Low-pass)

This implementation uses fixed (non-trainable) complex weights for frequency filtering.
It contains two parallel resamplers:
1. High-pass resampler: emphasizes high-frequency components
2. Low-pass resampler: emphasizes low-frequency components

The outputs from both resamplers are concatenated along the sequence dimension,
resulting in a dual-frequency representation.

Author: Modified from wan_video_resampler_3dfft_fixed.py
Date: 2025-11-06
"""

from loguru import logger

import torch
import torch.nn as nn
from typing import Tuple, Optional

__all__ = [
    'FilterLayer3D_HighFix',
    'FilterLayer3D_LowFix',
    'Conv3dFFTBlock_HighFix',
    'Conv3dFFTBlock_LowFix',
    'Resampler_3DFFT_HighFix',
    'Resampler_3DFFT_LowFix',
    'DualResampler3DFFTAtten',
]


class FilterLayer3D_HighFix(nn.Module):
    """
    3D Frequency Domain Filter Layer for video features with FIXED (non-trainable) weights.
    
    Applies 3D FFT-based filtering on temporal and spatial dimensions of 3D convolutional features.
    Input: [B, C, D, H, W] where D is the temporal dimension, H and W are spatial dimensions
    Output: [B, C, D, H, W] (same shape)
    
    The filtering process:
    1. Spatial-temporal domain → Frequency domain (rfftn on dimensions D, H, W)
    2. Frequency filtering (FIXED complex weights, initialized with high-pass pattern)
    3. Frequency domain → Spatial-temporal domain (irfftn)
    4. Residual connection + BatchNorm
    
    Note: The complex_weight is NOT trainable and initialized using high-pass filter pattern.
    """
    
    def __init__(self, 
                 num_channels: int,
                 dropout_prob: float = 0.1,
                 max_freq: Optional[Tuple[int, int, int]] = None):
        """
        Args:
            num_channels: Number of channels (C) in the feature map
            dropout_prob: Dropout probability
            max_freq: Maximum frequency dimensions (D_freq, H_freq, W_freq)
        """
        super().__init__()
        self.num_channels = num_channels
        if max_freq is not None:
            if any(v <= 0 for v in max_freq):
                raise ValueError("max_freq entries must be positive")
        self.max_freq = max_freq
        
        # Initialize with minimal shape - will be resized on first forward pass
        init_shape = (
            num_channels,
            1 if max_freq is None else min(1, max_freq[0]),
            1 if max_freq is None else min(1, max_freq[1]),
            1 if max_freq is None else min(1, max_freq[2]),
            2,
        )
        
        # Register as buffer (non-trainable) instead of Parameter
        self.register_buffer('complex_weight', torch.zeros(init_shape, dtype=torch.float32))
        
        self.D_freq = init_shape[1]
        self.H_freq = init_shape[2]
        self.W_freq = init_shape[3]
        
        self.dropout = nn.Dropout(dropout_prob)
        self.norm = nn.BatchNorm3d(num_channels)

    def _init_weights(self, D_freq: int, H_freq: int, W_freq: int, device: torch.device):
        """Initialize the fixed frequency weights using high-pass filter pattern."""
        if self.max_freq is not None:
            max_d, max_h, max_w = self.max_freq
            if D_freq > max_d or H_freq > max_h or W_freq > max_w:
                raise ValueError(
                    f"FilterLayer3D requested ({D_freq}, {H_freq}, {W_freq}) but max_freq={self.max_freq}. "
                    "Increase max_freq when instantiating the layer."
                )

        with torch.no_grad():
            current = self.complex_weight.to(device=device)
            target_shape = (self.num_channels, D_freq, H_freq, W_freq, 2)
            
            if current.shape != target_shape:
                # Initialize with high-pass filter pattern
                freq_d = torch.linspace(0, 1, D_freq, dtype=torch.float32, device=device)
                freq_h = torch.linspace(0, 1, H_freq, dtype=torch.float32, device=device)
                freq_w = torch.linspace(0, 1, W_freq, dtype=torch.float32, device=device)
                
                # Create frequency grid
                grid_d, grid_h, grid_w = torch.meshgrid(freq_d, freq_h, freq_w, indexing='ij')
                
                # Calculate frequency magnitude
                freq_magnitude = torch.sqrt(grid_d**2 + grid_h**2 + grid_w**2)
                freq_magnitude = freq_magnitude / torch.sqrt(torch.tensor(3.0, device=device))
                
                # High-pass filter curve: emphasizes high frequencies
                base_curve = 0.2 + 0.8 * torch.sqrt(freq_magnitude)
                
                # Per-channel variation
                channel_slopes = torch.linspace(0.7, 1.3, self.num_channels, dtype=torch.float32, device=device)
                channel_offsets = torch.linspace(-0.05, 0.05, self.num_channels, dtype=torch.float32, device=device)
                
                # Real part: modulated high-pass filter
                real_part = (
                    channel_slopes.view(-1, 1, 1, 1) * base_curve.unsqueeze(0)
                    + channel_offsets.view(-1, 1, 1, 1)
                ).clamp_(min=0.1, max=1.2)
                
                # Imaginary part: set to zero for phase-neutral filtering
                imag_part = torch.zeros(self.num_channels, D_freq, H_freq, W_freq, dtype=torch.float32, device=device)
                
                # Stack real and imaginary parts
                init_data = torch.stack([real_part, imag_part], dim=-1)
                
                # Add small random noise for diversity
                init_data.add_(torch.randn_like(init_data) * 0.05)
                
                # Clamp real part to ensure stable filtering
                init_data[..., 0].clamp_(min=0.05, max=1.5)
                
                # Update the buffer
                self.complex_weight.data = init_data
                self.D_freq, self.H_freq, self.W_freq = D_freq, H_freq, W_freq
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply 3D frequency domain filtering on temporal and spatial dimensions.
        
        Args:
            x: Input tensor [B, C, D, H, W]
            
        Returns:
            Filtered tensor [B, C, D, H, W]
        """
        B, C, D, H, W = x.shape
        identity = x  # For residual connection
        
        # Lazy initialization of complex_weight based on input spatial-temporal dimensions
        D_freq = D
        H_freq = H
        W_freq = W // 2 + 1  # rfftn only computes half of the last dimension
        
        if (self.D_freq != D_freq or 
            self.H_freq != H_freq or 
            self.W_freq != W_freq):
            self._init_weights(D_freq, H_freq, W_freq, x.device)
        
        # Save original dtype for later restoration
        original_dtype = x.dtype
        
        # Convert to float32 if necessary (FFT doesn't support bfloat16)
        if x.dtype == torch.bfloat16:
            x = x.float()
        
        # Step 1: Spatial-temporal domain → Frequency domain (rfftn on D, H, W dimensions)
        x_freq = torch.fft.rfftn(x, dim=(2, 3, 4), norm='ortho')
        
        # Step 2: Frequency domain filtering with FIXED weights
        # Convert stored real weights to complex format
        complex_weight = self.complex_weight
        if complex_weight.dtype == torch.bfloat16:
            complex_weight = complex_weight.float()
        weight = torch.view_as_complex(complex_weight)  # [C, D_freq, H_freq, W_freq]
        
        # Reshape weight for broadcasting: [C, D_freq, H_freq, W_freq] → [1, C, D_freq, H_freq, W_freq]
        weight = weight.view(1, C, self.D_freq, self.H_freq, self.W_freq)
        
        # Element-wise complex multiplication (frequency filtering)
        x_freq = x_freq * weight
        
        # Step 3: Frequency domain → Spatial-temporal domain (irfftn)
        x_filtered = torch.fft.irfftn(x_freq, s=(D, H, W), dim=(2, 3, 4), norm='ortho')
        
        # Convert back to original dtype if necessary
        if original_dtype == torch.bfloat16:
            x_filtered = x_filtered.to(torch.bfloat16)
        
        # Step 4: Dropout + Residual connection + Normalization
        x_filtered = self.dropout(x_filtered)
        x_out = self.norm(x_filtered + identity)
        
        return x_out


class Conv3dFFTBlock_HighFix(nn.Module):
    """
    A building block combining Conv3d with FFT-based frequency filtering (fixed weights).
    
    Structure:
        Conv3d → FilterLayer3D (FFT with fixed weights) → BatchNorm3d → ReLU → Optional MaxPool3d
    """
    
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 padding: int = 1,
                 use_maxpool: bool = False,
                 maxpool_kernel: int = 2,
                 dropout_prob: float = 0.1):
        """
        Args:
            in_channels: Input channels
            out_channels: Output channels
            kernel_size: Conv3d kernel size
            stride: Conv3d stride
            padding: Conv3d padding
            use_maxpool: Whether to apply MaxPool3d after activation
            maxpool_kernel: MaxPool3d kernel size
            dropout_prob: Dropout probability for FFT layer
        """
        super().__init__()
        
        self.conv = nn.Conv3d(in_channels, out_channels, 
                             kernel_size=kernel_size, 
                             stride=stride, 
                             padding=padding)
        
        # FFT-based frequency filtering with FIXED weights (high-pass initialized)
        self.fft_filter = FilterLayer3D_HighFix(
            num_channels=out_channels,
            dropout_prob=dropout_prob
        )
        
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.maxpool = nn.MaxPool3d(kernel_size=maxpool_kernel) if use_maxpool else None
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Conv3d → FFT Filter (fixed) → BN → ReLU → MaxPool.
        
        Args:
            x: Input tensor [B, C_in, D, H, W]
            
        Returns:
            Output tensor [B, C_out, D', H', W']
        """
        # Conv3d
        x = self.conv(x)
        
        # FFT-based frequency filtering with fixed weights
        x = self.fft_filter(x)
        
        # Activation
        x = self.relu(x)
        
        # Optional MaxPooling
        if self.maxpool is not None:
            x = self.maxpool(x)
            
        return x


class Resampler_3DFFT_HighFix(nn.Module):
    """
    3D FFT-Enhanced Resampler model with FIXED (non-trainable) 3D frequency domain filtering.
    
    This version uses FilterLayer3D with fixed weights after each Conv3d to enhance 
    spatial-temporal modeling through 3D frequency domain filtering using rfftn.
    """
    
    def __init__(self, 
                 in_channels: int = 16, 
                 target_seq_len: int = 100,
                 target_feat_dim: int = 1536,
                 latent_shape: Optional[Tuple[int, int, int]] = None,
                 fft_dropout: float = 0.5):
        """
        Args:
            in_channels: Input channels (from VAE encoder)
            target_seq_len: Target sequence length for output
            target_feat_dim: Target feature dimension for output
            fft_dropout: Dropout probability for FFT layers
        """
        super().__init__()
        self.target_seq_len = target_seq_len
        self.target_feat_dim = target_feat_dim
        self.in_channels = in_channels
        
        # Build CNN with FFT filtering (fixed weights)
        # Layer 1: Conv3d → FFT Filter (fixed) → BN → ReLU → MaxPool
        self.conv_block1 = Conv3dFFTBlock_HighFix(
            in_channels=in_channels,
            out_channels=512,
            kernel_size=7,
            stride=4,
            padding=3,
            use_maxpool=True,
            maxpool_kernel=2,
            dropout_prob=fft_dropout
        )
        
        # Layer 2: Conv3d → FFT Filter (fixed) → BN → ReLU → MaxPool
        self.conv_block2 = Conv3dFFTBlock_HighFix(
            in_channels=512,
            out_channels=768,
            kernel_size=3,
            stride=1,
            padding=1,
            use_maxpool=True,
            maxpool_kernel=2,
            dropout_prob=fft_dropout
        )
        
        self.cnn = nn.Sequential(
            self.conv_block1,
            self.conv_block2
            )
        
        # Initialize feature projection
        if latent_shape is None:
            latent_shape = (21, 60, 104)
        self._estimated_latent_shape = latent_shape
        self._init_feature_projection()

    def forward(self, x: torch.Tensor, shuffle: bool = True) -> torch.Tensor:
        """
        Process retrieved video latents and return reference tokens.
        
        Args:
            x: Input tensor with shape [B, N, C, D, H, W]
               where B=batch size, N=number of retrieved videos (e.g., 4)
            shuffle: Whether to shuffle tokens. If True, shuffles only during training.
                    During inference, tokens are kept in retrieval order. Default: True.
        
        Returns:
            Output tensor with shape [B, N, target_seq_len, target_feat_dim]
        """
        B, N, C, D, H, W = x.shape  # e.g., [1, 4, 16, 21, 60, 104]
        
        # Process each retrieved video separately
        x = x.reshape(B * N, C, D, H, W)  # [B*N, C, D, H, W]
        
        # Extract features via 3D CNN with FFT filtering (fixed weights)
        x = self.cnn(x)  # [B*N, 768, D', H', W']
        
        # Ensure CNN output matches expected shape
        if x.shape[2:] != self._cnn_output_shape:
            x = nn.functional.adaptive_avg_pool3d(x, self._cnn_output_shape)
        
        # Flatten spatial dimensions while preserving temporal dimension
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(B * N, self._cnn_output_shape[0], -1)
        
        # Adaptive pooling on temporal dimension to get target_seq_len tokens
        x = nn.functional.adaptive_avg_pool1d(x.transpose(1, 2), self.target_seq_len).transpose(1, 2)
        
        # Project to target feature dimension
        x = x.reshape(B * N * self.target_seq_len, -1)
        x = self.feature_proj(x)
        x = x.reshape(B, N, self.target_seq_len, self.target_feat_dim)  # x: [B, N, target_seq_len, target_feat_dim] e.g., [1, 4, 100, 1536]
        
        return x

    def _init_feature_projection(self):
        """Pre-compute CNN output shape and initialize projection layer."""
        with torch.no_grad():
            dummy = torch.zeros(
                (1, self.in_channels, *self._estimated_latent_shape),
                dtype=torch.float32,
            )
            current_mode = self.cnn.training
            try:
                self.cnn.eval()
                cnn_out = self.cnn(dummy)
            finally:
                self.cnn.train(current_mode)
        
        self._cnn_output_shape = cnn_out.shape[2:]  # (D', H', W')
        flattened_dim = cnn_out.shape[1] * cnn_out.shape[3] * cnn_out.shape[4]
        self.feature_proj = nn.Linear(flattened_dim, self.target_feat_dim)
        
        logger.info(f"Resampler_3DFFT_HighFix initialized: CNN output shape = {self._cnn_output_shape}, "
                   f"Flattened dim = {flattened_dim}")



print("================================================")
print("Resampler_3DFFT_HighFix initialized")
print("================================================")

class FilterLayer3D_LowFix(nn.Module):
    """
    3D Frequency Domain Filter Layer for video features with FIXED (non-trainable) LOW-PASS weights.
    
    Applies 3D FFT-based filtering on temporal and spatial dimensions of 3D convolutional features.
    Input: [B, C, D, H, W] where D is the temporal dimension, H and W are spatial dimensions
    Output: [B, C, D, H, W] (same shape)
    
    The filtering process:
    1. Spatial-temporal domain → Frequency domain (rfftn on dimensions D, H, W)
    2. Frequency filtering (FIXED complex weights, initialized with low-pass pattern)
    3. Frequency domain → Spatial-temporal domain (irfftn)
    4. Residual connection + BatchNorm
    
    Note: The complex_weight is NOT trainable and initialized using low-pass filter pattern.
    """
    
    def __init__(self, 
                 num_channels: int,
                 dropout_prob: float = 0.1,
                 max_freq: Optional[Tuple[int, int, int]] = None):
        """
        Args:
            num_channels: Number of channels (C) in the feature map
            dropout_prob: Dropout probability
            max_freq: Maximum frequency dimensions (D_freq, H_freq, W_freq)
        """
        super().__init__()
        self.num_channels = num_channels
        if max_freq is not None:
            if any(v <= 0 for v in max_freq):
                raise ValueError("max_freq entries must be positive")
        self.max_freq = max_freq
        
        # Initialize with minimal shape - will be resized on first forward pass
        init_shape = (
            num_channels,
            1 if max_freq is None else min(1, max_freq[0]),
            1 if max_freq is None else min(1, max_freq[1]),
            1 if max_freq is None else min(1, max_freq[2]),
            2,
        )
        
        # Register as buffer (non-trainable) instead of Parameter
        self.register_buffer('complex_weight', torch.zeros(init_shape, dtype=torch.float32))
        
        self.D_freq = init_shape[1]
        self.H_freq = init_shape[2]
        self.W_freq = init_shape[3]
        
        self.dropout = nn.Dropout(dropout_prob)
        self.norm = nn.BatchNorm3d(num_channels)

    def _init_weights(self, D_freq: int, H_freq: int, W_freq: int, device: torch.device):
        """Initialize the fixed frequency weights using low-pass filter pattern."""
        if self.max_freq is not None:
            max_d, max_h, max_w = self.max_freq
            if D_freq > max_d or H_freq > max_h or W_freq > max_w:
                raise ValueError(
                    f"FilterLayer3D requested ({D_freq}, {H_freq}, {W_freq}) but max_freq={self.max_freq}. "
                    "Increase max_freq when instantiating the layer."
                )

        with torch.no_grad():
            current = self.complex_weight.to(device=device)
            target_shape = (self.num_channels, D_freq, H_freq, W_freq, 2)
            
            if current.shape != target_shape:
                # Initialize with low-pass filter pattern
                freq_d = torch.linspace(0, 1, D_freq, dtype=torch.float32, device=device)
                freq_h = torch.linspace(0, 1, H_freq, dtype=torch.float32, device=device)
                freq_w = torch.linspace(0, 1, W_freq, dtype=torch.float32, device=device)
                
                # Create frequency grid
                grid_d, grid_h, grid_w = torch.meshgrid(freq_d, freq_h, freq_w, indexing='ij')
                
                # Calculate frequency magnitude
                freq_magnitude = torch.sqrt(grid_d**2 + grid_h**2 + grid_w**2)
                freq_magnitude = freq_magnitude / torch.sqrt(torch.tensor(3.0, device=device))
                
                # Low-pass filter curve: emphasizes low frequencies, attenuates high frequencies
                base_curve = 1.2 - 0.8 * torch.sqrt(freq_magnitude)
                
                # Per-channel variation
                channel_slopes = torch.linspace(0.7, 1.3, self.num_channels, dtype=torch.float32, device=device)
                channel_offsets = torch.linspace(-0.05, 0.05, self.num_channels, dtype=torch.float32, device=device)
                
                # Real part: modulated low-pass filter
                real_part = (
                    channel_slopes.view(-1, 1, 1, 1) * base_curve.unsqueeze(0)
                    + channel_offsets.view(-1, 1, 1, 1)
                ).clamp_(min=0.1, max=1.2)
                
                # Imaginary part: set to zero for phase-neutral filtering
                imag_part = torch.zeros(self.num_channels, D_freq, H_freq, W_freq, dtype=torch.float32, device=device)
                
                # Stack real and imaginary parts
                init_data = torch.stack([real_part, imag_part], dim=-1)
                
                # Add small random noise for diversity
                init_data.add_(torch.randn_like(init_data) * 0.05)
                
                # Clamp real part to ensure stable filtering
                init_data[..., 0].clamp_(min=0.05, max=1.5)
                
                # Update the buffer
                self.complex_weight.data = init_data
                self.D_freq, self.H_freq, self.W_freq = D_freq, H_freq, W_freq
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply 3D frequency domain filtering on temporal and spatial dimensions.
        
        Args:
            x: Input tensor [B, C, D, H, W]
            
        Returns:
            Filtered tensor [B, C, D, H, W]
        """
        B, C, D, H, W = x.shape
        identity = x  # For residual connection
        
        # Lazy initialization of complex_weight based on input spatial-temporal dimensions
        D_freq = D
        H_freq = H
        W_freq = W // 2 + 1  # rfftn only computes half of the last dimension
        
        if (self.D_freq != D_freq or 
            self.H_freq != H_freq or 
            self.W_freq != W_freq):
            self._init_weights(D_freq, H_freq, W_freq, x.device)
        
        # Save original dtype for later restoration
        original_dtype = x.dtype
        
        # Convert to float32 if necessary (FFT doesn't support bfloat16)
        if x.dtype == torch.bfloat16:
            x = x.float()
        
        # Step 1: Spatial-temporal domain → Frequency domain (rfftn on D, H, W dimensions)
        x_freq = torch.fft.rfftn(x, dim=(2, 3, 4), norm='ortho')
        
        # Step 2: Frequency domain filtering with FIXED weights
        # Convert stored real weights to complex format
        complex_weight = self.complex_weight
        if complex_weight.dtype == torch.bfloat16:
            complex_weight = complex_weight.float()
        weight = torch.view_as_complex(complex_weight)  # [C, D_freq, H_freq, W_freq]
        
        # Reshape weight for broadcasting: [C, D_freq, H_freq, W_freq] → [1, C, D_freq, H_freq, W_freq]
        weight = weight.view(1, C, self.D_freq, self.H_freq, self.W_freq)
        
        # Element-wise complex multiplication (frequency filtering)
        x_freq = x_freq * weight
        
        # Step 3: Frequency domain → Spatial-temporal domain (irfftn)
        x_filtered = torch.fft.irfftn(x_freq, s=(D, H, W), dim=(2, 3, 4), norm='ortho')
        
        # Convert back to original dtype if necessary
        if original_dtype == torch.bfloat16:
            x_filtered = x_filtered.to(torch.bfloat16)
        
        # Step 4: Dropout + Residual connection + Normalization
        x_filtered = self.dropout(x_filtered)
        x_out = self.norm(x_filtered + identity)
        
        return x_out


class Conv3dFFTBlock_LowFix(nn.Module):
    """
    A building block combining Conv3d with FFT-based frequency filtering (fixed weights).
    
    Structure:
        Conv3d → FilterLayer3D (FFT with fixed weights) → BatchNorm3d → ReLU → Optional MaxPool3d
    """
    
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 padding: int = 1,
                 use_maxpool: bool = False,
                 maxpool_kernel: int = 2,
                 dropout_prob: float = 0.1):
        """
        Args:
            in_channels: Input channels
            out_channels: Output channels
            kernel_size: Conv3d kernel size
            stride: Conv3d stride
            padding: Conv3d padding
            use_maxpool: Whether to apply MaxPool3d after activation
            maxpool_kernel: MaxPool3d kernel size
            dropout_prob: Dropout probability for FFT layer
        """
        super().__init__()
        
        self.conv = nn.Conv3d(in_channels, out_channels, 
                             kernel_size=kernel_size, 
                             stride=stride, 
                             padding=padding)
        
        # FFT-based frequency filtering with FIXED weights (low-pass initialized)
        self.fft_filter = FilterLayer3D_LowFix(
            num_channels=out_channels,
            dropout_prob=dropout_prob
        )
        
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.maxpool = nn.MaxPool3d(kernel_size=maxpool_kernel) if use_maxpool else None
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Conv3d → FFT Filter (fixed) → BN → ReLU → MaxPool.
        
        Args:
            x: Input tensor [B, C_in, D, H, W]
            
        Returns:
            Output tensor [B, C_out, D', H', W']
        """
        # Conv3d
        x = self.conv(x)
        
        # FFT-based frequency filtering with fixed weights
        x = self.fft_filter(x)
        
        # Activation
        x = self.relu(x)
        
        # Optional MaxPooling
        if self.maxpool is not None:
            x = self.maxpool(x)
            
        return x


class Resampler_3DFFT_LowFix(nn.Module):
    """
    3D FFT-Enhanced Resampler model with FIXED (non-trainable) 3D frequency domain filtering.
    
    This version uses FilterLayer3D with fixed weights after each Conv3d to enhance 
    spatial-temporal modeling through 3D frequency domain filtering using rfftn.
    """
    
    def __init__(self, 
                 in_channels: int = 16, 
                 target_seq_len: int = 100,
                 target_feat_dim: int = 1536,
                 latent_shape: Optional[Tuple[int, int, int]] = None,
                 fft_dropout: float = 0.5):
        """
        Args:
            in_channels: Input channels (from VAE encoder)
            target_seq_len: Target sequence length for output
            target_feat_dim: Target feature dimension for output
            fft_dropout: Dropout probability for FFT layers
        """
        super().__init__()
        self.target_seq_len = target_seq_len
        self.target_feat_dim = target_feat_dim
        self.in_channels = in_channels
        
        # Build CNN with FFT filtering (fixed weights)
        # Layer 1: Conv3d → FFT Filter (fixed) → BN → ReLU → MaxPool
        self.conv_block1 = Conv3dFFTBlock_LowFix(
            in_channels=in_channels,
            out_channels=512,
            kernel_size=7,
            stride=4,
            padding=3,
            use_maxpool=True,
            maxpool_kernel=2,
            dropout_prob=fft_dropout
        )
        
        # Layer 2: Conv3d → FFT Filter (fixed) → BN → ReLU → MaxPool
        self.conv_block2 = Conv3dFFTBlock_LowFix(  
            in_channels=512,
            out_channels=768,
            kernel_size=3,
            stride=1,
            padding=1,
            use_maxpool=True,
            maxpool_kernel=2,
            dropout_prob=fft_dropout
        )
        
        self.cnn = nn.Sequential(
            self.conv_block1,
            self.conv_block2
            )
        
        # Initialize feature projection
        if latent_shape is None:
            latent_shape = (21, 60, 104)
        self._estimated_latent_shape = latent_shape
        self._init_feature_projection()

    def forward(self, x: torch.Tensor, shuffle: bool = True) -> torch.Tensor:
        """
        Process retrieved video latents and return reference tokens.
        
        Args:
            x: Input tensor with shape [B, N, C, D, H, W]
               where B=batch size, N=number of retrieved videos (e.g., 4)
            shuffle: Whether to shuffle tokens. If True, shuffles only during training.
                    During inference, tokens are kept in retrieval order. Default: True.
        
        Returns:
            Output tensor with shape [B, N, target_seq_len, target_feat_dim]
        """
        B, N, C, D, H, W = x.shape  # e.g., [1, 4, 16, 21, 60, 104]
        
        # Process each retrieved video separately
        x = x.reshape(B * N, C, D, H, W)  # [B*N, C, D, H, W]
        
        # Extract features via 3D CNN with FFT filtering (fixed weights)
        x = self.cnn(x)  # [B*N, 768, D', H', W']
        
        # Ensure CNN output matches expected shape
        if x.shape[2:] != self._cnn_output_shape:
            x = nn.functional.adaptive_avg_pool3d(x, self._cnn_output_shape)
        
        # Flatten spatial dimensions while preserving temporal dimension
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(B * N, self._cnn_output_shape[0], -1)
        
        # Adaptive pooling on temporal dimension to get target_seq_len tokens
        x = nn.functional.adaptive_avg_pool1d(x.transpose(1, 2), self.target_seq_len).transpose(1, 2)
        
        # Project to target feature dimension
        x = x.reshape(B * N * self.target_seq_len, -1)
        x = self.feature_proj(x)
        x = x.reshape(B, N, self.target_seq_len, self.target_feat_dim)  # x: [B, N, target_seq_len, target_feat_dim] e.g., [1, 4, 100, 1536]
        
        return x

    def _init_feature_projection(self):
        """Pre-compute CNN output shape and initialize projection layer."""
        with torch.no_grad():
            dummy = torch.zeros(
                (1, self.in_channels, *self._estimated_latent_shape),
                dtype=torch.float32,
            )
            current_mode = self.cnn.training
            try:
                self.cnn.eval()
                cnn_out = self.cnn(dummy)
            finally:
                self.cnn.train(current_mode)
        
        self._cnn_output_shape = cnn_out.shape[2:]  # (D', H', W')
        flattened_dim = cnn_out.shape[1] * cnn_out.shape[3] * cnn_out.shape[4]
        self.feature_proj = nn.Linear(flattened_dim, self.target_feat_dim)
        
        logger.info(f"Resampler_3DFFT_LowFix initialized: CNN output shape = {self._cnn_output_shape}, "
                   f"Flattened dim = {flattened_dim}")

from einops import rearrange
import torch.nn.functional as F
try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

try:
    from sageattention import sageattn
    SAGE_ATTN_AVAILABLE = True
except ModuleNotFoundError:
    SAGE_ATTN_AVAILABLE = False
    
    
def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_heads: int, compatibility_mode=False):
    if compatibility_mode:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    elif FLASH_ATTN_3_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
        x = flash_attn_interface.flash_attn_func(q, k, v)
        if isinstance(x,tuple):
            x = x[0]
        x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
    elif FLASH_ATTN_2_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
        x = flash_attn.flash_attn_func(q, k, v)
        x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
    elif SAGE_ATTN_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = sageattn(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    else:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    return x

class AttentionModule(nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads
        
    def forward(self, q, k, v):
        x = flash_attention(q=q, k=k, v=v, num_heads=self.num_heads)
        return x

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        dtype = x.dtype
        return self.norm(x.float()).to(dtype) * self.weight

class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)
        
        self.attn = AttentionModule(self.num_heads)

    def forward(self, x):
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(x))
        v = self.v(x)
        x = self.attn(q, k, v)
        return self.o(x)

class DualResampler3DFFTAtten(nn.Module):
    """
    Dual Resampler combining HIGH-PASS and LOW-PASS 3D FFT filtering.
    
    This module processes the input through two parallel resamplers:
    1. High-pass resampler: captures high-frequency details and textures
    2. Low-pass resampler: captures low-frequency structures and smooth variations
    
    The outputs from both resamplers are concatenated along the sequence dimension,
    providing a comprehensive dual-frequency representation.
    
    Architecture:
        Input [B, N, C, D, H, W] (N = number of retrieved videos)
        ├─→ Resampler_3DFFT_HighFix → [B, N, target_seq_len, target_feat_dim]
        └─→ Resampler_3DFFT_LowFix  → [B, N, target_seq_len, target_feat_dim]
        → Optional shuffle along N dimension (training only, same permutation for both)
        → Reshape to [B, N*target_seq_len, target_feat_dim]
        → Concatenate → [B, N*2*target_seq_len, target_feat_dim]
    """
    
    def __init__(self, 
                 in_channels: int = 16, 
                 target_seq_len: int = 100,
                 target_feat_dim: int = 1536,
                 latent_shape: Optional[Tuple[int, int, int]] = None,
                 fft_dropout: float = 0.5):
        """
        Args:
            in_channels: Number of input channels from video encoder
            target_seq_len: Target sequence length per video (total output will be N*2*target_seq_len)
            target_feat_dim: Dimension of output embeddings
            latent_shape: Expected input shape (D, H, W) for initialization
            fft_dropout: Dropout probability for FFT layers
        """
        super().__init__()

        self.in_channels = in_channels
        self.target_seq_len = target_seq_len
        self.target_feat_dim = target_feat_dim

        self.latent_shape = latent_shape
        self.fft_dropout = fft_dropout
        # High-pass resampler
        self.high_pass_resampler = Resampler_3DFFT_HighFix(
            in_channels=in_channels,
            target_seq_len=target_seq_len,
            target_feat_dim=target_feat_dim,
            latent_shape=latent_shape,
            fft_dropout=fft_dropout
        )
        
        # Low-pass resampler
        self.low_pass_resampler = Resampler_3DFFT_LowFix(
            in_channels=in_channels,
            target_seq_len=target_seq_len,
            target_feat_dim=target_feat_dim,
            latent_shape=latent_shape,
            fft_dropout=fft_dropout
        )      
        # Self-attention
        self.self_attention = SelfAttention(target_feat_dim, num_heads=8)

    def forward(self, x: torch.Tensor, shuffle: bool = True) -> torch.Tensor:
        """
        Forward pass through dual resampler.
        
        Args:
            x: Input video features with shape [B, N, C, D, H, W]
               where B=batch size, N=number of retrieved videos (e.g., 4)
            shuffle: Whether to shuffle tokens. If True, shuffles only during training.
                    The same permutation is applied to both high and low tokens.
        
        Returns:
            Dual-frequency tokens [B, N*2*target_seq_len, target_feat_dim]
        """
        # Get high-pass and low-pass tokens: [B, N, target_seq_len, target_feat_dim]
        high_tokens = self.high_pass_resampler(x)
        low_tokens = self.low_pass_resampler(x)
        B, N, seq_len, feat_dim = high_tokens.shape
        # Shuffle along N dimension with the same permutation (only during training)
        if shuffle and self.training:
            # Generate random permutation for N dimension (e.g., [1,3,2,4] for N=4)
            perm_idx = torch.randperm(N, device=high_tokens.device)
            # Apply same permutation to both high and low tokens
            high_tokens = high_tokens[:, perm_idx, :, :]
            low_tokens = low_tokens[:, perm_idx, :, :]
        
        # Reshape to [B, N*target_seq_len, target_feat_dim] and concatenate
        high_tokens = high_tokens.reshape(B, N * seq_len, feat_dim)
        low_tokens = low_tokens.reshape(B, N * seq_len, feat_dim)
        # Apply self-attention to high and low tokens
        high_tokens = self.self_attention(high_tokens)
        low_tokens = self.self_attention(low_tokens)
        dual_tokens = torch.cat([high_tokens, low_tokens], dim=1)
        
        return dual_tokens
    


# Example usage and testing
if __name__ == "__main__":
    # Test DualResampler3DFFT
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = DualResampler3DFFTAtten(
        in_channels=16,
        target_seq_len=100,
        target_feat_dim=1536,
        latent_shape=(21, 60, 104),
        fft_dropout=0.5
    ).to(device)
    
    print("="*60)
    print("Test 1: 6D Input [B, N, C, D, H, W] - Multiple Retrieved Videos")
    print("="*60)
    
    # Test 6D input (multiple retrieved videos)
    batch_size = 1
    num_videos = 4  # Number of retrieved videos
    channels = 16
    temporal_dim = 21
    height = 60
    width = 104
    
    x_6d = torch.randn(batch_size, num_videos, channels, temporal_dim, height, width).to(device)
    
    print(f"Input shape: {x_6d.shape}")
    print(f"Expected: [1, 4, 16, 21, 60, 104]")
    
    # Forward pass
    output_6d = model(x_6d, shuffle=False)
    print(f"\nDual output shape: {output_6d.shape}")