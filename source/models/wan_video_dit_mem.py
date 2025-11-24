import torch
import torch.nn as nn
from typing import Tuple, Optional

__all__ = [
    'FilterLayer3D_HighFix',
    'FilterLayer3D_LowFix',
    'Conv3dFFTBlock_HighFix',
    'Conv3dFFTBlock_LowFix',
    'DiTMem_HighFix',
    'DiTMem_LowFix',
    'DiTMem',
]


class FilterLayer3D_HighFix(nn.Module):
    """
    3D frequency domain filter layer with fixed non-trainable weights.
    
    Applies FFT-based filtering on spatiotemporal dimensions using high-pass initialization.
    Transforms input [B, C, D, H, W] through frequency domain and returns [B, C, D, H, W].
    """
    
    def __init__(self, 
                 num_channels: int,
                 dropout_prob: float = 0.1,
                 max_freq: Optional[Tuple[int, int, int]] = None):
        """
        Args:
            num_channels: Number of channels
            dropout_prob: Dropout probability
            max_freq: Maximum frequency dimensions
        """
        super().__init__()
        self.num_channels = num_channels
        if max_freq is not None:
            if any(v <= 0 for v in max_freq):
                raise ValueError("max_freq entries must be positive")
        self.max_freq = max_freq
        
        init_shape = (
            num_channels,
            1 if max_freq is None else min(1, max_freq[0]),
            1 if max_freq is None else min(1, max_freq[1]),
            1 if max_freq is None else min(1, max_freq[2]),
            2,
        )
        
        self.register_buffer('complex_weight', torch.zeros(init_shape, dtype=torch.float32))
        
        self.D_freq = init_shape[1]
        self.H_freq = init_shape[2]
        self.W_freq = init_shape[3]
        
        self.dropout = nn.Dropout(dropout_prob)
        self.norm = nn.BatchNorm3d(num_channels)

    def _init_weights(self, D_freq: int, H_freq: int, W_freq: int, device: torch.device):
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
                freq_d = torch.linspace(0, 1, D_freq, dtype=torch.float32, device=device)
                freq_h = torch.linspace(0, 1, H_freq, dtype=torch.float32, device=device)
                freq_w = torch.linspace(0, 1, W_freq, dtype=torch.float32, device=device)
                
                grid_d, grid_h, grid_w = torch.meshgrid(freq_d, freq_h, freq_w, indexing='ij')
                freq_magnitude = torch.sqrt(grid_d**2 + grid_h**2 + grid_w**2)
                freq_magnitude = freq_magnitude / torch.sqrt(torch.tensor(3.0, device=device))
                
                base_curve = 0.2 + 0.8 * torch.sqrt(freq_magnitude)
                channel_slopes = torch.linspace(0.7, 1.3, self.num_channels, dtype=torch.float32, device=device)
                channel_offsets = torch.linspace(-0.05, 0.05, self.num_channels, dtype=torch.float32, device=device)
                
                real_part = (
                    channel_slopes.view(-1, 1, 1, 1) * base_curve.unsqueeze(0)
                    + channel_offsets.view(-1, 1, 1, 1)
                ).clamp_(min=0.2, max=1.2)
                
                imag_part = torch.zeros(self.num_channels, D_freq, H_freq, W_freq, dtype=torch.float32, device=device)
                init_data = torch.stack([real_part, imag_part], dim=-1)
                init_data.add_(torch.randn_like(init_data) * 0.05)
                init_data[..., 0].clamp_(min=0.05, max=1.2)
                
                self.complex_weight.data = init_data
                self.D_freq, self.H_freq, self.W_freq = D_freq, H_freq, W_freq
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, C, D, H, W]
        Returns:
            Filtered tensor [B, C, D, H, W]
        """
        B, C, D, H, W = x.shape
        identity = x
        
        D_freq = D
        H_freq = H
        W_freq = W // 2 + 1
        
        if (self.D_freq != D_freq or 
            self.H_freq != H_freq or 
            self.W_freq != W_freq):
            self._init_weights(D_freq, H_freq, W_freq, x.device)
        
        original_dtype = x.dtype
        if x.dtype == torch.bfloat16:
            x = x.float()
        
        x_freq = torch.fft.rfftn(x, dim=(2, 3, 4), norm='ortho')
        
        complex_weight = self.complex_weight
        if complex_weight.dtype == torch.bfloat16:
            complex_weight = complex_weight.float()
        weight = torch.view_as_complex(complex_weight)
        weight = weight.view(1, C, self.D_freq, self.H_freq, self.W_freq)
        
        x_freq = x_freq * weight
        x_filtered = torch.fft.irfftn(x_freq, s=(D, H, W), dim=(2, 3, 4), norm='ortho')
        
        if original_dtype == torch.bfloat16:
            x_filtered = x_filtered.to(torch.bfloat16)
        
        x_filtered = self.dropout(x_filtered)
        x_out = self.norm(x_filtered + identity)
        
        return x_out


class Conv3dFFTBlock_HighFix(nn.Module):
    """
    Convolutional block with 3D FFT-based frequency filtering.
    
    Architecture: Conv3d → FilterLayer3D → BatchNorm3d → ReLU → MaxPool3d (optional).
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
        
        self.fft_filter = FilterLayer3D_HighFix(
            num_channels=out_channels,
            dropout_prob=dropout_prob
        )
        
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.maxpool = nn.MaxPool3d(kernel_size=maxpool_kernel) if use_maxpool else None
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, C_in, D, H, W]
        Returns:
            Output tensor [B, C_out, D', H', W']
        """
        x = self.conv(x)
        x = self.fft_filter(x)
        x = self.relu(x)
        if self.maxpool is not None:
            x = self.maxpool(x)
            
        return x


class DiTMem_HighFix(nn.Module):
    """
    3D FFT-enhanced model with fixed frequency domain filtering.
    
    Processes video latents through convolutional layers with high-pass frequency filtering
    to extract spatiotemporal features for memory retrieval.
    """
    
    def __init__(self, 
                 in_channels: int = 16, 
                 target_seq_len: int = 100,
                 target_feat_dim: int = 1536,
                 latent_shape: Optional[Tuple[int, int, int]] = None,
                 fft_dropout: float = 0.5):
        """
        Args:
            in_channels: Input channels
            target_seq_len: Target sequence length
            target_feat_dim: Target feature dimension
            fft_dropout: Dropout probability for FFT layers
        """
        super().__init__()
        self.target_seq_len = target_seq_len
        self.target_feat_dim = target_feat_dim
        self.in_channels = in_channels
        
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
        
        if latent_shape is None:
            latent_shape = (21, 60, 104)
        self._estimated_latent_shape = latent_shape
        self._init_feature_projection()

    def forward(self, x: torch.Tensor, shuffle: bool = True) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, N, C, D, H, W] where N is number of retrieved videos
            shuffle: Whether to shuffle tokens during training
        Returns:
            Output tensor [B, N, target_seq_len, target_feat_dim]
        """
        B, N, C, D, H, W = x.shape
        
        x = x.reshape(B * N, C, D, H, W)
        x = self.cnn(x)
        
        if x.shape[2:] != self._cnn_output_shape:
            x = nn.functional.adaptive_avg_pool3d(x, self._cnn_output_shape)
        
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(B * N, self._cnn_output_shape[0], -1)
        x = nn.functional.adaptive_avg_pool1d(x.transpose(1, 2), self.target_seq_len).transpose(1, 2)
        
        x = x.reshape(B * N * self.target_seq_len, -1)
        x = self.feature_proj(x)
        x = x.reshape(B, N, self.target_seq_len, self.target_feat_dim)
        
        return x

    def _init_feature_projection(self):
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
        
        self._cnn_output_shape = cnn_out.shape[2:]
        flattened_dim = cnn_out.shape[1] * cnn_out.shape[3] * cnn_out.shape[4]
        self.feature_proj = nn.Linear(flattened_dim, self.target_feat_dim)

class FilterLayer3D_LowFix(nn.Module):
    """
    3D frequency domain filter layer with fixed non-trainable weights.
    
    Applies FFT-based filtering on spatiotemporal dimensions using low-pass initialization.
    Transforms input [B, C, D, H, W] through frequency domain and returns [B, C, D, H, W].
    """
    
    def __init__(self, 
                 num_channels: int,
                 dropout_prob: float = 0.1,
                 max_freq: Optional[Tuple[int, int, int]] = None):
        """
        Args:
            num_channels: Number of channels
            dropout_prob: Dropout probability
            max_freq: Maximum frequency dimensions
        """
        super().__init__()
        self.num_channels = num_channels
        if max_freq is not None:
            if any(v <= 0 for v in max_freq):
                raise ValueError("max_freq entries must be positive")
        self.max_freq = max_freq
        
        init_shape = (
            num_channels,
            1 if max_freq is None else min(1, max_freq[0]),
            1 if max_freq is None else min(1, max_freq[1]),
            1 if max_freq is None else min(1, max_freq[2]),
            2,
        )
        
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
                freq_d = torch.linspace(0, 1, D_freq, dtype=torch.float32, device=device)
                freq_h = torch.linspace(0, 1, H_freq, dtype=torch.float32, device=device)
                freq_w = torch.linspace(0, 1, W_freq, dtype=torch.float32, device=device)
                
                grid_d, grid_h, grid_w = torch.meshgrid(freq_d, freq_h, freq_w, indexing='ij')
                freq_magnitude = torch.sqrt(grid_d**2 + grid_h**2 + grid_w**2)
                freq_magnitude = freq_magnitude / torch.sqrt(torch.tensor(3.0, device=device))
                
                base_curve = 1.2 - 0.8 * torch.sqrt(freq_magnitude)
                channel_slopes = torch.linspace(0.7, 1.3, self.num_channels, dtype=torch.float32, device=device)
                channel_offsets = torch.linspace(-0.05, 0.05, self.num_channels, dtype=torch.float32, device=device)
                
                real_part = (
                    channel_slopes.view(-1, 1, 1, 1) * base_curve.unsqueeze(0)
                    + channel_offsets.view(-1, 1, 1, 1)
                ).clamp_(min=0.2, max=1.2)
                
                imag_part = torch.zeros(self.num_channels, D_freq, H_freq, W_freq, dtype=torch.float32, device=device)
                init_data = torch.stack([real_part, imag_part], dim=-1)
                init_data.add_(torch.randn_like(init_data) * 0.05)
                init_data[..., 0].clamp_(min=0.05, max=1.2)
                
                self.complex_weight.data = init_data
                self.D_freq, self.H_freq, self.W_freq = D_freq, H_freq, W_freq
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, C, D, H, W]
        Returns:
            Filtered tensor [B, C, D, H, W]
        """
        B, C, D, H, W = x.shape
        identity = x
        
        D_freq = D
        H_freq = H
        W_freq = W // 2 + 1
        
        if (self.D_freq != D_freq or 
            self.H_freq != H_freq or 
            self.W_freq != W_freq):
            self._init_weights(D_freq, H_freq, W_freq, x.device)
        
        original_dtype = x.dtype
        if x.dtype == torch.bfloat16:
            x = x.float()
        
        x_freq = torch.fft.rfftn(x, dim=(2, 3, 4), norm='ortho')
        
        complex_weight = self.complex_weight
        if complex_weight.dtype == torch.bfloat16:
            complex_weight = complex_weight.float()
        weight = torch.view_as_complex(complex_weight)
        weight = weight.view(1, C, self.D_freq, self.H_freq, self.W_freq)
        
        x_freq = x_freq * weight
        x_filtered = torch.fft.irfftn(x_freq, s=(D, H, W), dim=(2, 3, 4), norm='ortho')
        
        if original_dtype == torch.bfloat16:
            x_filtered = x_filtered.to(torch.bfloat16)
        
        x_filtered = self.dropout(x_filtered)
        x_out = self.norm(x_filtered + identity)
        
        return x_out


class Conv3dFFTBlock_LowFix(nn.Module):
    """
    Convolutional block with 3D FFT-based frequency filtering.
    
    Architecture: Conv3d → FilterLayer3D → BatchNorm3d → ReLU → MaxPool3d (optional).
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
        
        self.fft_filter = FilterLayer3D_LowFix(
            num_channels=out_channels,
            dropout_prob=dropout_prob
        )
        
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.maxpool = nn.MaxPool3d(kernel_size=maxpool_kernel) if use_maxpool else None
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, C_in, D, H, W]
        Returns:
            Output tensor [B, C_out, D', H', W']
        """
        x = self.conv(x)
        x = self.fft_filter(x)
        x = self.relu(x)
        if self.maxpool is not None:
            x = self.maxpool(x)
            
        return x


class DiTMem_LowFix(nn.Module):
    """
    3D FFT-enhanced model with fixed frequency domain filtering.
    
    Processes video latents through convolutional layers with low-pass frequency filtering
    to extract spatiotemporal features for memory retrieval.
    """
    
    def __init__(self, 
                 in_channels: int = 16, 
                 target_seq_len: int = 100,
                 target_feat_dim: int = 1536,
                 latent_shape: Optional[Tuple[int, int, int]] = None,
                 fft_dropout: float = 0.5):
        """
        Args:
            in_channels: Input channels
            target_seq_len: Target sequence length
            target_feat_dim: Target feature dimension
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
        
        if latent_shape is None:
            latent_shape = (21, 60, 104)
        self._estimated_latent_shape = latent_shape
        self._init_feature_projection()

    def forward(self, x: torch.Tensor, shuffle: bool = True) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, N, C, D, H, W] where N is number of retrieved videos
            shuffle: Whether to shuffle tokens during training
        Returns:
            Output tensor [B, N, target_seq_len, target_feat_dim]
        """
        B, N, C, D, H, W = x.shape
        
        x = x.reshape(B * N, C, D, H, W)
        x = self.cnn(x)
        
        if x.shape[2:] != self._cnn_output_shape:
            x = nn.functional.adaptive_avg_pool3d(x, self._cnn_output_shape)
        
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(B * N, self._cnn_output_shape[0], -1)
        x = nn.functional.adaptive_avg_pool1d(x.transpose(1, 2), self.target_seq_len).transpose(1, 2)
        
        x = x.reshape(B * N * self.target_seq_len, -1)
        x = self.feature_proj(x)
        x = x.reshape(B, N, self.target_seq_len, self.target_feat_dim)
        
        return x

    def _init_feature_projection(self):
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
        
        self._cnn_output_shape = cnn_out.shape[2:]
        flattened_dim = cnn_out.shape[1] * cnn_out.shape[3] * cnn_out.shape[4]
        self.feature_proj = nn.Linear(flattened_dim, self.target_feat_dim)

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

class DiTMem(nn.Module):
    """
    Dual-frequency memory module for video retrieval.
    
    Processes input through parallel high-pass and low-pass frequency filtering paths,
    then concatenates outputs to form dual-frequency token representations.
    """
    
    def __init__(self, 
                 in_channels: int = 16, 
                 target_seq_len: int = 100,
                 target_feat_dim: int = 1536,
                 latent_shape: Optional[Tuple[int, int, int]] = None,
                 fft_dropout: float = 0.5):
        """
        Args:
            in_channels: Number of input channels
            target_seq_len: Target sequence length per video
            target_feat_dim: Dimension of output embeddings
            latent_shape: Expected input shape (D, H, W)
            fft_dropout: Dropout probability for FFT layers
        """
        super().__init__()

        self.in_channels = in_channels
        self.target_seq_len = target_seq_len
        self.target_feat_dim = target_feat_dim

        self.latent_shape = latent_shape
        self.fft_dropout = fft_dropout
        self.high_pass_module = DiTMem_HighFix(
            in_channels=in_channels,
            target_seq_len=target_seq_len,
            target_feat_dim=target_feat_dim,
            latent_shape=latent_shape,
            fft_dropout=fft_dropout
        )
        
        self.low_pass_module = DiTMem_LowFix(
            in_channels=in_channels,
            target_seq_len=target_seq_len,
            target_feat_dim=target_feat_dim,
            latent_shape=latent_shape,
            fft_dropout=fft_dropout
        )      
        self.self_attention = SelfAttention(target_feat_dim, num_heads=8)

    def forward(self, x: torch.Tensor, shuffle: bool = True) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, N, C, D, H, W] where N is number of retrieved videos
            shuffle: Whether to shuffle tokens during training
        Returns:
            Dual-frequency tokens [B, N*2*target_seq_len, target_feat_dim]
        """
        high_tokens = self.high_pass_module(x)
        low_tokens = self.low_pass_module(x)
        B, N, seq_len, feat_dim = high_tokens.shape
        
        if shuffle and self.training:
            perm_idx = torch.randperm(N, device=high_tokens.device)
            high_tokens = high_tokens[:, perm_idx, :, :]
            low_tokens = low_tokens[:, perm_idx, :, :]
        
        high_tokens = high_tokens.reshape(B, N * seq_len, feat_dim)
        low_tokens = low_tokens.reshape(B, N * seq_len, feat_dim)
        high_tokens = self.self_attention(high_tokens)
        low_tokens = self.self_attention(low_tokens)
        dual_tokens = torch.cat([high_tokens, low_tokens], dim=1)
        
        return dual_tokens
    

