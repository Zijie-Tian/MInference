# Copyright (c) 2024-2025 Microsoft
# Licensed under The MIT License [see LICENSE for details]

"""
CPU offload utilities for chunked prefill.

Provides KV cache management with CPU storage and on-demand GPU transfer.
"""

import torch
from typing import Optional, Tuple, Dict
from dataclasses import dataclass


@dataclass
class ChunkedKVCacheConfig:
    """Configuration for chunked KV cache."""
    num_layers: int
    num_heads: int
    num_kv_heads: int
    seq_len: int
    head_dim: int
    dtype: torch.dtype = torch.float16
    use_pinned_memory: bool = True


class ChunkedKVCache:
    """
    KV cache manager with CPU storage and chunked GPU transfer.

    Stores K, V tensors on CPU and provides methods to:
    - Store K, V from GPU to CPU
    - Load K, V chunks from CPU to GPU
    - Gather specific indices from CPU to GPU

    Example:
        cache = ChunkedKVCache(config)

        # Store K, V for a layer
        cache.store(layer_idx, k_gpu, v_gpu)

        # Load a chunk
        k_chunk, v_chunk = cache.load_chunk(layer_idx, head_idx, start=0, end=65536)

        # Gather specific indices
        k_gathered, v_gathered = cache.gather(layer_idx, head_idx, indices)
    """

    def __init__(self, config: ChunkedKVCacheConfig):
        self.config = config
        self.num_layers = config.num_layers
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.seq_len = config.seq_len
        self.head_dim = config.head_dim
        self.dtype = config.dtype
        self.heads_per_kv = config.num_heads // config.num_kv_heads

        # Allocate CPU storage
        # Shape: [num_layers, num_kv_heads, seq_len, head_dim]
        if config.use_pinned_memory:
            self.k_cache = torch.empty(
                (config.num_layers, config.num_kv_heads, config.seq_len, config.head_dim),
                dtype=config.dtype,
                device='cpu',
                pin_memory=True,
            )
            self.v_cache = torch.empty(
                (config.num_layers, config.num_kv_heads, config.seq_len, config.head_dim),
                dtype=config.dtype,
                device='cpu',
                pin_memory=True,
            )
        else:
            self.k_cache = torch.empty(
                (config.num_layers, config.num_kv_heads, config.seq_len, config.head_dim),
                dtype=config.dtype,
                device='cpu',
            )
            self.v_cache = torch.empty(
                (config.num_layers, config.num_kv_heads, config.seq_len, config.head_dim),
                dtype=config.dtype,
                device='cpu',
            )

        # Track which layers have been stored
        self.stored_layers = set()

    def store(
        self,
        layer_idx: int,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> None:
        """
        Store K, V tensors from GPU to CPU cache.

        Args:
            layer_idx: Layer index
            k: Key tensor [batch, num_kv_heads, seq_len, head_dim]
            v: Value tensor [batch, num_kv_heads, seq_len, head_dim]
        """
        # Move to CPU (non-blocking if pinned memory)
        self.k_cache[layer_idx] = k.squeeze(0).cpu()
        self.v_cache[layer_idx] = v.squeeze(0).cpu()
        self.stored_layers.add(layer_idx)

    def store_chunk(
        self,
        layer_idx: int,
        k_chunk: torch.Tensor,
        v_chunk: torch.Tensor,
        start: int,
        end: int,
    ) -> None:
        """
        Store a chunk of K, V tensors.

        Args:
            layer_idx: Layer index
            k_chunk: Key chunk [batch, num_kv_heads, chunk_len, head_dim]
            v_chunk: Value chunk
            start: Start position in sequence
            end: End position in sequence
        """
        self.k_cache[layer_idx, :, start:end, :] = k_chunk.squeeze(0).cpu()
        self.v_cache[layer_idx, :, start:end, :] = v_chunk.squeeze(0).cpu()
        self.stored_layers.add(layer_idx)

    def load_chunk(
        self,
        layer_idx: int,
        kv_head_idx: int,
        start: int,
        end: int,
        device: str = 'cuda',
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load a chunk of K, V from CPU to GPU.

        Args:
            layer_idx: Layer index
            kv_head_idx: KV head index
            start: Start position
            end: End position
            device: Target device

        Returns:
            k_chunk: [1, 1, chunk_len, head_dim]
            v_chunk: [1, 1, chunk_len, head_dim]
        """
        k_chunk = self.k_cache[layer_idx, kv_head_idx, start:end, :].to(device, non_blocking=True)
        v_chunk = self.v_cache[layer_idx, kv_head_idx, start:end, :].to(device, non_blocking=True)
        return k_chunk.unsqueeze(0).unsqueeze(0), v_chunk.unsqueeze(0).unsqueeze(0)

    def gather(
        self,
        layer_idx: int,
        kv_head_idx: int,
        indices: torch.Tensor,
        device: str = 'cuda',
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Gather K, V at specific indices from CPU to GPU.

        Args:
            layer_idx: Layer index
            kv_head_idx: KV head index
            indices: Position indices [num_indices] (on CPU or GPU)
            device: Target device

        Returns:
            k_gathered: [1, 1, num_indices, head_dim]
            v_gathered: [1, 1, num_indices, head_dim]
        """
        # Ensure indices are on CPU for indexing
        if indices.device.type != 'cpu':
            indices = indices.cpu()

        k_gathered = self.k_cache[layer_idx, kv_head_idx, indices, :].to(device, non_blocking=True)
        v_gathered = self.v_cache[layer_idx, kv_head_idx, indices, :].to(device, non_blocking=True)
        return k_gathered.unsqueeze(0).unsqueeze(0), v_gathered.unsqueeze(0).unsqueeze(0)

    def get_full_kv(
        self,
        layer_idx: int,
        kv_head_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get full K, V for a layer/head (stays on CPU).

        Returns:
            k: [1, 1, seq_len, head_dim] on CPU
            v: [1, 1, seq_len, head_dim] on CPU
        """
        k = self.k_cache[layer_idx, kv_head_idx, :, :].unsqueeze(0).unsqueeze(0)
        v = self.v_cache[layer_idx, kv_head_idx, :, :].unsqueeze(0).unsqueeze(0)
        return k, v

    def clear(self) -> None:
        """Clear the cache."""
        self.k_cache.zero_()
        self.v_cache.zero_()
        self.stored_layers.clear()

    def memory_usage_gb(self) -> float:
        """Return memory usage in GB."""
        bytes_per_element = 2 if self.dtype == torch.float16 else 4
        total_elements = 2 * self.num_layers * self.num_kv_heads * self.seq_len * self.head_dim
        return total_elements * bytes_per_element / (1024 ** 3)


class HiddenStatesCache:
    """
    Cache for hidden states with CPU storage.

    Used to store intermediate hidden states between layers
    when GPU memory is limited.
    """

    def __init__(
        self,
        seq_len: int,
        hidden_dim: int,
        dtype: torch.dtype = torch.float16,
        use_pinned_memory: bool = True,
    ):
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.dtype = dtype

        if use_pinned_memory:
            self.cache = torch.empty(
                (1, seq_len, hidden_dim),
                dtype=dtype,
                device='cpu',
                pin_memory=True,
            )
        else:
            self.cache = torch.empty(
                (1, seq_len, hidden_dim),
                dtype=dtype,
                device='cpu',
            )

    def store(self, hidden_states: torch.Tensor) -> None:
        """Store hidden states from GPU to CPU."""
        self.cache.copy_(hidden_states.cpu())

    def store_chunk(
        self,
        chunk: torch.Tensor,
        start: int,
        end: int,
    ) -> None:
        """Store a chunk of hidden states."""
        self.cache[:, start:end, :] = chunk.cpu()

    def load_chunk(
        self,
        start: int,
        end: int,
        device: str = 'cuda',
    ) -> torch.Tensor:
        """Load a chunk of hidden states to GPU."""
        return self.cache[:, start:end, :].to(device, non_blocking=True)

    def get_full(self) -> torch.Tensor:
        """Get full hidden states (on CPU)."""
        return self.cache

    def memory_usage_gb(self) -> float:
        """Return memory usage in GB."""
        bytes_per_element = 2 if self.dtype == torch.float16 else 4
        total_elements = self.seq_len * self.hidden_dim
        return total_elements * bytes_per_element / (1024 ** 3)
