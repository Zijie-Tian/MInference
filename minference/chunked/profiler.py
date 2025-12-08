# Copyright (c) 2024-2025 Microsoft
# Licensed under The MIT License [see LICENSE for details]

"""
Chunked MInference profiler for sparse attention pattern analysis.

This module provides the main profiler class for analyzing sparse attention
patterns on extremely long sequences (1M+ tokens) with limited GPU memory.
"""

import os
import json
import time
import math
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from tqdm import tqdm

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from .cpu_offload import ChunkedKVCache, ChunkedKVCacheConfig, HiddenStatesCache
from .attention import (
    chunked_probe_attention,
    chunked_sparse_attention,
    minference_sparse_attention,
    chunked_minference_sparse_attention,
)
from .pattern_discovery import extract_patterns, compute_sparsity_ratio

# Import MInference's sparse attention function directly
from minference.ops.pit_sparse_flash_attention_v2 import vertical_slash_sparse_attention


@dataclass
class ProfilerConfig:
    """Configuration for chunked profiler."""
    model_path: str
    seq_len: int = 1_000_000
    vertical_size: int = 1000
    slash_size: int = 6096
    q_chunk_size: int = 512
    kv_chunk_size: int = 65536
    embedding_chunk_size: int = 8192
    save_dir: str = "results/profile"
    compute_output: bool = True  # Whether to compute sparse attention output
    save_outputs: bool = False   # Whether to save attention outputs
    dtype: torch.dtype = torch.float16
    device: str = "cuda"


class ChunkedMInferenceProfiler:
    """
    Profiler for MInference sparse attention patterns on long sequences.

    This profiler can analyze sparse attention patterns on 1M+ token sequences
    with limited GPU memory (24GB) by:
    1. Using CPU offloading for K, V tensors
    2. Chunked probe attention for pattern discovery
    3. Online softmax for chunked sparse attention computation

    Example:
        config = ProfilerConfig(
            model_path="/path/to/model",
            seq_len=1_000_000,
        )
        profiler = ChunkedMInferenceProfiler(config)
        results = profiler.run()
    """

    def __init__(self, config: ProfilerConfig):
        self.config = config
        self.model_path = config.model_path
        self.seq_len = config.seq_len
        self.device = config.device

        # Load model config
        print(f"Loading model config from {config.model_path}...")
        self.model_config = AutoConfig.from_pretrained(config.model_path)

        self.num_layers = self.model_config.num_hidden_layers
        self.num_heads = self.model_config.num_attention_heads
        self.num_kv_heads = getattr(self.model_config, 'num_key_value_heads', self.num_heads)
        self.head_dim = self.model_config.hidden_size // self.num_heads
        self.hidden_dim = self.model_config.hidden_size
        self.heads_per_kv = self.num_heads // self.num_kv_heads

        print(f"Model: {self.num_layers} layers, {self.num_heads} heads, "
              f"{self.num_kv_heads} KV heads, {self.head_dim} head_dim")

        # Results storage
        self.patterns: Dict[int, Dict[int, Tuple[torch.Tensor, torch.Tensor]]] = {}
        self.profile_data: Dict[str, Any] = {}

        # Model and tokenizer (lazy loading)
        self._model = None
        self._tokenizer = None

    @property
    def model(self):
        """Lazy load model."""
        if self._model is None:
            print(f"Loading model weights...")
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=self.config.dtype,
                device_map="auto",
                trust_remote_code=True,
            )
            self._model.eval()
        return self._model

    @property
    def tokenizer(self):
        """Lazy load tokenizer."""
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True,
            )
        return self._tokenizer

    def _get_attention_layer(self, layer_idx: int):
        """Get the attention module for a specific layer."""
        # This assumes Llama-style architecture
        # Adjust for other architectures as needed
        if hasattr(self.model, 'model'):
            # Llama, Mistral, etc.
            return self.model.model.layers[layer_idx].self_attn
        elif hasattr(self.model, 'transformer'):
            # GPT-2, GPT-Neo, etc.
            return self.model.transformer.h[layer_idx].attn
        else:
            raise ValueError(f"Unknown model architecture: {type(self.model)}")

    def _get_layer_norm(self, layer_idx: int, which: str = 'input'):
        """Get layer norm for a specific layer."""
        if hasattr(self.model, 'model'):
            layer = self.model.model.layers[layer_idx]
            if which == 'input':
                return layer.input_layernorm
            else:
                return layer.post_attention_layernorm
        else:
            raise ValueError(f"Unknown model architecture")

    def _get_mlp(self, layer_idx: int):
        """Get MLP module for a specific layer."""
        if hasattr(self.model, 'model'):
            return self.model.model.layers[layer_idx].mlp
        else:
            raise ValueError(f"Unknown model architecture")

    def _get_decoder_layer(self, layer_idx: int):
        """Get the full decoder layer module."""
        if hasattr(self.model, 'model'):
            return self.model.model.layers[layer_idx]
        else:
            raise ValueError(f"Unknown model architecture")

    def _generate_input_tokens(self) -> torch.Tensor:
        """Generate random input tokens for profiling."""
        vocab_size = self.model_config.vocab_size
        # Generate random tokens, avoiding special tokens at the start
        input_ids = torch.randint(100, vocab_size - 100, (1, self.seq_len), dtype=torch.long)
        return input_ids

    def _chunked_embedding(
        self,
        input_ids: torch.Tensor,
        hidden_cache: HiddenStatesCache,
    ) -> None:
        """
        Compute embeddings in chunks and store to CPU.

        Args:
            input_ids: Input token IDs [1, seq_len]
            hidden_cache: Cache to store hidden states
        """
        print("Computing embeddings...")
        chunk_size = self.config.embedding_chunk_size

        # Get embedding layer
        if hasattr(self.model, 'model'):
            embed_tokens = self.model.model.embed_tokens
        else:
            embed_tokens = self.model.transformer.wte

        for start in tqdm(range(0, self.seq_len, chunk_size), desc="Embedding"):
            end = min(start + chunk_size, self.seq_len)
            chunk_ids = input_ids[:, start:end].to(self.device)

            with torch.no_grad():
                chunk_embeds = embed_tokens(chunk_ids)

            hidden_cache.store_chunk(chunk_embeds.cpu(), start, end)

            del chunk_ids, chunk_embeds
            torch.cuda.empty_cache()

    def _chunked_qkv_projection(
        self,
        layer_idx: int,
        hidden_cache: HiddenStatesCache,
        kv_cache: ChunkedKVCache,
    ) -> torch.Tensor:
        """
        Compute Q, K, V projections in chunks.

        K, V are stored to CPU cache.
        Q is returned on CPU (will be loaded to GPU in chunks during attention).

        Args:
            layer_idx: Layer index
            hidden_cache: Hidden states cache (on CPU)
            kv_cache: KV cache for storing K, V

        Returns:
            q: Query tensor [1, num_heads, seq_len, head_dim] on CPU
        """
        chunk_size = self.config.embedding_chunk_size
        attn = self._get_attention_layer(layer_idx)
        ln = self._get_layer_norm(layer_idx, 'input')

        # Allocate Q on CPU
        q_cpu = torch.empty(
            1, self.num_heads, self.seq_len, self.head_dim,
            dtype=self.config.dtype, device='cpu'
        )

        for start in range(0, self.seq_len, chunk_size):
            end = min(start + chunk_size, self.seq_len)
            chunk_len = end - start

            # Load hidden states chunk
            hidden_chunk = hidden_cache.load_chunk(start, end, self.device)

            with torch.no_grad():
                # Apply layer norm
                normed = ln(hidden_chunk)

                # Compute Q, K, V
                q = attn.q_proj(normed)
                k = attn.k_proj(normed)
                v = attn.v_proj(normed)

                # Reshape
                q = q.view(1, chunk_len, self.num_heads, self.head_dim).transpose(1, 2)
                k = k.view(1, chunk_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
                v = v.view(1, chunk_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

                # Apply rotary embeddings if available
                if hasattr(attn, 'rotary_emb'):
                    # Create position IDs for this chunk
                    position_ids = torch.arange(start, end, device=self.device).unsqueeze(0)
                    cos, sin = attn.rotary_emb(v, position_ids)

                    # Apply rotary embeddings
                    q, k = self._apply_rotary_pos_emb(q, k, cos, sin)

            # Store to CPU
            q_cpu[:, :, start:end, :] = q.cpu()
            kv_cache.store_chunk(layer_idx, k, v, start, end)

            del hidden_chunk, normed, q, k, v
            torch.cuda.empty_cache()

        return q_cpu

    def _apply_rotary_pos_emb(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary position embeddings."""
        # Reshape for rotary
        q_embed = (q * cos) + (self._rotate_half(q) * sin)
        k_embed = (k * cos) + (self._rotate_half(k) * sin)
        return q_embed, k_embed

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate half of the hidden dims."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def _process_layer(
        self,
        layer_idx: int,
        hidden_cache: HiddenStatesCache,
        kv_cache: ChunkedKVCache,
    ) -> Dict[int, Dict[str, Any]]:
        """
        Process a single layer: pattern discovery and optional sparse attention.

        Args:
            layer_idx: Layer index
            hidden_cache: Hidden states cache
            kv_cache: KV cache

        Returns:
            layer_results: Results for each head
        """
        print(f"\n=== Layer {layer_idx} ===")

        # Compute Q, K, V
        print("Computing Q, K, V projections...")
        q_cpu = self._chunked_qkv_projection(layer_idx, hidden_cache, kv_cache)

        layer_results = {}

        # Process each attention head
        for head_idx in tqdm(range(self.num_heads), desc=f"Layer {layer_idx} heads"):
            kv_head_idx = head_idx // self.heads_per_kv

            # Get Q for this head
            q_head = q_cpu[:, head_idx:head_idx+1, :, :]  # [1, 1, seq_len, head_dim]

            # Get K, V for this KV head (on CPU)
            k_head, v_head = kv_cache.get_full_kv(layer_idx, kv_head_idx)

            # Get probe queries (last 64)
            q_probe = q_head[:, :, -64:, :].cuda()

            # Compute probe attention
            probe_attn = chunked_probe_attention(
                q_probe, k_head, self.head_dim,
                chunk_size=self.config.kv_chunk_size,
            )

            # Extract patterns
            v_idx, s_idx = extract_patterns(
                probe_attn, self.seq_len,
                vertical_size=self.config.vertical_size,
                slash_size=self.config.slash_size,
            )

            # Store patterns
            self.patterns.setdefault(layer_idx, {})[head_idx] = (v_idx.cpu(), s_idx.cpu())

            # Compute sparsity
            sparsity = compute_sparsity_ratio(v_idx, s_idx, self.seq_len)

            # Store results
            layer_results[head_idx] = {
                'vertical_size': len(v_idx),
                'slash_size': len(s_idx),
                'sparsity_ratio': sparsity,
                'vertical_indices': v_idx.cpu().tolist(),
                'slash_indices': s_idx.cpu().tolist(),
            }

            # Optionally compute sparse attention output using MInference kernel
            if self.config.compute_output:
                # Try to use MInference's kernel directly if data fits in GPU
                # Otherwise fall back to chunked implementation
                try:
                    # Check if we can fit the data in GPU memory
                    gpu_memory_needed = (
                        self.seq_len * self.head_dim * 2 * 3  # Q, K, V in FP16
                    ) / (1024 ** 3)  # GB

                    if gpu_memory_needed < 8:  # If < 8GB, use MInference kernel directly
                        # Load all data to GPU
                        q_gpu = q_head.cuda()
                        k_gpu = k_head.cuda()
                        v_gpu = v_head.cuda()

                        # Call MInference's sparse attention kernel
                        output = minference_sparse_attention(
                            q_gpu, k_gpu, v_gpu, v_idx.cuda(), s_idx.cuda()
                        )
                        output = output.cpu()
                        del q_gpu, k_gpu, v_gpu
                    else:
                        # Use chunked MInference sparse attention
                        output = chunked_minference_sparse_attention(
                            q_head, k_head, v_head,
                            v_idx, s_idx, self.head_dim,
                            q_chunk_size=self.config.q_chunk_size,
                            block_size=64,
                        )
                except Exception as e:
                    print(f"Warning: MInference kernel failed ({e}), falling back to chunked attention")
                    # Fall back to pure PyTorch chunked sparse attention
                    output = chunked_sparse_attention(
                        q_head, k_head, v_head,
                        v_idx, s_idx, self.head_dim,
                        q_chunk_size=self.config.q_chunk_size,
                        kv_batch_size=self.config.kv_chunk_size // 16,
                    )

                if self.config.save_outputs:
                    output_dir = os.path.join(self.config.save_dir, str(self.seq_len), 'outputs')
                    os.makedirs(output_dir, exist_ok=True)
                    output_path = os.path.join(output_dir, f'layer_{layer_idx:02d}_head_{head_idx:02d}.pt')
                    torch.save(output.cpu(), output_path)

                del output

            del q_probe, probe_attn, v_idx, s_idx
            torch.cuda.empty_cache()

        # Update hidden states for next layer (if not last layer)
        if layer_idx < self.num_layers - 1:
            self._update_hidden_states(layer_idx, q_cpu, kv_cache, hidden_cache)

        del q_cpu
        torch.cuda.empty_cache()

        return layer_results

    def _update_hidden_states(
        self,
        layer_idx: int,
        q_cpu: torch.Tensor,
        kv_cache: ChunkedKVCache,
        hidden_cache: HiddenStatesCache,
    ) -> None:
        """
        Update hidden states after attention for next layer.

        Computes full layer forward: Attention + FFN with residual connections.
        Uses Flash Attention for memory efficiency.
        """
        try:
            from flash_attn import flash_attn_func
            use_flash = True
        except ImportError:
            use_flash = False

        # Use smaller chunk to fit FFN in memory
        chunk_size = min(2048, self.config.embedding_chunk_size)
        attn = self._get_attention_layer(layer_idx)

        for start in range(0, self.seq_len, chunk_size):
            end = min(start + chunk_size, self.seq_len)
            chunk_len = end - start

            # Load current hidden states - ensure on correct device
            hidden_chunk = hidden_cache.load_chunk(start, end, self.device).to(self.device)
            residual = hidden_chunk.clone().to(self.device)

            with torch.no_grad():
                # Input layernorm
                ln = self._get_layer_norm(layer_idx, 'input')
                normed = ln(hidden_chunk)

                # Get Q, K, V chunks
                q_chunk = q_cpu[:, :, start:end, :].to(self.device)  # [1, num_heads, chunk_len, head_dim]

                # For causal attention, we need K, V up to current position
                # Load K, V from start to end (local window approximation for efficiency)
                k_chunk = torch.empty(1, self.num_kv_heads, chunk_len, self.head_dim,
                                     dtype=self.config.dtype, device=self.device)
                v_chunk = torch.empty_like(k_chunk)

                for kv_head in range(self.num_kv_heads):
                    k_full, v_full = kv_cache.get_full_kv(layer_idx, kv_head)
                    k_chunk[:, kv_head] = k_full[:, :, start:end, :].to(self.device)
                    v_chunk[:, kv_head] = v_full[:, :, start:end, :].to(self.device)

                # Expand K, V for GQA
                k_expanded = k_chunk.repeat_interleave(self.heads_per_kv, dim=1)
                v_expanded = v_chunk.repeat_interleave(self.heads_per_kv, dim=1)

                if use_flash:
                    # Flash attention: [batch, seqlen, heads, head_dim]
                    q_fa = q_chunk.transpose(1, 2).contiguous()
                    k_fa = k_expanded.transpose(1, 2).contiguous()
                    v_fa = v_expanded.transpose(1, 2).contiguous()
                    attn_output = flash_attn_func(q_fa, k_fa, v_fa, causal=True)
                    attn_output = attn_output.to(self.device).view(1, chunk_len, -1)
                else:
                    # Standard attention with memory-efficient computation
                    scale = 1.0 / math.sqrt(self.head_dim)
                    attn_weights = torch.matmul(q_chunk, k_expanded.transpose(-2, -1)) * scale
                    causal_mask = torch.triu(torch.ones(chunk_len, chunk_len, device=self.device), diagonal=1)
                    attn_weights = attn_weights.masked_fill(causal_mask.bool(), float('-inf'))
                    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(self.config.dtype)
                    attn_output = torch.matmul(attn_weights, v_expanded)
                    attn_output = attn_output.transpose(1, 2).contiguous().view(1, chunk_len, -1)
                    del attn_weights

                # Apply output projection and ensure on correct device
                attn_output = attn.o_proj(attn_output).to(self.device)

                # Residual connection
                hidden_chunk = residual.to(self.device) + attn_output
                del attn_output

                # Post-attention layernorm + FFN
                post_ln = self._get_layer_norm(layer_idx, 'post')
                normed = post_ln(hidden_chunk)

                # FFN: gate_proj, up_proj, down_proj with SiLU activation
                mlp = self._get_mlp(layer_idx)
                if hasattr(mlp, 'gate_proj'):
                    gate = mlp.gate_proj(normed)
                    up = mlp.up_proj(normed)
                    ffn_output = mlp.down_proj(F.silu(gate) * up).to(self.device)
                    del gate, up
                else:
                    ffn_output = mlp(normed).to(self.device)

                # Residual connection
                hidden_chunk = hidden_chunk.to(self.device) + ffn_output

            # Store updated hidden states
            hidden_cache.store_chunk(hidden_chunk.cpu(), start, end)

            del hidden_chunk, residual, q_chunk, k_chunk, v_chunk, k_expanded, v_expanded, ffn_output, normed
            torch.cuda.empty_cache()

    def run(self) -> Dict[str, Any]:
        """
        Run the chunked profiling pipeline.

        Returns:
            Profile results including patterns for each layer/head
        """
        start_time = time.time()

        # Create output directory
        output_dir = os.path.join(self.config.save_dir, str(self.seq_len))
        os.makedirs(output_dir, exist_ok=True)

        # Generate input tokens
        print(f"Generating {self.seq_len:,} random tokens...")
        input_ids = self._generate_input_tokens()

        # Initialize caches
        print("Initializing caches...")
        hidden_cache = HiddenStatesCache(
            self.seq_len, self.hidden_dim,
            dtype=self.config.dtype,
        )

        kv_config = ChunkedKVCacheConfig(
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            seq_len=self.seq_len,
            head_dim=self.head_dim,
            dtype=self.config.dtype,
        )
        kv_cache = ChunkedKVCache(kv_config)

        print(f"KV cache memory: {kv_cache.memory_usage_gb():.2f} GB")
        print(f"Hidden cache memory: {hidden_cache.memory_usage_gb():.2f} GB")

        # Compute embeddings
        self._chunked_embedding(input_ids, hidden_cache)

        # Process each layer
        all_results = {}
        for layer_idx in range(self.num_layers):
            layer_results = self._process_layer(layer_idx, hidden_cache, kv_cache)
            all_results[layer_idx] = layer_results

        # Compile final results
        total_time = time.time() - start_time

        self.profile_data = {
            'model_path': self.model_path,
            'seq_len': self.seq_len,
            'num_layers': self.num_layers,
            'num_heads': self.num_heads,
            'num_kv_heads': self.num_kv_heads,
            'head_dim': self.head_dim,
            'vertical_size': self.config.vertical_size,
            'slash_size': self.config.slash_size,
            'total_time_seconds': total_time,
            'layers': {},
        }

        # Convert results to profile format
        for layer_idx, layer_results in all_results.items():
            layer_data = {'heads': {}}
            for head_idx, head_results in layer_results.items():
                layer_data['heads'][str(head_idx)] = head_results
            self.profile_data['layers'][str(layer_idx)] = layer_data

        # Save results
        self._save_results(output_dir)

        print(f"\nProfiling complete in {total_time:.1f} seconds")
        print(f"Results saved to {output_dir}")

        return self.profile_data

    def _save_results(self, output_dir: str) -> None:
        """Save profiling results."""
        # Save main profile JSON
        json_path = os.path.join(output_dir, 'profile.json')
        with open(json_path, 'w') as f:
            json.dump(self.profile_data, f, indent=2)
        print(f"Saved profile to {json_path}")

        # Save summary NPZ
        summary_data = {
            'seq_len': self.seq_len,
            'num_layers': self.num_layers,
            'num_heads': self.num_heads,
            'num_kv_heads': self.num_kv_heads,
        }

        # Compute summary statistics
        sparsities = []
        for layer_idx in range(self.num_layers):
            for head_idx in range(self.num_heads):
                head_data = self.profile_data['layers'][str(layer_idx)]['heads'][str(head_idx)]
                sparsities.append(head_data['sparsity_ratio'])

        summary_data['sparsities'] = np.array(sparsities).reshape(self.num_layers, self.num_heads)
        summary_data['avg_sparsity'] = np.mean(sparsities)

        npz_path = os.path.join(output_dir, 'summary.npz')
        np.savez(npz_path, **summary_data)
        print(f"Saved summary to {npz_path}")

        # Save per-layer NPZ files
        layers_dir = os.path.join(output_dir, 'layers')
        os.makedirs(layers_dir, exist_ok=True)

        for layer_idx in range(self.num_layers):
            layer_data = {}
            for head_idx in range(self.num_heads):
                v_idx, s_idx = self.patterns[layer_idx][head_idx]
                layer_data[f'head_{head_idx}_v_idx'] = v_idx.numpy()
                layer_data[f'head_{head_idx}_s_idx'] = s_idx.numpy()

            layer_path = os.path.join(layers_dir, f'layer_{layer_idx:02d}.npz')
            np.savez(layer_path, **layer_data)

        print(f"Saved {self.num_layers} layer files to {layers_dir}")


def run_profiler(
    model_path: str,
    seq_len: int = 1_000_000,
    save_dir: str = "results/profile",
    vertical_size: int = 1000,
    slash_size: int = 6096,
    q_chunk_size: int = 512,
    kv_chunk_size: int = 65536,
    compute_output: bool = False,
    save_outputs: bool = False,
) -> Dict[str, Any]:
    """
    Convenience function to run the chunked profiler.

    Args:
        model_path: Path to the model
        seq_len: Sequence length to profile
        save_dir: Directory to save results
        vertical_size: Number of vertical columns
        slash_size: Number of slash diagonals
        q_chunk_size: Query chunk size
        kv_chunk_size: KV chunk size
        compute_output: Whether to compute sparse attention output
        save_outputs: Whether to save attention outputs

    Returns:
        Profile results
    """
    config = ProfilerConfig(
        model_path=model_path,
        seq_len=seq_len,
        save_dir=save_dir,
        vertical_size=vertical_size,
        slash_size=slash_size,
        q_chunk_size=q_chunk_size,
        kv_chunk_size=kv_chunk_size,
        compute_output=compute_output,
        save_outputs=save_outputs,
    )

    profiler = ChunkedMInferenceProfiler(config)
    return profiler.run()
