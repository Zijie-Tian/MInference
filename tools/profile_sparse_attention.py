# Copyright (c) 2024-2025 Microsoft
# Licensed under The MIT License [see LICENSE for details]

"""
Profile MInference sparse attention patterns during real model inference.

This script traces the actual computation patterns during prefill:
1. Selected slash diagonal indices (topk)
2. Selected vertical column indices (topk)
3. convert_vertical_slash_indexes outputs (block_count, block_offset, etc.)

Results are saved to results/profile/ directory.

Usage:
    python tools/profile_sparse_attention.py --seq-len 16000
    python tools/profile_sparse_attention.py --visualize results/profile/profile_16000_xxx.json
"""

import os
import sys
import json
import time
from datetime import datetime
from functools import wraps
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

import numpy as np
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class HeadProfile:
    """Profile data for a single attention head."""
    layer_idx: int
    head_idx: int
    seq_len: int
    # Slash indices (diagonal offsets)
    slash_indices: List[int] = field(default_factory=list)
    slash_size: int = 0
    # Vertical indices (column positions)
    vertical_indices: List[int] = field(default_factory=list)
    vertical_size: int = 0
    # Block-level outputs
    block_counts: List[int] = field(default_factory=list)
    column_counts: List[int] = field(default_factory=list)
    total_slash_blocks: int = 0
    total_vertical_columns: int = 0
    compute_time_ms: float = 0.0


class SparseAttentionProfiler:
    """Profiler that captures sparse attention patterns via monkey-patching."""

    def __init__(self, save_dir: str = "results/profile"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        self.enabled = False
        self.data: Dict[str, Any] = {
            "model_name": "",
            "seq_len": 0,
            "num_layers": 0,
            "num_heads": 0,
            "timestamp": "",
            "layers": {},
        }

        # Current position tracking
        self.current_layer = 0
        self.current_head = 0

        # Call counter for sequential tracking
        self.call_count = 0

        # Store original functions
        self._originals = {}

    def start(self, model_name: str, seq_len: int, num_layers: int, num_heads: int):
        """Start profiling session."""
        self.enabled = True
        self.call_count = 0
        self.data = {
            "model_name": model_name,
            "seq_len": seq_len,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "timestamp": datetime.now().isoformat(),
            "layers": {},
            "total_time_ms": 0.0,
        }
        print(f"[Profiler] Started: seq_len={seq_len}, layers={num_layers}, heads={num_heads}")

    def set_position(self, layer_idx: int, head_idx: int):
        """Set current layer and head being processed."""
        self.current_layer = layer_idx
        self.current_head = head_idx

    def record_indices(self, v_idx: torch.Tensor, s_idx: torch.Tensor, seq_len: int):
        """Record vertical and slash indices for current head."""
        if not self.enabled:
            return

        num_heads = self.data.get("num_heads", 32)

        # Use call count to determine layer and head
        layer_idx = self.call_count // num_heads
        head_idx = self.call_count % num_heads
        self.call_count += 1

        layer_key = str(layer_idx)
        head_key = str(head_idx)

        if layer_key not in self.data["layers"]:
            self.data["layers"][layer_key] = {"heads": {}}

        # Convert to lists
        v_list = v_idx.detach().cpu().flatten().tolist()
        s_list = s_idx.detach().cpu().flatten().tolist()

        # Filter valid indices
        v_list = sorted(set(int(x) for x in v_list if 0 <= x < seq_len))
        s_list = sorted(set(int(x) for x in s_list if x >= 0), reverse=True)

        self.data["layers"][layer_key]["heads"][head_key] = {
            "layer_idx": layer_idx,
            "head_idx": head_idx,
            "seq_len": seq_len,
            "vertical_indices": v_list,
            "vertical_size": len(v_list),
            "slash_indices": s_list,
            "slash_size": len(s_list),
            "block_counts": [],
            "column_counts": [],
            "total_slash_blocks": 0,
            "total_vertical_columns": 0,
        }

        # Debug: print progress periodically
        if self.call_count % 100 == 0:
            print(f"  [Profiler] Captured {self.call_count} heads (L{layer_idx} H{head_idx})...")

    def record_blocks(self, block_count: torch.Tensor, column_count: torch.Tensor):
        """Record block-level information."""
        if not self.enabled:
            return

        num_heads = self.data.get("num_heads", 32)

        # Use call_count - 1 since record_indices already incremented it
        prev_call = self.call_count - 1
        if prev_call < 0:
            return

        layer_idx = prev_call // num_heads
        head_idx = prev_call % num_heads

        layer_key = str(layer_idx)
        head_key = str(head_idx)

        if layer_key not in self.data["layers"]:
            return
        if head_key not in self.data["layers"][layer_key]["heads"]:
            return

        bc = block_count.detach().cpu().flatten().tolist()
        cc = column_count.detach().cpu().flatten().tolist()

        head_data = self.data["layers"][layer_key]["heads"][head_key]
        head_data["block_counts"] = bc
        head_data["column_counts"] = cc
        head_data["total_slash_blocks"] = sum(bc)
        head_data["total_vertical_columns"] = sum(cc)

    def finish(self, total_time_ms: float = 0.0, num_kv_heads: int = None) -> str:
        """Finish profiling and save results."""
        self.enabled = False
        self.data["total_time_ms"] = total_time_ms
        if num_kv_heads is not None:
            self.data["num_kv_heads"] = num_kv_heads

        seq_len = self.data["seq_len"]

        # Create output directory: results/profile/{seq_len}/
        output_dir = os.path.join(self.save_dir, str(seq_len))
        os.makedirs(output_dir, exist_ok=True)

        # Save JSON (overwrites existing)
        json_path = os.path.join(output_dir, "profile.json")
        with open(json_path, 'w') as f:
            json.dump(self.data, f, indent=2)
        print(f"[Profiler] JSON saved to {json_path}")

        # Save numpy files (one per layer)
        self._save_numpy_per_layer(output_dir)

        return output_dir

    def _save_numpy_per_layer(self, output_dir: str):
        """Save numpy data with one file per layer."""
        num_layers = self.data["num_layers"]
        num_heads = self.data["num_heads"]
        seq_len = self.data["seq_len"]
        num_kv_heads = self.data.get("num_kv_heads", num_heads)

        # Create layers subdirectory
        layers_dir = os.path.join(output_dir, "layers")
        os.makedirs(layers_dir, exist_ok=True)

        # Summary arrays for the overview
        slash_sizes = np.zeros((num_layers, num_heads), dtype=np.int32)
        vertical_sizes = np.zeros((num_layers, num_heads), dtype=np.int32)
        total_blocks = np.zeros((num_layers, num_heads), dtype=np.int32)

        for layer_key, layer_data in self.data["layers"].items():
            l = int(layer_key)
            if l >= num_layers:
                continue

            # Per-layer data
            layer_slash = np.zeros(num_heads, dtype=np.int32)
            layer_vert = np.zeros(num_heads, dtype=np.int32)
            layer_blocks = np.zeros(num_heads, dtype=np.int32)

            # Collect all indices for this layer
            slash_indices_list = []
            vertical_indices_list = []

            for head_key, head_data in layer_data["heads"].items():
                h = int(head_key)
                if h < num_heads:
                    layer_slash[h] = head_data["slash_size"]
                    layer_vert[h] = head_data["vertical_size"]
                    layer_blocks[h] = head_data["total_slash_blocks"]

                    slash_sizes[l, h] = head_data["slash_size"]
                    vertical_sizes[l, h] = head_data["vertical_size"]
                    total_blocks[l, h] = head_data["total_slash_blocks"]

                    # Store indices (padded to max length)
                    slash_indices_list.append(head_data["slash_indices"])
                    vertical_indices_list.append(head_data["vertical_indices"])

            # Save per-layer NPZ
            layer_path = os.path.join(layers_dir, f"layer_{l:02d}.npz")
            np.savez_compressed(
                layer_path,
                layer_idx=l,
                slash_sizes=layer_slash,
                vertical_sizes=layer_vert,
                total_blocks=layer_blocks,
                # Note: indices have variable length, store as object array
            )

        # Save summary NPZ
        summary_path = os.path.join(output_dir, "summary.npz")
        np.savez_compressed(
            summary_path,
            model_name=self.data["model_name"],
            seq_len=seq_len,
            num_layers=num_layers,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            slash_sizes=slash_sizes,
            vertical_sizes=vertical_sizes,
            total_blocks=total_blocks,
        )
        print(f"[Profiler] Summary saved to {summary_path}")
        print(f"[Profiler] Per-layer data saved to {layers_dir}/")

    def install_hooks(self):
        """Install monkey-patches to capture sparse attention data."""
        import minference.ops.pit_sparse_flash_attention_v2 as sparse_module
        import minference.modules.minference_forward as forward_module

        # Save original - get it from the source module
        self._originals["vertical_slash_sparse_attention"] = sparse_module.vertical_slash_sparse_attention

        profiler = self  # Capture reference

        original_func = self._originals["vertical_slash_sparse_attention"]

        def hooked_vertical_slash_sparse_attention(
            query, key, value, v_idx, s_idx, block_size_M=64, block_size_N=64
        ):
            seq_len = query.shape[2]

            # Record indices
            profiler.record_indices(v_idx, s_idx, seq_len)

            # Call original
            result = original_func(query, key, value, v_idx, s_idx, block_size_M, block_size_N)

            return result

        # Patch in BOTH modules - the source module AND the forward module where it's imported
        sparse_module.vertical_slash_sparse_attention = hooked_vertical_slash_sparse_attention
        forward_module.vertical_slash_sparse_attention = hooked_vertical_slash_sparse_attention

        # Hook convert_vertical_slash_indexes from minference.cuda
        import minference.cuda as cuda_module

        if hasattr(cuda_module, 'convert_vertical_slash_indexes'):
            self._originals["convert_vertical_slash_indexes"] = cuda_module.convert_vertical_slash_indexes

            original_convert = self._originals["convert_vertical_slash_indexes"]

            def hooked_convert(seqlens, v_idx, s_idx, ctx_size, bsm, bsn):
                result = original_convert(seqlens, v_idx, s_idx, ctx_size, bsm, bsn)
                block_count, block_offset, column_count, column_index = result
                profiler.record_blocks(block_count, column_count)
                return result

            # Patch in both cuda module and sparse module (where it's imported)
            cuda_module.convert_vertical_slash_indexes = hooked_convert
            sparse_module.convert_vertical_slash_indexes = hooked_convert

        # Hook the optimized version (from sgl_kernel/vllm) if it exists
        if hasattr(sparse_module, 'convert_vertical_slash_indexes_opt') and sparse_module.convert_vertical_slash_indexes_opt is not None:
            self._originals["convert_vertical_slash_indexes_opt"] = sparse_module.convert_vertical_slash_indexes_opt

            original_convert_opt = self._originals["convert_vertical_slash_indexes_opt"]

            def hooked_convert_opt(seqlens, seqlens2, v_idx, s_idx, ctx_size, bsm, bsn, causal=True):
                result = original_convert_opt(seqlens, seqlens2, v_idx, s_idx, ctx_size, bsm, bsn, causal)
                block_count, block_offset, column_count, column_index = result
                profiler.record_blocks(block_count, column_count)
                return result

            sparse_module.convert_vertical_slash_indexes_opt = hooked_convert_opt

        print("[Profiler] Hooks installed on sparse_module, forward_module, and cuda_module")

    def uninstall_hooks(self):
        """Restore original functions."""
        import minference.ops.pit_sparse_flash_attention_v2 as sparse_module
        import minference.modules.minference_forward as forward_module
        import minference.cuda as cuda_module

        for name, func in self._originals.items():
            if hasattr(sparse_module, name):
                setattr(sparse_module, name, func)
            if hasattr(forward_module, name):
                setattr(forward_module, name, func)
            if hasattr(cuda_module, name):
                setattr(cuda_module, name, func)

        self._originals.clear()
        print("[Profiler] Hooks uninstalled")


# Global profiler
_profiler = SparseAttentionProfiler()


def generate_long_prompt(tokenizer, target_length: int = 16000) -> str:
    """Generate a prompt with approximately target_length tokens."""

    base_texts = [
        "The history of artificial intelligence began in antiquity, with myths and stories of artificial beings. The seeds of modern AI were planted by philosophers who described human thinking as mechanical symbol manipulation. This work culminated in the programmable digital computer in the 1940s. " * 5,

        "Machine learning is a subset of AI that enables systems to learn from experience without explicit programming. It focuses on developing programs that access data and learn for themselves. Learning begins with observations or data to look for patterns and make better decisions. " * 5,

        "Deep learning uses artificial neural networks with representation learning. Architectures like CNNs and RNNs have been applied to computer vision, speech recognition, NLP, and many other fields. These models can automatically learn hierarchical representations of data. " * 5,

        "Transformers revolutionized NLP with the self-attention mechanism introduced in 'Attention Is All You Need'. This allows models to weigh the importance of different input parts when producing outputs. The architecture has become the foundation for many state-of-the-art models. " * 5,

        "Large language models represent a significant AI advancement. Trained on vast text data, they generate human-like text, answer questions, and write code. Their scale, often billions of parameters, allows capturing complex language patterns and demonstrating emergent abilities. " * 5,
    ]

    prompt = ""
    while len(tokenizer.encode(prompt)) < target_length:
        for text in base_texts:
            prompt += text + "\n\n"
            if len(tokenizer.encode(prompt)) >= target_length:
                break

    # Trim to target
    tokens = tokenizer.encode(prompt)[:target_length]
    return tokenizer.decode(tokens)


def run_profiled_inference(
    model_name: str = "/home/zijie/models/Llama-3-8B-Instruct-262k",
    target_seq_len: int = 16000,
    save_dir: str = "results/profile",
):
    """Run inference with profiling."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from minference import MInference

    print("=" * 70)
    print("MInference Sparse Attention Profiler")
    print("=" * 70)

    global _profiler
    _profiler = SparseAttentionProfiler(save_dir)

    # Load tokenizer
    print(f"\n[1/5] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Generate prompt
    print(f"\n[2/5] Generating {target_seq_len}-token prompt...")
    prompt = generate_long_prompt(tokenizer, target_seq_len)
    actual_len = len(tokenizer.encode(prompt))
    print(f"       Actual length: {actual_len} tokens")

    # Load model
    print(f"\n[3/5] Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="sdpa",
    )

    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    num_kv_heads = getattr(model.config, 'num_key_value_heads', num_heads)
    print(f"       {num_layers} layers, {num_heads} heads, {num_kv_heads} KV heads")

    # Apply MInference
    print(f"\n[4/5] Applying MInference patch...")
    minference_patch = MInference(
        attn_type="minference",
        model_name=model_name,
        kv_type="dense"
    )
    model = minference_patch(model)

    # Install hooks AFTER MInference patch
    _profiler.install_hooks()

    # Start profiling
    _profiler.start(model_name, actual_len, num_layers, num_heads)

    # Run inference
    print(f"\n[5/5] Running prefill...")
    batch_inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    torch.cuda.synchronize()
    start = time.perf_counter()

    with torch.no_grad():
        outputs = model(**batch_inputs, use_cache=True, return_dict=True)

    torch.cuda.synchronize()
    total_ms = (time.perf_counter() - start) * 1000
    print(f"       Completed in {total_ms:.2f} ms")

    # Finish and uninstall hooks
    _profiler.uninstall_hooks()

    output_dir = _profiler.finish(total_ms, num_kv_heads=num_kv_heads)

    # Print summary
    print_summary(_profiler.data)

    return output_dir


def print_summary(data: dict):
    """Print profiling summary."""
    print("\n" + "=" * 70)
    print("PROFILING SUMMARY")
    print("=" * 70)

    print(f"\nSequence Length: {data['seq_len']}")
    print(f"Total Time: {data['total_time_ms']:.2f} ms")

    total_slash = 0
    total_vert = 0
    total_blocks = 0
    count = 0

    for layer_data in data["layers"].values():
        for head_data in layer_data["heads"].values():
            total_slash += head_data["slash_size"]
            total_vert += head_data["vertical_size"]
            total_blocks += head_data["total_slash_blocks"]
            count += 1

    if count > 0:
        print(f"\nProfiled {count} attention heads")
        print(f"Avg slash diagonals: {total_slash/count:.1f}")
        print(f"Avg vertical columns: {total_vert/count:.1f}")
        print(f"Avg computed blocks: {total_blocks/count:.1f}")

    print("=" * 70)


def visualize_profile(profile_dir: str):
    """Visualize saved profile with GQA KV head grouping.

    Args:
        profile_dir: Path to profile directory (e.g., results/profile/16001/)
                     or path to legacy JSON file for backward compatibility.
    """
    import matplotlib.pyplot as plt

    # Handle both new directory structure and legacy JSON path
    if profile_dir.endswith('.json'):
        # Legacy: convert to directory path
        profile_dir = os.path.dirname(profile_dir)

    summary_path = os.path.join(profile_dir, "summary.npz")
    if not os.path.exists(summary_path):
        # Try legacy path
        legacy_npz = profile_dir.replace('.json', '.npz') if profile_dir.endswith('.json') else None
        if legacy_npz and os.path.exists(legacy_npz):
            summary_path = legacy_npz
        else:
            print(f"Summary file not found: {summary_path}")
            return

    data = np.load(summary_path, allow_pickle=True)
    seq_len = int(data['seq_len'])
    num_heads = int(data['num_heads'])
    num_kv_heads = int(data.get('num_kv_heads', num_heads))
    slash_sizes = data['slash_sizes']
    vertical_sizes = data['vertical_sizes']
    total_blocks = data['total_blocks']

    # Calculate GQA group size
    heads_per_kv = num_heads // num_kv_heads

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    def add_kv_group_lines(ax, num_heads, heads_per_kv):
        """Add vertical lines to separate KV head groups."""
        if heads_per_kv > 1:
            for i in range(1, num_kv_heads):
                x = i * heads_per_kv - 0.5
                ax.axvline(x=x, color='white', linewidth=1.5, linestyle='-')
                ax.axvline(x=x, color='black', linewidth=0.5, linestyle='--', alpha=0.5)

    # Slash sizes
    im1 = axes[0, 0].imshow(slash_sizes, cmap='YlOrRd', aspect='auto')
    add_kv_group_lines(axes[0, 0], num_heads, heads_per_kv)
    axes[0, 0].set_xlabel(f'Head (grouped by {heads_per_kv} per KV head)')
    axes[0, 0].set_ylabel('Layer')
    axes[0, 0].set_title(f'Slash Sizes (seq_len={seq_len})')
    plt.colorbar(im1, ax=axes[0, 0])

    # Vertical sizes
    im2 = axes[0, 1].imshow(vertical_sizes, cmap='YlOrRd', aspect='auto')
    add_kv_group_lines(axes[0, 1], num_heads, heads_per_kv)
    axes[0, 1].set_xlabel(f'Head (grouped by {heads_per_kv} per KV head)')
    axes[0, 1].set_ylabel('Layer')
    axes[0, 1].set_title('Vertical Sizes')
    plt.colorbar(im2, ax=axes[0, 1])

    # Total blocks
    im3 = axes[1, 0].imshow(total_blocks, cmap='YlOrRd', aspect='auto')
    add_kv_group_lines(axes[1, 0], num_heads, heads_per_kv)
    axes[1, 0].set_xlabel(f'Head (grouped by {heads_per_kv} per KV head)')
    axes[1, 0].set_ylabel('Layer')
    axes[1, 0].set_title('Total Computed Blocks')
    plt.colorbar(im3, ax=axes[1, 0])

    # Per-layer averages with KV group breakdown
    layers = np.arange(slash_sizes.shape[0])
    axes[1, 1].plot(layers, slash_sizes.mean(1), 'b-o', label='Slash (avg)', ms=3)
    axes[1, 1].plot(layers, vertical_sizes.mean(1), 'r-s', label='Vertical (avg)', ms=3)

    # Add per-KV-group averages if GQA is used
    if heads_per_kv > 1:
        # Reshape to (layers, kv_heads, heads_per_kv) and average within each KV group
        slash_per_kv = slash_sizes.reshape(slash_sizes.shape[0], num_kv_heads, heads_per_kv)
        kv_group_std = slash_per_kv.std(axis=2).mean(axis=1)  # std within KV groups
        axes[1, 1].fill_between(layers,
                                slash_sizes.mean(1) - kv_group_std,
                                slash_sizes.mean(1) + kv_group_std,
                                alpha=0.2, color='blue', label='±std within KV group')

    axes[1, 1].set_xlabel('Layer')
    axes[1, 1].set_ylabel('Average Size')
    axes[1, 1].set_title(f'Average per Layer ({num_kv_heads} KV heads, {heads_per_kv} Q heads each)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    viz_path = os.path.join(profile_dir, "profile_viz.png")
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to {viz_path}")
    plt.close()


def visualize_single_layer_head(profile_dir: str, layer_idx: int = 0, head_idx: int = 0):
    """Visualize detailed pattern for a specific layer/head.

    Args:
        profile_dir: Path to profile directory or legacy JSON file.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    # Handle both new directory structure and legacy JSON path
    if profile_dir.endswith('.json'):
        json_path = profile_dir
        profile_dir = os.path.dirname(profile_dir)
    else:
        json_path = os.path.join(profile_dir, "profile.json")

    with open(json_path, 'r') as f:
        data = json.load(f)

    seq_len = data["seq_len"]
    layer_key = str(layer_idx)
    head_key = str(head_idx)

    if layer_key not in data["layers"]:
        print(f"Layer {layer_idx} not found")
        return
    if head_key not in data["layers"][layer_key]["heads"]:
        print(f"Head {head_idx} not found in layer {layer_idx}")
        return

    head_data = data["layers"][layer_key]["heads"][head_key]
    slash_indices = head_data["slash_indices"]
    vertical_indices = head_data["vertical_indices"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Plot 1: Full attention matrix with patterns
    block_size = 64
    num_blocks = (seq_len + block_size - 1) // block_size

    # Create sparse pattern visualization (downsampled)
    display_size = min(512, seq_len)
    scale = seq_len / display_size

    pattern = np.zeros((display_size, display_size))

    # Mark causal region
    for i in range(display_size):
        for j in range(int(i * scale / scale) + 1):
            pattern[i, j] = 0.1

    # Mark slash diagonals
    for d in slash_indices:
        for q in range(display_size):
            k = int((q * scale - d) / scale)
            if 0 <= k < display_size and k <= q:
                pattern[q, k] = 1.0

    # Mark vertical columns
    for v in vertical_indices:
        v_scaled = int(v / scale)
        if v_scaled < display_size:
            for q in range(v_scaled, display_size):
                if pattern[q, v_scaled] < 1.0:
                    pattern[q, v_scaled] = 0.6

    im1 = axes[0].imshow(pattern, cmap='YlOrRd', aspect='equal', origin='upper')
    axes[0].set_xlabel('Key Position')
    axes[0].set_ylabel('Query Position')
    axes[0].set_title(f'Layer {layer_idx}, Head {head_idx}\n'
                      f'Slash: {len(slash_indices)}, Vertical: {len(vertical_indices)}')
    plt.colorbar(im1, ax=axes[0])

    # Plot 2: Distribution of slash indices
    axes[1].hist(slash_indices, bins=50, color='coral', edgecolor='black', alpha=0.7)
    axes[1].set_xlabel('Diagonal Index d')
    axes[1].set_ylabel('Count')
    axes[1].set_title(f'Distribution of Selected Slash Diagonals\n(d=0 is main diagonal)')
    axes[1].axvline(x=0, color='red', linestyle='--', label='Main diagonal')
    axes[1].legend()

    plt.tight_layout()

    viz_path = os.path.join(profile_dir, f"pattern_L{layer_idx}_H{head_idx}.png")
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    print(f"Pattern visualization saved to {viz_path}")
    plt.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Profile MInference sparse attention")
    parser.add_argument("--model", type=str, default="/home/zijie/models/Llama-3-8B-Instruct-262k")
    parser.add_argument("--seq-len", type=int, default=16000)
    parser.add_argument("--save-dir", type=str, default="results/profile")
    parser.add_argument("--visualize", type=str, default=None,
                        help="Profile directory to visualize (e.g., results/profile/16001)")
    parser.add_argument("--layer", type=int, default=0, help="Layer for detailed viz")
    parser.add_argument("--head", type=int, default=0, help="Head for detailed viz")

    args = parser.parse_args()

    if args.visualize:
        visualize_profile(args.visualize)
        visualize_single_layer_head(args.visualize, args.layer, args.head)
    else:
        output_dir = run_profiled_inference(
            model_name=args.model,
            target_seq_len=args.seq_len,
            save_dir=args.save_dir,
        )
        if output_dir:
            visualize_profile(output_dir)
