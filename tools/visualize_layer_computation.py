# Copyright (c) 2024-2025 Microsoft
# Licensed under The MIT License [see LICENSE for details]

"""
Visualize actual computed blocks for specific layers based on profiled data.

Shows which 64x64 blocks are actually computed by the sparse attention kernel.

Usage:
    python tools/visualize_layer_computation.py --profile-dir results/profile/16001 --layer 0
"""

import os
import sys
import json
import argparse
from typing import List, Optional

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_layer_data(profile_dir: str, layer_idx: int) -> dict:
    """Load profile data for a specific layer."""
    json_path = os.path.join(profile_dir, "profile.json")
    with open(json_path, 'r') as f:
        data = json.load(f)

    layer_key = str(layer_idx)
    if layer_key not in data["layers"]:
        raise ValueError(f"Layer {layer_idx} not found in profile data")

    return {
        "seq_len": data["seq_len"],
        "num_heads": data["num_heads"],
        "num_kv_heads": data.get("num_kv_heads", data["num_heads"]),
        "layer_data": data["layers"][layer_key],
    }


def compute_block_mask(
    seq_len: int,
    slash_indices: List[int],
    vertical_indices: List[int],
    block_size: int = 64,
) -> np.ndarray:
    """
    Compute block-level mask showing which blocks are computed.
    Fully vectorized for speed.

    Returns:
        block_mask: (num_blocks, num_blocks) array
    """
    num_blocks = (seq_len + block_size - 1) // block_size
    block_mask = np.zeros((num_blocks, num_blocks), dtype=np.float32)

    # Vectorized: for each diagonal d, compute which (q_block, k_block) pairs are hit
    # diagonal d means: key = query - d, so k_block = q_block - d // block_size (approx)
    slash_arr = np.array(slash_indices, dtype=np.int64)

    # Convert diagonals to block-level diagonals
    # A diagonal d hits block (q, k) if any (q_pos, k_pos) in that block satisfies q_pos - k_pos = d
    # This means: q_block * B <= q_pos < (q_block+1) * B
    #             k_block * B <= k_pos < (k_block+1) * B
    #             q_pos - k_pos = d
    # So: (q_block - k_block - 1) * B < d <= (q_block - k_block + 1) * B
    # Simplified: block diagonal bd = q_block - k_block, and d is in range [(bd-1)*B+1, (bd+1)*B]

    # For efficiency, convert slash indices to the set of block diagonals they cover
    block_diags_covered = set()
    for d in slash_arr:
        # Block diagonal that this element diagonal d primarily hits
        bd_min = (d - block_size + 1) // block_size
        bd_max = (d + block_size - 1) // block_size
        for bd in range(max(0, bd_min), bd_max + 1):
            block_diags_covered.add(bd)

    # Fill in the block mask based on covered block diagonals
    for bd in block_diags_covered:
        # Block diagonal bd: q_block - k_block = bd
        # Valid range: k_block from 0 to num_blocks-1, q_block = k_block + bd
        for k_block in range(num_blocks):
            q_block = k_block + bd
            if 0 <= q_block < num_blocks:
                block_mask[q_block, k_block] = 1.0

    # Mark vertical columns
    vert_arr = np.array(vertical_indices, dtype=np.int64)
    v_blocks = np.unique(vert_arr // block_size)
    v_blocks = v_blocks[v_blocks < num_blocks]

    for v_block in v_blocks:
        # Column v_block is attended by all query blocks >= v_block
        mask_slice = block_mask[v_block:, v_block]
        block_mask[v_block:, v_block] = np.where(mask_slice == 0, 0.5, mask_slice)

    return block_mask


def visualize_layer(
    profile_dir: str,
    layer_idx: int,
    block_size: int = 64,
    output_path: Optional[str] = None,
):
    """Visualize block-level computation for a layer."""

    data = load_layer_data(profile_dir, layer_idx)
    seq_len = data["seq_len"]
    num_heads = data["num_heads"]
    num_kv_heads = data["num_kv_heads"]
    layer_data = data["layer_data"]
    heads_per_kv = num_heads // num_kv_heads

    num_blocks = (seq_len + block_size - 1) // block_size
    total_causal = num_blocks * (num_blocks + 1) // 2

    # Create figure: show first 8 heads (one per KV head if GQA)
    # Plus aggregate view
    n_cols = 4
    n_rows = 3

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 12))

    fig.suptitle(
        f'Layer {layer_idx}: Block-Level Computation Pattern\n'
        f'seq_len={seq_len}, {num_blocks} blocks, block_size={block_size}',
        fontsize=14
    )

    # Aggregate mask across all heads
    aggregate_mask = np.zeros((num_blocks, num_blocks), dtype=np.float32)

    # Show first 8 heads
    display_heads = list(range(min(8, num_heads)))

    for idx, head_idx in enumerate(display_heads):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        head_key = str(head_idx)
        if head_key in layer_data["heads"]:
            head_data = layer_data["heads"][head_key]

            block_mask = compute_block_mask(
                seq_len,
                head_data["slash_indices"],
                head_data["vertical_indices"],
                block_size
            )

            aggregate_mask = np.maximum(aggregate_mask, block_mask)

            # Count computed blocks
            computed = np.sum(block_mask > 0)
            sparsity = computed / total_causal * 100

            im = ax.imshow(block_mask, cmap='YlOrRd', aspect='equal',
                          origin='upper', vmin=0, vmax=1)

            kv_group = head_idx // heads_per_kv
            ax.set_title(
                f'Head {head_idx} (KV{kv_group})\n'
                f'{int(computed)}/{total_causal} blocks ({sparsity:.1f}%)',
                fontsize=10
            )
            ax.set_xlabel('Key Block')
            ax.set_ylabel('Query Block')

    # Bottom row: aggregate + stats
    # Aggregate view
    ax_agg = axes[2, 0]
    computed_agg = np.sum(aggregate_mask > 0)
    sparsity_agg = computed_agg / total_causal * 100

    im = ax_agg.imshow(aggregate_mask, cmap='YlOrRd', aspect='equal',
                       origin='upper', vmin=0, vmax=1)
    ax_agg.set_title(f'Aggregate (Union)\n{int(computed_agg)}/{total_causal} ({sparsity_agg:.1f}%)')
    ax_agg.set_xlabel('Key Block')
    ax_agg.set_ylabel('Query Block')

    # Per-head stats
    ax_stats = axes[2, 1]
    sparsities = []
    slash_sizes = []
    vert_sizes = []

    for h in range(num_heads):
        hkey = str(h)
        if hkey in layer_data["heads"]:
            hd = layer_data["heads"][hkey]
            slash_sizes.append(hd["slash_size"])
            vert_sizes.append(hd["vertical_size"])

            mask = compute_block_mask(seq_len, hd["slash_indices"],
                                      hd["vertical_indices"], block_size)
            sparsities.append(np.sum(mask > 0) / total_causal * 100)

    x = np.arange(num_heads)
    colors = plt.cm.Set3(np.arange(num_heads) // heads_per_kv % 12)
    ax_stats.bar(x, sparsities, color=colors, edgecolor='black', linewidth=0.3)

    for i in range(1, num_kv_heads):
        ax_stats.axvline(x=i * heads_per_kv - 0.5, color='red', linewidth=1, linestyle='--')

    ax_stats.set_xlabel('Head')
    ax_stats.set_ylabel('Sparsity (%)')
    ax_stats.set_title('Per-Head Block Sparsity')
    ax_stats.set_ylim(0, 100)

    # Slash vs Vertical
    ax_sv = axes[2, 2]
    ax_sv.bar(x, slash_sizes, label='Slash', color='coral', alpha=0.8)
    ax_sv.bar(x, vert_sizes, bottom=slash_sizes, label='Vertical', color='steelblue', alpha=0.8)
    ax_sv.set_xlabel('Head')
    ax_sv.set_ylabel('Pattern Size')
    ax_sv.set_title('Slash + Vertical Sizes')
    ax_sv.legend(fontsize=8)

    # Legend
    ax_leg = axes[2, 3]
    ax_leg.axis('off')
    legend_text = (
        f"Layer {layer_idx} Summary\n"
        f"─────────────────\n"
        f"Sequence length: {seq_len}\n"
        f"Block size: {block_size}\n"
        f"Num blocks: {num_blocks}\n"
        f"Causal blocks: {total_causal}\n"
        f"─────────────────\n"
        f"Num heads: {num_heads}\n"
        f"KV heads: {num_kv_heads}\n"
        f"Heads/KV: {heads_per_kv}\n"
        f"─────────────────\n"
        f"Avg sparsity: {np.mean(sparsities):.1f}%\n"
        f"Avg slash: {np.mean(slash_sizes):.0f}\n"
        f"Avg vert: {np.mean(vert_sizes):.0f}"
    )
    ax_leg.text(0.1, 0.9, legend_text, transform=ax_leg.transAxes,
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if output_path is None:
        viz_dir = os.path.join(profile_dir, f"viz_bs{block_size}")
        os.makedirs(viz_dir, exist_ok=True)
        output_path = os.path.join(viz_dir, f"layer_{layer_idx:02d}.png")

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved to {output_path}")
    plt.close()

    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize layer block computation")
    parser.add_argument("--profile-dir", type=str, default="results/profile/16001")
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--block-size", type=int, default=64)

    args = parser.parse_args()
    visualize_layer(args.profile_dir, args.layer, args.block_size)
