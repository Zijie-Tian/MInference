# Copyright (c) 2024-2025 Microsoft
# Licensed under The MIT License [see LICENSE for details]

"""
KV Cache Overlap Analysis Tool

Analyzes the overlap of KV cache requirements across different attention heads
during chunked prefill, to evaluate the effectiveness of different offload strategies.

Usage:
    python tools/analyze_kv_overlap.py --profile_dir ./tools/profiles --chunk_size 1024 --output overlap_analysis.png
"""

import argparse
import os
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict


def load_profile(profile_path: str) -> dict:
    """Load profile data from npz file."""
    data = np.load(profile_path, allow_pickle=True)
    return {
        "mask": data["mask"],
        "pattern_type": str(data["pattern_type"]),
        "vertical_size": int(data["vertical_size"]),
        "slash_size": int(data["slash_size"]),
        "sparsity_ratio": float(data["sparsity_ratio"]),
    }


def extract_pattern_info(mask: np.ndarray, vertical_size: int, slash_size: int) -> dict:
    """
    Extract vertical indices and effective slash window from mask.

    Returns:
        dict with 'vertical_indices' and 'slash_window'
    """
    seq_len = mask.shape[0]

    # Extract vertical columns: columns with high coverage (>50% of queries access them)
    col_coverage = mask.sum(axis=0) / seq_len
    # Use a dynamic threshold based on vertical_size
    if vertical_size > 0:
        # Sort by coverage and take top vertical_size
        sorted_indices = np.argsort(col_coverage)[::-1]
        vertical_indices = sorted(sorted_indices[:vertical_size].tolist())
    else:
        vertical_indices = []

    # Slash window is directly from the config
    slash_window = slash_size

    return {
        "vertical_indices": set(vertical_indices),
        "slash_window": slash_window,
    }


def compute_kv_requirements(
    head_info: dict,
    chunk_start: int,
    chunk_end: int,
    seq_len: int,
) -> set:
    """
    Compute the set of KV positions required for a head during a chunk.

    For query positions in [chunk_start, chunk_end):
      - Need vertical indices (always)
      - Need slash window: [max(0, q - slash_window), q] for each q in chunk
      - Combined slash range: [max(0, chunk_start - slash_window), chunk_end)

    Returns:
        Set of KV position indices needed
    """
    vertical = head_info["vertical_indices"]
    slash_window = head_info["slash_window"]

    # Slash range for this chunk (causal: only look at past)
    slash_start = max(0, chunk_start - slash_window)
    slash_end = chunk_end  # Causal: can see up to current position
    slash_positions = set(range(slash_start, slash_end))

    # Union of vertical and slash
    required = vertical | slash_positions

    return required


def compute_overlap_metrics(head_requirements: dict) -> dict:
    """
    Compute overlap metrics between different heads.

    Args:
        head_requirements: dict mapping head_id to set of required KV positions

    Returns:
        dict with various overlap metrics
    """
    heads = list(head_requirements.keys())
    num_heads = len(heads)

    if num_heads == 0:
        return {}

    # Union of all heads
    union_all = set()
    for req in head_requirements.values():
        union_all |= req

    # Intersection of all heads (positions needed by ALL heads)
    intersection_all = head_requirements[heads[0]].copy()
    for head in heads[1:]:
        intersection_all &= head_requirements[head]

    # Per-head sizes
    sizes = [len(head_requirements[h]) for h in heads]

    # Pairwise Jaccard similarity
    jaccard_matrix = np.zeros((num_heads, num_heads))
    for i, h1 in enumerate(heads):
        for j, h2 in enumerate(heads):
            if i == j:
                jaccard_matrix[i, j] = 1.0
            else:
                intersection = len(head_requirements[h1] & head_requirements[h2])
                union = len(head_requirements[h1] | head_requirements[h2])
                jaccard_matrix[i, j] = intersection / union if union > 0 else 0

    # Compute waste ratio if using union strategy
    # Waste = (union - actual_needed) / actual_needed for each head
    waste_ratios = []
    for h in heads:
        actual = len(head_requirements[h])
        waste = len(union_all) - actual
        waste_ratios.append(waste / actual if actual > 0 else 0)

    return {
        "num_heads": num_heads,
        "union_size": len(union_all),
        "intersection_size": len(intersection_all),
        "per_head_sizes": sizes,
        "avg_head_size": np.mean(sizes),
        "jaccard_matrix": jaccard_matrix,
        "avg_jaccard": np.mean(jaccard_matrix[np.triu_indices(num_heads, k=1)]),
        "waste_ratios": waste_ratios,
        "avg_waste_ratio": np.mean(waste_ratios),
        "union_vs_avg_ratio": len(union_all) / np.mean(sizes) if np.mean(sizes) > 0 else 0,
    }


def analyze_chunked_prefill(
    profile_dir: str,
    layer: int,
    chunk_size: int,
    output_path: str = None,
):
    """
    Analyze KV cache overlap during chunked prefill.
    """
    # Load all head profiles for this layer
    pattern = os.path.join(profile_dir, f"layer{layer}_head*.npz")
    files = sorted(glob.glob(pattern), key=lambda x: int(x.split("head")[-1].replace(".npz", "")))

    if not files:
        print(f"No profile files found for layer {layer} in {profile_dir}")
        return

    print(f"Found {len(files)} heads for layer {layer}")

    # Extract pattern info for each head
    head_infos = {}
    for file_path in files:
        head_idx = int(file_path.split("head")[-1].replace(".npz", ""))
        data = load_profile(file_path)
        info = extract_pattern_info(data["mask"], data["vertical_size"], data["slash_size"])
        info["pattern_type"] = data["pattern_type"]
        head_infos[head_idx] = info
        print(f"  Head {head_idx}: {data['pattern_type']}, vertical={len(info['vertical_indices'])}, window={info['slash_window']}")

    # Get sequence length from first file
    first_data = load_profile(files[0])
    seq_len = first_data["mask"].shape[0]
    num_chunks = (seq_len + chunk_size - 1) // chunk_size

    print(f"\nSequence length: {seq_len}, Chunk size: {chunk_size}, Num chunks: {num_chunks}")

    # Analyze each chunk
    chunk_metrics = []

    print("\n" + "="*80)
    print("Chunk-by-Chunk Analysis")
    print("="*80)

    for chunk_idx in range(num_chunks):
        chunk_start = chunk_idx * chunk_size
        chunk_end = min((chunk_idx + 1) * chunk_size, seq_len)

        # Compute KV requirements for each head
        head_requirements = {}
        for head_idx, info in head_infos.items():
            required = compute_kv_requirements(info, chunk_start, chunk_end, seq_len)
            head_requirements[head_idx] = required

        # Compute overlap metrics
        metrics = compute_overlap_metrics(head_requirements)
        metrics["chunk_idx"] = chunk_idx
        metrics["chunk_start"] = chunk_start
        metrics["chunk_end"] = chunk_end
        chunk_metrics.append(metrics)

        # Print summary for this chunk
        print(f"\nChunk {chunk_idx} [{chunk_start}:{chunk_end}]:")
        print(f"  Per-head avg KV needed: {metrics['avg_head_size']:.0f}")
        print(f"  Union (all heads):      {metrics['union_size']}")
        print(f"  Intersection (common):  {metrics['intersection_size']}")
        print(f"  Union/Avg ratio:        {metrics['union_vs_avg_ratio']:.2f}x")
        print(f"  Avg pairwise Jaccard:   {metrics['avg_jaccard']:.3f}")
        print(f"  Avg waste ratio:        {metrics['avg_waste_ratio']:.2f} ({metrics['avg_waste_ratio']*100:.1f}% extra)")

    # Summary statistics
    print("\n" + "="*80)
    print("Overall Summary")
    print("="*80)

    avg_union_ratio = np.mean([m['union_vs_avg_ratio'] for m in chunk_metrics])
    avg_jaccard = np.mean([m['avg_jaccard'] for m in chunk_metrics])
    avg_waste = np.mean([m['avg_waste_ratio'] for m in chunk_metrics])

    print(f"Average Union/PerHead ratio: {avg_union_ratio:.2f}x")
    print(f"Average Jaccard similarity:  {avg_jaccard:.3f}")
    print(f"Average waste ratio:         {avg_waste:.2f} ({avg_waste*100:.1f}% extra if using union)")

    # Analyze by window size groups
    print("\n" + "="*80)
    print("Head Grouping Analysis (by slash window size)")
    print("="*80)

    # Group heads by window size
    window_groups = defaultdict(list)
    buckets = [256, 512, 1024, 2048, 4096, 8192]
    for head_idx, info in head_infos.items():
        window = info['slash_window']
        bucket = min(b for b in buckets if window <= b)
        window_groups[bucket].append(head_idx)

    for bucket in sorted(window_groups.keys()):
        heads_in_group = window_groups[bucket]
        print(f"\nGroup (window <= {bucket}): {len(heads_in_group)} heads")
        print(f"  Heads: {heads_in_group}")

        # Compute group-specific metrics for last chunk (most representative)
        last_chunk = chunk_metrics[-1]
        group_requirements = {h: compute_kv_requirements(head_infos[h],
                             last_chunk['chunk_start'], last_chunk['chunk_end'], seq_len)
                             for h in heads_in_group}
        group_metrics = compute_overlap_metrics(group_requirements)

        if group_metrics:
            print(f"  Group union size:      {group_metrics['union_size']}")
            print(f"  Group avg per-head:    {group_metrics['avg_head_size']:.0f}")
            print(f"  Group union/avg ratio: {group_metrics['union_vs_avg_ratio']:.2f}x")
            print(f"  Group avg Jaccard:     {group_metrics['avg_jaccard']:.3f}")

    # Plot results
    if output_path:
        _plot_analysis(chunk_metrics, head_infos, output_path, layer, chunk_size)

    return chunk_metrics, head_infos


def _plot_analysis(chunk_metrics, head_infos, output_path, layer, chunk_size):
    """Generate visualization of the analysis."""

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 1. KV requirements over chunks
    ax = axes[0, 0]
    chunks = [m['chunk_idx'] for m in chunk_metrics]
    union_sizes = [m['union_size'] for m in chunk_metrics]
    avg_sizes = [m['avg_head_size'] for m in chunk_metrics]
    intersection_sizes = [m['intersection_size'] for m in chunk_metrics]

    ax.plot(chunks, union_sizes, 'r-', label='Union (all heads)', linewidth=2)
    ax.plot(chunks, avg_sizes, 'b-', label='Avg per-head', linewidth=2)
    ax.plot(chunks, intersection_sizes, 'g-', label='Intersection (common)', linewidth=2)
    ax.fill_between(chunks, avg_sizes, union_sizes, alpha=0.3, color='red', label='Waste if union')
    ax.set_xlabel('Chunk Index')
    ax.set_ylabel('Number of KV Positions')
    ax.set_title('KV Requirements per Chunk')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Union/Avg ratio over chunks
    ax = axes[0, 1]
    ratios = [m['union_vs_avg_ratio'] for m in chunk_metrics]
    ax.plot(chunks, ratios, 'purple', linewidth=2, marker='o')
    ax.axhline(y=1.0, color='gray', linestyle='--', label='Ideal (no waste)')
    ax.set_xlabel('Chunk Index')
    ax.set_ylabel('Union / Avg Ratio')
    ax.set_title('Overhead Ratio per Chunk')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Average Jaccard similarity over chunks
    ax = axes[0, 2]
    jaccards = [m['avg_jaccard'] for m in chunk_metrics]
    ax.plot(chunks, jaccards, 'green', linewidth=2, marker='s')
    ax.set_xlabel('Chunk Index')
    ax.set_ylabel('Average Pairwise Jaccard')
    ax.set_title('Head Overlap (Jaccard Similarity)')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    # 4. Jaccard matrix heatmap (last chunk)
    ax = axes[1, 0]
    last_metrics = chunk_metrics[-1]
    im = ax.imshow(last_metrics['jaccard_matrix'], cmap='YlOrRd', vmin=0, vmax=1)
    ax.set_xlabel('Head Index')
    ax.set_ylabel('Head Index')
    ax.set_title(f'Pairwise Jaccard (Last Chunk)')
    plt.colorbar(im, ax=ax)

    # 5. Per-head window sizes
    ax = axes[1, 1]
    head_ids = sorted(head_infos.keys())
    windows = [head_infos[h]['slash_window'] for h in head_ids]
    verticals = [len(head_infos[h]['vertical_indices']) for h in head_ids]

    x = np.arange(len(head_ids))
    width = 0.35
    ax.bar(x - width/2, windows, width, label='Slash Window', color='steelblue')
    ax.bar(x + width/2, verticals, width, label='Vertical Size', color='coral')
    ax.set_xlabel('Head Index')
    ax.set_ylabel('Size')
    ax.set_title('Per-Head Pattern Parameters')
    ax.set_xticks(x)
    ax.set_xticklabels(head_ids)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # 6. Waste distribution
    ax = axes[1, 2]
    waste_ratios = [m['avg_waste_ratio'] * 100 for m in chunk_metrics]
    ax.bar(chunks, waste_ratios, color='salmon', edgecolor='darkred')
    ax.set_xlabel('Chunk Index')
    ax.set_ylabel('Waste Ratio (%)')
    ax.set_title('Memory Waste if Using Union Strategy')
    ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle(f'Layer {layer} - KV Cache Overlap Analysis (Chunk Size = {chunk_size})',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved analysis plot to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze KV cache overlap during chunked prefill",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--profile_dir",
        type=str,
        required=True,
        help="Directory containing profile npz files",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=0,
        help="Layer index to analyze (default: 0)",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=1024,
        help="Chunk size for simulated chunked prefill (default: 1024)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for the analysis plot (default: overlap_analysis.png)",
    )

    args = parser.parse_args()

    output_path = args.output or f"./tools/profiles/overlap_analysis_chunk{args.chunk_size}.png"

    analyze_chunked_prefill(
        profile_dir=args.profile_dir,
        layer=args.layer,
        chunk_size=args.chunk_size,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()
