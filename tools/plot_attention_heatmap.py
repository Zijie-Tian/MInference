# Copyright (c) 2024-2025 Microsoft
# Licensed under The MIT License [see LICENSE for details]

"""
Attention Pattern Visualization Tool

Visualizes attention weights and sparse masks from profile data.

Usage:
    # Plot single head
    python tools/plot_attention_heatmap.py --profile_dir ./tools/profiles --layer 0 --head 0

    # Plot all heads in a layer
    python tools/plot_attention_heatmap.py --profile_dir ./tools/profiles --layer 0 --all_heads

    # Plot with custom output
    python tools/plot_attention_heatmap.py --profile_dir ./tools/profiles --layer 0 --head 0 --output ./my_plot.png
"""

import argparse
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def load_profile(profile_path: str) -> dict:
    """Load profile data from npz file."""
    data = np.load(profile_path, allow_pickle=True)
    return {
        "attn_weights": data["attn_weights"],
        "mask": data["mask"],
        "pattern_type": str(data["pattern_type"]),
        "vertical_size": int(data["vertical_size"]),
        "slash_size": int(data["slash_size"]),
        "score": float(data["score"]),
        "sparsity_ratio": float(data["sparsity_ratio"]),
    }


def plot_single_head(
    profile_path: str,
    output_path: str = None,
    downsample: int = 1,
    show: bool = True,
):
    """
    Plot attention weights and mask for a single head.

    Args:
        profile_path: Path to the npz profile file
        output_path: Path to save the figure (optional)
        downsample: Downsample factor for large matrices (e.g., 4 means 1/4 resolution)
        show: Whether to display the plot
    """
    data = load_profile(profile_path)
    attn = data["attn_weights"]
    mask = data["mask"]

    # Downsample if needed
    if downsample > 1:
        attn = attn[::downsample, ::downsample]
        mask = mask[::downsample, ::downsample]

    seq_len = attn.shape[0]

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Extract layer and head from filename
    filename = os.path.basename(profile_path)
    layer_head = filename.replace(".npz", "").replace("layer", "L").replace("_head", "H")

    # 1. Full attention weights (log scale for better visibility)
    im1 = axes[0].imshow(
        attn + 1e-10,  # Add small value to avoid log(0)
        cmap="viridis",
        norm=LogNorm(vmin=1e-6, vmax=1),
        aspect="auto",
    )
    axes[0].set_title(f"{layer_head} Attention Weights (log scale)")
    axes[0].set_xlabel("Key Position")
    axes[0].set_ylabel("Query Position")
    plt.colorbar(im1, ax=axes[0], shrink=0.8)

    # 2. Sparse mask
    im2 = axes[1].imshow(
        mask,
        cmap="Blues",
        aspect="auto",
        vmin=0,
        vmax=1,
    )
    axes[1].set_title(f"{layer_head} Sparse Mask\n({data['pattern_type']}, v={data['vertical_size']}, s={data['slash_size']})")
    axes[1].set_xlabel("Key Position")
    axes[1].set_ylabel("Query Position")
    plt.colorbar(im2, ax=axes[1], shrink=0.8)

    # 3. Masked attention (attention * mask)
    masked_attn = attn * mask
    im3 = axes[2].imshow(
        masked_attn + 1e-10,
        cmap="viridis",
        norm=LogNorm(vmin=1e-6, vmax=1),
        aspect="auto",
    )
    axes[2].set_title(f"{layer_head} Masked Attention\n(score={data['score']:.4f}, sparsity={data['sparsity_ratio']:.4f})")
    axes[2].set_xlabel("Key Position")
    axes[2].set_ylabel("Query Position")
    plt.colorbar(im3, ax=axes[2], shrink=0.8)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved to: {output_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_layer_overview(
    profile_dir: str,
    layer: int,
    output_path: str = None,
    downsample: int = 8,
    show: bool = True,
):
    """
    Plot overview of all heads in a layer.

    Args:
        profile_dir: Directory containing profile npz files
        layer: Layer index
        output_path: Path to save the figure (optional)
        downsample: Downsample factor for large matrices
        show: Whether to display the plot
    """
    # Find all head files for this layer
    pattern = os.path.join(profile_dir, f"layer{layer}_head*.npz")
    files = sorted(glob.glob(pattern), key=lambda x: int(x.split("head")[-1].replace(".npz", "")))

    if not files:
        print(f"No profile files found for layer {layer} in {profile_dir}")
        return

    num_heads = len(files)
    print(f"Found {num_heads} heads for layer {layer}")

    # Determine grid layout
    cols = min(4, num_heads)
    rows = (num_heads + cols - 1) // cols

    # Create figure for masks
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    if num_heads == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)

    for idx, file_path in enumerate(files):
        row, col = idx // cols, idx % cols
        ax = axes[row, col]

        data = load_profile(file_path)
        mask = data["mask"]

        # Downsample
        if downsample > 1:
            mask = mask[::downsample, ::downsample]

        head_idx = int(file_path.split("head")[-1].replace(".npz", ""))

        im = ax.imshow(mask, cmap="Blues", aspect="auto", vmin=0, vmax=1)
        ax.set_title(f"H{head_idx}: {data['pattern_type'][:8]}\nv={data['vertical_size']}, s={data['slash_size']}\nscore={data['score']:.3f}, sp={data['sparsity_ratio']:.3f}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    # Hide empty subplots
    for idx in range(num_heads, rows * cols):
        row, col = idx // cols, idx % cols
        axes[row, col].axis("off")

    plt.suptitle(f"Layer {layer} - Sparse Masks Overview", fontsize=14)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved to: {output_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_attention_statistics(
    profile_dir: str,
    output_path: str = None,
    show: bool = True,
):
    """
    Plot statistics across all layers and heads.

    Args:
        profile_dir: Directory containing profile npz files
        output_path: Path to save the figure (optional)
        show: Whether to display the plot
    """
    # Find all profile files
    files = sorted(glob.glob(os.path.join(profile_dir, "layer*_head*.npz")))

    if not files:
        print(f"No profile files found in {profile_dir}")
        return

    # Extract data
    layers = []
    heads = []
    scores = []
    sparsities = []
    patterns = []

    for file_path in files:
        filename = os.path.basename(file_path)
        parts = filename.replace(".npz", "").split("_")
        layer = int(parts[0].replace("layer", ""))
        head = int(parts[1].replace("head", ""))

        data = load_profile(file_path)

        layers.append(layer)
        heads.append(head)
        scores.append(data["score"])
        sparsities.append(data["sparsity_ratio"])
        patterns.append(data["pattern_type"])

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Score distribution
    axes[0, 0].hist(scores, bins=20, edgecolor="black", alpha=0.7)
    axes[0, 0].set_xlabel("Score")
    axes[0, 0].set_ylabel("Count")
    axes[0, 0].set_title(f"Score Distribution (mean={np.mean(scores):.4f})")
    axes[0, 0].axvline(np.mean(scores), color="red", linestyle="--", label=f"Mean: {np.mean(scores):.4f}")
    axes[0, 0].legend()

    # 2. Sparsity distribution
    axes[0, 1].hist(sparsities, bins=20, edgecolor="black", alpha=0.7, color="orange")
    axes[0, 1].set_xlabel("Sparsity Ratio")
    axes[0, 1].set_ylabel("Count")
    axes[0, 1].set_title(f"Sparsity Distribution (mean={np.mean(sparsities):.4f})")
    axes[0, 1].axvline(np.mean(sparsities), color="red", linestyle="--", label=f"Mean: {np.mean(sparsities):.4f}")
    axes[0, 1].legend()

    # 3. Score vs Sparsity scatter
    axes[1, 0].scatter(sparsities, scores, alpha=0.6)
    axes[1, 0].set_xlabel("Sparsity Ratio")
    axes[1, 0].set_ylabel("Score")
    axes[1, 0].set_title("Score vs Sparsity")

    # 4. Pattern type distribution
    pattern_counts = {}
    for p in patterns:
        pattern_counts[p] = pattern_counts.get(p, 0) + 1
    axes[1, 1].bar(pattern_counts.keys(), pattern_counts.values(), edgecolor="black")
    axes[1, 1].set_xlabel("Pattern Type")
    axes[1, 1].set_ylabel("Count")
    axes[1, 1].set_title("Pattern Type Distribution")
    axes[1, 1].tick_params(axis="x", rotation=15)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved to: {output_path}")

    if show:
        plt.show()
    else:
        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize attention patterns from profile data",
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
        default=None,
        help="Layer index to visualize",
    )
    parser.add_argument(
        "--head",
        type=int,
        default=None,
        help="Head index to visualize (requires --layer)",
    )
    parser.add_argument(
        "--all_heads",
        action="store_true",
        help="Plot all heads overview for the specified layer",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Plot statistics across all layers/heads",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for the figure",
    )
    parser.add_argument(
        "--downsample",
        type=int,
        default=1,
        help="Downsample factor for large matrices (default: 1)",
    )
    parser.add_argument(
        "--no_show",
        action="store_true",
        help="Don't display the plot (just save)",
    )

    args = parser.parse_args()

    if args.stats:
        # Plot statistics
        plot_attention_statistics(
            profile_dir=args.profile_dir,
            output_path=args.output,
            show=not args.no_show,
        )
    elif args.layer is not None and args.head is not None:
        # Plot single head
        profile_path = os.path.join(args.profile_dir, f"layer{args.layer}_head{args.head}.npz")
        if not os.path.exists(profile_path):
            print(f"Profile file not found: {profile_path}")
            return
        plot_single_head(
            profile_path=profile_path,
            output_path=args.output,
            downsample=args.downsample,
            show=not args.no_show,
        )
    elif args.layer is not None and args.all_heads:
        # Plot layer overview
        plot_layer_overview(
            profile_dir=args.profile_dir,
            layer=args.layer,
            output_path=args.output,
            downsample=args.downsample,
            show=not args.no_show,
        )
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python tools/plot_attention_heatmap.py --profile_dir ./tools/profiles --layer 0 --head 0")
        print("  python tools/plot_attention_heatmap.py --profile_dir ./tools/profiles --layer 0 --all_heads")
        print("  python tools/plot_attention_heatmap.py --profile_dir ./tools/profiles --stats")


if __name__ == "__main__":
    main()
