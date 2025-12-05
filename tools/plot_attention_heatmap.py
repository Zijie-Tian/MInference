# Copyright (c) 2024-2025 Microsoft
# Licensed under The MIT License [see LICENSE for details]

"""
Attention Pattern Visualization Tool

Visualizes attention weights and sparse masks from profile data.
Designed for headless server environments - outputs to files only.

Usage:
    # Plot single head
    python tools/plot_attention_heatmap.py --profile_dir ./profiles --layer 0 --head 0 --output head0.png

    # Plot all heads in a layer (2D grid)
    python tools/plot_attention_heatmap.py --profile_dir ./profiles --layer 0 --all_heads --output layer0_overview.png

    # Plot 3D visualization (head × query × key)
    python tools/plot_attention_heatmap.py --profile_dir ./profiles --layer 0 --plot_3d --output layer0_3d.html
    python tools/plot_attention_heatmap.py --profile_dir ./profiles --layer 0 --plot_3d --backend matplotlib --output layer0_3d.png

    # Plot statistics
    python tools/plot_attention_heatmap.py --profile_dir ./profiles --stats --output stats.png
"""

import argparse
import os
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless servers
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
    output_path: str,
    downsample: int = 1,
):
    """
    Plot attention weights and mask for a single head.

    Args:
        profile_path: Path to the npz profile file
        output_path: Path to save the figure
        downsample: Downsample factor for large matrices (e.g., 4 means 1/4 resolution)
    """
    data = load_profile(profile_path)
    attn = data["attn_weights"]
    mask = data["mask"]

    # Downsample if needed
    if downsample > 1:
        attn = attn[::downsample, ::downsample]
        mask = mask[::downsample, ::downsample]

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
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved to: {output_path}")


def plot_layer_overview(
    profile_dir: str,
    layer: int,
    output_path: str,
    downsample: int = 8,
):
    """
    Plot overview of all heads in a layer.

    Args:
        profile_dir: Directory containing profile npz files
        layer: Layer index
        output_path: Path to save the figure
        downsample: Downsample factor for large matrices
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

        ax.imshow(mask, cmap="Blues", aspect="auto", vmin=0, vmax=1)
        ax.set_title(f"H{head_idx}: {data['pattern_type'][:8]}\nv={data['vertical_size']}, s={data['slash_size']}\nscore={data['score']:.3f}, sp={data['sparsity_ratio']:.3f}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    # Hide empty subplots
    for idx in range(num_heads, rows * cols):
        row, col = idx // cols, idx % cols
        axes[row, col].axis("off")

    plt.suptitle(f"Layer {layer} - Sparse Masks Overview", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved to: {output_path}")


def plot_layer_3d(
    profile_dir: str,
    layer: int,
    output_path: str,
    downsample: int = 16,
    backend: str = "plotly",
    opacity: float = 0.6,
    colorscale: str = "Viridis",
):
    """
    Plot 3D visualization of attention masks across all heads in a layer.

    Creates a 3D heatmap where:
    - X axis: Key Position (kv_seq_len)
    - Y axis: Query Position (q_seq_len)
    - Z axis: Head Index

    Args:
        profile_dir: Directory containing profile npz files
        layer: Layer index
        output_path: Path to save the figure (.html for plotly, .png for matplotlib)
        downsample: Downsample factor for large matrices (default: 16)
        backend: "plotly" for interactive 3D, "matplotlib" for static 3D
        opacity: Opacity of the 3D points/voxels (0.0-1.0)
        colorscale: Color scale for the visualization
    """
    # Find all head files for this layer
    pattern = os.path.join(profile_dir, f"layer{layer}_head*.npz")
    files = sorted(glob.glob(pattern), key=lambda x: int(x.split("head")[-1].replace(".npz", "")))

    if not files:
        print(f"No profile files found for layer {layer} in {profile_dir}")
        return

    num_heads = len(files)
    print(f"Found {num_heads} heads for layer {layer}")

    # Load first file to get dimensions
    first_data = load_profile(files[0])
    seq_len = first_data["mask"].shape[0]
    downsampled_len = seq_len // downsample

    print(f"Original seq_len: {seq_len}, downsampled to: {downsampled_len}")

    # Stack all masks into 3D array [num_heads, q_seq_len, kv_seq_len]
    masks_3d = np.zeros((num_heads, downsampled_len, downsampled_len), dtype=np.float32)
    head_info = []

    for idx, file_path in enumerate(files):
        data = load_profile(file_path)
        mask = data["mask"]

        # Downsample using max pooling to preserve sparse structure
        if downsample > 1:
            # Use block max to preserve 1s in sparse mask
            h, w = mask.shape
            new_h, new_w = h // downsample, w // downsample
            mask = mask[:new_h * downsample, :new_w * downsample]
            mask = mask.reshape(new_h, downsample, new_w, downsample).max(axis=(1, 3))

        masks_3d[idx] = mask

        head_idx = int(file_path.split("head")[-1].replace(".npz", ""))
        head_info.append({
            "head_idx": head_idx,
            "pattern_type": data["pattern_type"],
            "score": data["score"],
            "sparsity": data["sparsity_ratio"],
        })

    if backend == "plotly":
        _plot_3d_plotly(
            masks_3d=masks_3d,
            layer=layer,
            head_info=head_info,
            output_path=output_path,
            opacity=opacity,
            colorscale=colorscale,
        )
    else:
        _plot_3d_matplotlib(
            masks_3d=masks_3d,
            layer=layer,
            head_info=head_info,
            output_path=output_path,
            opacity=opacity,
        )


def _plot_3d_plotly(
    masks_3d: np.ndarray,
    layer: int,
    head_info: list,
    output_path: str,
    opacity: float = 0.6,
    colorscale: str = "Viridis",
):
    """
    Create interactive 3D visualization using Plotly.
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("Plotly not installed. Install with: pip install plotly")
        print("Falling back to matplotlib...")
        _plot_3d_matplotlib(masks_3d, layer, head_info, output_path, opacity)
        return

    num_heads, q_len, k_len = masks_3d.shape
    print(f"Creating 3D visualization: {num_heads} heads × {q_len} queries × {k_len} keys")

    # Extract coordinates where mask == 1
    head_coords, q_coords, k_coords = np.where(masks_3d > 0.5)

    # Create color array based on head index
    colors = head_coords / max(num_heads - 1, 1)

    # Create hover text with head info
    hover_texts = []
    for h, q, k in zip(head_coords, q_coords, k_coords):
        info = head_info[h]
        hover_texts.append(
            f"Head {info['head_idx']}: {info['pattern_type']}<br>"
            f"Query: {q}, Key: {k}<br>"
            f"Score: {info['score']:.4f}, Sparsity: {info['sparsity']:.4f}"
        )

    # Create the figure
    fig = go.Figure()

    # Add scatter3d trace
    fig.add_trace(go.Scatter3d(
        x=k_coords,  # Key position
        y=q_coords,  # Query position
        z=head_coords,  # Head index
        mode='markers',
        marker=dict(
            size=2,
            color=colors,
            colorscale=colorscale,
            opacity=opacity,
            colorbar=dict(
                title="Head Index",
                tickvals=np.linspace(0, 1, min(num_heads, 10)),
                ticktext=[str(int(i)) for i in np.linspace(0, num_heads-1, min(num_heads, 10))],
            ),
        ),
        text=hover_texts,
        hoverinfo='text',
        name='Attention Mask',
    ))

    # Update layout
    fig.update_layout(
        title=dict(
            text=f"Layer {layer} - 3D Attention Mask Visualization<br>"
                 f"<sup>{num_heads} heads, {q_len}×{k_len} sequence (downsampled)</sup>",
            x=0.5,
        ),
        scene=dict(
            xaxis_title="Key Position",
            yaxis_title="Query Position",
            zaxis_title="Head Index",
            xaxis=dict(range=[0, k_len]),
            yaxis=dict(range=[0, q_len]),
            zaxis=dict(range=[0, num_heads]),
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=0.5),
        ),
        width=1000,
        height=800,
        margin=dict(l=0, r=0, t=50, b=0),
    )

    # Add per-head statistics as annotations
    pattern_summary = {}
    for info in head_info:
        pt = info['pattern_type']
        pattern_summary[pt] = pattern_summary.get(pt, 0) + 1

    pattern_text = ", ".join([f"{k}: {v}" for k, v in pattern_summary.items()])
    fig.add_annotation(
        text=f"Pattern distribution: {pattern_text}",
        xref="paper", yref="paper",
        x=0.5, y=-0.05,
        showarrow=False,
        font=dict(size=12),
    )

    # Save to file
    if output_path.endswith('.html'):
        fig.write_html(output_path)
    else:
        # For image formats, use kaleido
        try:
            fig.write_image(output_path, scale=2)
        except Exception as e:
            print(f"Failed to save as image ({e}). Saving as HTML instead.")
            html_path = output_path.rsplit('.', 1)[0] + '.html'
            fig.write_html(html_path)
            output_path = html_path
    print(f"Saved to: {output_path}")


def _plot_3d_matplotlib(
    masks_3d: np.ndarray,
    layer: int,
    head_info: list,
    output_path: str,
    opacity: float = 0.6,
):
    """
    Create static 3D voxel visualization using Matplotlib.
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    from matplotlib.colors import Normalize

    num_heads, q_len, k_len = masks_3d.shape
    print(f"Creating 3D voxel visualization: {num_heads} heads × {q_len} queries × {k_len} keys")

    # Create boolean array for voxels (swap axes for better visualization)
    # Shape: [k_len, q_len, num_heads] for voxels (x, y, z)
    voxels = masks_3d.transpose(2, 1, 0) > 0.5  # [k, q, h]

    # Create color array based on head index
    norm = Normalize(vmin=0, vmax=num_heads - 1)
    cmap = plt.get_cmap('viridis')

    # Create facecolors array with shape matching voxels
    facecolors = np.zeros(voxels.shape + (4,), dtype=np.float32)
    for h in range(num_heads):
        color = cmap(norm(h))
        # Set color with opacity for this head's layer
        facecolors[:, :, h, :] = (*color[:3], opacity)

    # Create figure
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Draw voxels
    ax.voxels(
        voxels,
        facecolors=facecolors,
        edgecolor='none',  # No edge for cleaner look with many voxels
    )

    # Set labels
    ax.set_xlabel('Key Position')
    ax.set_ylabel('Query Position')
    ax.set_zlabel('Head Index')

    # Set title
    ax.set_title(f"Layer {layer} - 3D Attention Mask (Voxel)\n({num_heads} heads × {q_len}×{k_len} sequence)")

    # Add colorbar manually
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label('Head Index')

    # Add pattern summary
    pattern_summary = {}
    for info in head_info:
        pt = info['pattern_type']
        pattern_summary[pt] = pattern_summary.get(pt, 0) + 1
    pattern_text = ", ".join([f"{k}: {v}" for k, v in pattern_summary.items()])
    fig.text(0.5, 0.02, f"Patterns: {pattern_text}", ha='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved to: {output_path}")


def plot_layer_3d_volume(
    profile_dir: str,
    layer: int,
    output_path: str,
    downsample: int = 16,
    threshold: float = 0.5,
):
    """
    Plot 3D volume rendering of attention masks using isosurfaces.

    This creates a more visually appealing 3D representation showing
    the "shape" of attention patterns across heads.

    Args:
        profile_dir: Directory containing profile npz files
        layer: Layer index
        output_path: Path to save the figure (.html)
        downsample: Downsample factor
        threshold: Isosurface threshold value
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("Plotly not installed. Install with: pip install plotly")
        return

    # Find all head files for this layer
    pattern = os.path.join(profile_dir, f"layer{layer}_head*.npz")
    files = sorted(glob.glob(pattern), key=lambda x: int(x.split("head")[-1].replace(".npz", "")))

    if not files:
        print(f"No profile files found for layer {layer} in {profile_dir}")
        return

    num_heads = len(files)
    print(f"Found {num_heads} heads for layer {layer}")

    # Load and stack masks
    first_data = load_profile(files[0])
    seq_len = first_data["mask"].shape[0]
    downsampled_len = seq_len // downsample

    print(f"Original seq_len: {seq_len}, downsampled to: {downsampled_len}")

    masks_3d = np.zeros((num_heads, downsampled_len, downsampled_len), dtype=np.float32)

    for idx, file_path in enumerate(files):
        data = load_profile(file_path)
        mask = data["mask"]

        if downsample > 1:
            h, w = mask.shape
            new_h, new_w = h // downsample, w // downsample
            mask = mask[:new_h * downsample, :new_w * downsample]
            mask = mask.reshape(new_h, downsample, new_w, downsample).max(axis=(1, 3))

        masks_3d[idx] = mask

    # Create meshgrid for coordinates
    Z, Y, X = np.mgrid[0:num_heads, 0:downsampled_len, 0:downsampled_len]

    # Create isosurface
    fig = go.Figure(data=go.Isosurface(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=masks_3d.flatten(),
        isomin=threshold,
        isomax=1.0,
        surface_count=3,
        colorscale='Viridis',
        caps=dict(x_show=False, y_show=False, z_show=False),
        opacity=0.6,
    ))

    fig.update_layout(
        title=f"Layer {layer} - 3D Attention Mask Volume",
        scene=dict(
            xaxis_title="Key Position",
            yaxis_title="Query Position",
            zaxis_title="Head Index",
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=0.5),
        ),
        width=1000,
        height=800,
    )

    # Ensure output is HTML for plotly
    if not output_path.endswith('.html'):
        output_path = output_path.rsplit('.', 1)[0] + '.html'
    fig.write_html(output_path)
    print(f"Saved to: {output_path}")


def plot_attention_statistics(
    profile_dir: str,
    output_path: str,
):
    """
    Plot statistics across all layers and heads.

    Args:
        profile_dir: Directory containing profile npz files
        output_path: Path to save the figure
    """
    # Find all profile files
    files = sorted(glob.glob(os.path.join(profile_dir, "layer*_head*.npz")))

    if not files:
        print(f"No profile files found in {profile_dir}")
        return

    # Extract data
    scores = []
    sparsities = []
    patterns = []

    for file_path in files:
        data = load_profile(file_path)
        scores.append(data["score"])
        sparsities.append(data["sparsity_ratio"])
        patterns.append(data["pattern_type"])

    print(f"Found {len(files)} profile files")

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
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize attention patterns from profile data (headless server mode)",
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
        help="Plot all heads overview for the specified layer (2D grid)",
    )
    parser.add_argument(
        "--plot_3d",
        action="store_true",
        help="Plot 3D visualization (head × query × key) for the specified layer",
    )
    parser.add_argument(
        "--plot_3d_volume",
        action="store_true",
        help="Plot 3D volume/isosurface visualization for the specified layer",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="plotly",
        choices=["plotly", "matplotlib"],
        help="Backend for 3D plotting: 'plotly' (interactive HTML) or 'matplotlib' (static PNG)",
    )
    parser.add_argument(
        "--opacity",
        type=float,
        default=0.6,
        help="Opacity for 3D visualization (0.0-1.0, default: 0.6)",
    )
    parser.add_argument(
        "--colorscale",
        type=str,
        default="Viridis",
        help="Color scale for 3D plotly visualization (default: Viridis)",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Plot statistics across all layers/heads",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for the figure (.html for plotly 3D, .png for others)",
    )
    parser.add_argument(
        "--downsample",
        type=int,
        default=1,
        help="Downsample factor for large matrices (default: 1, recommended 8-16 for 3D)",
    )

    args = parser.parse_args()

    if args.stats:
        # Plot statistics
        plot_attention_statistics(
            profile_dir=args.profile_dir,
            output_path=args.output,
        )
    elif args.layer is not None and args.plot_3d:
        # Plot 3D visualization
        downsample = args.downsample if args.downsample > 1 else 16  # Default to 16 for 3D
        plot_layer_3d(
            profile_dir=args.profile_dir,
            layer=args.layer,
            output_path=args.output,
            downsample=downsample,
            backend=args.backend,
            opacity=args.opacity,
            colorscale=args.colorscale,
        )
    elif args.layer is not None and args.plot_3d_volume:
        # Plot 3D volume visualization
        downsample = args.downsample if args.downsample > 1 else 16
        plot_layer_3d_volume(
            profile_dir=args.profile_dir,
            layer=args.layer,
            output_path=args.output,
            downsample=downsample,
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
        )
    elif args.layer is not None and args.all_heads:
        # Plot layer overview (2D grid)
        plot_layer_overview(
            profile_dir=args.profile_dir,
            layer=args.layer,
            output_path=args.output,
            downsample=args.downsample,
        )
    else:
        parser.print_help()
        print("\nExamples:")
        print("  # Single head visualization")
        print("  python tools/plot_attention_heatmap.py --profile_dir ./profiles --layer 0 --head 0 --output head0.png")
        print("")
        print("  # 2D grid overview of all heads")
        print("  python tools/plot_attention_heatmap.py --profile_dir ./profiles --layer 0 --all_heads --output layer0_overview.png")
        print("")
        print("  # 3D visualization (plotly HTML)")
        print("  python tools/plot_attention_heatmap.py --profile_dir ./profiles --layer 0 --plot_3d --output layer0_3d.html")
        print("")
        print("  # 3D visualization (matplotlib PNG)")
        print("  python tools/plot_attention_heatmap.py --profile_dir ./profiles --layer 0 --plot_3d --backend matplotlib --output layer0_3d.png")
        print("")
        print("  # 3D volume/isosurface visualization")
        print("  python tools/plot_attention_heatmap.py --profile_dir ./profiles --layer 0 --plot_3d_volume --output layer0_volume.html")
        print("")
        print("  # Statistics across all layers/heads")
        print("  python tools/plot_attention_heatmap.py --profile_dir ./profiles --stats --output stats.png")


if __name__ == "__main__":
    main()
