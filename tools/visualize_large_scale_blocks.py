# Copyright (c) 2024-2025 Microsoft
# Licensed under The MIT License [see LICENSE for details]

"""
Visualization of MInference sparse attention mechanisms.

This script provides comprehensive visualizations for:
1. Block-level computation patterns
2. Slash diagonal extraction process (sum_all_diagonal_matrix)
3. Vertical column selection
4. Block size comparisons
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import torch


def sum_all_diagonal_matrix_numpy(mat: np.ndarray) -> np.ndarray:
    """
    NumPy implementation of sum_all_diagonal_matrix.

    Computes the sum of attention weights along each diagonal.
    Diagonal d = query_pos - key_pos.

    Args:
        mat: Attention matrix [n, m] or [b, h, n, m]

    Returns:
        Diagonal sums [n + m - 1] or [b, h, n + m - 1]
    """
    if mat.ndim == 2:
        n, m = mat.shape
        # Pad matrix
        zero_mat = np.zeros((n, n))
        mat_padded = np.concatenate([zero_mat, mat, zero_mat], axis=-1)

        # Use stride tricks to align diagonals
        # This is equivalent to the PyTorch as_strided operation
        result = np.zeros(n + m)
        for d in range(n + m):
            # Diagonal d corresponds to positions where query - key = (n - 1) - d
            diag_sum = 0
            for q in range(n):
                k = q - ((n - 1) - d)
                if 0 <= k < m:
                    diag_sum += mat[q, k]
            result[d] = diag_sum
        return result[1:]  # Skip first element to match PyTorch implementation
    else:
        raise NotImplementedError("Only 2D matrices supported in this visualization")


def visualize_slash_extraction_process(
    seq_len: int = 256,
    last_q: int = 64,
    slash_size: int = 100,
    save_path: str = None,
):
    """
    Visualize how slash diagonals are extracted from probe attention.

    This demonstrates the complete process:
    1. Compute probe attention (last 64 queries)
    2. Apply sum_all_diagonal_matrix
    3. Select top-k diagonals
    """

    np.random.seed(42)

    # Simulate attention scores (last_q queries attending to all keys)
    # In reality, this comes from: softmax(Q[-64:] @ K.T / sqrt(d))

    # Create realistic attention pattern:
    # - Strong attention to recent tokens (local context)
    # - Some attention to BOS/important tokens
    # - Decaying attention to distant tokens

    probe_attention = np.zeros((last_q, seq_len))

    for q_idx in range(last_q):
        q_pos = seq_len - last_q + q_idx  # Actual query position

        for k_pos in range(q_pos + 1):  # Causal: only attend to past
            distance = q_pos - k_pos

            # Local attention (recent tokens) - strong
            if distance < 100:
                probe_attention[q_idx, k_pos] = np.exp(-distance / 30) + np.random.uniform(0, 0.1)
            # BOS token - always important
            elif k_pos < 5:
                probe_attention[q_idx, k_pos] = 0.3 + np.random.uniform(0, 0.1)
            # Some random important positions
            elif k_pos in [20, 50, 80, 120]:
                probe_attention[q_idx, k_pos] = 0.2 + np.random.uniform(0, 0.1)
            # Distant tokens - weak
            else:
                probe_attention[q_idx, k_pos] = np.random.uniform(0, 0.05)

    # Normalize each row (softmax-like)
    for q_idx in range(last_q):
        row_sum = probe_attention[q_idx, :].sum()
        if row_sum > 0:
            probe_attention[q_idx, :] /= row_sum

    # =========================================================================
    # Step 1: Compute diagonal sums
    # =========================================================================

    # Manual diagonal sum computation for visualization
    diagonal_sums = np.zeros(seq_len)
    for d in range(seq_len):
        # Diagonal d: positions where query_pos - key_pos = d
        diag_sum = 0
        count = 0
        for q_idx in range(last_q):
            q_pos = seq_len - last_q + q_idx
            k_pos = q_pos - d
            if 0 <= k_pos < seq_len:
                diag_sum += probe_attention[q_idx, k_pos]
                count += 1
        diagonal_sums[d] = diag_sum

    # =========================================================================
    # Step 2: Select top-k diagonals
    # =========================================================================

    # Force keep last 30 diagonals (recent context)
    diagonal_sums_for_selection = diagonal_sums.copy()
    diagonal_sums_for_selection[-30:] = np.inf

    # Select top-k
    top_k_indices = np.argsort(diagonal_sums_for_selection)[-slash_size:]
    top_k_indices = np.sort(top_k_indices)[::-1]  # Descending order

    # =========================================================================
    # Create visualization
    # =========================================================================

    fig = plt.figure(figsize=(18, 14))

    # Layout: 2x3 grid
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 0.8], hspace=0.3, wspace=0.3)

    # =========================================================================
    # Plot 1: Probe Attention Matrix
    # =========================================================================
    ax1 = fig.add_subplot(gs[0, 0])

    im1 = ax1.imshow(probe_attention, cmap='YlOrRd', aspect='auto', origin='upper')
    ax1.set_xlabel('Key Position', fontsize=10)
    ax1.set_ylabel('Probe Query Index (0-63)', fontsize=10)
    ax1.set_title('Step 1: Probe Attention\n(Last 64 queries)', fontsize=12, fontweight='bold')
    plt.colorbar(im1, ax=ax1, label='Attention Weight')

    # Draw diagonal lines to show what we're summing
    for d in [0, 30, 60, 100]:
        # Draw line for diagonal d
        q_start, q_end = 0, last_q - 1
        k_start = (seq_len - last_q) - d
        k_end = (seq_len - 1) - d
        if k_start >= 0 and k_end >= 0:
            ax1.plot([k_start, k_end], [q_start, q_end], 'b-', linewidth=1, alpha=0.5)

    # =========================================================================
    # Plot 2: Diagonal Sum Visualization
    # =========================================================================
    ax2 = fig.add_subplot(gs[0, 1])

    # Create a matrix showing diagonal assignments
    diag_viz = np.zeros((last_q, seq_len))
    for q_idx in range(last_q):
        q_pos = seq_len - last_q + q_idx
        for k_pos in range(min(q_pos + 1, seq_len)):
            d = q_pos - k_pos
            if d < seq_len:
                diag_viz[q_idx, k_pos] = d

    im2 = ax2.imshow(diag_viz, cmap='viridis', aspect='auto', origin='upper')
    ax2.set_xlabel('Key Position', fontsize=10)
    ax2.set_ylabel('Probe Query Index', fontsize=10)
    ax2.set_title('Step 2: Diagonal Index (d = q - k)\nSame color = same diagonal', fontsize=12, fontweight='bold')
    plt.colorbar(im2, ax=ax2, label='Diagonal Index d')

    # =========================================================================
    # Plot 3: Diagonal Sums Bar Chart
    # =========================================================================
    ax3 = fig.add_subplot(gs[0, 2])

    colors = ['red' if i in top_k_indices else 'lightgray' for i in range(len(diagonal_sums))]
    ax3.bar(range(len(diagonal_sums)), diagonal_sums, color=colors, width=1.0, edgecolor='none')
    ax3.set_xlabel('Diagonal Index d', fontsize=10)
    ax3.set_ylabel('Sum of Attention Weights', fontsize=10)
    ax3.set_title(f'Step 3: sum_all_diagonal_matrix Result\nRed = Selected Top-{slash_size}', fontsize=12, fontweight='bold')
    ax3.set_xlim(0, seq_len)

    # Mark the forced region
    ax3.axvspan(seq_len - 30, seq_len, alpha=0.3, color='green', label='Forced keep (last 30)')
    ax3.legend(loc='upper right')

    # =========================================================================
    # Plot 4: Selected Diagonals on Attention Matrix
    # =========================================================================
    ax4 = fig.add_subplot(gs[1, 0])

    # Create full attention matrix showing selected diagonals
    full_attn_viz = np.zeros((seq_len, seq_len))

    # Mark causal region
    for q in range(seq_len):
        for k in range(q + 1):
            full_attn_viz[q, k] = 0.1

    # Mark selected diagonals
    for d in top_k_indices:
        for q in range(seq_len):
            k = q - d
            if 0 <= k <= q:
                full_attn_viz[q, k] = 1.0

    im4 = ax4.imshow(full_attn_viz, cmap='YlOrRd', aspect='equal', origin='upper')
    ax4.set_xlabel('Key Position', fontsize=10)
    ax4.set_ylabel('Query Position', fontsize=10)
    ax4.set_title(f'Step 4: Selected Slash Diagonals\n({slash_size} diagonals selected)', fontsize=12, fontweight='bold')

    # =========================================================================
    # Plot 5: Zoom into a query block showing diagonal pattern
    # =========================================================================
    ax5 = fig.add_subplot(gs[1, 1])

    block_size = 64
    query_block = (seq_len // block_size) - 1  # Second to last block
    q_start = query_block * block_size
    q_end = q_start + block_size

    block_viz = np.zeros((block_size, seq_len))

    # Mark computed (selected diagonals within this block)
    for d in top_k_indices:
        for q_off in range(block_size):
            q = q_start + q_off
            k = q - d
            if 0 <= k <= q:
                block_viz[q_off, k] = 1.0

    im5 = ax5.imshow(block_viz, cmap='YlOrRd', aspect='auto', origin='upper')
    ax5.set_xlabel('Key Position', fontsize=10)
    ax5.set_ylabel(f'Query Offset (Block {query_block})', fontsize=10)
    ax5.set_title(f'Query Block {query_block} [{q_start}-{q_end-1}]\nExact Diagonal Positions', fontsize=12, fontweight='bold')

    # Add block grid
    for i in range(0, seq_len + 1, block_size):
        ax5.axvline(x=i - 0.5, color='blue', linewidth=0.5, alpha=0.5)

    # =========================================================================
    # Plot 6: Block-level computation
    # =========================================================================
    ax6 = fig.add_subplot(gs[1, 2])

    # Determine which blocks are computed
    computed_blocks = set()
    for d in top_k_indices:
        k_start = max(0, q_start - d)
        k_end = max(0, q_end - 1 - d)
        if k_end >= 0:
            for b in range(k_start // block_size, k_end // block_size + 1):
                computed_blocks.add(b)

    block_level_viz = np.zeros((block_size, seq_len))

    # Mark block regions
    for b in computed_blocks:
        b_start = b * block_size
        b_end = min((b + 1) * block_size, seq_len)
        for q_off in range(block_size):
            q = q_start + q_off
            for k in range(b_start, b_end):
                if k <= q:
                    block_level_viz[q_off, k] = 0.4

    # Overlay exact diagonals
    for d in top_k_indices:
        for q_off in range(block_size):
            q = q_start + q_off
            k = q - d
            if 0 <= k <= q:
                block_level_viz[q_off, k] = 1.0

    im6 = ax6.imshow(block_level_viz, cmap='YlOrRd', aspect='auto', origin='upper')
    ax6.set_xlabel('Key Position', fontsize=10)
    ax6.set_ylabel(f'Query Offset (Block {query_block})', fontsize=10)
    ax6.set_title(f'Block-Level Computation\nOrange=Block fill, Red=Diagonal', fontsize=12, fontweight='bold')

    for i in range(0, seq_len + 1, block_size):
        ax6.axvline(x=i - 0.5, color='blue', linewidth=0.5, alpha=0.5)

    # Highlight computed blocks
    for b in sorted(computed_blocks):
        rect = patches.Rectangle((b * block_size - 0.5, -0.5), block_size, block_size,
                                 linewidth=2, edgecolor='green', facecolor='none')
        ax6.add_patch(rect)

    # =========================================================================
    # Plot 7: Process Flow Diagram
    # =========================================================================
    ax7 = fig.add_subplot(gs[2, :])
    ax7.axis('off')

    process_text = """
╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
║                                    SLASH DIAGONAL EXTRACTION PROCESS                                                   ║
╠══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                                                        ║
║   ┌─────────────────┐      ┌─────────────────────┐      ┌──────────────────┐      ┌─────────────────────┐            ║
║   │  Q[-64:] @ K.T  │ ──── │  sum_all_diagonal   │ ──── │   topk(slash,k)  │ ──── │ convert_v_s_indexes │            ║
║   │   / sqrt(d)     │      │     (as_strided)    │      │                  │      │   (CUDA kernel)     │            ║
║   │   + softmax     │      │                     │      │                  │      │                     │            ║
║   └─────────────────┘      └─────────────────────┘      └──────────────────┘      └─────────────────────┘            ║
║          ↓                         ↓                           ↓                          ↓                           ║
║   Probe Attention           Diagonal Sums              Selected Diagonal          Block Offsets for                   ║
║   [64 × seq_len]            [seq_len]                  Indices [slash_size]        Triton Kernel                      ║
║                                                                                                                        ║
╠══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                                                        ║
║   KEY INSIGHT: Diagonals capture LOCAL CONTEXT patterns                                                                ║
║   ─────────────────────────────────────────────────────                                                                ║
║   • d=0: Self-attention (query attends to itself)                                                                      ║
║   • d=1: Previous token                                                                                                ║
║   • d=k: Token k positions ago                                                                                         ║
║                                                                                                                        ║
║   Most attention concentrates on recent tokens (small d), so slash captures local context efficiently.                 ║
║                                                                                                                        ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
"""

    ax7.text(0.5, 0.5, process_text, transform=ax7.transAxes,
             fontsize=9, fontfamily='monospace',
             verticalalignment='center', horizontalalignment='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.suptitle('MInference: Slash Diagonal Extraction & Computation', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.show()

    # Print statistics
    print("\n" + "="*60)
    print("SLASH EXTRACTION STATISTICS")
    print("="*60)
    print(f"Sequence length: {seq_len}")
    print(f"Probe queries: {last_q}")
    print(f"Selected diagonals: {slash_size}")
    print(f"Computed blocks for query block {query_block}: {len(computed_blocks)}")
    print(f"Selected diagonal indices (top 20): {sorted(top_k_indices)[:20]}...")

    return top_k_indices


def visualize_sum_all_diagonal_matrix(
    save_path: str = None,
):
    """
    Detailed visualization of how sum_all_diagonal_matrix works with as_strided.
    """

    # Small example for clear visualization
    n, m = 6, 6

    # Create sample attention matrix
    np.random.seed(42)
    attn = np.zeros((n, m))
    for i in range(n):
        for j in range(min(i + 1, m)):
            attn[i, j] = np.round(np.random.uniform(0.1, 1.0), 2)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # =========================================================================
    # Plot 1: Original attention matrix
    # =========================================================================
    ax1 = axes[0, 0]
    im1 = ax1.imshow(attn, cmap='YlOrRd', aspect='equal')
    ax1.set_title('Original Attention Matrix\n(Causal)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Key Position')
    ax1.set_ylabel('Query Position')

    # Add value annotations
    for i in range(n):
        for j in range(m):
            if attn[i, j] > 0:
                ax1.text(j, i, f'{attn[i,j]:.2f}', ha='center', va='center', fontsize=8)

    plt.colorbar(im1, ax=ax1)

    # =========================================================================
    # Plot 2: Show diagonal indices
    # =========================================================================
    ax2 = axes[0, 1]
    diag_indices = np.full((n, m), -1.0)
    for i in range(n):
        for j in range(min(i + 1, m)):
            diag_indices[i, j] = i - j

    im2 = ax2.imshow(diag_indices, cmap='tab10', aspect='equal', vmin=-1, vmax=n-1)
    ax2.set_title('Diagonal Index (d = query - key)\nSame color = same diagonal', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Key Position')
    ax2.set_ylabel('Query Position')

    for i in range(n):
        for j in range(m):
            if diag_indices[i, j] >= 0:
                ax2.text(j, i, f'd={int(diag_indices[i,j])}', ha='center', va='center', fontsize=8)

    plt.colorbar(im2, ax=ax2, label='Diagonal d')

    # =========================================================================
    # Plot 3: Padded matrix (for as_strided)
    # =========================================================================
    ax3 = axes[0, 2]

    zero_pad = np.zeros((n, n))
    attn_padded = np.concatenate([zero_pad, attn, zero_pad], axis=1)

    im3 = ax3.imshow(attn_padded, cmap='YlOrRd', aspect='auto')
    ax3.set_title('Padded Matrix\n(zeros on both sides)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Padded Key Position')
    ax3.set_ylabel('Query Position')

    # Mark original region
    ax3.axvline(x=n - 0.5, color='blue', linewidth=2, linestyle='--')
    ax3.axvline(x=n + m - 0.5, color='blue', linewidth=2, linestyle='--')
    ax3.text(n + m/2, -0.8, 'Original', ha='center', fontsize=10, color='blue')

    plt.colorbar(im3, ax=ax3)

    # =========================================================================
    # Plot 4: as_strided view (conceptual)
    # =========================================================================
    ax4 = axes[1, 0]

    # Create strided view visualization
    # The stride trick aligns diagonals to columns
    strided_view = np.zeros((n, n + m))
    for i in range(n):
        for d in range(n + m):
            # In strided view, column d contains diagonal (n-1-d) from original
            orig_d = (n - 1) - d
            j = i - orig_d
            if 0 <= j < m and j <= i:
                strided_view[i, d] = attn[i, j]

    im4 = ax4.imshow(strided_view, cmap='YlOrRd', aspect='auto')
    ax4.set_title('as_strided View\n(Diagonals aligned to columns)', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Column = Diagonal d (reversed)')
    ax4.set_ylabel('Query Position')

    # Add column labels
    for d in range(min(n + m, 10)):
        ax4.text(d, -0.8, f'd={n-1-d}', ha='center', fontsize=8, rotation=45)

    plt.colorbar(im4, ax=ax4)

    # =========================================================================
    # Plot 5: Column sums = Diagonal sums
    # =========================================================================
    ax5 = axes[1, 1]

    # Compute diagonal sums
    diag_sums = np.zeros(n + m - 1)
    for d in range(n):
        for i in range(n):
            j = i - d
            if 0 <= j < m:
                diag_sums[d] += attn[i, j]

    bars = ax5.bar(range(len(diag_sums)), diag_sums, color='coral', edgecolor='black')
    ax5.set_xlabel('Diagonal Index d', fontsize=10)
    ax5.set_ylabel('Sum of Attention Weights', fontsize=10)
    ax5.set_title('sum_all_diagonal_matrix Result\n(Sum along each diagonal)', fontsize=12, fontweight='bold')

    # Annotate bars
    for i, v in enumerate(diag_sums):
        if v > 0:
            ax5.text(i, v + 0.02, f'{v:.2f}', ha='center', fontsize=8)

    # =========================================================================
    # Plot 6: Formula and explanation
    # =========================================================================
    ax6 = axes[1, 2]
    ax6.axis('off')

    explanation = """
    sum_all_diagonal_matrix Implementation:
    ═══════════════════════════════════════

    Input: Attention matrix A[n, m]

    Step 1: Pad with zeros
    ──────────────────────
    A_padded = [0_{n×n} | A | 0_{n×n}]
    Shape: [n, 2n + m]

    Step 2: as_strided trick
    ────────────────────────
    stride = (2n + m + 1, 1)

    This makes each row "shift" by one,
    aligning diagonals to columns:

    Original:        Strided view:
    [a b c]         [. . a b c .]
    [d e f]    →    [. d e f . .]
    [g h i]         [g h i . . .]

    Step 3: Sum columns
    ───────────────────
    result[d] = sum(strided[:, d])
              = sum of diagonal d in original

    ═══════════════════════════════════════
    Why this works:
    • d=0 (main diagonal): self-attention
    • d=1: previous token attention
    • d=k: attention to k-th previous token

    Large sum → important diagonal pattern
    """

    ax6.text(0.05, 0.95, explanation, transform=ax6.transAxes,
             fontsize=10, fontfamily='monospace',
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.suptitle('sum_all_diagonal_matrix: How It Works', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.show()


def simulate_convert_vertical_slash_indexes(
    seq_len: int,
    block_size: int,
    slash_indices: np.ndarray,
    query_block_idx: int,
):
    """
    Simulate the CUDA kernel convert_vertical_slash_indexes logic.
    """
    q_start = query_block_idx * block_size
    q_end = min((query_block_idx + 1) * block_size, seq_len)

    computed_blocks = set()

    for d in slash_indices:
        k_start = q_start - d
        k_end = q_end - 1 - d
        k_start = max(0, k_start)
        k_end = max(0, k_end)

        if k_end >= 0 and k_start < q_end:
            block_start = k_start // block_size
            block_end = k_end // block_size
            for b in range(block_start, block_end + 1):
                if b * block_size < q_end:
                    computed_blocks.add(b)

    return sorted(computed_blocks)


def visualize_large_scale(
    seq_len: int = 4096,
    block_size: int = 64,
    slash_indices: list = None,
    vertical_indices: list = None,
    query_block_idx: int = None,
    save_path: str = None,
):
    """
    Visualize large-scale attention pattern with blocks.
    """

    num_blocks = (seq_len + block_size - 1) // block_size

    if query_block_idx is None:
        query_block_idx = num_blocks - 1

    if slash_indices is None:
        slash_indices = list(range(0, 500, 1))
        slash_indices += [600, 800, 1000, 1500, 2000, 2500, 3000, 3500]

    if vertical_indices is None:
        vertical_indices = [0, 5, 10, 50, 100, 500, 1000]

    slash_indices = np.array(slash_indices)

    q_start = query_block_idx * block_size
    q_end = min((query_block_idx + 1) * block_size, seq_len)

    computed_blocks = simulate_convert_vertical_slash_indexes(
        seq_len, block_size, slash_indices, query_block_idx
    )

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # Plot 1: Block-level view
    ax1 = axes[0, 0]
    block_matrix = np.zeros((num_blocks, num_blocks))

    for i in range(num_blocks):
        for j in range(i + 1):
            block_matrix[i, j] = 0.2

    all_computed = {}
    for qb in range(num_blocks):
        computed = simulate_convert_vertical_slash_indexes(
            seq_len, block_size, slash_indices, qb
        )
        all_computed[qb] = computed
        for kb in computed:
            if kb <= qb:
                block_matrix[qb, kb] = 0.8

    for kb in computed_blocks:
        block_matrix[query_block_idx, kb] = 1.0

    im1 = ax1.imshow(block_matrix, cmap='YlOrRd', aspect='equal', origin='upper',
                     extent=[-0.5, num_blocks-0.5, num_blocks-0.5, -0.5])

    ax1.axhline(y=query_block_idx, color='green', linewidth=2, linestyle='--')
    ax1.axhline(y=query_block_idx + 1, color='green', linewidth=2, linestyle='--')

    ax1.set_xlabel('Key Block Index', fontsize=12)
    ax1.set_ylabel('Query Block Index', fontsize=12)
    ax1.set_title(f'Block-Level Attention Pattern\n'
                  f'({num_blocks}x{num_blocks} blocks, each {block_size}x{block_size})\n'
                  f'Highlighted: Query Block {query_block_idx}', fontsize=12)
    plt.colorbar(im1, ax=ax1, label='Computation')

    # Plot 2: Key blocks bar chart
    ax2 = axes[0, 1]
    colors = ['red' if kb in computed_blocks else 'lightgray' for kb in range(num_blocks)]
    ax2.bar(range(num_blocks), [1]*num_blocks, color=colors, edgecolor='black', linewidth=0.5)

    ax2.set_xlabel('Key Block Index', fontsize=12)
    ax2.set_ylabel('Computed?', fontsize=12)
    ax2.set_title(f'Query Block {query_block_idx} [{q_start}-{q_end-1}]\n'
                  f'Computed Key Blocks: {len(computed_blocks)} / {query_block_idx + 1} '
                  f'({100*len(computed_blocks)/(query_block_idx+1):.1f}%)', fontsize=12)
    ax2.set_xlim(-1, num_blocks)
    ax2.set_yticks([])

    if computed_blocks:
        ax2.axvline(x=min(computed_blocks)-0.5, color='green', linewidth=2, linestyle='--')
        ax2.axvline(x=max(computed_blocks)+0.5, color='green', linewidth=2, linestyle='--')

    # Plot 3: Detail view
    ax3 = axes[1, 0]
    display_key_range = min(seq_len, 2048)

    detail_matrix = np.zeros((block_size, display_key_range))

    for kb in computed_blocks:
        k_start = kb * block_size
        k_end = min((kb + 1) * block_size, display_key_range)
        if k_start < display_key_range:
            for q_off in range(block_size):
                q = q_start + q_off
                for k in range(k_start, k_end):
                    if k <= q:
                        detail_matrix[q_off, k] = 0.4

    for d in slash_indices:
        for q_off in range(block_size):
            q = q_start + q_off
            k = q - d
            if 0 <= k < display_key_range and k <= q:
                detail_matrix[q_off, k] = 1.0

    im3 = ax3.imshow(detail_matrix, cmap='YlOrRd', aspect='auto', origin='upper',
                     extent=[-0.5, display_key_range-0.5, block_size-0.5, -0.5])

    for i in range(0, display_key_range + 1, block_size):
        ax3.axvline(x=i - 0.5, color='blue', linewidth=0.5, alpha=0.5)

    ax3.set_xlabel('Key Position', fontsize=12)
    ax3.set_ylabel(f'Query Offset (in block {query_block_idx})', fontsize=12)
    ax3.set_title(f'Detail: Query Block {query_block_idx}, Keys [0-{display_key_range-1}]\n'
                  f'Red=Diagonal (needed), Orange=Block fill (over-computed)', fontsize=12)
    plt.colorbar(im3, ax=ax3, label='Intensity')

    # Plot 4: Statistics
    ax4 = axes[1, 1]
    ax4.axis('off')

    exact_diagonal_count = 0
    for d in slash_indices:
        for q_off in range(block_size):
            q = q_start + q_off
            k = q - d
            if 0 <= k <= q:
                exact_diagonal_count += 1

    computed_elements = 0
    for kb in computed_blocks:
        k_start = kb * block_size
        k_end = min((kb + 1) * block_size, seq_len)
        for q_off in range(block_size):
            q = q_start + q_off
            for k in range(k_start, k_end):
                if k <= q:
                    computed_elements += 1

    dense_elements = sum(q_start + q_off + 1 for q_off in range(block_size))

    stats_text = f"""
╔══════════════════════════════════════════════════════════════╗
║                    COMPUTATION STATISTICS                      ║
╠══════════════════════════════════════════════════════════════╣
║  Sequence Length:          {seq_len:,}
║  Block Size:               {block_size}
║  Total Blocks:             {num_blocks}
║
║  Query Block:              {query_block_idx} (positions {q_start}-{q_end-1})
║  Selected Slash Diagonals: {len(slash_indices)}
║
╠══════════════════════════════════════════════════════════════╣
║  COMPARISON
╠══════════════════════════════════════════════════════════════╣
║
║  Dense Attention:
║    - Elements: {dense_elements:,}
║    - 100% of causal region
║
║  Exact Diagonal (ideal):
║    - Elements: {exact_diagonal_count:,}
║    - {100*exact_diagonal_count/dense_elements:.2f}% of dense
║
║  Block-based (MInference):
║    - Computed blocks: {len(computed_blocks)} / {query_block_idx+1}
║    - Elements: {computed_elements:,}
║    - {100*computed_elements/dense_elements:.2f}% of dense
║
║  Over-computation ratio:
║    - vs exact diagonal: {computed_elements/max(exact_diagonal_count,1):.2f}x
║    - Efficiency: {100*exact_diagonal_count/max(computed_elements,1):.1f}%
║
║  Speedup vs Dense:
║    - {dense_elements/max(computed_elements,1):.2f}x fewer elements computed
║
╚══════════════════════════════════════════════════════════════╝
"""

    ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.show()

    return computed_blocks, exact_diagonal_count, computed_elements, dense_elements


def compare_block_sizes(
    seq_len: int = 4096,
    slash_indices: list = None,
    query_block_idx_ratio: float = 0.9,
    save_path: str = None,
):
    """
    Compare different block sizes and their effect on computation.
    """

    if slash_indices is None:
        slash_indices = list(range(0, 500, 1)) + [600, 800, 1000, 1500, 2000, 2500, 3000]

    slash_indices = np.array(slash_indices)

    block_sizes = [32, 64, 128]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    stats = {}

    for idx, block_size in enumerate(block_sizes):
        num_blocks = (seq_len + block_size - 1) // block_size
        query_block_idx = int(num_blocks * query_block_idx_ratio)

        q_start = query_block_idx * block_size
        q_end = min((query_block_idx + 1) * block_size, seq_len)

        computed_blocks = simulate_convert_vertical_slash_indexes(
            seq_len, block_size, slash_indices, query_block_idx
        )

        ax_top = axes[0, idx]
        block_matrix = np.zeros((num_blocks, num_blocks))
        for i in range(num_blocks):
            for j in range(i + 1):
                block_matrix[i, j] = 0.2

        for qb in range(num_blocks):
            computed = simulate_convert_vertical_slash_indexes(
                seq_len, block_size, slash_indices, qb
            )
            for kb in computed:
                if kb <= qb:
                    block_matrix[qb, kb] = 0.8

        for kb in computed_blocks:
            block_matrix[query_block_idx, kb] = 1.0

        im = ax_top.imshow(block_matrix, cmap='YlOrRd', aspect='equal', origin='upper')
        ax_top.axhline(y=query_block_idx, color='green', linewidth=1, linestyle='--')
        ax_top.set_title(f'Block Size = {block_size}\n({num_blocks}x{num_blocks} blocks)', fontsize=12)
        ax_top.set_xlabel('Key Block')
        ax_top.set_ylabel('Query Block')

        ax_bot = axes[1, idx]

        display_range = min(2048, seq_len)
        detail_matrix = np.zeros((block_size, display_range))

        for kb in computed_blocks:
            k_start = kb * block_size
            k_end = min((kb + 1) * block_size, display_range)
            if k_start < display_range:
                for q_off in range(block_size):
                    q = q_start + q_off
                    for k in range(k_start, k_end):
                        if k <= q:
                            detail_matrix[q_off, k] = 0.4

        for d in slash_indices:
            for q_off in range(block_size):
                q = q_start + q_off
                k = q - d
                if 0 <= k < display_range and k <= q:
                    detail_matrix[q_off, k] = 1.0

        im2 = ax_bot.imshow(detail_matrix, cmap='YlOrRd', aspect='auto', origin='upper')

        for i in range(0, display_range + 1, block_size):
            ax_bot.axvline(x=i - 0.5, color='blue', linewidth=0.5, alpha=0.5)

        exact_count = sum(1 for d in slash_indices
                        for q_off in range(block_size)
                        if 0 <= (q_start + q_off - d) <= (q_start + q_off))

        computed_count = sum(
            1 for kb in computed_blocks
            for q_off in range(block_size)
            for k in range(kb * block_size, min((kb + 1) * block_size, seq_len))
            if k <= q_start + q_off
        )

        dense_count = sum(q_start + q_off + 1 for q_off in range(block_size))

        stats[block_size] = {
            'computed_blocks': len(computed_blocks),
            'total_blocks': query_block_idx + 1,
            'exact_count': exact_count,
            'computed_count': computed_count,
            'dense_count': dense_count,
        }

        ax_bot.set_xlabel('Key Position')
        ax_bot.set_ylabel('Query Offset')
        ax_bot.set_title(f'Computed: {len(computed_blocks)} blocks\n'
                        f'Overhead: {computed_count/max(exact_count,1):.1f}x, '
                        f'Speedup vs dense: {dense_count/max(computed_count,1):.1f}x', fontsize=10)

    plt.suptitle(f'Block Size Comparison (seq_len={seq_len}, {len(slash_indices)} slash diagonals)',
                fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.show()

    print("\n" + "="*70)
    print("BLOCK SIZE COMPARISON")
    print("="*70)
    print(f"{'Block Size':<12} {'Blocks Computed':<18} {'Elements':<15} {'Overhead':<12} {'Speedup':<10}")
    print("-"*70)
    for bs in block_sizes:
        s = stats[bs]
        print(f"{bs:<12} {s['computed_blocks']}/{s['total_blocks']:<14} "
              f"{s['computed_count']:,}/{s['dense_count']:,}  "
              f"{s['computed_count']/max(s['exact_count'],1):.2f}x         "
              f"{s['dense_count']/max(s['computed_count'],1):.1f}x")
    print("="*70)

    return stats


if __name__ == "__main__":
    print("="*70)
    print("1. Visualizing sum_all_diagonal_matrix mechanism")
    print("="*70)

    visualize_sum_all_diagonal_matrix(
        save_path="/home/zijie/Code/MInference/tools/sum_all_diagonal_explained.png"
    )

    print("\n" + "="*70)
    print("2. Visualizing Slash Extraction Process")
    print("="*70)

    visualize_slash_extraction_process(
        seq_len=256,
        last_q=64,
        slash_size=100,
        save_path="/home/zijie/Code/MInference/tools/slash_extraction_process.png"
    )

    print("\n" + "="*70)
    print("3. Large Scale Visualization: 4096x4096 with block_size=64")
    print("="*70)

    slash_indices = list(range(0, 500))
    slash_indices += [600, 800, 1000, 1200, 1500, 2000, 2500, 3000, 3500]

    visualize_large_scale(
        seq_len=4096,
        block_size=64,
        slash_indices=slash_indices,
        vertical_indices=[0, 5, 10, 50, 100],
        query_block_idx=60,
        save_path="/home/zijie/Code/MInference/tools/large_scale_4096_block64.png"
    )

    print("\n" + "="*70)
    print("4. Block Size Comparison: 32 vs 64 vs 128")
    print("="*70)

    compare_block_sizes(
        seq_len=4096,
        slash_indices=slash_indices,
        query_block_idx_ratio=0.9,
        save_path="/home/zijie/Code/MInference/tools/block_size_comparison.png"
    )
