# Copyright (c) 2024-2025 Microsoft
# Licensed under The MIT License [see LICENSE for details]

"""
Pattern discovery for sparse attention.

Implements vertical and slash pattern extraction from probe attention.
Uses MInference's sum_all_diagonal_matrix function directly.
"""

import torch
from typing import Tuple, Optional

# Import from MInference - use the same function as the main codebase
from minference.modules.minference_forward import sum_all_diagonal_matrix


def extract_patterns(
    probe_attn: torch.Tensor,
    seq_len: int,
    vertical_size: int = 1000,
    slash_size: int = 6096,
    keep_first_n: int = 30,
    keep_last_n: int = 30,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract vertical and slash indices from probe attention.

    This implements the pattern discovery from minference_forward.py:248-258.

    Vertical Pattern:
        - Sum attention weights across probe queries for each key position
        - Select top-k columns with highest total attention
        - These represent globally important tokens (e.g., BOS, keywords)

    Slash Pattern:
        - Sum attention weights along each diagonal
        - Select top-k diagonals with highest total attention
        - These represent local context patterns (recent tokens)

    Args:
        probe_attn: Attention weights [1, 1, 64, seq_len] from probe queries
        seq_len: Total sequence length
        vertical_size: Number of vertical columns to keep
        slash_size: Number of slash diagonals to keep
        keep_first_n: Force keep first N columns (BOS tokens)
        keep_last_n: Force keep last N diagonals (recent context)

    Returns:
        v_idx: Vertical column indices [vertical_size], sorted ascending
        s_idx: Slash diagonal offsets [slash_size], negative values, sorted descending
    """
    last_q = probe_attn.shape[2]  # Should be 64
    device = probe_attn.device

    # ===== Vertical Pattern =====
    # Sum attention across probe queries to get importance per column
    # Shape: [1, 1, 1, seq_len]
    vertical = probe_attn.sum(dim=-2, keepdim=True)

    # Force keep first N columns by setting their "importance" to infinity
    # This ensures BOS and initial tokens are always included
    vertical[..., :keep_first_n] = float('inf')

    # Select top-k columns (highest attention = most important)
    # We use topk to get vertical_size most important columns
    v_idx = torch.topk(vertical.squeeze(), vertical_size, dim=-1).indices

    # Sort indices for easier processing later
    v_idx = v_idx.sort().values

    # ===== Slash Pattern =====
    # Sum attention along each diagonal
    # Shape: [1, 1, seq_len + 63]
    slash = sum_all_diagonal_matrix(probe_attn)

    # We only care about diagonals that can be computed for all probe queries
    # The valid range is [0, seq_len - last_q + 1) in the diagonal sum output
    # which corresponds to diagonals d ∈ [-(seq_len-1), last_q-1] in actual indices
    slash = slash[..., :-last_q + 1]

    # Force keep last N diagonals (most recent context)
    slash[..., -keep_last_n:] = float('inf')

    # Select top-k diagonals
    # The diagonal index in the sum output needs to be converted to actual diagonal offset
    # MInference uses: s_idx = (seq_len - 1) - topk_indices (positive values)
    # This represents the distance from the query position (key_pos = query_pos - s_idx)
    slash_topk_indices = torch.topk(slash.squeeze(), slash_size, dim=-1).indices
    s_idx = (seq_len - 1) - slash_topk_indices

    # Sort indices (descending for slash - larger offset first)
    s_idx = s_idx.sort(descending=True).values

    return v_idx, s_idx


def compute_sparsity_ratio(
    v_idx: torch.Tensor,
    s_idx: torch.Tensor,
    seq_len: int,
) -> float:
    """
    Compute the sparsity ratio of the sparse pattern.

    Sparsity = (computed elements) / (full causal elements)

    For causal attention:
        Full elements = seq_len * (seq_len + 1) / 2

    For sparse pattern:
        Vertical contributes ~vertical_size * seq_len (some overlap)
        Slash contributes ~slash_size * seq_len (some overlap)

    This is an approximation as vertical and slash may overlap.

    Args:
        v_idx: Vertical indices
        s_idx: Slash indices
        seq_len: Sequence length

    Returns:
        Sparsity ratio (0 to 1)
    """
    # Full causal attention elements
    full_elements = seq_len * (seq_len + 1) / 2

    # Estimate sparse elements (upper bound, ignores overlap)
    # Each vertical column contributes roughly (seq_len + 1) / 2 elements on average
    # Each slash diagonal contributes roughly seq_len elements
    vertical_elements = len(v_idx) * (seq_len + 1) / 2
    slash_elements = len(s_idx) * seq_len

    # This is an upper bound (actual may be less due to overlap)
    sparse_elements = min(vertical_elements + slash_elements, full_elements)

    return sparse_elements / full_elements


def visualize_pattern_coverage(
    v_idx: torch.Tensor,
    s_idx: torch.Tensor,
    seq_len: int,
    block_size: int = 64,
) -> torch.Tensor:
    """
    Create a block-level mask showing pattern coverage.

    Args:
        v_idx: Vertical column indices
        s_idx: Slash diagonal offsets (negative)
        seq_len: Sequence length
        block_size: Block size for visualization

    Returns:
        block_mask: [num_blocks, num_blocks] showing covered blocks
    """
    num_blocks = (seq_len + block_size - 1) // block_size
    block_mask = torch.zeros(num_blocks, num_blocks)

    # Convert to CPU numpy for easier processing
    v_idx_np = v_idx.cpu().numpy() if v_idx.is_cuda else v_idx.numpy()
    s_idx_np = s_idx.cpu().numpy() if s_idx.is_cuda else s_idx.numpy()

    # Mark blocks covered by vertical columns
    v_blocks = set(v_idx_np // block_size)
    for v_block in v_blocks:
        if v_block < num_blocks:
            # Vertical column is attended by all query blocks after it
            block_mask[v_block:, v_block] = 1.0

    # Mark blocks covered by slash diagonals
    # Convert element diagonals to block diagonals
    block_diags_covered = set()
    for d in s_idx_np:
        # Block diagonal that this element diagonal primarily hits
        bd_min = (abs(d) - block_size + 1) // block_size
        bd_max = (abs(d) + block_size - 1) // block_size
        for bd in range(max(0, bd_min), bd_max + 1):
            block_diags_covered.add(bd)

    # Fill block mask based on covered block diagonals
    for bd in block_diags_covered:
        for k_block in range(num_blocks):
            q_block = k_block + bd
            if 0 <= q_block < num_blocks:
                block_mask[q_block, k_block] = 1.0

    return block_mask
