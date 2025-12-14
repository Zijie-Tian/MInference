# Copyright (c) 2024-2025 Microsoft
# Licensed under The MIT License [see LICENSE for details]

"""
CPU SIMD Sparse Attention Implementation

This module provides CPU-based sparse attention using AVX-512 SIMD instructions.
Key advantages over GPU:
1. Fine-grained scheduling - can skip causal mask regions (saves ~40% compute)
2. No kernel launch overhead - better for short sequences
3. Precise computation - no block-level waste

Usage:
    from minference.ops.cpu_sparse_attention import vertical_slash_attention_cpu

    # Full pipeline (estimate + compute)
    output = vertical_slash_attention_cpu(Q, K, V, vertical_size=100, slash_size=500)

    # Or step by step for analysis
    v_idx, s_idx = estimate_pattern_cpu(Q[0,0], K[0,0], 100, 500)
    output = sparse_attention_cpu(Q[0,0], K[0,0], V[0,0], v_idx, s_idx)
"""

import torch
import math
from typing import Tuple, Optional

# Try to import C++ extension
try:
    from minference._C import cpu_sparse_attention as _C
    HAS_CPU_EXTENSION = True
except ImportError:
    try:
        # Fallback for different import paths during development
        import minference._C.cpu_sparse_attention as _C
        HAS_CPU_EXTENSION = True
    except ImportError:
        HAS_CPU_EXTENSION = False
        # Only print warning if not in a headless/silent environment
        import sys
        if sys.stderr.isatty():
            print("Warning: CPU sparse attention extension not found. Using pure Python fallback.")


def _check_cpu_extension():
    if not HAS_CPU_EXTENSION:
        raise RuntimeError(
            "CPU sparse attention extension not available. "
            "Please compile with: python setup.py build_ext --inplace"
        )


# =============================================================================
# Pure Python Reference Implementation (for testing and fallback)
# =============================================================================

def estimate_pattern_python(
    Q: torch.Tensor,  # [seq_len, head_dim]
    K: torch.Tensor,  # [seq_len, head_dim]
    vertical_size: int,
    slash_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pure Python implementation of pattern estimation.
    Equivalent to the C++ estimate_pattern_cpu function.
    """
    seq_len, head_dim = Q.shape
    last_q = min(64, seq_len)
    scale = 1.0 / math.sqrt(head_dim)

    # Probe GEMM: Q[-last_q:] @ K^T
    qk = torch.matmul(Q[-last_q:], K.t()) * scale  # [last_q, seq_len]

    # Causal mask
    q_positions = torch.arange(seq_len - last_q, seq_len, device=Q.device)
    k_positions = torch.arange(seq_len, device=Q.device)
    causal_mask = k_positions[None, :] <= q_positions[:, None]
    qk = qk.masked_fill(~causal_mask, float('-inf'))

    # Softmax
    qk = torch.softmax(qk, dim=-1)

    # Vertical extraction: sum along query dim
    vertical = qk.sum(dim=0)  # [seq_len]
    vertical[:30] = float('inf')  # force keep first 30
    v_idx = torch.topk(vertical, min(vertical_size, seq_len)).indices

    # Slash extraction: diagonal sum
    slash = torch.zeros(seq_len, device=Q.device)
    for q in range(last_q):
        q_pos = seq_len - last_q + q
        for k in range(min(q_pos + 1, seq_len)):
            diag = q_pos - k
            if diag < seq_len:
                slash[diag] += qk[q, k].item()

    slash[-100:] = float('inf')  # force keep last 100 diagonals
    s_idx = torch.topk(slash, min(slash_size, seq_len)).indices

    return v_idx.int(), s_idx.int()


def sparse_attention_python(
    Q: torch.Tensor,  # [seq_len, head_dim]
    K: torch.Tensor,  # [seq_len, head_dim]
    V: torch.Tensor,  # [seq_len, head_dim]
    v_idx: torch.Tensor,  # [num_vertical]
    s_idx: torch.Tensor,  # [num_slash]
) -> torch.Tensor:
    """
    Pure Python implementation of sparse attention with online softmax.
    """
    seq_len, head_dim = Q.shape
    scale = 1.0 / math.sqrt(head_dim)
    output = torch.zeros_like(Q)

    v_idx_set = set(v_idx.tolist())
    s_idx_list = s_idx.tolist()

    for q in range(seq_len):
        # Expand pattern: collect valid key positions
        valid_keys = set()

        # Slash: diagonal offsets
        for s in s_idx_list:
            k = q - s
            if 0 <= k <= q:
                valid_keys.add(k)

        # Vertical: column indices
        for v in v_idx_set:
            if v <= q:
                valid_keys.add(v)

        valid_keys = sorted(valid_keys)

        if not valid_keys:
            continue

        # Online softmax attention
        m_i = float('-inf')
        l_i = 0.0
        acc = torch.zeros(head_dim, device=Q.device)

        for k in valid_keys:
            score = (Q[q] @ K[k]) * scale

            m_new = max(m_i, score.item())
            alpha = math.exp(m_i - m_new) if m_i > float('-inf') else 0.0
            p = math.exp(score.item() - m_new)

            acc = acc * alpha + V[k] * p
            l_i = l_i * alpha + p
            m_i = m_new

        if l_i > 0:
            output[q] = acc / l_i

    return output


# =============================================================================
# Main API Functions
# =============================================================================

def vertical_slash_attention_cpu(
    Q: torch.Tensor,  # [batch, heads, seq_len, head_dim]
    K: torch.Tensor,
    V: torch.Tensor,
    vertical_size: int = 100,
    slash_size: int = 500,
    use_cpp: bool = True,
) -> torch.Tensor:
    """
    CPU implementation of vertical-slash sparse attention.

    This function performs:
    1. Estimate stage: Extract sparse pattern from probe attention
    2. Compute stage: Sparse attention with fine-grained scheduling

    Args:
        Q, K, V: Input tensors [batch, heads, seq_len, head_dim]
        vertical_size: Number of important vertical columns to keep
        slash_size: Number of important diagonal offsets to keep
        use_cpp: Use C++ extension if available (default True)

    Returns:
        Output tensor [batch, heads, seq_len, head_dim]
    """
    # Convert to float32 and ensure contiguous
    Q = Q.float().contiguous()
    K = K.float().contiguous()
    V = V.float().contiguous()

    if use_cpp and HAS_CPU_EXTENSION:
        return _C.vertical_slash_attention(Q, K, V, vertical_size, slash_size)
    else:
        # Python fallback (slow but correct)
        batch, heads, seq_len, head_dim = Q.shape
        output = torch.zeros_like(Q)

        for b in range(batch):
            for h in range(heads):
                q = Q[b, h]
                k = K[b, h]
                v = V[b, h]

                v_idx, s_idx = estimate_pattern_python(q, k, vertical_size, slash_size)
                output[b, h] = sparse_attention_python(q, k, v, v_idx, s_idx)

        return output


def estimate_pattern_cpu(
    Q: torch.Tensor,  # [seq_len, head_dim]
    K: torch.Tensor,
    vertical_size: int,
    slash_size: int,
    use_cpp: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Estimate sparse pattern from Q, K.

    Returns:
        v_idx: [vertical_size] vertical column indices
        s_idx: [slash_size] diagonal offset indices
    """
    Q = Q.float().contiguous()
    K = K.float().contiguous()

    if use_cpp and HAS_CPU_EXTENSION:
        return _C.estimate_pattern(Q, K, vertical_size, slash_size)
    else:
        return estimate_pattern_python(Q, K, vertical_size, slash_size)


def sparse_attention_cpu(
    Q: torch.Tensor,  # [seq_len, head_dim]
    K: torch.Tensor,
    V: torch.Tensor,
    v_idx: torch.Tensor,
    s_idx: torch.Tensor,
    use_cpp: bool = True,
) -> torch.Tensor:
    """
    Sparse attention with pre-computed pattern.
    """
    Q = Q.float().contiguous()
    K = K.float().contiguous()
    V = V.float().contiguous()
    v_idx = v_idx.int().contiguous()
    s_idx = s_idx.int().contiguous()

    if use_cpp and HAS_CPU_EXTENSION:
        return _C.sparse_attention(Q, K, V, v_idx, s_idx)
    else:
        return sparse_attention_python(Q, K, V, v_idx, s_idx)


def analyze_sparsity_cpu(
    seq_len: int,
    v_idx: torch.Tensor,
    s_idx: torch.Tensor,
) -> dict:
    """
    Analyze sparsity pattern statistics.

    Returns:
        Dictionary with:
        - total_keys_computed: Number of (q,k) pairs in sparse pattern
        - total_keys_dense: Number of (q,k) pairs in dense causal attention
        - sparsity_ratio: sparse / dense
        - causal_savings: Estimated savings from skipping causal mask regions
    """
    if HAS_CPU_EXTENSION:
        computed, dense, ratio, savings = _C.analyze_sparsity(
            seq_len, v_idx.int(), s_idx.int()
        )
        return {
            'total_keys_computed': computed,
            'total_keys_dense': dense,
            'sparsity_ratio': ratio,
            'causal_savings': savings,
        }
    else:
        # Python fallback
        v_set = set(v_idx.tolist())
        s_list = s_idx.tolist()

        computed = 0
        for q in range(seq_len):
            valid = set()
            for s in s_list:
                k = q - s
                if 0 <= k <= q:
                    valid.add(k)
            for v in v_set:
                if v <= q:
                    valid.add(v)
            computed += len(valid)

        dense = seq_len * (seq_len + 1) // 2
        return {
            'total_keys_computed': computed,
            'total_keys_dense': dense,
            'sparsity_ratio': computed / dense,
            'causal_savings': 0.0,  # Not computed in Python
        }
