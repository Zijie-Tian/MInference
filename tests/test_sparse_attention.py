# Copyright (c) 2024-2025 Microsoft
# Licensed under The MIT License [see LICENSE for details]

"""
Minimal test script to understand sgl_kernel sparse attention call flow.

Usage:
    python tests/test_sparse_attention.py
"""

import math
import torch
from transformers.models.llama.modeling_llama import repeat_kv

from minference.modules.minference_forward import LAST_Q_MASK, sum_all_diagonal_matrix
from minference.ops.pit_sparse_flash_attention_v2 import (
    vertical_slash_sparse_attention,
    convert_vertical_slash_indexes_opt,
    sparse_attn_func,
)


# ============================================================================
# Utility Functions
# ============================================================================

def create_qkv_tensors(batch_size, num_heads, seq_len, head_dim, dtype=torch.bfloat16, device="cuda"):
    """Create query, key, value tensors for testing."""
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype, device=device)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype, device=device)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype, device=device)
    return q, k, v


def create_sparse_indices(seq_len, vertical_size, slash_size, device="cuda"):
    """Create vertical and slash indices for sparse attention."""
    v_idx = torch.randint(0, seq_len, (1, 1, vertical_size), dtype=torch.int32, device=device)
    v_idx = v_idx.sort(dim=-1, descending=False)[0]

    s_idx = torch.randint(0, seq_len, (1, 1, slash_size), dtype=torch.int32, device=device)
    s_idx = s_idx.sort(dim=-1, descending=True)[0]

    return v_idx, s_idx


def create_gqa_tensors(batch_size, seq_len, num_heads, num_kv_heads, head_dim, dtype=torch.bfloat16, device="cuda"):
    """Create GQA-like tensors (non-contiguous query after transpose)."""
    num_key_value_groups = num_heads // num_kv_heads

    # Query: created from hidden states -> view -> transpose (becomes non-contiguous)
    query_hidden = torch.randn(batch_size, seq_len, num_heads * head_dim, dtype=dtype, device=device)
    query_states = query_hidden.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)

    # Key/Value: created similarly but then expanded via repeat_kv
    kv_hidden = torch.randn(batch_size, seq_len, num_kv_heads * head_dim, dtype=dtype, device=device)
    key_states = repeat_kv(
        kv_hidden.view(batch_size, seq_len, num_kv_heads, head_dim).transpose(1, 2),
        num_key_value_groups
    )
    value_states = repeat_kv(
        kv_hidden.view(batch_size, seq_len, num_kv_heads, head_dim).transpose(1, 2),
        num_key_value_groups
    )

    return query_states, key_states, value_states


def compute_sparse_indices_from_attention(q, k, head_dim, vertical_size, slash_size):
    """
    Compute sparse indices using the actual vertical_and_slash_kernel logic.
    This replicates the probe-based pattern discovery from minference_forward.py.
    """
    q_len = q.shape[2]

    # Probe-based pattern discovery using last 64 queries
    last_q = min(64, q_len)
    qk = torch.einsum('bhmk, bhnk -> bhmn', q[:, :, -last_q:, :], k) / math.sqrt(head_dim)

    # Apply causal mask for the last_q region
    qk[:, :, :, -last_q:] = torch.where(
        LAST_Q_MASK[..., -last_q:, -last_q:].to(q.device),
        qk[:, :, :, -last_q:],
        -torch.inf
    )
    qk = torch.nn.functional.softmax(qk, dim=-1, dtype=torch.float32)

    # Extract vertical indices (important columns)
    vertical = qk.sum(-2, keepdim=True)
    vertical[..., :30] = torch.inf  # Always include first 30 tokens
    vertical_topk = torch.topk(vertical, min(vertical_size, q_len), -1).indices

    # Extract slash indices (important diagonals)
    slash = sum_all_diagonal_matrix(qk)[..., :-last_q + 1]
    slash[..., -100:] = torch.inf  # Always include recent tokens
    actual_slash_size = min(slash_size, slash.shape[-1])
    slash_idx = (q_len - 1) - torch.topk(slash, actual_slash_size, -1).indices

    return vertical_topk, slash_idx


# ============================================================================
# Main Test
# ============================================================================

def main():
    print("=" * 60)
    print("SGLang Sparse Attention Call Flow Test")
    print("=" * 60)

    # =========================================================================
    # Step 1: Create GQA-like tensors (simulating real model behavior)
    # =========================================================================
    print("\n[Step 1] Creating GQA-like tensors...")

    batch_size, seq_len, num_heads, num_kv_heads, head_dim = 1, 4096, 32, 8, 128

    query_states, key_states, value_states = create_gqa_tensors(
        batch_size, seq_len, num_heads, num_kv_heads, head_dim
    )

    print(f"  query_states: shape={query_states.shape}, contiguous={query_states.is_contiguous()}")
    print(f"  key_states: shape={key_states.shape}, contiguous={key_states.is_contiguous()}")
    print(f"  value_states: shape={value_states.shape}, contiguous={value_states.is_contiguous()}")

    # =========================================================================
    # Step 2: Extract single head (as done in minference_prefill_forward)
    # =========================================================================
    print("\n[Step 2] Extracting single head (head_id=0)...")

    head_id = 0
    q = query_states[:, head_id, :, :].unsqueeze(1)  # [1, 1, 4096, 128]
    k = key_states[:, head_id, :, :].unsqueeze(1)
    v = value_states[:, head_id, :, :].unsqueeze(1)

    print(f"  q: shape={q.shape}, contiguous={q.is_contiguous()}, stride={q.stride()}")
    print(f"  k: shape={k.shape}, contiguous={k.is_contiguous()}, stride={k.stride()}")
    print(f"  v: shape={v.shape}, contiguous={v.is_contiguous()}, stride={v.stride()}")

    # =========================================================================
    # Step 3: Compute sparse indices (vertical_and_slash_kernel logic)
    # =========================================================================
    print("\n[Step 3] Computing sparse indices...")

    vertical_size, slash_size = 1000, 2000
    vertical_topk, slash_idx = compute_sparse_indices_from_attention(q, k, head_dim, vertical_size, slash_size)

    print(f"  vertical_topk: shape={vertical_topk.shape}, min={vertical_topk.min()}, max={vertical_topk.max()}")
    print(f"  slash_idx: shape={slash_idx.shape}, min={slash_idx.min()}, max={slash_idx.max()}")

    # =========================================================================
    # Step 4: Call vertical_slash_sparse_attention (high-level API)
    # =========================================================================
    print("\n[Step 4] Calling vertical_slash_sparse_attention...")

    out = vertical_slash_sparse_attention(q, k, v, vertical_topk, slash_idx)
    print(f"  Output: shape={out.shape}, dtype={out.dtype}")
    print(f"  Has NaN: {torch.isnan(out).any()}")

    # =========================================================================
    # Step 5: Detailed breakdown of what happens inside
    # =========================================================================
    print("\n[Step 5] Detailed breakdown of internal calls...")

    # 5a. Reshape and sort indices
    v_idx = vertical_topk.to(torch.int32).reshape((batch_size, 1, -1)).sort(dim=-1, descending=False)[0]
    s_idx = slash_idx.to(torch.int32).reshape((batch_size, 1, -1)).sort(dim=-1, descending=True)[0]
    print(f"  v_idx after reshape/sort: shape={v_idx.shape}")
    print(f"  s_idx after reshape/sort: shape={s_idx.shape}")

    # 5b. Convert indices to block format (sgl_kernel's convert_vertical_slash_indexes)
    seqlens = torch.tensor([seq_len], dtype=torch.int32, device="cuda")
    block_count, block_offset, column_count, column_index = convert_vertical_slash_indexes_opt(
        seqlens,      # q_seqlens
        seqlens,      # kv_seqlens
        v_idx,        # vertical_indexes
        s_idx,        # slash_indexes
        seq_len,      # context_size
        64,           # block_size_M
        64,           # block_size_N
        causal=True,
    )
    print(f"  block_count: shape={block_count.shape}, max={block_count.max()}")
    print(f"  block_offset: shape={block_offset.shape}")
    print(f"  column_count: shape={column_count.shape}, max={column_count.max()}")
    print(f"  column_index: shape={column_index.shape}")

    # 5c. Prepare tensors for sparse_attn_func (transpose + contiguous)
    qt = q.transpose(1, 2).contiguous()  # [1, 4096, 1, 128]
    kt = k.transpose(1, 2).contiguous()
    vt = v.transpose(1, 2).contiguous()
    print(f"  qt (after transpose+contiguous): shape={qt.shape}, contiguous={qt.is_contiguous()}")

    # 5d. Call sgl_kernel's sparse_attn_func
    print("\n  Calling sparse_attn_func (sgl_kernel)...")
    out2 = sparse_attn_func(
        qt, kt, vt,
        block_count,
        block_offset,
        column_count,
        column_index,
        causal=True,
        return_softmax_lse=False,
    )
    print(f"  sparse_attn_func output: shape={out2.shape}")

    # 5e. Transpose back
    out2 = out2.transpose(1, 2).contiguous()  # [1, 1, 4096, 128]
    print(f"  Final output: shape={out2.shape}")

    print("\n" + "=" * 60)
    print("Test completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
