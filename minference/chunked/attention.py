# Copyright (c) 2024-2025 Microsoft
# Licensed under The MIT License [see LICENSE for details]

"""
Chunked attention algorithms for sparse attention profiling.

Extends MInference to support chunk prefill for long sequences (1M+).

Implements:
1. Chunked probe attention for pattern discovery
2. Online softmax for chunked computation
3. Chunked sparse attention with vertical + slash patterns
4. Integration with MInference's vertical_slash_sparse_attention kernel
"""

import math
import torch
import torch.nn.functional as F
from typing import Tuple, Optional
from tqdm import tqdm

# Import MInference's sparse attention function
from minference.ops.pit_sparse_flash_attention_v2 import (
    vertical_slash_sparse_attention,
    convert_vertical_slash_indexes,
)


def chunked_probe_attention(
    q_probe: torch.Tensor,
    k_cpu: torch.Tensor,
    head_dim: int,
    chunk_size: int = 65536,
    seq_len: Optional[int] = None,
    show_progress: bool = False,
) -> torch.Tensor:
    """
    Compute probe attention in chunks: softmax(Q_probe @ K.T / sqrt(d))

    This function computes attention scores for the last 64 queries (probes)
    against all keys in the sequence. The computation is chunked to fit in
    GPU memory.

    Args:
        q_probe: Probe queries [1, 1, 64, head_dim] on GPU
        k_cpu: All keys [1, 1, seq_len, head_dim] on CPU
        head_dim: Dimension of each head
        chunk_size: Number of keys to process per chunk
        seq_len: Total sequence length (inferred from k_cpu if not provided)
        show_progress: Show progress bar

    Returns:
        probe_attn: Attention weights [1, 1, 64, seq_len] on GPU
    """
    if seq_len is None:
        seq_len = k_cpu.shape[2]

    last_q = q_probe.shape[2]  # Should be 64
    device = q_probe.device

    # Allocate output on GPU
    scores = torch.empty(1, 1, last_q, seq_len, device=device, dtype=torch.float32)

    # Probe query positions (last `last_q` positions of the sequence)
    probe_pos = torch.arange(seq_len - last_q, seq_len, device=device)

    num_chunks = (seq_len + chunk_size - 1) // chunk_size
    iterator = range(0, seq_len, chunk_size)
    if show_progress:
        iterator = tqdm(iterator, desc="Probe attention", total=num_chunks)

    for start in iterator:
        end = min(start + chunk_size, seq_len)

        # Load K chunk to GPU
        k_chunk = k_cpu[:, :, start:end, :].to(device, non_blocking=True)

        # Compute attention scores: [1, 1, 64, chunk_len]
        qk = torch.matmul(q_probe, k_chunk.transpose(-2, -1)) / math.sqrt(head_dim)

        # Apply causal mask
        # Probe queries at [seq_len-64, seq_len), keys at [start, end)
        key_pos = torch.arange(start, end, device=device)
        causal_mask = probe_pos[:, None] >= key_pos[None, :]
        qk = torch.where(causal_mask, qk, torch.tensor(float('-inf'), device=device, dtype=qk.dtype))

        scores[:, :, :, start:end] = qk

        del k_chunk, qk
        torch.cuda.empty_cache()

    # Apply softmax over full sequence
    probe_attn = F.softmax(scores, dim=-1)

    return probe_attn


def online_softmax_update(
    qk: torch.Tensor,
    v: torch.Tensor,
    m_i: torch.Tensor,
    l_i: torch.Tensor,
    acc: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Online softmax accumulation update.

    Implements the online softmax algorithm from FlashAttention:
        m_new = max(m_i, max(qk))
        alpha = exp(m_i - m_new)
        p = exp(qk - m_new)
        acc = acc * alpha + p @ V
        l_i = l_i * alpha + sum(p)
        m_i = m_new

    Args:
        qk: Attention scores [q_len, kv_len]
        v: Value tensor [kv_len, head_dim]
        m_i: Running max [q_len]
        l_i: Running sum [q_len]
        acc: Output accumulator [q_len, head_dim]

    Returns:
        Updated (m_i, l_i, acc)
    """
    # Get max of current block
    m_ij = qk.max(dim=-1).values  # [q_len]

    # New running max
    m_new = torch.maximum(m_i, m_ij)

    # Avoid numerical issues with -inf
    # When m_i is -inf and m_ij is -inf, alpha should be 0
    alpha = torch.where(
        m_i == float('-inf'),
        torch.zeros_like(m_i),
        torch.exp(m_i - m_new)
    )

    # Softmax numerator for current block
    p = torch.exp(qk - m_new[:, None])

    # Handle -inf in qk
    p = torch.where(torch.isinf(qk) & (qk < 0), torch.zeros_like(p), p)

    # Update accumulator
    acc = acc * alpha[:, None] + torch.matmul(p, v)

    # Update running sum
    l_i = l_i * alpha + p.sum(dim=-1)

    # Update running max
    m_i = m_new

    return m_i, l_i, acc


def online_softmax_update_single(
    qk: torch.Tensor,
    v: torch.Tensor,
    m_i: torch.Tensor,
    l_i: torch.Tensor,
    acc: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Online softmax update for a single query.

    Args:
        qk: Attention scores [kv_len]
        v: Value tensor [kv_len, head_dim]
        m_i: Running max (scalar)
        l_i: Running sum (scalar)
        acc: Output accumulator [head_dim]

    Returns:
        Updated (m_i, l_i, acc)
    """
    # Get max of current block
    m_ij = qk.max()

    # New running max
    m_new = torch.maximum(m_i, m_ij)

    # Rescale factor
    if m_i == float('-inf'):
        alpha = torch.tensor(0.0, device=m_i.device, dtype=m_i.dtype)
    else:
        alpha = torch.exp(m_i - m_new)

    # Softmax numerator
    p = torch.exp(qk - m_new)
    p = torch.where(torch.isinf(qk) & (qk < 0), torch.zeros_like(p), p)

    # Update
    acc = acc * alpha + torch.matmul(p, v)
    l_i = l_i * alpha + p.sum()
    m_i = m_new

    return m_i, l_i, acc


def chunked_sparse_attention(
    q_cpu: torch.Tensor,
    k_cpu: torch.Tensor,
    v_cpu: torch.Tensor,
    v_idx: torch.Tensor,
    s_idx: torch.Tensor,
    head_dim: int,
    q_chunk_size: int = 512,
    kv_batch_size: int = 4096,
    show_progress: bool = False,
) -> torch.Tensor:
    """
    Compute sparse attention output using chunked online softmax.

    This function computes attention output for sparse patterns defined by
    vertical indices (important columns) and slash indices (diagonal offsets).

    Args:
        q_cpu: Queries [1, 1, seq_len, head_dim] on CPU
        k_cpu: Keys [1, 1, seq_len, head_dim] on CPU
        v_cpu: Values [1, 1, seq_len, head_dim] on CPU
        v_idx: Vertical column indices [nnz_v] (sorted ascending)
        s_idx: Slash diagonal offsets [nnz_s] (negative, sorted descending)
        head_dim: Dimension of each head
        q_chunk_size: Number of queries per chunk
        kv_batch_size: Number of K/V positions per batch
        show_progress: Show progress bar

    Returns:
        output: Attention output [1, 1, seq_len, head_dim] on CPU
    """
    seq_len = q_cpu.shape[2]
    output = torch.zeros_like(q_cpu)

    # Move indices to GPU
    v_idx_gpu = v_idx.cuda()
    s_idx_gpu = s_idx.cuda()

    num_q_chunks = (seq_len + q_chunk_size - 1) // q_chunk_size
    iterator = range(0, seq_len, q_chunk_size)
    if show_progress:
        iterator = tqdm(iterator, desc="Sparse attention", total=num_q_chunks)

    for q_start in iterator:
        q_end = min(q_start + q_chunk_size, seq_len)
        chunk_len = q_end - q_start

        # Load query chunk to GPU
        q_chunk = q_cpu[:, :, q_start:q_end, :].cuda().squeeze(0).squeeze(0)

        # Initialize online softmax state
        m_i = torch.full((chunk_len,), float('-inf'), device='cuda', dtype=torch.float32)
        l_i = torch.zeros(chunk_len, device='cuda', dtype=torch.float32)
        acc = torch.zeros(chunk_len, head_dim, device='cuda', dtype=torch.float32)

        query_positions = torch.arange(q_start, q_end, device='cuda')

        # Process vertical columns in batches
        for v_start in range(0, len(v_idx), kv_batch_size):
            v_end_batch = min(v_start + kv_batch_size, len(v_idx))
            v_batch = v_idx_gpu[v_start:v_end_batch]

            # Gather K, V for vertical positions
            k_vert = k_cpu[:, :, v_batch.cpu(), :].cuda().squeeze(0).squeeze(0)
            v_vert = v_cpu[:, :, v_batch.cpu(), :].cuda().squeeze(0).squeeze(0)

            # Compute attention scores
            qk = torch.matmul(q_chunk.float(), k_vert.float().T) / math.sqrt(head_dim)

            # Apply causal mask: only attend to keys where key_pos <= query_pos
            causal_mask = query_positions[:, None] >= v_batch[None, :]
            qk = torch.where(causal_mask, qk, torch.tensor(float('-inf'), device='cuda', dtype=qk.dtype))

            # Online softmax update
            m_i, l_i, acc = online_softmax_update(qk, v_vert.float(), m_i, l_i, acc)

            del k_vert, v_vert, qk

        # Process slash diagonals
        # For efficiency, we process queries in mini-batches and vectorize where possible
        slash_batch_size = min(64, chunk_len)
        for q_batch_start in range(0, chunk_len, slash_batch_size):
            q_batch_end = min(q_batch_start + slash_batch_size, chunk_len)

            for q_local_idx in range(q_batch_start, q_batch_end):
                query_pos = q_start + q_local_idx

                # Compute key positions for slash diagonals
                # s_idx contains positive distances: key_pos = query_pos - s_idx
                key_positions = query_pos - s_idx_gpu

                # Filter valid positions: 0 <= key_pos < seq_len and key_pos <= query_pos
                valid = (key_positions >= 0) & (key_positions < seq_len) & (key_positions <= query_pos)

                if not valid.any():
                    continue

                valid_keys = key_positions[valid]

                # Gather K, V for valid slash positions
                k_slash = k_cpu[:, :, valid_keys.cpu(), :].cuda().squeeze(0).squeeze(0)
                v_slash = v_cpu[:, :, valid_keys.cpu(), :].cuda().squeeze(0).squeeze(0)

                # Compute attention score for single query
                qk = torch.matmul(q_chunk[q_local_idx:q_local_idx+1].float(), k_slash.float().T)
                qk = qk.squeeze(0) / math.sqrt(head_dim)

                # Online softmax update for single query
                m_i[q_local_idx], l_i[q_local_idx], acc[q_local_idx] = online_softmax_update_single(
                    qk, v_slash.float(),
                    m_i[q_local_idx], l_i[q_local_idx], acc[q_local_idx]
                )

                del k_slash, v_slash, qk

        # Normalize and store output
        # Handle case where l_i is 0 (no valid attention)
        l_i_safe = torch.where(l_i == 0, torch.ones_like(l_i), l_i)
        chunk_output = acc / l_i_safe[:, None]

        output[:, :, q_start:q_end, :] = chunk_output.unsqueeze(0).unsqueeze(0).cpu().to(output.dtype)

        del q_chunk, m_i, l_i, acc, chunk_output
        torch.cuda.empty_cache()

    return output


def chunked_dense_attention(
    q_cpu: torch.Tensor,
    k_cpu: torch.Tensor,
    v_cpu: torch.Tensor,
    head_dim: int,
    q_chunk_size: int = 64,
    kv_chunk_size: int = 65536,
    show_progress: bool = False,
) -> torch.Tensor:
    """
    Compute dense causal attention using chunked online softmax.

    This is a reference implementation for validation. Not optimized for speed.

    Args:
        q_cpu: Queries [1, 1, seq_len, head_dim] on CPU
        k_cpu: Keys [1, 1, seq_len, head_dim] on CPU
        v_cpu: Values [1, 1, seq_len, head_dim] on CPU
        head_dim: Dimension of each head
        q_chunk_size: Number of queries per chunk
        kv_chunk_size: Number of K/V positions per chunk
        show_progress: Show progress bar

    Returns:
        output: Attention output [1, 1, seq_len, head_dim] on CPU
    """
    seq_len = q_cpu.shape[2]
    output = torch.zeros_like(q_cpu)

    num_q_chunks = (seq_len + q_chunk_size - 1) // q_chunk_size
    iterator = range(0, seq_len, q_chunk_size)
    if show_progress:
        iterator = tqdm(iterator, desc="Dense attention", total=num_q_chunks)

    for q_start in iterator:
        q_end = min(q_start + q_chunk_size, seq_len)
        chunk_len = q_end - q_start

        # Load query chunk to GPU
        q_chunk = q_cpu[:, :, q_start:q_end, :].cuda().squeeze(0).squeeze(0)

        # Initialize online softmax state
        m_i = torch.full((chunk_len,), float('-inf'), device='cuda', dtype=torch.float32)
        l_i = torch.zeros(chunk_len, device='cuda', dtype=torch.float32)
        acc = torch.zeros(chunk_len, head_dim, device='cuda', dtype=torch.float32)

        query_positions = torch.arange(q_start, q_end, device='cuda')

        # Process K, V in chunks
        # For causal attention, we only need keys up to q_end
        for kv_start in range(0, q_end, kv_chunk_size):
            kv_end = min(kv_start + kv_chunk_size, q_end)

            # Load K, V chunk
            k_chunk = k_cpu[:, :, kv_start:kv_end, :].cuda().squeeze(0).squeeze(0)
            v_chunk = v_cpu[:, :, kv_start:kv_end, :].cuda().squeeze(0).squeeze(0)

            # Compute attention scores
            qk = torch.matmul(q_chunk.float(), k_chunk.float().T) / math.sqrt(head_dim)

            # Causal mask
            key_positions = torch.arange(kv_start, kv_end, device='cuda')
            causal_mask = query_positions[:, None] >= key_positions[None, :]
            qk = torch.where(causal_mask, qk, torch.tensor(float('-inf'), device='cuda', dtype=qk.dtype))

            # Online softmax update
            m_i, l_i, acc = online_softmax_update(qk, v_chunk.float(), m_i, l_i, acc)

            del k_chunk, v_chunk, qk

        # Normalize
        l_i_safe = torch.where(l_i == 0, torch.ones_like(l_i), l_i)
        chunk_output = acc / l_i_safe[:, None]

        output[:, :, q_start:q_end, :] = chunk_output.unsqueeze(0).unsqueeze(0).cpu().to(output.dtype)

        del q_chunk, m_i, l_i, acc, chunk_output
        torch.cuda.empty_cache()

    return output


def minference_sparse_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    v_idx: torch.Tensor,
    s_idx: torch.Tensor,
) -> torch.Tensor:
    """
    Call MInference's vertical_slash_sparse_attention kernel directly.

    This function is a thin wrapper around MInference's sparse attention kernel.
    Use this when the full Q, K, V can fit in GPU memory.

    Args:
        q: Query tensor [1, 1, seq_len, head_dim] on GPU
        k: Key tensor [1, 1, seq_len, head_dim] on GPU
        v: Value tensor [1, 1, seq_len, head_dim] on GPU
        v_idx: Vertical column indices [nnz_v]
        s_idx: Slash diagonal offsets [nnz_s] (negative values)

    Returns:
        output: Attention output [1, 1, seq_len, head_dim] on GPU
    """
    batch_size, num_heads, seq_len, head_dim = q.shape

    # Reshape v_idx and s_idx for MInference kernel
    # MInference expects [batch, heads, nnz]
    v_idx_reshaped = v_idx.unsqueeze(0).unsqueeze(0)  # [1, 1, nnz_v]
    s_idx_reshaped = s_idx.unsqueeze(0).unsqueeze(0)  # [1, 1, nnz_s]

    # Call MInference's sparse attention kernel
    output = vertical_slash_sparse_attention(
        q, k, v, v_idx_reshaped, s_idx_reshaped
    )

    return output


def chunked_minference_sparse_attention(
    q_cpu: torch.Tensor,
    k_cpu: torch.Tensor,
    v_cpu: torch.Tensor,
    v_idx: torch.Tensor,
    s_idx: torch.Tensor,
    head_dim: int,
    q_chunk_size: int = 8192,
    block_size: int = 64,
    show_progress: bool = False,
) -> torch.Tensor:
    """
    Chunked sparse attention using MInference's kernel for each chunk.

    This function processes the sequence in chunks, calling MInference's
    sparse attention kernel for each chunk. This enables processing of
    sequences that don't fit in GPU memory.

    Note: This is an approximation since the sparse pattern was computed
    for the full sequence, but applied to chunks. For accurate results
    on very long sequences, use chunked_sparse_attention() which uses
    online softmax.

    Args:
        q_cpu: Queries [1, 1, seq_len, head_dim] on CPU
        k_cpu: Keys [1, 1, seq_len, head_dim] on CPU
        v_cpu: Values [1, 1, seq_len, head_dim] on CPU
        v_idx: Vertical column indices [nnz_v]
        s_idx: Slash diagonal offsets [nnz_s]
        head_dim: Dimension of each head
        q_chunk_size: Number of queries per chunk
        block_size: Block size for MInference kernel
        show_progress: Show progress bar

    Returns:
        output: Attention output [1, 1, seq_len, head_dim] on CPU
    """
    seq_len = q_cpu.shape[2]
    output = torch.zeros_like(q_cpu)

    num_chunks = (seq_len + q_chunk_size - 1) // q_chunk_size
    iterator = range(0, seq_len, q_chunk_size)
    if show_progress:
        iterator = tqdm(iterator, desc="MInference chunked attention", total=num_chunks)

    for q_start in iterator:
        q_end = min(q_start + q_chunk_size, seq_len)

        # For this chunk, we need to compute attention for queries [q_start:q_end]
        # Against all keys that they can attend to (causal: keys <= query position)

        # Load Q chunk
        q_chunk = q_cpu[:, :, q_start:q_end, :].cuda()

        # For causal attention, we need keys from [0, q_end]
        k_chunk = k_cpu[:, :, :q_end, :].cuda()
        v_chunk = v_cpu[:, :, :q_end, :].cuda()

        # Filter v_idx to only include positions < q_end
        valid_v_mask = v_idx < q_end
        valid_v_idx = v_idx[valid_v_mask]

        # Filter s_idx to only include diagonals that are valid for this chunk
        # For query at position q, diagonal d means key at q + d (d is negative)
        # We need keys in [0, q_end], so we need q_start + d >= 0
        valid_s_mask = (q_start + s_idx) >= 0
        valid_s_idx = s_idx[valid_s_mask]

        if len(valid_v_idx) > 0 and len(valid_s_idx) > 0:
            # Create adjusted indices for the chunk
            # Note: We're computing attention for a submatrix
            # This is an approximation - the kernel expects the full sequence context

            # Call MInference sparse attention on the chunk
            # We need to handle the offset: queries at [q_start:q_end] attending to keys [0:q_end]
            chunk_output = vertical_slash_sparse_attention(
                q_chunk, k_chunk, v_chunk,
                valid_v_idx.unsqueeze(0).unsqueeze(0),
                valid_s_idx.unsqueeze(0).unsqueeze(0),
                block_size_M=block_size,
                block_size_N=block_size,
            )

            # Extract only the part for our queries (last q_end - q_start positions)
            output[:, :, q_start:q_end, :] = chunk_output[:, :, q_start:q_end, :].cpu()
        else:
            # Fall back to zero output if no valid patterns
            pass

        del q_chunk, k_chunk, v_chunk
        torch.cuda.empty_cache()

    return output
