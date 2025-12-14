# Copyright (c) 2024-2025 Microsoft
# Licensed under The MIT License [see LICENSE for details]

"""
Correctness Verification: Vertical-Slash Sparse Attention

This script verifies the numerical correctness of MInference's Vertical-Slash
sparse attention kernel by comparing its output against dense Flash Attention.

Usage:
    python experiments/microbench/verify_sparse.py
    python experiments/microbench/verify_sparse.py --seq_lens 1024 2048 4096
    python experiments/microbench/verify_sparse.py --verbose
"""

import argparse
import math
from typing import List, Tuple, Dict

import torch
import triton
import triton.language as tl


# ============================================================================
# Triton Kernels (copied from minference/ops for standalone execution)
# ============================================================================

@triton.jit
def _triton_mixed_sparse_attn_fwd_kernel(
    Q, K, V, seqlens, sm_scale,
    block_count, block_offset, column_count, column_index,
    Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    Z, H, N_CTX,
    NUM_ROWS, NNZ_S, NNZ_V,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    dtype: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)

    seqlen = tl.load(seqlens + off_hz // H)
    if start_m * BLOCK_M >= seqlen:
        return

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    qo_offset = (off_hz // H) * stride_qz + (off_hz % H) * stride_qh
    kv_offset = (off_hz // H) * stride_kz + (off_hz % H) * stride_kh

    q_ptrs = Q + qo_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    k_ptrs = K + kv_offset + offs_d[:, None] * stride_kk
    v_ptrs = V + kv_offset + offs_d[None, :] * stride_vk
    o_ptrs = Out + qo_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok

    num_blks = tl.load(block_count + off_hz * NUM_ROWS + start_m)
    blks_ptr = block_offset + (off_hz * NUM_ROWS + start_m) * NNZ_S
    num_cols = tl.load(column_count + off_hz * NUM_ROWS + start_m)
    cols_ptr = column_index + (off_hz * NUM_ROWS + start_m) * NNZ_V

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    qk_scale = sm_scale * 1.44269504
    q = tl.load(q_ptrs)
    q = (q * qk_scale).to(dtype)

    m_mask = offs_m[:, None] < seqlen

    # Loop over slash blocks
    for block_index in range(num_blks):
        start_n = tl.load(blks_ptr + block_index)
        cols = start_n + offs_n
        n_mask = cols < seqlen
        k = tl.load(k_ptrs + cols[None, :] * stride_kn, mask=n_mask[None, :], other=0.0)
        v = tl.load(v_ptrs + cols[:, None] * stride_vn, mask=n_mask[:, None], other=0.0)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        causal_mask = cols[None, :] <= offs_m[:, None]
        qk = tl.where(m_mask & causal_mask, qk, float("-inf"))
        qk += tl.dot(q, k)
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.math.exp2(m_i - m_i_new)
        p = tl.math.exp2(qk - m_i_new[:, None])
        acc_scale = l_i * 0 + alpha
        acc *= acc_scale[:, None]
        acc += tl.dot(p.to(dtype), v)
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new

    # Loop over vertical columns
    for start_n in range(0, num_cols, BLOCK_N):
        n_mask = start_n + offs_n < num_cols
        cols = tl.load(cols_ptr + start_n + offs_n, mask=n_mask, other=0)
        k = tl.load(k_ptrs + cols[None, :] * stride_kn, mask=n_mask[None, :], other=0.0)
        v = tl.load(v_ptrs + cols[:, None] * stride_vn, mask=n_mask[:, None], other=0.0)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk = tl.where(m_mask & n_mask, qk, float("-inf"))
        qk += tl.dot(q, k)
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.math.exp2(m_i - m_i_new)
        p = tl.math.exp2(qk - m_i_new[:, None])
        acc_scale = l_i * 0 + alpha
        acc *= acc_scale[:, None]
        acc += tl.dot(p.to(dtype), v)
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new

    acc /= l_i[:, None]
    tl.store(o_ptrs, acc.to(dtype), mask=m_mask)


def triton_mixed_sparse_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seqlens: torch.Tensor,
    block_count: torch.Tensor,
    block_offset: torch.Tensor,
    column_count: torch.Tensor,
    column_index: torch.Tensor,
    sm_scale: float,
    block_size_M: int = 64,
    block_size_N: int = 64,
) -> torch.Tensor:
    Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
    assert Lq == Lk and Lk == Lv
    assert Lk in {16, 32, 64, 128}
    o = torch.zeros_like(q)
    grid = (triton.cdiv(q.shape[2], block_size_M), q.shape[0] * q.shape[1], 1)
    dtype = tl.bfloat16 if q.dtype == torch.bfloat16 else tl.float16
    _triton_mixed_sparse_attn_fwd_kernel[grid](
        q, k, v, seqlens, sm_scale,
        block_count, block_offset, column_count, column_index,
        o,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        q.shape[0], q.shape[1], q.shape[2],
        block_count.shape[-1], block_offset.shape[-1], column_index.shape[-1],
        BLOCK_M=block_size_M, BLOCK_N=block_size_N,
        BLOCK_DMODEL=Lk,
        dtype=dtype,
        num_warps=4, num_stages=2,
    )
    return o


# ============================================================================
# Index Conversion (Python implementation for standalone use)
# ============================================================================

def convert_vertical_slash_indexes_python(
    seqlens: torch.Tensor,
    vertical_indexes: torch.Tensor,
    slash_indexes: torch.Tensor,
    context_size: int,
    block_size_M: int = 64,
    block_size_N: int = 64,
):
    """
    Python implementation of convert_vertical_slash_indexes.
    Converts vertical and slash indices into block-based format for the kernel.
    """
    batch_size = vertical_indexes.shape[0]
    num_heads = vertical_indexes.shape[1]
    nnz_v = vertical_indexes.shape[2]
    nnz_s = slash_indexes.shape[2]
    num_rows = (context_size + block_size_M - 1) // block_size_M

    block_count = torch.zeros((batch_size, num_heads, num_rows), dtype=torch.int32, device=vertical_indexes.device)
    block_offset = torch.zeros((batch_size, num_heads, num_rows, nnz_s), dtype=torch.int32, device=vertical_indexes.device)
    column_count = torch.zeros((batch_size, num_heads, num_rows), dtype=torch.int32, device=vertical_indexes.device)
    column_index = torch.zeros((batch_size, num_heads, num_rows, nnz_v), dtype=torch.int32, device=vertical_indexes.device)

    for b in range(batch_size):
        for h in range(num_heads):
            v_idx = vertical_indexes[b, h].cpu().numpy()
            s_idx = slash_indexes[b, h].cpu().numpy()

            for row in range(num_rows):
                start_m = row * block_size_M
                end_m = min(start_m + block_size_M, context_size)

                # Process slash blocks
                blk_cnt = 0
                processed_ranges = []

                for s in s_idx:
                    if s >= end_m:
                        continue
                    # Convert slash index to block start
                    blk_start = max(0, end_m - s - block_size_M)
                    blk_start = (blk_start // block_size_N) * block_size_N

                    # Check if overlaps with existing ranges
                    merged = False
                    for i, (rs, re) in enumerate(processed_ranges):
                        if blk_start <= re and blk_start + block_size_N >= rs:
                            processed_ranges[i] = (min(rs, blk_start), max(re, blk_start + block_size_N))
                            merged = True
                            break
                    if not merged:
                        processed_ranges.append((blk_start, blk_start + block_size_N))

                # Save block offsets
                for rs, re in sorted(processed_ranges):
                    for blk in range(rs, min(re, end_m), block_size_N):
                        if blk_cnt < nnz_s:
                            block_offset[b, h, row, blk_cnt] = blk
                            blk_cnt += 1
                block_count[b, h, row] = blk_cnt

                # Process vertical columns (exclude those in slash blocks)
                col_cnt = 0
                for v in v_idx:
                    if v >= end_m:
                        continue
                    # Check if in any slash block
                    in_slash = False
                    for rs, re in processed_ranges:
                        if rs <= v < re:
                            in_slash = True
                            break
                    if not in_slash and col_cnt < nnz_v:
                        column_index[b, h, row, col_cnt] = v
                        col_cnt += 1
                column_count[b, h, row] = col_cnt

    return block_count, block_offset, column_count, column_index


# ============================================================================
# Flash Attention Reference
# ============================================================================

try:
    from flash_attn import flash_attn_func
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False


@triton.jit
def _flash_attn_fwd_kernel(
    Q, K, V, Out,
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_kn, stride_kk,
    stride_vb, stride_vh, stride_vn, stride_vk,
    stride_ob, stride_oh, stride_om, stride_ok,
    seqlen, sm_scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    dtype: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hb = tl.program_id(1)

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    q_ptrs = Q + off_hb * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    k_ptrs = K + off_hb * stride_kh + offs_d[:, None] * stride_kk
    v_ptrs = V + off_hb * stride_vh + offs_d[None, :] * stride_vk
    o_ptrs = Out + off_hb * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    qk_scale = sm_scale * 1.44269504
    q = tl.load(q_ptrs, mask=offs_m[:, None] < seqlen, other=0.0)
    q = (q * qk_scale).to(dtype)

    end_n = seqlen if not IS_CAUSAL else min((start_m + 1) * BLOCK_M, seqlen)

    for start_n in range(0, end_n, BLOCK_N):
        cols = start_n + offs_n
        n_mask = cols < seqlen
        k = tl.load(k_ptrs + cols[None, :] * stride_kn, mask=n_mask[None, :], other=0.0)
        v = tl.load(v_ptrs + cols[:, None] * stride_vn, mask=n_mask[:, None], other=0.0)

        qk = tl.dot(q, k)
        if IS_CAUSAL:
            causal_mask = cols[None, :] <= offs_m[:, None]
            qk = tl.where(causal_mask & n_mask[None, :], qk, float("-inf"))
        else:
            qk = tl.where(n_mask[None, :], qk, float("-inf"))

        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.math.exp2(m_i - m_i_new)
        p = tl.math.exp2(qk - m_i_new[:, None])
        acc = acc * alpha[:, None] + tl.dot(p.to(dtype), v)
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new

    acc /= l_i[:, None]
    tl.store(o_ptrs, acc.to(dtype), mask=offs_m[:, None] < seqlen)


def triton_flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = True,
) -> torch.Tensor:
    """Triton Flash Attention implementation."""
    batch, heads, seqlen, head_dim = q.shape
    assert head_dim in [16, 32, 64, 128]

    o = torch.zeros_like(q)
    sm_scale = head_dim ** -0.5

    BLOCK_M = 64
    BLOCK_N = 64

    grid = (triton.cdiv(seqlen, BLOCK_M), batch * heads)
    dtype = tl.bfloat16 if q.dtype == torch.bfloat16 else tl.float16

    _flash_attn_fwd_kernel[grid](
        q, k, v, o,
        q.stride(0) * q.stride(1), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0) * k.stride(1), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0) * v.stride(1), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0) * o.stride(1), o.stride(1), o.stride(2), o.stride(3),
        seqlen, sm_scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        BLOCK_DMODEL=head_dim,
        IS_CAUSAL=causal,
        dtype=dtype,
        num_warps=4, num_stages=2,
    )
    return o


def flash_attention_reference(q, k, v, causal=True):
    """Flash Attention reference implementation."""
    if HAS_FLASH_ATTN:
        q_t = q.transpose(1, 2).contiguous()
        k_t = k.transpose(1, 2).contiguous()
        v_t = v.transpose(1, 2).contiguous()
        out = flash_attn_func(q_t, k_t, v_t, causal=causal)
        return out.transpose(1, 2).contiguous()
    else:
        return triton_flash_attention(q, k, v, causal=causal)


# ============================================================================
# Vertical-Slash Sparse Attention Wrapper
# ============================================================================

def prepare_sparse_indices(
    batch_size: int,
    num_heads: int,
    context_size: int,
    v_idx: torch.Tensor,
    s_idx: torch.Tensor,
    block_size_M: int = 64,
    block_size_N: int = 64,
):
    """Pre-compute sparse indices."""
    pad = (block_size_M - context_size % block_size_M) % block_size_M
    padded_size = context_size + pad

    v_idx = v_idx.to(torch.int32).reshape((batch_size, num_heads, -1))
    v_idx = v_idx.sort(dim=-1, descending=False)[0]
    s_idx = s_idx.to(torch.int32).reshape((batch_size, num_heads, -1))
    s_idx = s_idx.sort(dim=-1, descending=True)[0]

    seqlens = torch.tensor([context_size], dtype=torch.int32, device=v_idx.device)

    block_count, block_offset, column_count, column_index = convert_vertical_slash_indexes_python(
        seqlens, v_idx, s_idx, padded_size, block_size_M, block_size_N,
    )

    return {
        'seqlens': seqlens,
        'block_count': block_count,
        'block_offset': block_offset,
        'column_count': column_count,
        'column_index': column_index,
        'pad': pad,
    }


def vertical_slash_sparse_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    v_idx: torch.Tensor,
    s_idx: torch.Tensor,
    block_size_M: int = 64,
    block_size_N: int = 64,
) -> torch.Tensor:
    """Vertical-Slash sparse attention."""
    batch_size, num_heads, context_size, head_dim = query.shape
    indices = prepare_sparse_indices(batch_size, num_heads, context_size, v_idx, s_idx, block_size_M, block_size_N)
    pad = indices['pad']

    if pad > 0:
        query = torch.nn.functional.pad(query, [0, 0, 0, pad])
        key = torch.nn.functional.pad(key, [0, 0, 0, pad])
        value = torch.nn.functional.pad(value, [0, 0, 0, pad])

    sm_scale = head_dim ** -0.5

    out = triton_mixed_sparse_attention(
        query, key, value, indices['seqlens'],
        indices['block_count'], indices['block_offset'],
        indices['column_count'], indices['column_index'],
        sm_scale, block_size_M, block_size_N,
    )

    return out[..., :context_size, :]


# ============================================================================
# Verification Functions
# ============================================================================

def generate_sparse_pattern(
    batch_size: int,
    num_heads: int,
    seq_len: int,
    num_vertical: int,
    num_slash: int,
    device: str = "cuda",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate synthetic sparse patterns for testing."""
    v_idx = torch.randint(0, seq_len // 2, (batch_size, num_heads, num_vertical),
                          device=device, dtype=torch.int32)
    s_idx = torch.randint(0, seq_len, (batch_size, num_heads, num_slash),
                          device=device, dtype=torch.int32)
    return v_idx, s_idx


def compute_metrics(sparse_out: torch.Tensor, dense_out: torch.Tensor) -> Dict[str, float]:
    """Compute various error metrics between sparse and dense outputs."""
    # Flatten for easier computation
    sparse_flat = sparse_out.float().flatten()
    dense_flat = dense_out.float().flatten()

    # Absolute error
    abs_error = (sparse_flat - dense_flat).abs()
    max_abs_error = abs_error.max().item()
    mean_abs_error = abs_error.mean().item()

    # Relative error (avoid division by zero)
    rel_error = abs_error / (dense_flat.abs() + 1e-8)
    max_rel_error = rel_error.max().item()
    mean_rel_error = rel_error.mean().item()

    # Cosine similarity
    cosine_sim = torch.nn.functional.cosine_similarity(
        sparse_flat.unsqueeze(0), dense_flat.unsqueeze(0)
    ).item()

    # Check for NaN/Inf
    has_nan = torch.isnan(sparse_out).any().item()
    has_inf = torch.isinf(sparse_out).any().item()

    return {
        'max_abs_error': max_abs_error,
        'mean_abs_error': mean_abs_error,
        'max_rel_error': max_rel_error,
        'mean_rel_error': mean_rel_error,
        'cosine_similarity': cosine_sim,
        'has_nan': has_nan,
        'has_inf': has_inf,
    }


def verify_single_config(
    seq_len: int,
    batch_size: int = 1,
    num_heads: int = 4,
    head_dim: int = 128,
    num_vertical: int = 64,
    num_slash: int = 32,
    dtype: torch.dtype = torch.bfloat16,
    verbose: bool = False,
) -> Dict:
    """Verify correctness for a single configuration."""
    device = "cuda"

    # Generate inputs
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)

    # Generate sparse pattern
    v_idx, s_idx = generate_sparse_pattern(batch_size, num_heads, seq_len, num_vertical, num_slash, device)

    # Compute outputs
    sparse_out = vertical_slash_sparse_attention(q, k, v, v_idx, s_idx)
    dense_out = flash_attention_reference(q, k, v)

    # Compute metrics
    metrics = compute_metrics(sparse_out, dense_out)

    result = {
        'seq_len': seq_len,
        'batch_size': batch_size,
        'num_heads': num_heads,
        'head_dim': head_dim,
        'num_vertical': num_vertical,
        'num_slash': num_slash,
        'dtype': str(dtype),
        **metrics,
    }

    if verbose:
        print(f"\n  Config: seq_len={seq_len}, batch={batch_size}, heads={num_heads}, "
              f"head_dim={head_dim}")
        print(f"  Sparse pattern: num_vertical={num_vertical}, num_slash={num_slash}")
        print(f"  Output shape: {sparse_out.shape}")
        print(f"  Max absolute error: {metrics['max_abs_error']:.6e}")
        print(f"  Mean absolute error: {metrics['mean_abs_error']:.6e}")
        print(f"  Max relative error: {metrics['max_rel_error']:.6e}")
        print(f"  Mean relative error: {metrics['mean_rel_error']:.6e}")
        print(f"  Cosine similarity: {metrics['cosine_similarity']:.8f}")
        print(f"  Has NaN: {metrics['has_nan']}, Has Inf: {metrics['has_inf']}")

    return result


def run_verification(
    seq_lens: List[int],
    batch_size: int = 1,
    num_heads: int = 4,
    head_dim: int = 128,
    sparsity_ratios: List[float] = [0.05, 0.1],
    dtype: torch.dtype = torch.bfloat16,
    verbose: bool = False,
):
    """Run comprehensive verification across multiple configurations."""
    print("=" * 70)
    print("Vertical-Slash Sparse Attention Correctness Verification")
    print("=" * 70)
    print(f"Config: batch={batch_size}, heads={num_heads}, head_dim={head_dim}")
    print(f"Dtype: {dtype}")
    print(f"Flash Attention backend: {'flash_attn' if HAS_FLASH_ATTN else 'triton'}")
    print("=" * 70)

    results = []
    all_passed = True

    for seq_len in seq_lens:
        print(f"\n{'='*50}")
        print(f"Sequence Length: {seq_len:,}")
        print(f"{'='*50}")

        for sparsity in sparsity_ratios:
            block_size = 64
            num_vertical = max(32, min(512, int(seq_len * sparsity * 0.5)))
            num_slash = max(8, min(128, int(seq_len * sparsity * 0.5 / block_size)))

            print(f"\n[Sparsity ~{sparsity:.0%}] num_vertical={num_vertical}, num_slash={num_slash}")

            result = verify_single_config(
                seq_len=seq_len,
                batch_size=batch_size,
                num_heads=num_heads,
                head_dim=head_dim,
                num_vertical=num_vertical,
                num_slash=num_slash,
                dtype=dtype,
                verbose=verbose,
            )
            results.append(result)

            # Check if test passed
            passed = (
                not result['has_nan'] and
                not result['has_inf'] and
                result['cosine_similarity'] > 0.99
            )

            if passed:
                print(f"  Status: PASSED (cosine_sim={result['cosine_similarity']:.6f})")
            else:
                print(f"  Status: FAILED")
                print(f"    - Has NaN: {result['has_nan']}")
                print(f"    - Has Inf: {result['has_inf']}")
                print(f"    - Cosine similarity: {result['cosine_similarity']:.6f}")
                all_passed = False

    # Summary
    print("\n" + "=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)
    print(f"{'Seq Len':>10} | {'Sparsity':>10} | {'Max Abs Err':>12} | {'Cosine Sim':>12} | {'Status':>8}")
    print("-" * 70)

    for r in results:
        sparsity = r['num_vertical'] * r['seq_len'] / (r['seq_len'] * (r['seq_len'] + 1) // 2)
        passed = not r['has_nan'] and not r['has_inf'] and r['cosine_similarity'] > 0.99
        status = "PASS" if passed else "FAIL"
        print(f"{r['seq_len']:>10,} | {sparsity:>10.2%} | {r['max_abs_error']:>12.2e} | "
              f"{r['cosine_similarity']:>12.8f} | {status:>8}")

    print("-" * 70)
    if all_passed:
        print("Overall: ALL TESTS PASSED")
    else:
        print("Overall: SOME TESTS FAILED")

    return results, all_passed


def main():
    parser = argparse.ArgumentParser(description="Verify Vertical-Slash Sparse Attention Correctness")
    parser.add_argument("--seq_lens", type=int, nargs="+", default=[512, 1024, 2048, 4096])
    parser.add_argument("--sparsity", type=float, nargs="+", default=[0.05, 0.1])
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--head_dim", type=int, default=128)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["fp16", "bf16"])
    parser.add_argument("--verbose", action="store_true", help="Print detailed metrics")

    args = parser.parse_args()
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    results, all_passed = run_verification(
        seq_lens=args.seq_lens,
        batch_size=args.batch_size,
        num_heads=args.num_heads,
        head_dim=args.head_dim,
        sparsity_ratios=args.sparsity,
        dtype=dtype,
        verbose=args.verbose,
    )

    # Exit with non-zero code if tests failed
    exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
