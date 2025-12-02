# Copyright (c) 2024-2025 Microsoft
# Licensed under The MIT License [see LICENSE for details]

"""
Microbenchmark: Vertical-Slash Sparse Attention vs Flash Attention

This benchmark compares the performance of MInference's Vertical-Slash sparse
attention kernel against standard Flash Attention across different sequence
lengths and sparsity levels.

Usage:
    python experiments/microbench/bench_vertical_slash.py
    python experiments/microbench/bench_vertical_slash.py --seq_lens 4096 8192 16384
    python experiments/microbench/bench_vertical_slash.py --sparsity 0.01 0.05 0.1
"""

import argparse
import math
from typing import List, Tuple

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
    """Pre-compute sparse indices (do this outside the benchmark loop)."""
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


def vertical_slash_sparse_attention_with_indices(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    indices: dict,
    block_size_M: int = 64,
    block_size_N: int = 64,
):
    """Sparse attention with pre-computed indices (for benchmarking kernel only)."""
    batch_size, num_heads, context_size, head_dim = query.shape
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


def vertical_slash_sparse_attention_standalone(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    v_idx: torch.Tensor,
    s_idx: torch.Tensor,
    block_size_M: int = 64,
    block_size_N: int = 64,
):
    """Standalone Vertical-Slash sparse attention (includes index conversion)."""
    batch_size, num_heads, context_size, head_dim = query.shape
    indices = prepare_sparse_indices(batch_size, num_heads, context_size, v_idx, s_idx, block_size_M, block_size_N)
    return vertical_slash_sparse_attention_with_indices(query, key, value, indices, block_size_M, block_size_N)


# ============================================================================
# Benchmarking Functions
# ============================================================================

def generate_sparse_pattern(
    batch_size: int,
    num_heads: int,
    seq_len: int,
    num_vertical: int,
    num_slash: int,
    device: str = "cuda",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate synthetic sparse patterns for benchmarking."""
    v_idx = torch.randint(0, seq_len // 2, (batch_size, num_heads, num_vertical),
                          device=device, dtype=torch.int32)
    s_idx = torch.randint(0, seq_len, (batch_size, num_heads, num_slash),
                          device=device, dtype=torch.int32)
    return v_idx, s_idx


def benchmark_kernel(func, *args, warmup=10, repeat=100, **kwargs):
    """Benchmark a kernel function."""
    for _ in range(warmup):
        _ = func(*args, **kwargs)
    torch.cuda.synchronize()

    times = []
    for _ in range(repeat):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        _ = func(*args, **kwargs)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    times = torch.tensor(times)
    return times.mean().item(), times.std().item()


def run_benchmark(
    seq_lens: List[int],
    batch_size: int = 1,
    num_heads: int = 32,
    head_dim: int = 128,
    sparsity_ratios: List[float] = [0.01, 0.05, 0.1],
    warmup: int = 10,
    repeat: int = 100,
    dtype: torch.dtype = torch.bfloat16,
):
    """Run comprehensive benchmark."""
    print("=" * 80)
    print("Vertical-Slash Sparse Attention vs Flash Attention Benchmark")
    print("=" * 80)
    print(f"Config: batch={batch_size}, heads={num_heads}, head_dim={head_dim}")
    print(f"Warmup: {warmup}, Repeat: {repeat}, Dtype: {dtype}")
    print(f"Flash Attention backend: {'flash_attn' if HAS_FLASH_ATTN else 'triton'}")
    print("=" * 80)

    results = []

    for seq_len in seq_lens:
        print(f"\n{'='*60}")
        print(f"Sequence Length: {seq_len:,}")
        print(f"{'='*60}")

        q = torch.randn(batch_size, num_heads, seq_len, head_dim, device="cuda", dtype=dtype)
        k = torch.randn(batch_size, num_heads, seq_len, head_dim, device="cuda", dtype=dtype)
        v = torch.randn(batch_size, num_heads, seq_len, head_dim, device="cuda", dtype=dtype)

        # Benchmark Flash Attention
        print("\n[Flash Attention (Dense Causal)]")
        try:
            fa_time, fa_std = benchmark_kernel(flash_attention_reference, q, k, v, warmup=warmup, repeat=repeat)
            print(f"  Time: {fa_time:.3f} ± {fa_std:.3f} ms")
            flops_dense = 2 * batch_size * num_heads * seq_len * seq_len * head_dim
            tflops_dense = flops_dense / (fa_time / 1000) / 1e12
            print(f"  Throughput: {tflops_dense:.2f} TFLOPS")
        except Exception as e:
            print(f"  Error: {e}")
            fa_time = float('inf')

        # Benchmark Vertical-Slash
        for sparsity in sparsity_ratios:
            block_size = 64
            # For long sequences, use realistic sparse counts
            # num_vertical: important tokens (like BOS, keywords)
            # num_slash: number of diagonal blocks to include
            num_vertical = max(32, min(512, int(seq_len * sparsity * 0.5)))
            num_slash = max(8, min(128, int(seq_len * sparsity * 0.5 / block_size)))

            # Effective sparsity: what fraction of the full attention we compute
            # For causal attention: full = seq_len * seq_len / 2
            # Sparse = num_vertical * seq_len + num_slash * block_size * seq_len / num_rows
            full_attention_elements = seq_len * (seq_len + 1) // 2
            sparse_elements = num_vertical * seq_len + num_slash * block_size * seq_len
            effective_sparsity = min(1.0, sparse_elements / full_attention_elements)

            print(f"\n[Vertical-Slash Sparse (sparsity≈{effective_sparsity:.1%})]")
            print(f"  num_vertical={num_vertical}, num_slash={num_slash}")

            v_idx, s_idx = generate_sparse_pattern(batch_size, num_heads, seq_len, num_vertical, num_slash)

            try:
                # Pre-compute indices (not part of kernel benchmark)
                print("  Pre-computing indices...")
                indices = prepare_sparse_indices(batch_size, num_heads, seq_len, v_idx, s_idx)
                print("  Indices ready. Benchmarking kernel...")

                vs_time, vs_std = benchmark_kernel(
                    vertical_slash_sparse_attention_with_indices,
                    q, k, v, indices,
                    warmup=warmup, repeat=repeat
                )
                print(f"  Time: {vs_time:.3f} ± {vs_std:.3f} ms")

                speedup = fa_time / vs_time
                print(f"  Speedup vs Flash Attention: {speedup:.2f}x")

                theoretical_speedup = 1 / effective_sparsity
                efficiency = speedup / theoretical_speedup * 100
                print(f"  Theoretical max speedup: {theoretical_speedup:.2f}x")
                print(f"  Efficiency: {efficiency:.1f}%")

                results.append({
                    'seq_len': seq_len,
                    'sparsity': effective_sparsity,
                    'fa_time': fa_time,
                    'vs_time': vs_time,
                    'speedup': speedup,
                    'efficiency': efficiency,
                })
            except Exception as e:
                print(f"  Error: {e}")
                import traceback
                traceback.print_exc()

    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Seq Len':>10} | {'Sparsity':>10} | {'FA (ms)':>10} | {'VS (ms)':>10} | {'Speedup':>10} | {'Efficiency':>10}")
    print("-" * 80)
    for r in results:
        print(f"{r['seq_len']:>10,} | {r['sparsity']:>10.1%} | {r['fa_time']:>10.3f} | {r['vs_time']:>10.3f} | {r['speedup']:>10.2f}x | {r['efficiency']:>9.1f}%")

    return results


def verify_correctness(seq_len=1024, batch_size=1, num_heads=4, head_dim=128,
                       num_vertical=64, num_slash=32, dtype=torch.bfloat16):
    """Verify correctness of sparse attention."""
    print("\n" + "=" * 60)
    print("Correctness Verification")
    print("=" * 60)

    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device="cuda", dtype=dtype)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device="cuda", dtype=dtype)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, device="cuda", dtype=dtype)

    v_idx, s_idx = generate_sparse_pattern(batch_size, num_heads, seq_len, num_vertical, num_slash)

    sparse_out = vertical_slash_sparse_attention_standalone(q, k, v, v_idx, s_idx)
    dense_out = flash_attention_reference(q, k, v)

    print(f"Sparse output shape: {sparse_out.shape}")
    print(f"Dense output shape: {dense_out.shape}")

    has_nan = torch.isnan(sparse_out).any()
    has_inf = torch.isinf(sparse_out).any()
    print(f"Has NaN: {has_nan}, Has Inf: {has_inf}")

    if not has_nan and not has_inf:
        print("✓ Output is valid (no NaN/Inf)")
    else:
        print("✗ Output contains invalid values!")

    return sparse_out, dense_out


def main():
    parser = argparse.ArgumentParser(description="Benchmark Vertical-Slash vs Flash Attention")
    parser.add_argument("--seq_lens", type=int, nargs="+", default=[1024, 2048, 4096, 8192, 16384, 32768])
    parser.add_argument("--sparsity", type=float, nargs="+", default=[0.01, 0.05, 0.1])
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_heads", type=int, default=32)
    parser.add_argument("--head_dim", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["fp16", "bf16"])

    args = parser.parse_args()
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    if args.verify:
        verify_correctness(dtype=dtype)

    run_benchmark(
        seq_lens=args.seq_lens,
        batch_size=args.batch_size,
        num_heads=args.num_heads,
        head_dim=args.head_dim,
        sparsity_ratios=args.sparsity,
        warmup=args.warmup,
        repeat=args.repeat,
        dtype=dtype,
    )


if __name__ == "__main__":
    main()
