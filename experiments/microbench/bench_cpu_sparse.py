# Copyright (c) 2024-2025 Microsoft
# Licensed under The MIT License [see LICENSE for details]

"""
Benchmark CPU SIMD Sparse Attention vs GPU implementations.

This script compares:
1. CPU C++ AVX-512 implementation
2. GPU Flash Attention
3. GPU MInference sparse attention

Key research questions:
1. At what sequence length does CPU become competitive?
2. How much does fine-grained scheduling save (causal mask skipping)?
3. What is the overhead of pattern estimation?
"""

import time
import torch
import argparse

# Import CPU sparse attention
from minference.ops.cpu_sparse_attention import (
    vertical_slash_attention_cpu,
    estimate_pattern_cpu,
    analyze_sparsity_cpu,
    HAS_CPU_EXTENSION,
)

# Import GPU implementations for comparison
try:
    from flash_attn.flash_attn_interface import flash_attn_func
    HAS_FLASH_ATTN = True
except ImportError:
    try:
        from flash_attn import flash_attn_func
        HAS_FLASH_ATTN = True
    except ImportError:
        HAS_FLASH_ATTN = False

try:
    from minference.ops.pit_sparse_flash_attention_v2 import vertical_slash_sparse_attention
    HAS_MINFERENCE = True
except ImportError:
    HAS_MINFERENCE = False


def benchmark_kernel(func, *args, warmup=3, repeat=10, sync_cuda=False, **kwargs):
    """Benchmark a kernel function."""
    # Warmup
    for _ in range(warmup):
        _ = func(*args, **kwargs)

    if sync_cuda:
        torch.cuda.synchronize()

    # Measure
    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        _ = func(*args, **kwargs)
        if sync_cuda:
            torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000)  # ms

    times = torch.tensor(times)
    return times.mean().item(), times.std().item()


def run_benchmark(seq_len: int, head_dim: int = 128, num_heads: int = 1,
                  vertical_size: int = 100, slash_size: int = 500,
                  warmup: int = 3, repeat: int = 10):
    """Run benchmark for a specific configuration."""
    print(f"\n{'='*60}")
    print(f"Benchmark: seq_len={seq_len}, head_dim={head_dim}, heads={num_heads}")
    print(f"Pattern: vertical={vertical_size}, slash={slash_size}")
    print(f"{'='*60}")

    # Generate random Q, K, V
    Q = torch.randn(1, num_heads, seq_len, head_dim)
    K = torch.randn(1, num_heads, seq_len, head_dim)
    V = torch.randn(1, num_heads, seq_len, head_dim)

    results = {}

    # Estimate pattern and analyze sparsity
    v_idx, s_idx = estimate_pattern_cpu(Q[0, 0], K[0, 0], vertical_size, slash_size)
    stats = analyze_sparsity_cpu(seq_len, v_idx, s_idx)
    print(f"\nSparsity: {stats['sparsity_ratio']:.2%}")
    print(f"Keys computed: {stats['total_keys_computed']:,} / {stats['total_keys_dense']:,}")

    # 1. CPU C++ AVX-512
    if HAS_CPU_EXTENSION:
        print("\n[1] CPU C++ AVX-512...")
        time_cpp, std_cpp = benchmark_kernel(
            vertical_slash_attention_cpu,
            Q, K, V, vertical_size, slash_size,
            use_cpp=True,
            warmup=warmup, repeat=repeat
        )
        results['cpu_cpp'] = time_cpp
        print(f"  Time: {time_cpp:.3f} +/- {std_cpp:.3f} ms")
    else:
        print("\n[1] CPU C++ AVX-512: Not available (extension not compiled)")

    # 2. GPU Flash Attention
    if HAS_FLASH_ATTN and torch.cuda.is_available():
        print("\n[2] GPU Flash Attention...")
        Q_gpu = Q.cuda().bfloat16().transpose(1, 2).contiguous()  # [B, S, H, D]
        K_gpu = K.cuda().bfloat16().transpose(1, 2).contiguous()
        V_gpu = V.cuda().bfloat16().transpose(1, 2).contiguous()

        time_flash, std_flash = benchmark_kernel(
            flash_attn_func,
            Q_gpu, K_gpu, V_gpu, causal=True,
            sync_cuda=True,
            warmup=warmup, repeat=repeat
        )
        results['gpu_flash'] = time_flash
        print(f"  Time: {time_flash:.3f} +/- {std_flash:.3f} ms")
    else:
        print("\n[2] GPU Flash Attention: Not available")

    # 3. GPU MInference Sparse
    if HAS_MINFERENCE and torch.cuda.is_available():
        print("\n[3] GPU MInference Sparse...")
        Q_gpu = Q.cuda().bfloat16()
        K_gpu = K.cuda().bfloat16()
        V_gpu = V.cuda().bfloat16()

        # Prepare indices
        v_idx_gpu = v_idx.cuda().unsqueeze(0).unsqueeze(0)
        s_idx_gpu = s_idx.cuda().unsqueeze(0).unsqueeze(0)

        try:
            time_mi, std_mi = benchmark_kernel(
                vertical_slash_sparse_attention,
                Q_gpu, K_gpu, V_gpu, v_idx_gpu, s_idx_gpu,
                sync_cuda=True,
                warmup=warmup, repeat=repeat
            )
            results['gpu_sparse'] = time_mi
            print(f"  Time: {time_mi:.3f} +/- {std_mi:.3f} ms")
        except Exception as e:
            print(f"  Error: {e}")

    # Summary
    print(f"\n{'='*60}")
    print("Summary:")
    print(f"{'='*60}")

    baseline = results.get('gpu_flash', results.get('gpu_sparse', results.get('cpu_cpp', 1.0)))

    for name, time_ms in results.items():
        speedup = baseline / time_ms if time_ms > 0 else 0
        print(f"  {name:<20}: {time_ms:8.3f} ms  ({speedup:.2f}x vs baseline)")

    return results


def main():
    parser = argparse.ArgumentParser(description='Benchmark CPU Sparse Attention')
    parser.add_argument('--seq_lens', type=int, nargs='+',
                        default=[512, 1024, 2048, 4096, 8192],
                        help='Sequence lengths to test')
    parser.add_argument('--head_dim', type=int, default=128,
                        help='Head dimension')
    parser.add_argument('--num_heads', type=int, default=1,
                        help='Number of heads')
    parser.add_argument('--vertical_size', type=int, default=100,
                        help='Number of vertical columns')
    parser.add_argument('--slash_size', type=int, default=500,
                        help='Number of slash diagonals')
    parser.add_argument('--warmup', type=int, default=3,
                        help='Warmup iterations')
    parser.add_argument('--repeat', type=int, default=10,
                        help='Measurement iterations')
    args = parser.parse_args()

    print("="*60)
    print("CPU SIMD Sparse Attention Benchmark")
    print("="*60)
    print(f"CPU Extension available: {HAS_CPU_EXTENSION}")
    print(f"Flash Attention available: {HAS_FLASH_ATTN}")
    print(f"MInference available: {HAS_MINFERENCE}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    all_results = {}
    for seq_len in args.seq_lens:
        results = run_benchmark(
            seq_len=seq_len,
            head_dim=args.head_dim,
            num_heads=args.num_heads,
            vertical_size=args.vertical_size,
            slash_size=args.slash_size,
            warmup=args.warmup,
            repeat=args.repeat,
        )
        all_results[seq_len] = results

    # Final comparison table
    print(f"\n{'='*70}")
    print("Final Comparison Table")
    print(f"{'='*70}")
    print(f"{'seq_len':<10}", end="")
    methods = ['cpu_cpp', 'gpu_flash', 'gpu_sparse']
    for m in methods:
        print(f"{m:<18}", end="")
    print()
    print("-" * 70)

    for seq_len, results in all_results.items():
        print(f"{seq_len:<10}", end="")
        for m in methods:
            if m in results:
                print(f"{results[m]:>8.3f} ms       ", end="")
            else:
                print(f"{'N/A':>8}          ", end="")
        print()


if __name__ == "__main__":
    main()
