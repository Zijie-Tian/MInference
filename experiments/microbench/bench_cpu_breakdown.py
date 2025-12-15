# Copyright (c) 2024-2025 Microsoft
# Licensed under The MIT License [see LICENSE for details]

"""
Benchmark CPU sparse attention breakdown: Estimate vs Compute stages.

This script separates the two stages:
1. Estimate stage: Pattern discovery from probe attention
2. Compute stage: Sparse attention with given pattern
"""

import time
import torch
import argparse

from minference.ops.cpu_sparse_attention import HAS_CPU_EXTENSION

if not HAS_CPU_EXTENSION:
    print("ERROR: CPU extension not available. Please compile first.")
    exit(1)

from minference._C import cpu_sparse_attention as _C


def benchmark(func, *args, warmup=3, repeat=10, **kwargs):
    """Benchmark a function."""
    # Warmup
    for _ in range(warmup):
        _ = func(*args, **kwargs)

    # Measure
    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        _ = func(*args, **kwargs)
        end = time.perf_counter()
        times.append((end - start) * 1000)  # ms

    times = torch.tensor(times)
    return times.mean().item(), times.std().item()


def benchmark_breakdown(seq_len, head_dim=128, num_heads=1,
                       vertical_size=100, slash_size=500,
                       warmup=3, repeat=10):
    """Benchmark estimate and compute stages separately."""
    print(f"\n{'='*60}")
    print(f"seq_len={seq_len}, head_dim={head_dim}, heads={num_heads}")
    print(f"Pattern: vertical={vertical_size}, slash={slash_size}")
    print(f"{'='*60}")

    # Generate test data
    Q = torch.randn(num_heads, seq_len, head_dim)
    K = torch.randn(num_heads, seq_len, head_dim)
    V = torch.randn(num_heads, seq_len, head_dim)

    # Use first head for single-head tests
    q_single = Q[0].float()
    k_single = K[0].float()
    v_single = V[0].float()

    results = {}

    # 1. Estimate stage only
    print(f"\n[1] Estimate Stage (pattern discovery)...")

    def estimate_all_heads():
        for h in range(num_heads):
            v_idx, s_idx = _C.estimate_pattern(
                Q[h].float(), K[h].float(),
                vertical_size, slash_size
            )

    time_est, std_est = benchmark(estimate_all_heads, warmup=warmup, repeat=repeat)
    results['estimate'] = time_est
    print(f"  Total: {time_est:.3f} +/- {std_est:.3f} ms")
    print(f"  Per-head: {time_est / num_heads:.3f} ms")

    # 2. Compute stage only (with pre-computed pattern)
    print(f"\n[2] Compute Stage (sparse attention)...")

    # Pre-compute pattern
    v_idx, s_idx = _C.estimate_pattern(q_single, k_single, vertical_size, slash_size)

    def compute_all_heads():
        for h in range(num_heads):
            output = _C.sparse_attention(
                Q[h].float(), K[h].float(), V[h].float(),
                v_idx, s_idx
            )

    time_comp, std_comp = benchmark(compute_all_heads, warmup=warmup, repeat=repeat)
    results['compute'] = time_comp
    print(f"  Total: {time_comp:.3f} +/- {std_comp:.3f} ms")
    print(f"  Per-head: {time_comp / num_heads:.3f} ms")

    # 3. Full pipeline (for comparison)
    print(f"\n[3] Full Pipeline (estimate + compute)...")

    Q_4d = Q.unsqueeze(0)  # [1, heads, seq_len, head_dim]
    K_4d = K.unsqueeze(0)
    V_4d = V.unsqueeze(0)

    time_full, std_full = benchmark(
        _C.vertical_slash_attention,
        Q_4d, K_4d, V_4d, vertical_size, slash_size,
        warmup=warmup, repeat=repeat
    )
    results['full'] = time_full
    print(f"  Total: {time_full:.3f} +/- {std_full:.3f} ms")
    print(f"  Per-head: {time_full / num_heads:.3f} ms")

    # Analysis
    print(f"\n{'='*60}")
    print("Breakdown Analysis:")
    print(f"{'='*60}")

    est_pct = 100 * time_est / time_full
    comp_pct = 100 * time_comp / time_full
    overhead = time_full - (time_est + time_comp)
    overhead_pct = 100 * overhead / time_full

    print(f"  Estimate:     {time_est:7.3f} ms ({est_pct:5.1f}%)")
    print(f"  Compute:      {time_comp:7.3f} ms ({comp_pct:5.1f}%)")
    print(f"  Overhead:     {overhead:7.3f} ms ({overhead_pct:5.1f}%)")
    print(f"  Total (sum):  {time_est + time_comp:7.3f} ms")
    print(f"  Total (full): {time_full:7.3f} ms")

    return results


def main():
    parser = argparse.ArgumentParser(description='CPU Sparse Attention Breakdown')
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
    print("CPU Sparse Attention Breakdown Benchmark")
    print("="*60)
    print(f"CPU Extension: {HAS_CPU_EXTENSION}")
    print(f"PyTorch threads: {torch.get_num_threads()}")

    all_results = {}
    for seq_len in args.seq_lens:
        results = benchmark_breakdown(
            seq_len=seq_len,
            head_dim=args.head_dim,
            num_heads=args.num_heads,
            vertical_size=args.vertical_size,
            slash_size=args.slash_size,
            warmup=args.warmup,
            repeat=args.repeat,
        )
        all_results[seq_len] = results

    # Summary table
    print(f"\n{'='*70}")
    print("Summary Table")
    print(f"{'='*70}")
    print(f"{'seq_len':<10} {'Estimate':<12} {'Compute':<12} {'Full':<12} {'Est%':<8}")
    print("-" * 70)

    for seq_len, res in all_results.items():
        est = res['estimate']
        comp = res['compute']
        full = res['full']
        est_pct = 100 * est / full
        print(f"{seq_len:<10} {est:>8.3f} ms  {comp:>8.3f} ms  {full:>8.3f} ms  {est_pct:>5.1f}%")


if __name__ == "__main__":
    main()
