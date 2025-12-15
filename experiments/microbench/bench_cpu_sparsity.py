# Copyright (c) 2024-2025 Microsoft
# Licensed under The MIT License [see LICENSE for details]

"""
Benchmark CPU vs GPU across different sparsity levels.
"""

import time
import torch
import argparse

from minference.ops.cpu_sparse_attention import (
    vertical_slash_attention_cpu,
    estimate_pattern_cpu,
    analyze_sparsity_cpu,
    HAS_CPU_EXTENSION,
)

try:
    from flash_attn.flash_attn_interface import flash_attn_func
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False

try:
    from minference.ops.pit_sparse_flash_attention_v2 import vertical_slash_sparse_attention
    HAS_MINFERENCE = True
except ImportError:
    HAS_MINFERENCE = False


def benchmark(func, *args, warmup=3, repeat=10, sync_cuda=False, **kwargs):
    """Benchmark a function."""
    for _ in range(warmup):
        _ = func(*args, **kwargs)
    if sync_cuda:
        torch.cuda.synchronize()

    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        _ = func(*args, **kwargs)
        if sync_cuda:
            torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)

    times = torch.tensor(times)
    return times.mean().item(), times.std().item()


def test_sparsity_config(seq_len, vertical_size, slash_size, warmup=3, repeat=10):
    """Test one sparsity configuration."""
    head_dim = 128
    num_heads = 1

    Q = torch.randn(1, num_heads, seq_len, head_dim)
    K = torch.randn(1, num_heads, seq_len, head_dim)
    V = torch.randn(1, num_heads, seq_len, head_dim)

    # Get sparsity stats
    v_idx, s_idx = estimate_pattern_cpu(Q[0, 0], K[0, 0], vertical_size, slash_size)
    stats = analyze_sparsity_cpu(seq_len, v_idx, s_idx)

    results = {'sparsity': stats['sparsity_ratio']}

    # CPU
    if HAS_CPU_EXTENSION:
        time_cpu, _ = benchmark(
            vertical_slash_attention_cpu,
            Q, K, V, vertical_size, slash_size,
            use_cpp=True, warmup=warmup, repeat=repeat
        )
        results['cpu'] = time_cpu
    else:
        results['cpu'] = float('nan')

    # GPU Flash
    if HAS_FLASH_ATTN and torch.cuda.is_available():
        Q_gpu = Q.cuda().bfloat16().transpose(1, 2).contiguous()
        K_gpu = K.cuda().bfloat16().transpose(1, 2).contiguous()
        V_gpu = V.cuda().bfloat16().transpose(1, 2).contiguous()

        time_flash, _ = benchmark(
            flash_attn_func, Q_gpu, K_gpu, V_gpu, causal=True,
            sync_cuda=True, warmup=warmup, repeat=repeat
        )
        results['gpu_flash'] = time_flash
    else:
        results['gpu_flash'] = float('nan')

    # GPU Sparse
    if HAS_MINFERENCE and torch.cuda.is_available():
        Q_gpu = Q.cuda().bfloat16()
        K_gpu = K.cuda().bfloat16()
        V_gpu = V.cuda().bfloat16()

        v_idx_gpu = v_idx.cuda().unsqueeze(0).unsqueeze(0)
        s_idx_gpu = s_idx.cuda().unsqueeze(0).unsqueeze(0)

        try:
            time_sparse, _ = benchmark(
                vertical_slash_sparse_attention,
                Q_gpu, K_gpu, V_gpu, v_idx_gpu, s_idx_gpu,
                sync_cuda=True, warmup=warmup, repeat=repeat
            )
            results['gpu_sparse'] = time_sparse
        except Exception:
            results['gpu_sparse'] = float('nan')
    else:
        results['gpu_sparse'] = float('nan')

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq_lens', type=int, nargs='+',
                        default=[4096, 8192, 16384, 32768, 65536])
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--repeat', type=int, default=10)
    parser.add_argument('--compact', action='store_true',
                        help='Compact output format')
    args = parser.parse_args()

    print("="*80)
    print("CPU vs GPU Sparsity Sweep")
    print("="*80)
    print(f"CPU Extension: {HAS_CPU_EXTENSION}")
    print(f"Flash Attention: {HAS_FLASH_ATTN}")
    print(f"GPU Sparse: {HAS_MINFERENCE}")
    print()

    # Different sparsity configurations (from extreme to dense)
    configs = [
        {'name': 'Extreme Sparse', 'v': 10, 's': 50},
        {'name': 'Ultra Sparse', 'v': 20, 's': 80},
        {'name': 'Very Sparse', 'v': 30, 's': 100},
        {'name': 'Sparse', 'v': 50, 's': 200},
        {'name': 'Medium Sparse', 'v': 80, 's': 400},
        {'name': 'Medium', 'v': 100, 's': 500},
        {'name': 'Medium Dense', 'v': 150, 's': 800},
        {'name': 'Dense', 'v': 200, 's': 1000},
    ]

    if args.compact:
        # Compact format: all configs in one table per seq_len
        for seq_len in args.seq_lens:
            print(f"\n{'='*90}")
            print(f"seq_len = {seq_len}")
            print(f"{'='*90}")
            print(f"{'Config':<18} {'Sparsity':<10} {'CPU':<10} {'GPU Flash':<12} "
                  f"{'GPU Sparse':<12} {'CPU/GPUsparse'}")
            print("-"*90)

            for config in configs:
                res = test_sparsity_config(
                    seq_len, config['v'], config['s'],
                    warmup=args.warmup, repeat=args.repeat
                )
                ratio = res['cpu'] / res['gpu_sparse'] if res['gpu_sparse'] == res['gpu_sparse'] else 0
                print(f"{config['name']:<18} {res['sparsity']:<9.2%} "
                      f"{res['cpu']:>8.2f}ms  {res['gpu_flash']:>10.2f}ms  "
                      f"{res['gpu_sparse']:>10.2f}ms  {ratio:>10.1f}x")
    else:
        # Original format: one config per table
        for config in configs:
            print(f"\n{'='*80}")
            print(f"{config['name']}: vertical={config['v']}, slash={config['s']}")
            print(f"{'='*80}")
            print(f"{'seq_len':<10} {'Sparsity':<10} {'CPU (ms)':<12} {'GPU Flash':<12} "
                  f"{'GPU Sparse':<12} {'CPU/GPUsparse'}")
            print("-"*80)

            for seq_len in args.seq_lens:
                res = test_sparsity_config(
                    seq_len, config['v'], config['s'],
                    warmup=args.warmup, repeat=args.repeat
                )

                ratio = res['cpu'] / res['gpu_sparse'] if res['gpu_sparse'] == res['gpu_sparse'] else 0
                print(f"{seq_len:<10} {res['sparsity']:<9.2%} "
                      f"{res['cpu']:>10.2f}  {res['gpu_flash']:>10.2f}  "
                      f"{res['gpu_sparse']:>10.2f}  {ratio:>10.1f}x")


if __name__ == "__main__":
    main()
