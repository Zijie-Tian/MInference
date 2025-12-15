# Copyright (c) 2024-2025 Microsoft
# Licensed under The MIT License [see LICENSE for details]

"""
Benchmark vertical vs slash pattern balance.

Test extreme configurations:
1. Heavy Vertical + Light Slash (gather-heavy)
2. Light Vertical + Heavy Slash (block-heavy)
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


def test_pattern_config(seq_len, vertical_size, slash_size, warmup=3, repeat=10):
    """Test one pattern configuration."""
    head_dim = 128
    num_heads = 1

    Q = torch.randn(1, num_heads, seq_len, head_dim)
    K = torch.randn(1, num_heads, seq_len, head_dim)
    V = torch.randn(1, num_heads, seq_len, head_dim)

    # Get sparsity stats
    v_idx, s_idx = estimate_pattern_cpu(Q[0, 0], K[0, 0], vertical_size, slash_size)
    stats = analyze_sparsity_cpu(seq_len, v_idx, s_idx)

    results = {
        'sparsity': stats['sparsity_ratio'],
        'v_size': vertical_size,
        's_size': slash_size,
    }

    # CPU
    if HAS_CPU_EXTENSION:
        time_cpu, std_cpu = benchmark(
            vertical_slash_attention_cpu,
            Q, K, V, vertical_size, slash_size,
            use_cpp=True, warmup=warmup, repeat=repeat
        )
        results['cpu'] = time_cpu
        results['cpu_std'] = std_cpu
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
            time_sparse, std_sparse = benchmark(
                vertical_slash_sparse_attention,
                Q_gpu, K_gpu, V_gpu, v_idx_gpu, s_idx_gpu,
                sync_cuda=True, warmup=warmup, repeat=repeat
            )
            results['gpu_sparse'] = time_sparse
            results['gpu_sparse_std'] = std_sparse
        except Exception:
            results['gpu_sparse'] = float('nan')
    else:
        results['gpu_sparse'] = float('nan')

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq_lens', type=int, nargs='+',
                        default=[4096, 8192, 16384, 32768])
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--repeat', type=int, default=10)
    args = parser.parse_args()

    print("="*100)
    print("Vertical vs Slash Pattern Balance")
    print("="*100)
    print(f"CPU Extension: {HAS_CPU_EXTENSION}")
    print(f"Flash Attention: {HAS_FLASH_ATTN}")
    print(f"GPU Sparse: {HAS_MINFERENCE}")
    print()

    # Pattern configurations to test
    # Format: (vertical_size, slash_size, description)
    pattern_types = [
        # Extreme Vertical-Heavy (gather operations)
        ('Extreme Vertical', [
            (500, 10, 'V-extreme'),
            (400, 20, 'V-heavy'),
            (300, 50, 'V-moderate'),
            (200, 100, 'V-light'),
        ]),
        # Extreme Slash-Heavy (block operations)
        ('Extreme Slash', [
            (10, 500, 'S-extreme'),
            (20, 400, 'S-heavy'),
            (50, 300, 'S-moderate'),
            (100, 200, 'S-light'),
        ]),
        # Balanced
        ('Balanced', [
            (100, 100, 'Equal-100'),
            (200, 200, 'Equal-200'),
            (50, 50, 'Equal-50'),
        ]),
    ]

    for category_name, configs in pattern_types:
        print(f"\n{'='*100}")
        print(f"{category_name} Patterns")
        print(f"{'='*100}")

        for seq_len in args.seq_lens:
            print(f"\nseq_len = {seq_len}")
            print(f"{'-'*100}")
            print(f"{'Config':<15} {'V/S':<12} {'Sparsity':<10} {'CPU':<12} {'GPU Flash':<12} "
                  f"{'GPU Sparse':<12} {'CPU/Sparse'}")
            print(f"{'-'*100}")

            for v_size, s_size, name in configs:
                res = test_pattern_config(
                    seq_len, v_size, s_size,
                    warmup=args.warmup, repeat=args.repeat
                )

                ratio = res['cpu'] / res['gpu_sparse'] if res['gpu_sparse'] == res['gpu_sparse'] else 0
                v_s_ratio = v_size / s_size

                print(f"{name:<15} {v_size}/{s_size:<8} {res['sparsity']:<9.2%} "
                      f"{res['cpu']:>10.2f}ms  {res['gpu_flash']:>10.2f}ms  "
                      f"{res['gpu_sparse']:>10.2f}ms  {ratio:>8.1f}x")

    # Detailed analysis for one sequence length
    print(f"\n{'='*100}")
    print(f"Detailed Analysis @ seq_len=16384")
    print(f"{'='*100}")

    seq_len = 16384

    # Test a range from pure vertical to pure slash
    test_configs = [
        (500, 10, 'V500/S10'),
        (400, 50, 'V400/S50'),
        (300, 100, 'V300/S100'),
        (200, 200, 'V200/S200'),
        (100, 300, 'V100/S300'),
        (50, 400, 'V50/S400'),
        (10, 500, 'V10/S500'),
    ]

    print(f"\n{'Config':<12} {'V/S Ratio':<10} {'Sparsity':<10} {'CPU':<12} {'GPU Sparse':<12} "
          f"{'CPU/Sparse':<12} {'Speedup vs V500/S10'}")
    print(f"{'-'*100}")

    baseline_cpu = None
    baseline_gpu = None

    for v_size, s_size, name in test_configs:
        res = test_pattern_config(seq_len, v_size, s_size, warmup=args.warmup, repeat=args.repeat)

        if baseline_cpu is None:
            baseline_cpu = res['cpu']
            baseline_gpu = res['gpu_sparse']

        ratio = res['cpu'] / res['gpu_sparse'] if res['gpu_sparse'] == res['gpu_sparse'] else 0
        v_s_ratio = v_size / s_size
        cpu_speedup = baseline_cpu / res['cpu']
        gpu_speedup = baseline_gpu / res['gpu_sparse']

        print(f"{name:<12} {v_s_ratio:>8.1f}x   {res['sparsity']:<9.2%} "
              f"{res['cpu']:>10.2f}ms  {res['gpu_sparse']:>10.2f}ms  {ratio:>10.1f}x  "
              f"CPU:{cpu_speedup:>5.2f}x GPU:{gpu_speedup:>5.2f}x")


if __name__ == "__main__":
    main()
