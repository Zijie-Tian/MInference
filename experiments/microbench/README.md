# MInference Microbenchmarks

This directory contains microbenchmarks for comparing MInference's sparse attention kernels against standard Flash Attention.

## Vertical-Slash Sparse Attention Benchmark

`bench_vertical_slash.py` compares the Vertical-Slash sparse attention kernel against Flash Attention across different sequence lengths and sparsity levels.

### Usage

```bash
# Basic usage (default parameters)
python experiments/microbench/bench_vertical_slash.py

# Custom sequence lengths
python experiments/microbench/bench_vertical_slash.py --seq_lens 4096 8192 16384 32768

# Custom sparsity ratios
python experiments/microbench/bench_vertical_slash.py --sparsity 0.02 0.05 0.1

# Full benchmark with all options
python experiments/microbench/bench_vertical_slash.py \
    --seq_lens 4096 8192 16384 32768 \
    --sparsity 0.02 0.05 0.1 \
    --batch_size 1 \
    --num_heads 32 \
    --head_dim 128 \
    --warmup 10 \
    --repeat 50 \
    --dtype bf16

# With correctness verification
python experiments/microbench/bench_vertical_slash.py --verify
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--seq_lens` | 1024, 2048, 4096, 8192, 16384, 32768 | Sequence lengths to benchmark |
| `--sparsity` | 0.01, 0.05, 0.1 | Target sparsity ratios |
| `--batch_size` | 1 | Batch size |
| `--num_heads` | 32 | Number of attention heads |
| `--head_dim` | 128 | Head dimension |
| `--warmup` | 10 | Warmup iterations |
| `--repeat` | 100 | Benchmark iterations |
| `--dtype` | bf16 | Data type (bf16 or fp16) |
| `--verify` | False | Run correctness verification |

### Example Results

```
================================================================================
SUMMARY TABLE
================================================================================
   Seq Len |   Sparsity |    FA (ms) |    VS (ms) |    Speedup | Efficiency
--------------------------------------------------------------------------------
     4,096 |      26.9% |      3.505 |      0.726 |       4.83x |     130.0%
     8,192 |      14.5% |     12.304 |      1.413 |       8.71x |     126.0%
    16,384 |       8.2% |     48.132 |      3.120 |      15.42x |     127.1%
    32,768 |       5.1% |    197.524 |      7.425 |      26.60x |     136.2%
```

### Key Observations

1. **Speedup increases with sequence length**: As sequences get longer, the O(N²) complexity of dense attention becomes more pronounced, making sparse attention more beneficial.

2. **Lower sparsity = higher speedup**: Computing fewer attention elements results in proportionally faster execution.

3. **Efficiency > 100%**: This indicates that the sparse kernel has additional optimizations beyond just computing fewer elements (e.g., better memory access patterns).

### How It Works

The benchmark:
1. Creates random Q, K, V tensors
2. Generates synthetic sparse patterns (vertical indices + slash indices)
3. Pre-computes sparse indices (not included in kernel timing)
4. Benchmarks both Flash Attention and Vertical-Slash kernels
5. Reports timing, speedup, and efficiency metrics
