# MInference Technical Documentation

## Project Overview

MInference is a library for accelerating long-context LLM inference through sparse attention mechanisms, reducing O(n²) attention computation to O(n×k) while maintaining accuracy.

---

## 1. Sparse Attention Mechanism

### 1.1 Core Concept

MInference discovers that LLM attention patterns exhibit strong **structural sparsity** - most attention weights concentrate on specific positions.

**Sparse Pattern Types:**
| Pattern | Description | Use Case |
|---------|-------------|----------|
| Vertical-and-Slash | Combines vertical columns + diagonal stripes | Primary pattern for most heads |
| Block Sparse | Block-level sparsity | Heads with clustered attention |
| Streaming/A-shape | Sliding window + initial tokens | Heads with local + BOS attention |

---

## 2. Vertical-and-Slash Pattern

### 2.1 Pattern Semantics

**Attention Matrix Decomposition:**
```
Full Causal Attention:          Vertical-Slash Sparse:
+---------------------------+   +---------------------------+
|■                          |   |■ ■   ■                    |
|■ ■                        |   |■ ■   ■ ■                  |
|■ ■ ■                      |   |■ ■   ■ ■ ■                |
|■ ■ ■ ■                    |   |■ ■   ■   ■ ■              |
|■ ■ ■ ■ ■                  |   |■ ■   ■     ■ ■            |
|■ ■ ■ ■ ■ ■                |   |■ ■   ■       ■ ■          |
+---------------------------+   +---------------------------+
 O(N²) elements                  O(N×k) elements
                                 ↑       ↑
                              Vertical  Slash (diagonal)
```

**Pattern Comparison:**
| Pattern | Shape | Captures | Memory Access |
|---------|-------|----------|---------------|
| Vertical | Columns (││) | Global important tokens (BOS, keywords) | Gather (scattered) |
| Slash | Diagonals (╲╲) | Local context (recent tokens) | Coalesced (contiguous) |

### 2.2 Diagonal Representation in Attention Matrix

**Diagonal Index Meaning:**
```
Attention Matrix [query_pos × key_pos]:
        k0   k1   k2   k3   k4   k5   k6   k7
      +----------------------------------------+
  q0  | d=0                                    |
  q1  | d=1  d=0                               |
  q2  | d=2  d=1  d=0                          |
  q3  | d=3  d=2  d=1  d=0                     |
  q4  | d=4  d=3  d=2  d=1  d=0                |
  q5  | d=5  d=4  d=3  d=2  d=1  d=0           |
  q6  | d=6  d=5  d=4  d=3  d=2  d=1  d=0      |
  q7  | d=7  d=6  d=5  d=4  d=3  d=2  d=1  d=0 |
      +----------------------------------------+

d = query_pos - key_pos (relative distance)
```

**Diagonal Properties:**
| Diagonal d | Meaning | Position in Matrix |
|------------|---------|-------------------|
| d=0 | Self-attention (q[i] → k[i]) | Main diagonal |
| d=1 | Previous token (q[i] → k[i-1]) | First sub-diagonal |
| d=N | N tokens ago (q[i] → k[i-N]) | N-th sub-diagonal |

### 2.3 Core Function: `sum_all_diagonal_matrix`

**Location**: `minference/modules/minference_forward.py:110-116`

**as_strided Transformation:**
```
Input: Attention Matrix (4×4)        Output: Diagonal-aligned Matrix
      k0   k1   k2   k3                    d3   d2   d1   d0   d-1  d-2  d-3
    +--------------------+               +--------------------------------+
 q0 | a    -    -    -   |            q0 | 0    0    0    a    -    -    - |
 q1 | e    f    -    -   |    -->     q1 | 0    0    e    f    -    0    0 |
 q2 | h    i    j    -   |            q2 | 0    h    i    j    0    0    0 |
 q3 | k    l    m    n   |            q3 | k    l    m    n    0    0    0 |
    +--------------------+               +--------------------------------+
                                                    ↓ sum along rows
                                         [k, h+l, e+i+m, a+f+j+n, -, -, -]
                                          ↑              ↑
                                        d=3 sum      d=0 sum (main diagonal)
```

**Stride Calculation:**
| Parameter | Value | Meaning |
|-----------|-------|---------|
| Input shape | (b, h, n, m) | batch, heads, queries, keys |
| Padded shape | (b, h, n, 2n+m) | Add n zeros on each side |
| Output shape | (1, 1, n, n+m) | n rows, n+m diagonals |
| Stride | (1, n*(2n+m), 2n+m+1, 1) | Skip 2n+m+1 per row to align diagonals |

### 2.4 Probe-Based Pattern Discovery

**Location**: `minference/modules/minference_forward.py:381-396`

**Discovery Process:**
```
Step 1: Select Probe Queries (last 64)
+--------------------------------------------------+
| Full Sequence: [t0, t1, ..., t_{n-65}, t_{n-64}, ..., t_{n-1}] |
|                                        ↑─────────────────────↑ |
|                                           Probe region (64)    |
+--------------------------------------------------+

Step 2: Compute Probe Attention
+--------------------------------------------------+
| Q_probe [64 × head_dim] × K [seq_len × head_dim]ᵀ |
|                    ↓                               |
| Attention Scores [64 × seq_len]                   |
+--------------------------------------------------+

Step 3: Extract Patterns
+--------------------------------------------------+
| Vertical: sum along query dim → [1 × seq_len]    |
|           select top-k columns                    |
|                                                   |
| Slash: sum_all_diagonal → [1 × seq_len]          |
|        select top-k diagonals                     |
+--------------------------------------------------+
```

**Pattern Extraction Summary:**
| Pattern | Operation | Output Shape | Selection |
|---------|-----------|--------------|-----------|
| Vertical | `attn.sum(dim=-2)` | [1, seq_len] | Top-k column indices |
| Slash | `sum_all_diagonal(attn)` | [1, seq_len] | Top-k diagonal indices |

---

## 3. Triton Kernel Implementation

### 3.1 Data Flow Architecture

```
Input: v_idx, s_idx (sparse indices from pattern discovery)
                ↓
+------------------------------------------+
| convert_vertical_slash_indexes (CUDA)    |
| - Merge overlapping slash blocks         |
| - Exclude covered vertical columns       |
+------------------------------------------+
                ↓
Output: block_count, block_offset, column_count, column_index
                ↓
+------------------------------------------+
| _triton_mixed_sparse_attn_fwd_kernel     |
| - Phase 1: Process slash blocks          |
| - Phase 2: Process vertical columns      |
| - Online Softmax accumulation            |
+------------------------------------------+
                ↓
Output: Sparse Attention Result
```

### 3.2 Index Conversion (CUDA Kernel)

**Location**: `csrc/vertical_slash_index.cu:27-99`

**Input/Output Specification:**
| Tensor | Shape | Description |
|--------|-------|-------------|
| vertical_indexes | [batch, heads, nnz_v] | Important column positions (ascending) |
| slash_indexes | [batch, heads, nnz_s] | Important diagonal offsets (descending) |
| block_count | [batch, heads, num_rows] | Number of slash blocks per query block |
| block_offset | [batch, heads, num_rows, nnz_s] | Starting positions of slash blocks |
| column_count | [batch, heads, num_rows] | Number of vertical columns per query block |
| column_index | [batch, heads, num_rows, nnz_v] | Vertical column positions (excl. covered) |

**Merge Logic:**
```
Query Block [64-127]:
+------------------------------------------+
| Slash blocks: [0-63], [64-127]           |
| Vertical cols: [3, 17, 45, 89, 102]      |
+------------------------------------------+
                ↓ After merge
+------------------------------------------+
| block_offset: [0, 64]  (2 blocks)        |
| column_index: [3, 17]  (45, 89, 102 covered by slash) |
+------------------------------------------+
```

### 3.3 Triton Attention Kernel

**Location**: `minference/ops/pit_sparse_flash_attention_v2.py:49-154`

**Online Softmax Algorithm:**
```
Initialize:
+------------------------------------------+
| m_i = [-∞, ..., -∞]    shape: [BLOCK_M]  |  running max
| l_i = [0, ..., 0]      shape: [BLOCK_M]  |  running sum
| acc = zeros            shape: [BLOCK_M, D]|  output accumulator
+------------------------------------------+

For each block:
+------------------------------------------+
| qk = dot(q, k)                           |  compute scores
| m_new = max(m_i, rowmax(qk))             |  update max
| alpha = exp2(m_i - m_new)                |  rescale factor
| p = exp2(qk - m_new)                     |  softmax numerator
| acc = acc * alpha + dot(p, v)            |  rescale + accumulate
| l_i = l_i * alpha + rowsum(p)            |  update denominator
| m_i = m_new                              |  update max
+------------------------------------------+

Finalize:
+------------------------------------------+
| output = acc / l_i                       |  normalize
+------------------------------------------+
```

**Two-Phase Loop Structure:**
| Phase | Target | Access Pattern | Masking |
|-------|--------|----------------|---------|
| Phase 1 | Slash blocks | Coalesced (contiguous 64×64) | Causal mask needed |
| Phase 2 | Vertical columns | Gather (scattered positions) | No mask (all valid) |

### 3.4 Slash Block Processing Detail

**Causal Mask in Slash Blocks:**
```
Query Block [64-127], Key Block [64-127] (main diagonal block):
           k64  k65  k66  ...  k127
         +---------------------------+
   q64   | ✓    ✗    ✗    ...   ✗   |
   q65   | ✓    ✓    ✗    ...   ✗   |
   q66   | ✓    ✓    ✓    ...   ✗   |
    :    | :    :    :     ⋱    :   |
   q127  | ✓    ✓    ✓    ...   ✓   |
         +---------------------------+

✓ = valid (key_pos <= query_pos)
✗ = masked (key_pos > query_pos) → set to -∞
```

**Causal Mask Computation:**
```python
causal_mask = cols[None, :] <= offs_m[:, None]
# cols = [64, 65, ..., 127]  (key positions)
# offs_m = [64, 65, ..., 127] (query positions)
# Result: lower triangular True matrix
```

### 3.5 Vertical Column Processing Detail

**Gather Operation:**
```
K matrix layout in memory:
+------------------------------------------+
| addr:  0     128   256   384   ...       |
| token: k0    k1    k2    k3    ...       |
|       [128d] [128d] [128d] [128d]        |
+------------------------------------------+

column_index = [3, 17, 45, 89]
                ↓ gather
+------------------------------------------+
| k_gathered: [k3, k17, k45, k89]          |
|             [128d each]                   |
+------------------------------------------+
```

**Why No Causal Mask for Vertical:**
| Scenario | Key Position | Query Position | Valid? |
|----------|--------------|----------------|--------|
| Vertical | [3, 17, 45, 89] | [128-191] | Always (key << query) |
| Slash | [128-191] | [128-191] | Depends (need mask) |

---

## 4. vLLM Integration

### 4.1 API Changes in vLLM 0.9.0+

| Version | API Method | Access Pattern |
|---------|------------|----------------|
| < 0.9.0 | `llm.llm_engine.model_executor...model` | Direct attribute access |
| >= 0.9.0 | `llm.collective_rpc(func, kwargs)` | RPC to workers |

**Location**: `minference/patch.py:1294-1315`

### 4.2 Required Environment Variables

| Variable | Value | Purpose |
|----------|-------|---------|
| `VLLM_USE_V1` | `0` | Use V0 engine (V1 FlashAttentionImpl incompatible) |
| `VLLM_ALLOW_INSECURE_SERIALIZATION` | `1` | Enable pickle for RPC |
| `VLLM_WORKER_MULTIPROC_METHOD` | `spawn` | Avoid CUDA context issues |

---

## 5. Key Code Locations

| Component | File Path | Lines |
|-----------|-----------|-------|
| Pattern search | `minference/modules/minference_forward.py` | 129-227 |
| Diagonal sum | `minference/modules/minference_forward.py` | 110-116 |
| V-S index extraction | `minference/modules/minference_forward.py` | 381-396 |
| Index conversion CUDA | `csrc/vertical_slash_index.cu` | 27-99 |
| V-S Triton kernel | `minference/ops/pit_sparse_flash_attention_v2.py` | 49-154 |

---

## 6. Microbenchmarks

### 6.1 Performance Results

| Seq Length | Sparsity | Flash Attn (ms) | Sparse Attn (ms) | Speedup |
|------------|----------|-----------------|------------------|---------|
| 4,096 | ~27% | 3.50 | 0.73 | 4.8x |
| 8,192 | ~15% | 12.30 | 1.41 | 8.7x |
| 16,384 | ~8% | 48.13 | 3.12 | 15.4x |
| 32,768 | ~5% | 197.52 | 7.43 | 26.6x |

### 6.2 Sparsity Definition

```
Sparsity = Computed Elements / Full Causal Elements

Full Causal Elements: N × (N+1) / 2
Sparse Elements: num_vertical × N + num_slash × block_size × N
```

### 6.3 Running Benchmarks

```bash
python experiments/microbench/bench_vertical_slash.py \
    --seq_lens 4096 8192 16384 32768 \
    --sparsity 0.02 0.05 0.1 \
    --num_heads 32
```

---

## 7. KV Cache Quantization Integration

### 7.1 Execution Order: Attention First, Then Quantization

**Standard Engineering Practice:**
```
+================================================================+
|                    PREFILL PHASE                                |
+================================================================+
| Step 1: Compute Q, K, V (FP16)                                  |
| Step 2: Compute Attention with FP16 K, V (full precision!)      |
| Step 3: Quantize K, V → INT4/FP8                                |
| Step 4: Store quantized K, V to cache                           |
+================================================================+

+================================================================+
|                    DECODE PHASE                                 |
+================================================================+
| Step 1: Load quantized K, V from cache                          |
| Step 2: Dequantize OR Mixed-Precision Attention                 |
| Step 3: Quantize new token's K, V                               |
| Step 4: Append to cache                                         |
+================================================================+
```

**Why Attention Before Quantization?**
| Reason | Explanation |
|--------|-------------|
| Precision | Prefill uses original FP16, no quantization error |
| Data Locality | K, V just computed, still in SRAM/registers |
| Error Accumulation | Quantization error only affects decode phase |

### 7.2 Mixed-Precision GEMV (FP16 × INT4)

KIVI uses fused dequantization + GEMM instead of explicit dequantization:

```
Explicit Dequant (slow):          Fused Mixed-Precision (fast):
+---------------------------+     +---------------------------+
| Load K_int4               |     | Load K_int4               |
| Dequant → K_fp16 (store)  |     | Fused: K*scale+zero → GEMM|
| Load K_fp16               |     | (no intermediate tensor)  |
| GEMM: Q @ K_fp16          |     +---------------------------+
+---------------------------+
Memory: 4.5x                      Memory: 0.5x
```

### 7.3 Combining Sparse Attention + KV Quantization

| Optimization | Target | Reduction |
|--------------|--------|-----------|
| Sparse Attention | Computation | 10-20× fewer blocks |
| KV Quantization | Memory | 4-8× less bandwidth |
| **Combined** | **Both** | **40-160× improvement** |

**For detailed analysis, see: `docs/kv_cache_quantization.md`**

---

## 8. Prefill vs Decode Optimization

### 8.1 Phase Characteristics

| Phase | Complexity | Bottleneck | MInference Optimization |
|-------|-----------|------------|------------------------|
| Prefill | O(N²) | Compute | Sparse Attention (V-S pattern) |
| Decode | O(N) | Memory | Dense/Quest (KV cache) |

### 8.2 Why Vertical-Slash is Prefill-Only

```
Prefill: N×N Attention Matrix       Decode: 1×N Attention Vector
+---------------------------+       +---------------------------+
|■                          |       |                           |
|■ ■                        |       | q  [? ? ? ? ? ? ? ?]      |
|■ ■ ■                      |       |     ↑                     |
|■ ■ ■ ■                    |       |   Only 1 row!             |
|■ ■ ■ ■ ■                  |       |   No diagonal pattern     |
|■ ■ ■ ■ ■ ■                |       |                           |
+---------------------------+       +---------------------------+
  2D pattern (V-S works)              1D vector (V-S not applicable)
```

**Decode Optimization Options:**
| Method | Description | Use Case |
|--------|-------------|----------|
| Dense | Standard attention | Short context |
| Quest | Query-aware chunk selection | Long context |
| StreamingLLM | Keep first + recent tokens | Very long context |

---

## 9. Summary

**MInference Core Innovations:**
| Innovation | Description |
|------------|-------------|
| Structural Sparsity | Attention concentrates on vertical + diagonal patterns |
| Probe-based Discovery | Last 64 queries reveal full sequence patterns |
| Per-head Customization | Each head uses independent sparse configuration |
| Efficient GPU Kernels | Triton + Online Softmax for sparse computation |

**Performance Characteristic:**
| Sparsity | Information Retained | Typical Speedup |
|----------|---------------------|-----------------|
| 5-10% | 90%+ | 10-30x |

---

## 10. Documentation Index

| Document | Content |
|----------|---------|
| `docs/sparse_attention_overview.md` | Sparse pattern types, search process |
| `docs/vertical_slash_pattern.md` | V-S pattern extraction details |
| `docs/triton_kernel_implementation.md` | Kernel implementation, online softmax |
| `docs/vllm_integration.md` | vLLM 0.9.0+ compatibility |
| `docs/kv_cache_quantization.md` | KV cache quantization + attention order |
