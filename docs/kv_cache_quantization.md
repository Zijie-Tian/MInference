# KV Cache Quantization

## 概述

KV Cache 量化是一种通过降低 Key-Value 缓存精度来减少内存占用的技术。本文档详细分析 Attention 计算与 KV Cache 量化的执行顺序，以及工程实现中的关键考量。

---

## Attention 与 KV Cache 量化的执行顺序

### 核心结论

**实际工程实践：先 Attention 后量化**

| 阶段 | K, V 来源 | Attention 精度 | 量化时机 |
|------|----------|---------------|---------|
| Prefill | 刚计算的 FP16 | FP16 (无损) | Attention 后存储 |
| Decode | 从 cache 加载 INT4/FP8 | 反量化后 FP16 或 Mixed | 新 token 计算后 |

---

## Prefill 阶段

### 执行流程

```
+------------------------------------------+
| Step 1: 计算 Q, K, V (FP16/BF16)          |
+------------------------------------------+
                ↓
+------------------------------------------+
| Step 2: 计算 Attention (FP16/BF16)        |
|   output = softmax(QK^T / √d) × V        |
+------------------------------------------+
                ↓
+------------------------------------------+
| Step 3: 量化 K, V (FP16 → INT4/FP8)       |
|   K_quant = quantize(K_fp16)             |
|   V_quant = quantize(V_fp16)             |
+------------------------------------------+
                ↓
+------------------------------------------+
| Step 4: 存储量化后的 K, V 到 cache         |
+------------------------------------------+
```

### 为什么 Prefill 使用 FP16 计算 Attention？

**原因 1: K, V 刚计算出来，无需量化**
```
+------------------------------------------+
| Prefill 阶段的数据流                      |
+------------------------------------------+
| K, V 刚从 Linear 层输出                   |
| → 还在 GPU SRAM/寄存器中                  |
| → 直接用于 Attention，无需从 HBM 读取      |
| → 量化只是为了后续 Decode 阶段的存储       |
+------------------------------------------+
```

**原因 2: 避免不必要的精度损失**
```
+------------------------------------------+
| 方案对比                                  |
+------------------------------------------+
| 方案 A: 先量化后 Attention (不推荐)        |
|   K_int4 = quant(K_fp16)                 |
|   K_dequant = dequant(K_int4)            |
|   output = attn(Q, K_dequant, V_dequant) |
|   → 引入量化误差                          |
+------------------------------------------+
| 方案 B: 先 Attention 后量化 (推荐)         |
|   output = attn(Q, K_fp16, V_fp16)       |
|   K_int4 = quant(K_fp16)  // for storage |
|   → Prefill 阶段无精度损失                |
+------------------------------------------+
```

---

## Decode 阶段

### 执行流程

```
+------------------------------------------+
| Step 1: 加载量化的 K, V (INT4/FP8)         |
+------------------------------------------+
                ↓
+------------------------------------------+
| Step 2: 反量化或融合计算                   |
|   方式 A: 显式反量化                       |
|     K_fp16 = dequant(K_int4)             |
|     QK = Q @ K_fp16.T                    |
|   方式 B: 融合计算 (Mixed-Precision)       |
|     QK = mixed_gemv(Q_fp16, K_int4, s, z)|
+------------------------------------------+
                ↓
+------------------------------------------+
| Step 3: Softmax + V 计算                  |
+------------------------------------------+
                ↓
+------------------------------------------+
| Step 4: 量化新 token 的 K, V               |
+------------------------------------------+
                ↓
+------------------------------------------+
| Step 5: 追加到 cache                      |
+------------------------------------------+
```

### 两种实现方式对比

| 特性 | 显式反量化 | 融合计算 (Mixed-Precision) |
|-----|-----------|---------------------------|
| 内存流量 | 2x (加载 int4, 写 fp16, 读 fp16) | 1x (只加载 int4) |
| 延迟 | 较高 | 较低 |
| 实现复杂度 | 简单 | 需要专用 kernel |
| 代表实现 | vLLM (当前) | KIVI |

**显式反量化流程:**
```
Load K_int4 → Dequant → Store K_fp16 → Load K_fp16 → GEMM
     ↑                      ↑              ↑
   0.5x                    2x             2x        = 4.5x memory traffic
```

**融合计算流程:**
```
Load K_int4 → Fused Dequant + GEMM
     ↑
   0.5x                              = 0.5x memory traffic
```

---

## Mixed-Precision GEMV 详解

### 计算原理

KIVI 等方法使用 Mixed-Precision GEMV (FP16 × INT4):

```
+------------------------------------------+
| Mixed-Precision GEMV Kernel               |
+------------------------------------------+
| 输入:                                     |
|   A: [M, K] FP16 (Query 或 Attention权重) |
|   B: [K, N] INT4 (量化的 Key 或 Value)    |
|   scale: [K/g, N] FP16                   |
|   zero: [K/g, N] FP16                    |
+------------------------------------------+
| 计算 (per output element):                |
|   for k in range(K):                     |
|     b_fp16 = B_int4[k] * scale + zero    |
|     acc += A_fp16[k] * b_fp16            |
+------------------------------------------+
| 输出: [M, N] FP16                         |
+------------------------------------------+
```

### 为什么不用纯 INT Attention？

| 问题 | 说明 |
|-----|------|
| 硬件限制 | INT4/INT2 无原生 Tensor Core 支持 |
| Softmax 需要浮点 | exp() 运算无法用整数完成 |
| 累加溢出 | INT32 累加器容易溢出 |

**GPU Tensor Core 支持情况:**
| 格式 | A100 | H100 | 支持程度 |
|-----|------|------|---------|
| FP16×FP16 | ✅ | ✅ | 原生 GEMM |
| BF16×BF16 | ✅ | ✅ | 原生 GEMM |
| INT8×INT8 | ✅ | ✅ | 原生 GEMM |
| INT4×INT4 | ❌ | ⚠️ | 有限支持 |
| INT2×INT2 | ❌ | ❌ | 无支持 |

---

## KIVI 实现分析

### 代码位置

`minference/modules/kivi.py`

### KV Cache 结构

```
+------------------------------------------+
| KIVI KV Cache 结构                        |
+------------------------------------------+
| Quantized Part (INT2/INT4):              |
|   K[0 : N-residual]                      |
|   V[0 : N-residual]                      |
+------------------------------------------+
| Full Precision Part (FP16):              |
|   K[N-residual : N]  (residual_length=32)|
|   V[N-residual : N]                      |
+------------------------------------------+
```

**为什么保留 residual?**
- 最近的 token 最重要
- 减少最近 token 的量化误差
- 内存开销很小 (32 tokens << N tokens)

### Prefill 实现

```python
# kivi.py: KiviCache.update() - Prefill
if prefilling:
    # Step 1: 量化 (为存储准备)
    key_states_quant = triton_quantize_and_pack_along_last_dim(key_states, ...)
    value_states_quant = triton_quantize_and_pack_along_last_dim(value_states, ...)

    # Step 2: 存储到 cache
    self.kv_cache.append((key_states_quant, key_states_full, scale, zero, ...))

    # Step 3: 返回原始 FP16 用于 Attention!
    return (key_states, value_states)  # ← 原始 FP16
```

### Decode 实现

```python
# kivi.py: kivi_forward() - Decode
# Step 1: Mixed-Precision QK 计算
att_qkquant = cuda_bmm_fA_qB_outer(
    group_size,
    query_states.to(torch.float16),  # A: FP16
    key_states_quant_trans,          # B: INT4
    key_scale_trans,
    key_mn_trans,
    k_bits,
)

# Step 2: FP16 部分 (residual)
att_qkfull = torch.matmul(query_states, key_states_full.transpose(2, 3))

# Step 3: 合并
attn_weights = torch.cat([att_qkquant, att_qkfull], dim=-1) / math.sqrt(head_dim)

# Step 4: Softmax (FP32)
attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32)

# Step 5: Mixed-Precision PV 计算
attn_output = cuda_bmm_fA_qB_outer(group_size, attn_weights, value_states_quant, ...)
```

---

## 性能分析

### 内存带宽优化

| 配置 | K, V 加载量 | 相对 FP16 |
|-----|------------|----------|
| FP16 | N × D × 2 bytes | 1× |
| INT8 | N × D × 1 byte | 0.5× |
| INT4 | N × D × 0.5 bytes | 0.25× |
| INT2 | N × D × 0.25 bytes | 0.125× |

### Decode 阶段收益

```
+------------------------------------------+
| Decode 是 Memory-Bound                    |
+------------------------------------------+
| Arithmetic Intensity = FLOPs / Bytes      |
|                      = 2ND / 4ND          |
|                      = 0.5                |
|                                           |
| A100 需要 AI > 100 才能 compute-bound     |
| → Decode 严重受限于内存带宽               |
| → 量化 KV Cache 直接减少内存加载量        |
+------------------------------------------+
```

### 实际加速效果

| 方法 | Prefill 加速 | Decode 加速 | 内存节省 |
|-----|-------------|-------------|---------|
| KIVI (INT2) | 1× | 2-3× | 8× |
| KVQuant (INT4) | 1× | 1.5-2× | 4× |
| vLLM FP8 | 1× | 1.4× | 2× |

---

## 与 Sparse Attention 结合

### 组合优化架构

```
+================================================================+
|              SPARSE + QUANTIZED ATTENTION                       |
+================================================================+
|                                                                 |
| PREFILL:                                                        |
| +------------------------------------------+                    |
| | 1. 计算 Q, K, V (FP16)                   |                    |
| | 2. Pattern Search → v_idx, s_idx         |                    |
| | 3. Sparse Attention (FP16)               |                    |
| |    - 只计算选中的 blocks                  |                    |
| | 4. 量化 K, V → INT4 存储                  |                    |
| +------------------------------------------+                    |
|                                                                 |
| DECODE:                                                         |
| +------------------------------------------+                    |
| | 1. 加载 K_int4, V_int4                   |                    |
| | 2. Mixed-Precision Attention             |                    |
| | 3. 量化新 token                          |                    |
| +------------------------------------------+                    |
|                                                                 |
+================================================================+
```

### 预期收益

| 优化 | 计算减少 | 内存减少 | 组合效果 |
|-----|---------|---------|---------|
| Sparse Attention | 10-20× | 1× | - |
| KV Quantization | 1× | 4-8× | - |
| **Sparse + Quant** | **10-20×** | **4-8×** | **40-160×** |

---

## 参考资料

- [KIVI Paper (ICML 2024)](https://arxiv.org/abs/2402.02750)
- [KVQuant Paper (NeurIPS 2024)](https://github.com/SqueezeAILab/KVQuant)
- [HuggingFace KV Cache Quantization](https://huggingface.co/blog/kv-cache-quantization)
- [vLLM Quantized KV Cache](https://docs.vllm.ai/en/stable/features/quantization/quantized_kvcache/)
