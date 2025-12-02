# Triton Kernel Implementation

## 整体架构

**数据流:**
```
输入: v_idx, s_idx (稀疏模式发现结果)
                ↓
+------------------------------------------+
| convert_vertical_slash_indexes (CUDA)    |
| - 合并重叠的 slash 块                     |
| - 排除被 slash 覆盖的 vertical 列         |
+------------------------------------------+
                ↓
输出: block_count, block_offset, column_count, column_index
                ↓
+------------------------------------------+
| _triton_mixed_sparse_attn_fwd_kernel     |
| - Phase 1: 处理 slash 块                  |
| - Phase 2: 处理 vertical 列               |
| - Online Softmax 累加                     |
+------------------------------------------+
                ↓
输出: 稀疏注意力结果
```

---

## 索引转换 CUDA Kernel

**代码位置**: `csrc/vertical_slash_index.cu:27-99`

**输入/输出规范:**
| Tensor | Shape | 描述 |
|--------|-------|------|
| vertical_indexes | [batch, heads, nnz_v] | 重要列位置（升序） |
| slash_indexes | [batch, heads, nnz_s] | 重要对角线偏移（降序） |
| block_count | [batch, heads, num_rows] | 每个query块的slash块数量 |
| block_offset | [batch, heads, num_rows, nnz_s] | slash块的起始位置 |
| column_count | [batch, heads, num_rows] | 每个query块的vertical列数量 |
| column_index | [batch, heads, num_rows, nnz_v] | vertical列位置（排除已覆盖） |

**合并逻辑示例:**
```
Query Block [64-127]:
+------------------------------------------+
| Slash 块: [0-63], [64-127]               |
| Vertical 列: [3, 17, 45, 89, 102]        |
+------------------------------------------+
                ↓ 合并后
+------------------------------------------+
| block_offset: [0, 64]  (2个块)           |
| column_index: [3, 17]  (45, 89, 102 被slash覆盖) |
+------------------------------------------+
```

---

## Triton 注意力 Kernel

**代码位置**: `minference/ops/pit_sparse_flash_attention_v2.py:49-154`

### Online Softmax 算法

**初始化:**
```
+------------------------------------------+
| m_i = [-∞, ..., -∞]    shape: [BLOCK_M]  |  运行时最大值
| l_i = [0, ..., 0]      shape: [BLOCK_M]  |  运行时求和
| acc = zeros            shape: [BLOCK_M, D]|  输出累加器
+------------------------------------------+
```

**每个块的更新:**
```
+------------------------------------------+
| qk = dot(q, k)                           |  计算分数
| m_new = max(m_i, rowmax(qk))             |  更新最大值
| alpha = exp2(m_i - m_new)                |  重缩放因子
| p = exp2(qk - m_new)                     |  softmax 分子
| acc = acc * alpha + dot(p, v)            |  重缩放 + 累加
| l_i = l_i * alpha + rowsum(p)            |  更新分母
| m_i = m_new                              |  更新最大值
+------------------------------------------+
```

**最终归一化:**
```
+------------------------------------------+
| output = acc / l_i                       |  归一化
+------------------------------------------+
```

**为什么用 exp2 而不是 exp:**
| 函数 | 公式 | 优势 |
|-----|------|------|
| exp2(x) | 2^x | 硬件原生支持，更快 |
| exp(x) | e^x | 需要转换: exp(x) = exp2(x * log2(e)) |

### 两阶段循环结构

**阶段对比:**
| 阶段 | 目标 | 访问模式 | 掩码 |
|-----|------|---------|-----|
| Phase 1 | Slash 块 | Coalesced（连续64×64） | 需要因果掩码 |
| Phase 2 | Vertical 列 | Gather（散点位置） | 不需要掩码 |

---

## Slash 块处理细节

### 因果掩码

**主对角线块的掩码模式:**
```
Query Block [64-127], Key Block [64-127]:
           k64  k65  k66  ...  k127
         +---------------------------+
   q64   | ✓    ✗    ✗    ...   ✗   |
   q65   | ✓    ✓    ✗    ...   ✗   |
   q66   | ✓    ✓    ✓    ...   ✗   |
    :    | :    :    :     ⋱    :   |
   q127  | ✓    ✓    ✓    ...   ✓   |
         +---------------------------+

✓ = 有效 (key_pos <= query_pos)
✗ = 掩码 (key_pos > query_pos) → 设为 -∞
```

**掩码计算:**
```python
causal_mask = cols[None, :] <= offs_m[:, None]
# cols = [64, 65, ..., 127]    (key 位置)
# offs_m = [64, 65, ..., 127]  (query 位置)
# 结果: 下三角 True 矩阵
```

### 不同位置的 Slash 块

**块位置对比:**
| 块位置 | Key 范围 | Query 范围 | 掩码情况 |
|-------|---------|-----------|---------|
| 远离对角线 | [0-63] | [128-191] | 全部有效（完整矩形） |
| 主对角线 | [128-191] | [128-191] | 下三角有效 |

---

## Vertical 列处理细节

### Gather 操作

**K 矩阵内存布局:**
```
+------------------------------------------+
| addr:  0     128   256   384   ...       |
| token: k0    k1    k2    k3    ...       |
|       [128d] [128d] [128d] [128d]        |
+------------------------------------------+

column_index = [3, 17, 45, 89]
                ↓ gather
+------------------------------------------+
| k_gathered: [k3, k17, k45, k89]          |
|             [每个128维]                   |
+------------------------------------------+
```

### 为什么 Vertical 不需要因果掩码

**位置对比:**
| 场景 | Key 位置 | Query 位置 | 是否有效 |
|-----|---------|-----------|---------|
| Vertical | [3, 17, 45, 89] | [128-191] | 始终有效（key << query） |
| Slash | [128-191] | [128-191] | 取决于位置（需要掩码） |

---

## 性能优化技巧

**优化汇总:**
| 优化 | 描述 | 效果 |
|-----|------|-----|
| 固定块大小 64×64 | 充分利用 GPU 共享内存 | 最大化带宽利用 |
| 合并连续 slash 块 | 减少循环迭代次数 | 降低 kernel 开销 |
| 排除重复计算 | vertical 在 slash 范围内时跳过 | 避免冗余计算 |
| 内存对齐 | 输入 padding 到块大小整数倍 | 优化内存访问 |
| exp2 代替 exp | 使用硬件原生指令 | 提升计算速度 |

---

## 可选优化路径

**两条执行路径:**
| 路径 | 条件 | 实现 |
|-----|------|-----|
| A: SGLang/vLLM | `convert_vertical_slash_indexes_opt` 存在 | 使用优化的 CUDA 实现 |
| B: 纯 Triton | 上述不可用 | 使用 Python + Triton 实现 |

---

## 关键代码位置汇总

| 组件 | 文件路径 | 行号 |
|-----|---------|-----|
| 索引转换 CUDA kernel | `csrc/vertical_slash_index.cu` | 27-99 |
| 索引转换 Python 接口 | `csrc/vertical_slash_index.cu` | 127-167 |
| V-S Triton kernel | `minference/ops/pit_sparse_flash_attention_v2.py` | 49-154 |
| V-S Python wrapper | `minference/ops/pit_sparse_flash_attention_v2.py` | 195-274 |
| Block Sparse kernel | `minference/ops/block_sparse_flash_attention.py` | 30-187 |
