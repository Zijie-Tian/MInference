# vLLM Integration

## vLLM 0.9.0+ 兼容性

vLLM 0.9.0 重构了内部 API，需要使用新的 `collective_rpc` 方式进行模型 patch。

### API 变化

| 版本 | API | 说明 |
|------|-----|------|
| < 0.9.0 | `llm.llm_engine.model_executor.driver_worker.model_runner.model` | 直接访问模型 |
| >= 0.9.0 | `llm.collective_rpc(func, kwargs)` | 通过 RPC 调用 worker 函数 |

### 修复代码

**代码位置**: `minference/patch.py:1294-1315`

```python
def _apply_minference_patch_to_worker(worker, config_file, patch_config):
    """Worker-side function to apply minference patch to the model."""
    from minference.patch import minference_patch_vllm_executor
    model = worker.get_model()
    patch_executor = minference_patch_vllm_executor(config_file, patch_config)
    model.apply(patch_executor)


def minference_patch_vllm(llm, config_file, patch_config={}):
    # vLLM >= 0.9.0 uses collective_rpc API with V1 engine
    llm.collective_rpc(
        _apply_minference_patch_to_worker,
        kwargs={"config_file": config_file, "patch_config": patch_config},
    )
    print("Patched model for minference with vLLM..")
    return llm
```

---

## 运行环境变量

运行 vLLM 0.9.0+ 时需要设置以下环境变量：

```bash
# 使用V0引擎（V1引擎的FlashAttentionImpl不兼容）
VLLM_USE_V1=0 \
# 允许pickle序列化（用于RPC传递函数）
VLLM_ALLOW_INSECURE_SERIALIZATION=1 \
# 使用spawn进行多进程（避免CUDA初始化问题）
VLLM_WORKER_MULTIPROC_METHOD=spawn \
python your_script.py
```

### 环境变量说明

| 变量 | 值 | 说明 |
|------|-----|------|
| `VLLM_USE_V1` | `0` | 使用 V0 引擎，V1 引擎的 `FlashAttentionImpl` 没有 `attn_type` 属性 |
| `VLLM_ALLOW_INSECURE_SERIALIZATION` | `1` | 允许 pickle 序列化，否则 `collective_rpc` 无法传递函数 |
| `VLLM_WORKER_MULTIPROC_METHOD` | `spawn` | 使用 spawn 方式创建子进程，避免 CUDA 上下文问题 |

---

## 运行脚本示例

**位置**: `experiments/benchmarks/run_e2e_vllm_tp.sh`

```bash
#!/bin/bash
# vLLM 0.9.0+ requires these environment variables:
VLLM_USE_V1=0 \
VLLM_ALLOW_INSECURE_SERIALIZATION=1 \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
python experiments/benchmarks/benchmark_e2e_vllm_tp.py \
    --attn_type minference \
    --context_window 100_000 \
    --tensor_parallel_size 4
```

---

## 常见错误及解决方案

### 错误 1: `'LLMEngine' object has no attribute 'model_executor'`

**原因**: vLLM 0.9.0 移除了 `model_executor` 属性

**解决**: 使用 `collective_rpc` API 代替直接访问

### 错误 2: `TypeError: Object of type <class 'function'> is not serializable`

**原因**: `collective_rpc` 需要序列化函数

**解决**: 设置 `VLLM_ALLOW_INSECURE_SERIALIZATION=1`

### 错误 3: `'FlashAttentionImpl' object has no attribute 'attn_type'`

**原因**: V1 引擎的 `FlashAttentionImpl` 实现不同

**解决**: 设置 `VLLM_USE_V1=0` 使用 V0 引擎
