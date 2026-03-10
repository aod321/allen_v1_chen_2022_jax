# 工作现场保存 - 2026-03-08

## 概述

调试并修复 brainstate/brainevent/braintrace 集成实现的 V1 网络训练问题。

## 已完成的修复

### 1. GLIF3Brainstate `_current_factor` 单位修复 ✅

**问题**: 网络输出全为零，导致梯度为零

**根本原因**: `_current_factor` 将 pA 转换为 mV，但膜电位使用归一化单位（dimensionless）

**修复位置**: `src/v1_jax/nn/glif3_brainstate.py:156-165`

```python
# 修复前
self._current_factor = (1.0 / self.C_m) * (1.0 - jnp.exp(-dt / self.tau_m)) * self.tau_m

# 修复后
current_factor_mv = (1.0 / self.C_m) * (1.0 - jnp.exp(-dt / self.tau_m)) * self.tau_m
self._current_factor = current_factor_mv / self.voltage_scale  # 归一化
```

### 2. 训练数据形状和缩放修复 ✅

**修复位置**: `scripts/train_brainstate.py:create_synthetic_data()`

- 输入形状: `(seq_len, batch, n_neurons)` 而非 `(seq_len, batch, n_neurons * n_receptors)`
- 输入缩放: 值范围 ~0-300 pA（足以驱动脉冲）

### 3. Loss 缩放修复梯度太小问题 ✅ (2026-03-08)

**问题**: 梯度太小（~1e-5 到 1e-7），导致权重几乎不变

**根本原因**: `_current_factor` (~0.001) 将梯度缩小了 ~1000 倍

**解决方案**: 在 trainer 中添加 `loss_scale` 参数

**修复位置**: `src/v1_jax/training/trainer_brainstate.py`

```python
@dataclass
class IODimConfig:
    loss_scale: float = 1000.0  # 新增：放大梯度
    ...

def loss_fn_wrapper():
    ...
    return (mse_loss + regularization_loss) * loss_scale  # 放大
```

**验证结果** (30 神经元, loss_scale=10000, lr=1e-3, 50 epochs):
- Loss: 0.1402 → 0.1336 (-5%)
- Weight change: 0.000074 → 0.003882
- 训练成功！

## 已解决的问题

### 1. 发放率低（已分析，是正常现象）

**数据**:
- 200 pA 输入: 4.3% 发放率
- 500 pA 输入: 10.6% 发放率
- 2000 pA 输入: 17.8% 发放率

**原因**:
1. 不应期 3ms (dt=1ms) → 最大发放率 ~33%
2. ASC（适应性脉冲电流）进一步抑制 → 实际最大 ~22%
3. Billeh 网络有 111 种神经元类型，阈值电流分布宽 (25-635 pA)

**结论**: 正常行为，无需修复。

### 2. Loss 不下降 ✅ 已修复

**根本原因**: 梯度太小（~1e-5 到 1e-7），标准学习率无法产生有效更新

**详细分析**:
梯度通过 `_current_factor` (~0.001) 被缩小了约 1000 倍：
```
gradient = d_loss/d_output × d_output/d_V × d_V/d_I × d_I/d_weight
         ≈ 0.4 × 0.3 × 0.001 × 0.3 ≈ 3.6e-5
```

**解决方案**: 在 `IODimConfig` 中添加 `loss_scale` 参数

**推荐配置**:
```python
config = IODimConfig(
    learning_rate=1e-3,    # 标准学习率
    loss_scale=10000.0,    # 放大梯度
    etrace_decay=0.99,
    grad_clip_norm=1.0,
)
```

## 关键发现

### Billeh 网络参数分析

从 `scripts/debug_billeh_params.py` 输出:

```
Per-neuron parameter distributions:
  V_th: -57.64 to -16.33 mV, mean=-43.11
  E_L: -85.44 to -61.70 mV, mean=-75.73
  voltage_scale: 15.38 to 54.37 mV, mean=32.62
  I_threshold: 25.3 to 635.1 pA, mean=163.6

Fraction of neurons with I_threshold <= input current:
    50 pA:  1.8%
   100 pA: 12.3%
   200 pA: 65.0%
   300 pA: 94.3%
   500 pA: 99.8%
```

### 代理梯度行为

`brainstate.surrogate.sigmoid` 行为:
- 前向: 阶跃函数 (x<0 → 0, x>=0 → 1)
- 反向: 平滑梯度 (max=1.0 at x=0, 快速衰减)

```
input= -2.0 -> output=0.0000, grad=0.0013
input= -1.0 -> output=0.0000, grad=0.0707
input= -0.5 -> output=0.0000, grad=0.4200
input=  0.0 -> output=1.0000, grad=1.0000
input=  0.1 -> output=1.0000, grad=0.9610
```

## 下一步工作

### 优先级 1: 完整 Billeh 网络测试

1. 使用真实 Billeh 网络数据（51978 神经元）测试训练
2. 使用 `scripts/train_brainstate.py` 进行完整训练
3. 验证 IODim 内存效率提升

### 优先级 2: 真实任务数据测试

1. 加载真实 LGN 数据 (`alternate_small_stimuli.pkl`)
2. 使用 Garrett 任务或 Evidence 任务目标
3. 对比 BPTT baseline（如果内存允许）

### 优先级 3: 超参数优化

1. 调整 `loss_scale`（1000-10000 范围）
2. 调整 `etrace_decay`（0.9-0.99 范围）
3. 实验不同学习率策略（warmup, decay）

## 文件清单

### 核心实现
- `src/v1_jax/nn/glif3_brainstate.py` - GLIF3 神经元（已修复）
- `src/v1_jax/nn/connectivity_brainstate.py` - 稀疏连接和延迟缓冲
- `src/v1_jax/models/v1_network_brainstate.py` - V1 网络
- `src/v1_jax/training/trainer_brainstate.py` - IODim 训练器
- `src/v1_jax/compat/jax_compat.py` - JAX 0.9.x 兼容性 patch

### 训练脚本
- `scripts/train_brainstate.py` - 主训练入口
- `scripts/test_training_brainstate.py` - 小网络测试
- `scripts/test_billeh_brainstate.py` - 完整 Billeh 网络测试

### 调试脚本
- `scripts/debug_glif3_brainstate.py` - GLIF3 诊断
- `scripts/debug_spike_rate.py` - 发放率诊断
- `scripts/debug_billeh_params.py` - 网络参数分析
- `scripts/debug_iodim_gradients.py` - IODim 梯度调试（待运行）

## 运行命令

```bash
# 测试 GLIF3 修复
uv run python scripts/debug_glif3_brainstate.py

# 测试发放率
uv run python scripts/debug_spike_rate.py

# 分析 Billeh 参数
uv run python scripts/debug_billeh_params.py

# 调试 IODim 梯度（待运行）
uv run python scripts/debug_iodim_gradients.py

# 完整 Billeh 测试
uv run python scripts/test_billeh_brainstate.py

# 正式训练
uv run python scripts/train_brainstate.py data_dir=/nvmessd/yinzi/GLIF_network
```

## 恢复工作时的下一步

1. **测试完整 Billeh 网络**:
   ```bash
   uv run python scripts/train_brainstate.py data_dir=/nvmessd/yinzi/GLIF_network training.batch_size=8
   ```

2. **验证内存效率**:
   - IODim 应该将激活值内存从 ~8 GB 降到 ~1 MB
   - 预计可以支持更大的 batch_size

3. **Loss scale 调优**:
   - 默认 1000.0，可能需要调整为 10000.0
   - 观察 loss 曲线和权重变化

**注意**: NumPy 2.4 与 pandas 有兼容性问题，避免导入 `src/v1_jax/models/__init__.py`
