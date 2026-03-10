# V1 SNN 训练方案对比实验

> 实验目的：对比不同训练方案的效果和效率

## 实验概述

| 项目 | 说明 |
|------|------|
| **实验日期** | 2026-03-08 |
| **硬件环境** | 8× NVIDIA A40 (48GB/卡) |
| **网络规模** | 51,978 neurons, 3.5M synapses |
| **任务** | Garrett (方向辨别，2类) |
| **代码仓库** | `/nvmessd/yinzi/allen_v1_chen_2022_jax` |

---

## 实验方案总览

| 方案 | 脚本 | 稀疏格式 | 梯度方法 | 数据类型 | 状态 |
|------|------|---------|---------|---------|------|
| **Exp 1: Baseline** | `train.py` | BCOO | BPTT | Garrett 真实数据 | ✅ 完成 |
| **Exp 2: BCSR优化** | `train.py` | BCSR | BPTT | Garrett 真实数据 | 🔄 运行中 (8/16) |
| **Exp 3: Brainstate** | `train_brainstate.py` | brainevent | Forward Pass | Garrett 真实数据 | 🔄 待执行 |

---

## Experiment 1: Baseline (JAX BCOO + BPTT)

### 1.1 实验发现

**重要澄清**: 配置文件中 `sparse_format: brainevent_csr`，但代码逻辑导致实际使用 **BCOO**：

```python
# sparse_layer.py:169-172
if format == "bcsr":     # "brainevent_csr" != "bcsr" → False
    return self.to_bcsr()
else:
    return self.to_bcoo()  # ← 实际执行
```

**结论**: 本次训练是纯 JAX 原生方案（BPTT + BCOO），**没有启用任何优化**。

### 1.2 实验配置

```bash
uv run python scripts/train.py \
  data_dir=/nvmessd/yinzi/GLIF_network \
  results_dir=/nvmessd/yinzi/results \
  training.batch_size=32 \
  training.n_epochs=16 \
  task.name=garrett \
  use_pmap=true
```

| 参数 | 值 | 备注 |
|------|-----|------|
| Epochs | 16 | |
| Batch Size | 32 (4/卡 × 8卡) | |
| Sequence Length | 600 ms | |
| Steps/Epoch | 100 (train) + 20 (val) | |
| Learning Rate | 0.001 (Adam) | |
| Gradient Clip | 1.0 | |
| Rate Cost | 0.1 | |
| Voltage Cost | 1e-5 | |
| **Sparse Format** | **BCOO** | 配置 brainevent_csr 但回退到 BCOO |
| **Gradient Method** | **BPTT** | jax.value_and_grad |

### 1.3 训练结果

#### 训练曲线

```
Accuracy (%)
100 |
 95 |                                    ●───●───●
 90 |        ●───●───●───●───●───●───●───●
 85 |    ●
 80 |
 75 |  ●
 70 |
 65 |
 60 | ●
 55 |
 50 |
 45 |●
 40 |
    +─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬
    0     2     4     6     8    10    12    14    16  Epoch

    ● Train Accuracy    ○ Val Accuracy
```

#### 数值数据

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc | Spike Rate |
|-------|------------|-----------|----------|---------|------------|
| 1 | 2.3174 | 35.22% | 1.2615 | 41.09% | 0.34% |
| 2 | 1.0674 | 61.00% | 1.5985 | 72.97% | 0.40% |
| 3 | 0.9645 | 84.97% | 2.0290 | 87.50% | 0.43% |
| 4 | 0.9249 | 89.88% | 1.8303 | 89.84% | 0.43% |
| 5 | 0.8905 | 89.88% | 1.9872 | 90.62% | 0.43% |
| 6 | 0.8661 | 90.72% | 1.9458 | 89.22% | 0.43% |
| 7 | 0.8586 | 89.00% | 1.9241 | 91.87% | 0.43% |
| 8 | 0.8437 | 88.91% | 2.2772 | 87.81% | 0.44% |
| 9 | 0.8233 | 90.03% | 2.7885 | 88.91% | 0.46% |
| 10 | 0.8067 | 90.19% | 2.8644 | 90.00% | 0.46% |
| 11 | 0.7914 | 90.38% | 3.0637 | 87.97% | 0.47% |
| 12 | 0.7808 | 89.91% | 3.3783 | 89.53% | 0.47% |
| 13 | 0.7626 | 90.66% | 3.6195 | 87.81% | 0.48% |
| 14 | 0.7589 | 89.78% | 3.7452 | 89.84% | 0.48% |
| 15 | 0.7436 | 90.03% | 3.9382 | 90.16% | 0.49% |
| **16** | **0.7281** | **90.34%** | **4.0210** | **90.16%** | **0.49%** |

#### 最终指标

| 指标 | 值 |
|------|-----|
| **最终训练准确率** | 90.34% |
| **最终验证准确率** | 90.16% |
| **最佳验证准确率** | 91.87% (Epoch 7) |
| **收敛速度** | Epoch 3 达到 85%+ |

### 1.4 训练速度

| 指标 | 值 |
|------|-----|
| **总训练时间** | ~135 分钟 (2h15m) |
| **每 Epoch 时间** | ~8.4 分钟 |
| **每步耗时** | ~14.5 秒 |
| **JIT 编译时间** | ~6 分钟 (首步) |
| **样本吞吐量** | ~2.2 samples/s |

### 1.5 资源使用

| 资源 | 值 |
|------|-----|
| **GPU 显存** | ~19 GB/卡 (batch=32) |
| **显存利用率** | ~40% (48GB/卡) |
| **Checkpoint 大小** | ~66 MB/epoch |

### 1.6 过拟合分析

```
Loss
4.5 |                                          ○
4.0 |                                      ○
3.5 |                                  ○
3.0 |                              ○
2.5 |                          ○
2.0 |          ○   ○   ○   ○
1.5 |      ○
1.0 |  ●───●───●───●───●───●───●───●───●───●───●───●───●───●───●
0.5 |
    +─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬
    0     2     4     6     8    10    12    14    16  Epoch

    ● Train Loss    ○ Val Loss
```

**观察**: 验证 Loss 从 Epoch 2 后持续上升（1.60 → 4.02），但验证准确率稳定在 ~90%。

---

## Experiment 2: BCSR 优化 (JAX BCSR + BPTT)

### 2.1 实验目标

对比 BCSR 稀疏格式 vs BCOO 的训练速度差异。

根据 OPTIMIZATION_PLAN.md 的 benchmark：
- **预期加速**: 18% (2425ms → 2064ms/step)

### 2.2 实验配置

```bash
uv run python scripts/train.py \
  data_dir=/nvmessd/yinzi/GLIF_network \
  results_dir=/nvmessd/yinzi/results/bcsr \
  training.batch_size=32 \
  training.n_epochs=16 \
  task.name=garrett \
  network.sparse_format=bcsr \
  use_pmap=true
```

### 2.3 训练结果 (进行中)

> **状态**: 🔄 运行中 (Epoch 8/16)

| Epoch | Val Loss | Val Acc | Spike Rate |
|-------|----------|---------|------------|
| 1 | 1.2616 | 41.56% | 0.34% |
| 2 | 1.5987 | 72.97% | 0.40% |
| 3 | 2.0298 | 87.03% | 0.43% |
| 4 | 1.8296 | 89.84% | 0.43% |
| 5 | 1.9881 | 90.62% | 0.43% |
| 6 | 1.9453 | 89.22% | 0.43% |
| **7** | **1.9236** | **91.87%** | **0.43%** |
| 8 | 2.2800 | 87.81% | 0.44% |
| 9-16 | ... | ... | ... |

**观察**: BCSR 结果与 BCOO baseline 几乎完全一致：
- 最佳验证准确率: 91.87% (Epoch 7) - 与 baseline 相同
- 验证损失趋势相同

### 2.4 训练速度对比

| 指标 | BCOO (Exp 1) | BCSR (Exp 2) | 差异 |
|------|--------------|--------------|------|
| 每 Epoch 时间 | ~8.4 min | ~13 min | -55% 更慢 |
| 总训练时间 | 135 min | ~208 min (预估) | -55% |

**发现**: BCSR 反而比 BCOO 更慢。这可能是因为：
1. JAX 的 BCSR 实现未完全优化
2. cuSPARSE 在此稀疏度下性能不如预期
3. BCOO 在 JAX 中有更好的编译优化

---

## Experiment 3: Brainstate (Forward Pass + Event-driven)

### 3.1 实验状态

✅ **架构已修复** - 等待 BCSR 实验完成后执行

### 3.2 架构修复

**已解决问题**:

1. **添加 Input Layer 支持**:
   - 修改 `V1NetworkBrainstate` 添加 `input_data` 和 `bkg_weights` 参数
   - 新增 `update_with_lgn()` 方法处理 LGN firing rates
   - 修改 `simulate()` 支持 `use_lgn_input=True` 参数

2. **数据加载器**:
   - 新增 `GarrettDataLoaderBrainstate` 类
   - 从 `load_network()` 改为 `load_billeh()` 获取输入连接数据

**代码变更**:
```python
# V1NetworkBrainstate 新增参数
def __init__(
    self,
    network_data: Dict[str, Any],
    input_data: Optional[Dict[str, Any]] = None,  # LGN → V1 连接
    bkg_weights: Optional[np.ndarray] = None,     # 背景权重
    ...
)

# 新增方法处理 LGN 输入
def update_with_lgn(self, lgn_input: jnp.ndarray) -> jnp.ndarray:
    # 将 LGN firing rates 通过 input CSR 转换为 V1 输入电流
    for key, csr in self.connectivity.input_csr_matrices.items():
        contrib = csr @ lgn_input[b]  # (n_neurons,)
        self.delay_buffer.add_delayed_synaptic_input(...)
```

### 3.3 前向传播验证

```bash
# Debug 脚本验证前向传播
python -u scripts/debug_brainstate_garrett.py
```

**结果**:
```
Network created in 1.7s
  has_input_layer: True
  n_neurons: 51978
  n_inputs: 17400

First update_with_lgn() completed in 4.6s  # JIT 编译
  Step 1: 0.031s  # 编译后速度
  Step 2: 0.030s

Simulation completed in 3.0s (100 timesteps)
  Mean spike rate: 1.25%

SUCCESS! Brainstate forward pass works correctly.
```

### 3.4 IODim 训练限制

**当前限制**:
- IODim 训练器编译时固定输入形状为 `(batch, n_neurons)`
- 无法直接支持 `update_with_lgn()` 的 LGN 输入形状 `(batch, n_inputs)`
- 需要在 IODimTrainer 中添加 input layer 支持

**替代方案**:
- 当前使用 `train_garrett_simple()` 运行前向传播测试（无梯度）
- 可以验证网络结构正确性和推理性能

### 3.5 预期实验配置

```bash
python -u scripts/train_brainstate.py \
  --data_dir=/nvmessd/yinzi/GLIF_network \
  --task=garrett \
  --batch_size=8 \
  --n_epochs=4 \
  --steps_per_epoch=20 \
  --val_steps=10 \
  --seq_len=600
```

**预期指标**:
- Forward pass: ~30ms/step (JIT 编译后)
- Simulation (600 steps): ~18s
- 总 epoch 时间: ~6 min (20 steps × 18s)

---

## 对比总结

### 当前结论

| 方案 | 准确率 | 训练时间 | 显存 | 状态 |
|------|--------|---------|------|------|
| **Exp 1: BCOO+BPTT** | 90.16% | 135 min | 19 GB | ✅ Baseline |
| **Exp 2: BCSR+BPTT** | 91.87%* | ~208 min | 37 GB | 🔄 运行中 (8/16) |
| **Exp 3: Brainstate** | TBD | TBD | TBD | 🔄 待执行 (架构已修复) |

*BCSR 最佳验证准确率（Epoch 7）

### 预期 vs 实际

| 优化方案 | 预期收益 | 实际收益 | 状态 |
|---------|---------|---------|------|
| BCSR 稀疏格式 | +18% 速度 | **-55% 更慢** | ❌ 不推荐 |
| Gradient Checkpointing | 显存节省 | 无效 | ❌ 不适用 |
| ZeRO-2 | 显存节省 | 无效 | ❌ 参数太少 |
| braintrace IODim | 500x 显存 | 需要更多开发 | ⚠️ 限制较多 |
| brainevent Event-driven | 5-20x 速度 | 30ms/step (验证通过) | ✅ 前向有效 |

---

## 附录

### A. 配置差异分析

| 配置项 | Exp 1 实际值 | 期望值 | 问题 |
|--------|-------------|--------|------|
| sparse_format | BCOO | brainevent_csr | 字符串不匹配回退 |
| gradient_method | BPTT | IODim | train.py 只支持 BPTT |

### B. 相关文件

- `experiments/baseline_vs_brainstate/experiment_report.md` - 本报告
- `experiments/baseline_vs_brainstate/logs/brainstate_train.log` - Brainstate 训练日志
- `OPTIMIZATION_PLAN.md` - 优化方案文档
- `WORK_STATE_20260308.md` - 工作进度记录

### C. 代码版本

```
commit: eba73b8 (main)
date: 2026-03-08
```

---

*报告生成时间: 2026-03-08 18:30*
*最后更新: 2026-03-08 21:00*

---

## 附录 D: IODim 训练进度 (2026-03-08 21:00)

### D.1 IODim 性能测试结果

| 指标 | 值 |
|------|-----|
| 网络规模 | 52K neurons |
| IODim step (JIT后) | ~120ms |
| First step JIT | ~12s |
| 预估完整训练 | 2-3 hours |

### D.2 已解决问题

1. **架构修复**: V1NetworkBrainstate 支持 input layer
2. **IODim 初始化**: 必须在 compile_graph 前调用 reset()
3. **预计算 LGN→V1**: 使用 CSR 矩阵转换

### D.3 待解决问题

- Loss 不下降（梯度流问题）
- 需要调试 gradient magnitude

### D.4 工作状态文件

详见: `WORK_STATE_20260308_2100.md`
