# V1 SNN 训练方案对比实验报告

**项目**: Allen V1 GLIF Network Training with JAX
**实验日期**: 2026-03-08
**报告生成时间**: 2026-03-08 21:30

---

## 目录

1. [实验背景与目标](#1-实验背景与目标)
2. [实验设计](#2-实验设计)
3. [实验环境](#3-实验环境)
4. [实验结果](#4-实验结果)
5. [可视化分析](#5-可视化分析)
6. [结论](#6-结论)
7. [下一步计划](#7-下一步计划)

---

## 1. 实验背景与目标

### 1.1 研究背景

本项目旨在训练一个基于生物学约束的 V1 视觉皮层脉冲神经网络 (SNN)，用于方向辨别任务。网络结构来自 Allen Institute 的 GLIF 模型，包含约 52,000 个神经元和 350 万个突触连接。

**核心挑战**:
- 大规模稀疏网络的高效训练
- BPTT (Backpropagation Through Time) 的内存消耗
- 稀疏矩阵运算的性能优化

### 1.2 实验目标

1. **建立 Baseline**: 使用 JAX 原生稀疏格式 (BCOO) + BPTT 完成完整训练
2. **评估 BCSR 优化**: 测试 BCSR 稀疏格式对训练速度的影响
3. **验证 Brainstate 方案**: 探索 brainstate/brainevent/braintrace 生态系统的可行性

### 1.3 为什么做这些实验

| 实验 | 动机 |
|------|------|
| **Exp 1: BCOO Baseline** | 确立性能基准，验证网络可训练性 |
| **Exp 2: BCSR 优化** | OPTIMIZATION_PLAN.md 显示 BCSR 可能带来 18% 加速 |
| **Exp 3: Brainstate** | 探索显存节省 500x (IODim) 和速度提升 5-20x (event-driven) |

---

## 2. 实验设计

### 2.1 实验方案总览

| 方案 | 稀疏格式 | 梯度方法 | 状态 |
|------|---------|---------|------|
| **Exp 1: Baseline** | BCOO | BPTT | ✅ 完成 |
| **Exp 2: BCSR优化** | BCSR | BPTT | ✅ 完成 |
| **Exp 3: Brainstate** | brainevent | IODim | 🔄 架构已修复，待执行 |

### 2.2 训练任务

**Garrett 方向辨别任务**:
- 分类: 2 类 (左/右方向)
- 输入: LGN firing rates → V1 网络
- 时长: 600ms 模拟时间
- 输出: 最后 50ms 平均发放率 → 线性分类器

### 2.3 网络架构

```
Input (LGN)     →    V1 Network    →    Readout
17,400 neurons       51,978 neurons      2 classes
                     3.5M synapses
                     4 receptor types
```

### 2.4 训练配置

| 参数 | 值 |
|------|-----|
| Epochs | 16 |
| Batch Size | 32 (4/卡 × 8卡) |
| Learning Rate | 0.001 (Adam) |
| Gradient Clip | 1.0 |
| Rate Cost | 0.1 |
| Voltage Cost | 1e-5 |

---

## 3. 实验环境

### 3.1 硬件配置

| 资源 | 规格 |
|------|------|
| **GPU** | 8× NVIDIA A40 (48GB/卡) |
| **总显存** | 384 GB |
| **存储** | NVMe SSD |

### 3.2 软件环境

| 软件 | 版本 |
|------|------|
| **JAX** | 0.4.x |
| **brainstate** | latest |
| **brainevent** | latest |
| **braintrace** | latest |
| **Python** | 3.11 |

---

## 4. 实验结果

### 4.1 Experiment 1: BCOO Baseline

**重要发现**: 配置文件中设置 `sparse_format: brainevent_csr`，但由于代码逻辑问题实际使用了 BCOO:

```python
# sparse_layer.py:169-172
if format == "bcsr":     # "brainevent_csr" != "bcsr" → False
    return self.to_bcsr()
else:
    return self.to_bcoo()  # ← 实际执行
```

#### 训练数据 (16 Epochs)

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc | Spike Rate |
|-------|------------|-----------|----------|---------|------------|
| 1 | 2.3174 | 35.22% | 1.2615 | 41.09% | 0.34% |
| 2 | 1.0674 | 61.00% | 1.5985 | 72.97% | 0.40% |
| 3 | 0.9645 | 84.97% | 2.0290 | 87.50% | 0.43% |
| 4 | 0.9249 | 89.88% | 1.8303 | 89.84% | 0.43% |
| 5 | 0.8905 | 89.88% | 1.9872 | 90.62% | 0.43% |
| 6 | 0.8661 | 90.72% | 1.9458 | 89.22% | 0.43% |
| **7** | 0.8586 | 89.00% | **1.9241** | **91.87%** | 0.43% |
| 8 | 0.8437 | 88.91% | 2.2772 | 87.81% | 0.44% |
| 9 | 0.8233 | 90.03% | 2.7885 | 88.91% | 0.46% |
| 10 | 0.8067 | 90.19% | 2.8644 | 90.00% | 0.46% |
| 11 | 0.7914 | 90.38% | 3.0637 | 87.97% | 0.47% |
| 12 | 0.7808 | 89.91% | 3.3783 | 89.53% | 0.47% |
| 13 | 0.7626 | 90.66% | 3.6195 | 87.81% | 0.48% |
| 14 | 0.7589 | 89.78% | 3.7452 | 89.84% | 0.48% |
| 15 | 0.7436 | 90.03% | 3.9382 | 90.16% | 0.49% |
| **16** | **0.7281** | **90.34%** | 4.0210 | 90.16% | 0.49% |

#### 关键指标

| 指标 | 值 |
|------|-----|
| **最终训练准确率** | 90.34% |
| **最终验证准确率** | 90.16% |
| **最佳验证准确率** | 91.87% (Epoch 7) |
| **收敛速度** | Epoch 3 达到 85%+ |
| **总训练时间** | ~135 分钟 (2h15m) |
| **每 Epoch 时间** | ~8.4 分钟 |
| **GPU 显存** | ~19 GB/卡 |

### 4.2 Experiment 2: BCSR 优化

#### 训练数据 (16 Epochs)

| Epoch | Val Loss | Val Acc | Spike Rate |
|-------|----------|---------|------------|
| 1 | 1.2616 | 41.56% | 0.34% |
| 2 | 1.5987 | 72.97% | 0.40% |
| 3 | 2.0298 | 87.03% | 0.43% |
| 4 | 1.8296 | 89.84% | 0.43% |
| 5 | 1.9881 | 90.62% | 0.43% |
| 6 | 1.9453 | 89.22% | 0.43% |
| **7** | **1.9236** | **91.87%** | 0.43% |
| 8 | 2.2800 | 87.81% | 0.44% |
| 9 | 2.7954 | 88.91% | 0.46% |
| 10 | 2.8697 | 90.00% | 0.46% |
| 11 | 3.0788 | 87.97% | 0.47% |
| 12 | 3.3783 | 89.53% | 0.47% |
| 13 | 3.6195 | 87.81% | 0.48% |
| 14 | 3.7452 | 89.84% | 0.48% |
| 15 | 3.9382 | 90.16% | 0.49% |
| 16 | 4.0210 | 90.16% | 0.49% |

#### 速度对比

| 指标 | BCOO | BCSR | 差异 |
|------|------|------|------|
| 每 Epoch 时间 | ~8.4 min | ~13 min | **-55% 更慢** |
| 总训练时间 | ~135 min | ~208 min | **-55%** |
| GPU 显存 | ~19 GB | ~37 GB | **+95% 更多** |

**意外发现**: BCSR 反而比 BCOO **更慢**，且消耗更多显存！

可能原因:
1. JAX 的 BCSR 实现在此稀疏度 (~0.13%) 下效率不高
2. cuSPARSE 调用开销超过稀疏矩阵乘法收益
3. BCOO 在 JAX 编译器中有更好的优化

### 4.3 Experiment 3: Brainstate (IODim)

#### 架构修复

已完成的修复工作:

1. **添加 Input Layer 支持**:
   - `V1NetworkBrainstate` 新增 `input_data` 和 `bkg_weights` 参数
   - 新增 `update_with_lgn()` 方法处理 LGN firing rates

2. **数据加载器适配**:
   - 新增 `GarrettDataLoaderBrainstate` 类
   - 支持 LGN → V1 电流预计算

#### 前向传播验证

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

#### IODim 训练测试

| 指标 | 值 |
|------|-----|
| IODim step (JIT后) | ~120ms |
| First step JIT | ~12s |
| 预估完整训练 | 2-3 hours |

**当前限制**: Loss 不下降，需要调试梯度流。

---

## 5. 可视化分析

### 5.1 训练损失曲线对比

![Loss Comparison](figures/fig1_loss_comparison.png)

**分析**:
- BCOO 和 BCSR 的训练曲线几乎完全重合
- 验证损失从 Epoch 2 后持续上升 (过拟合信号)
- 训练损失稳定下降，最终收敛到 ~0.73

### 5.2 准确率曲线对比

![Accuracy Comparison](figures/fig2_accuracy_comparison.png)

**分析**:
- 两种稀疏格式在准确率上没有差异
- 训练准确率在 Epoch 3 后稳定在 ~90%
- 验证准确率在 Epoch 7 达到峰值 91.87%

### 5.3 过拟合分析

![Overfitting Analysis](figures/fig3_overfitting_analysis.png)

**观察**:
- 训练损失持续下降 (2.32 → 0.73)
- 验证损失先降后升 (1.26 → 1.92 → 4.02)
- Generalization Gap 从 Epoch 4 开始显著扩大

**可能原因**:
1. 训练数据有限 (仅 8 张图像循环使用)
2. 模型容量相对任务过大
3. 缺乏正则化技术 (如 dropout)

### 5.4 Spike Rate 演变

![Spike Rate Evolution](figures/fig4_spike_rate_evolution.png)

**分析**:
- Spike rate 从 0.34% 逐渐增加到 0.49%
- 符合生物学约束 (稀疏发放)
- 随训练增加可能表示网络活跃度提升

### 5.5 训练速度对比

![Training Speed Comparison](figures/fig5_training_speed_comparison.png)

**关键发现**:
- BCSR **比 BCOO 慢 55%**
- 总训练时间: BCOO 135min vs BCSR 208min
- 与预期的 18% 加速完全相反

### 5.6 综合学习曲线

![BCOO Learning Curves](figures/fig7_bcoo_learning_curves.png)

### 5.7 优化方案雷达图对比

![Radar Comparison](figures/fig8_radar_comparison.png)

**解读**:
- BCOO 在速度和实现复杂度上表现最好
- BCSR 在所有维度上都不如 BCOO
- Brainstate 预期在内存效率和可扩展性上有优势

---

## 6. 结论

### 6.1 主要发现

| 发现 | 详情 |
|------|------|
| **网络可训练** | V1 GLIF 网络在 Garrett 任务上可达到 ~90% 准确率 |
| **快速收敛** | Epoch 3 即达到 85%+，Epoch 7 达到最佳 91.87% |
| **BCSR 无效** | BCSR 比 BCOO 慢 55%，显存增加 95%，完全不推荐 |
| **过拟合明显** | 验证损失从 Epoch 2 后持续上升 |
| **Brainstate 可行** | 前向传播验证成功，IODim 训练需要调试 |

### 6.2 优化方案评估

| 优化方案 | 预期收益 | 实际收益 | 结论 |
|---------|---------|---------|------|
| BCSR 稀疏格式 | +18% 速度 | **-55% 更慢** | ❌ 不推荐 |
| Gradient Checkpointing | 显存节省 | 无效 | ❌ 不适用 |
| ZeRO-2 | 显存节省 | 无效 | ❌ 参数太少 |
| braintrace IODim | 500x 显存 | 需要调试 | ⚠️ 待验证 |
| brainevent Event-driven | 5-20x 速度 | 30ms/step | ✅ 前向有效 |

### 6.3 技术债务

1. 配置文件中 `brainevent_csr` 字符串匹配问题导致意外回退到 BCOO
2. IODim 训练器不支持外部输入层
3. 梯度流调试尚未完成

---

## 7. 下一步计划

### 7.1 短期 (1-2天)

1. **修复 IODim 梯度问题**
   - 检查 gradient magnitude
   - 验证 loss scale 设置
   - 调试前向/反向传播

2. **完成 Brainstate 训练**
   - 在相同配置下与 BCOO baseline 对比
   - 记录显存和速度指标

### 7.2 中期 (1周)

1. **解决过拟合**
   - 实现 dropout 正则化
   - 增加训练数据多样性
   - 早停策略

2. **性能优化**
   - 探索混合精度训练 (bf16)
   - 批大小调优

### 7.3 长期

1. **扩展到更复杂任务**
   - 多类别分类
   - 更长时间序列

2. **生物学验证**
   - 对比真实 V1 响应特性
   - 分析网络动力学

---

## 附录

### A. 代码仓库结构

```
allen_v1_chen_2022_jax/
├── scripts/
│   ├── train.py              # BPTT 训练脚本
│   └── train_brainstate.py   # Brainstate 训练脚本
├── v1_model/
│   ├── model.py              # V1Network 定义
│   ├── model_brainstate.py   # V1NetworkBrainstate 定义
│   └── sparse_layer.py       # 稀疏矩阵处理
├── experiments/
│   └── baseline_vs_brainstate/
│       ├── experiment_report.md
│       ├── DETAILED_EXPERIMENT_REPORT.md  # 本报告
│       ├── detailed_experiment_report.py  # 图表生成脚本
│       ├── figures/          # 生成的图表
│       └── logs/             # 训练日志
└── configs/
    └── default.yaml          # 默认配置
```

### B. 复现命令

```bash
# Experiment 1: BCOO Baseline
uv run python scripts/train.py \
  data_dir=/nvmessd/yinzi/GLIF_network \
  results_dir=/nvmessd/yinzi/results \
  training.batch_size=32 \
  training.n_epochs=16 \
  task.name=garrett \
  use_pmap=true

# Experiment 2: BCSR
uv run python scripts/train.py \
  data_dir=/nvmessd/yinzi/GLIF_network \
  results_dir=/nvmessd/yinzi/results/bcsr \
  training.batch_size=32 \
  training.n_epochs=16 \
  task.name=garrett \
  network.sparse_format=bcsr \
  use_pmap=true

# Experiment 3: Brainstate (前向测试)
python -u scripts/debug_brainstate_garrett.py
```

### C. 参考文献

1. Chen et al. (2022). "Highly accurate prediction of cortical responses to natural movies through biologically-based neural network simulations"
2. Allen Institute for Brain Science. GLIF Models Documentation.
3. JAX Documentation: Sparse Arrays

---

*报告生成: 2026-03-08*
*代码版本: eba73b8 (main)*
