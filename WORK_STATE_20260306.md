# 工作现场保存 - 2026-03-06

## 项目概述
JAX实现的Allen Institute V1视觉皮层模型（Chen et al., Science Advances 2022）
- 原始实现：TensorFlow/Keras (`/nvmessd/yinzi/Training-data-driven-V1-model/`)
- JAX重构：`/nvmessd/yinzi/allen_v1_chen_2022_jax/`

## 运行命令
```bash
# JAX (Hydra配置，无--前缀)
uv run python scripts/train.py \
    data_dir=/nvmessd/yinzi/GLIF_network \
    results_dir=/nvmessd/yinzi/results \
    training.batch_size=4 \
    training.n_epochs=16 \
    task.name=garrett

# TF原始
cd /nvmessd/yinzi/Training-data-driven-V1-model
python multi_training.py --task_name=garrett ...
```

---

## 已完成的修复

### 1. 添加缺失函数 `create_drifting_grating_batch`
**文件**: `src/v1_jax/data/stim_generator.py:756-870`
**问题**: train.py导入了不存在的函数
**修复**: 添加了该函数，后来又重写以返回连续firing rates而非binary spikes

### 2. 修复 `load_billeh` 调用
**文件**: `scripts/train.py:595-603`
**问题**: 原代码 `load_billeh(config.data_dir)` 签名错误
**修复**:
```python
input_pop, network_data, bkg_weights = load_billeh(
    n_input=config.n_input,
    n_neurons=config.neurons,
    core_only=config.core_only,
    data_dir=config.data_dir,
    seed=config.seed,
    use_dale_law=config.use_dale_law,
)
n_neurons = network_data['n_nodes']
```

### 3. 修复 `V1Network.from_billeh` 接受预加载数据
**文件**: `src/v1_jax/models/v1_network.py:138-176`
**修复**: 添加 `network_data` 和 `input_pop` 可选参数，避免重复加载

### 4. 修复 `MultiClassReadout` 构造函数
**文件**: `scripts/train.py:617-621`
**问题**: 原代码缺少 `n_neurons` 参数，使用了错误的 `pool_method`
**修复**:
```python
readout = MultiClassReadout(
    n_neurons=network.n_neurons,
    n_classes=n_output,
    temporal_pooling='mean',
)
```

### 5. 修复 voltage_regularization 使用正确的 voltage_scale
**文件**: `src/v1_jax/training/trainer.py:262-274`
**问题**: 使用默认 `voltage_scale=1.0`，但应该使用网络的实际值（~20-30 mV）
**修复**:
```python
voltage_loss = voltage_regularization(
    output.voltages,
    v_th=jnp.ones((self.network.n_neurons,)),
    v_reset=jnp.zeros((self.network.n_neurons,)),
    voltage_cost=self.config.voltage_cost,
    voltage_scale=self.network.glif3_params.voltage_scale,  # 从网络获取
    voltage_offset=self.network.glif3_params.voltage_offset,
)
```

### 6. voltage_regularization 计算方式
**文件**: `src/v1_jax/training/regularizers.py:48-56`
**当前状态**: 使用 `sum over neurons, mean over batch/time`（与TF一致）
```python
voltage_loss = jnp.mean(jnp.sum(v_pos + v_neg, axis=-1))
return voltage_cost * voltage_loss
```

---

## 当前核心问题：Firing Rate 差异巨大

### 数值对比
| 指标 | TF | JAX | 差异 |
|------|-----|-----|------|
| Loss | ~1.9 | ~225-7600 | 100-4000x |
| RLoss | ~1.4 | ~224-766 | 160-500x |
| VLoss | ~0.32 | ~0.14-6800 | 不稳定 |
| Accuracy | ~0.60 | ~0.12 | 差很多 |
| **Rate** | **0.0016** | **0.068-0.24** | **40-150x 太高!** |

### 根本原因分析

#### 1. 输入数据格式差异
**TF使用的真实LGN数据**:
- 文件: `/nvmessd/yinzi/alternate_small_stimuli.pkl` 或 `many_small_stimuli.pkl`
- 格式: dict，每个key是图像名，value是 `(1000, 17400)` 的firing rates
- 数值范围: 0-94 Hz，mean ~5.4
- 代码位置: `Training-data-driven-V1-model/multi_training.py:140-156`

```python
# TF数据加载
path = os.path.join(flags.data_dir, '../alternate_small_stimuli.pkl')
_data_set = stim_dataset.generate_data_set_continuing(
    path, batch_size=per_replica_batch_size, seq_len=flags.seq_len,
    current_input=True, ...)  # current_input=True 表示连续firing rates
```

**JAX目前使用合成数据**:
- 文件: `src/v1_jax/data/stim_generator.py:create_drifting_grating_batch`
- 最初版本：生成binary spikes，~10% spike概率（太高）
- 修改后版本：生成连续firing rates，scale=5.0（仍然问题）

#### 2. 输入权重缩放差异（关键！未修复）
**TF代码** (`models.py:234-236`):
```python
input_weights = input_population['weights'].astype(np.float32)
# 关键：输入权重除以voltage_scale!
input_weights = input_weights / voltage_scale[self._node_type_ids[input_indices[:, 0] // self._n_receptors]]
```

**JAX代码**:
- 搜索 `network_loader.py` 未找到对输入权重的voltage_scale缩放
- **这可能是导致firing rate过高的主要原因！**

#### 3. voltage_scale 值
- 定义: `V_th - E_L`（阈值电压 - 静息电位）
- 典型值: 20-30 mV
- TF: 用于缩放输入权重和循环权重
- JAX: GLIF3Params中存储了voltage_scale，但输入权重可能未缩放

---

## 待检查/修复的问题

### 高优先级
1. **检查JAX的input_weights是否除以voltage_scale**
   - 文件: `src/v1_jax/data/network_loader.py`
   - 参考TF: `models.py:236`

2. **检查recurrent_weights的voltage_scale处理**
   - TF (`models.py:228`): `weights = weights / voltage_scale[...]`

3. **检查bkg_weights的处理**
   - TF (`models.py:270`): `bkg_weights = bkg_weights / np.repeat(voltage_scale[...], self._n_receptors)`

### 中优先级
4. **实现真实LGN数据加载**
   - 已添加 `LGNStimulusLoader` 类但未在train.py中使用
   - 需要修改train.py的数据迭代器使用pkl文件

5. **检查ASC (Adaptive Spike Currents) 的缩放**
   - TF (`models.py:151`): `asc_amps = asc_amps / voltage_scale[..., None]`
   - JAX (`glif3_cell.py:142`): `asc_amps_type = node_params['asc_amps'] / voltage_scale[..., None]` ✓ 已有

---

## 关键文件对照

### TF核心文件
```
/nvmessd/yinzi/Training-data-driven-V1-model/
├── models.py          # GLIF3Cell, 权重缩放, voltage regularization
├── multi_training.py  # 训练循环, 数据加载, loss计算
├── stim_dataset.py    # 数据生成器
└── load_sparse.py     # 网络数据加载
```

### JAX核心文件
```
/nvmessd/yinzi/allen_v1_chen_2022_jax/
├── scripts/train.py                     # 训练脚本
├── src/v1_jax/
│   ├── nn/
│   │   ├── glif3_cell.py               # GLIF3神经元
│   │   └── sparse_layer.py             # 稀疏层
│   ├── models/
│   │   ├── v1_network.py               # V1网络
│   │   └── readout.py                  # 读出层
│   ├── training/
│   │   ├── trainer.py                  # 训练器
│   │   └── regularizers.py             # 正则化
│   └── data/
│       ├── network_loader.py           # 网络加载
│       └── stim_generator.py           # 刺激生成
└── configs/
    └── training/default.yaml           # 训练配置
```

---

## 配置参数对比

### voltage_cost
- TF默认: `0.00001` (1e-5) - `multi_training.py:469`
- JAX默认: `1e-5` - `configs/training/default.yaml:41`
- 相同 ✓

### rate_cost
- TF默认: `0.1` - `multi_training.py:468`
- JAX默认: `0.1` - `configs/training/default.yaml:38`
- 相同 ✓

### learning_rate
- TF默认: `0.001` - `multi_training.py:467`
- JAX默认: `0.001` - `configs/training/default.yaml:23`
- 相同 ✓

---

## 数据文件位置

```
/nvmessd/yinzi/
├── GLIF_network/                    # 网络数据目录
│   ├── node_params/
│   ├── edges/
│   └── ...
├── alternate_small_stimuli.pkl      # 556 MB, 8张图像的LGN响应
├── many_small_stimuli.pkl           # 2.8 GB, 更多图像
├── garrett_firing_rates.pkl         # 208 KB, 目标firing rates
├── input_dat.pkl                    # 95 MB
└── network_dat.pkl                  # 1.7 GB
```

### alternate_small_stimuli.pkl 格式
```python
# dict, 8个图像
{
    'n02510455_5188.JPEG': array(1000, 17400),  # (time, n_lgn)
    'n02493509_12717.JPEG': array(1000, 17400),
    ...
}
# dtype: float32
# 值范围: 0-94 Hz, mean ~5.4
```

---

## 下一步建议

1. **首先检查并修复 input_weights 的 voltage_scale 缩放**
   - 这很可能是firing rate过高的主要原因
   - 检查 `network_loader.py` 的 `load_billeh` 函数

2. **验证修复后重新测试**
   ```bash
   rm -rf /nvmessd/yinzi/results/garrett
   uv run python scripts/train.py data_dir=/nvmessd/yinzi/GLIF_network \
       results_dir=/nvmessd/yinzi/results \
       training.batch_size=4 training.n_epochs=1 \
       training.steps_per_epoch=10 task.name=garrett
   ```

3. **如果仍有问题，实现真实LGN数据加载**
   - 使用 `LGNStimulusLoader` 类加载pkl文件
   - 修改 train.py 的数据迭代器

---

## 重要代码片段

### TF voltage_regularization (models.py:390-399)
```python
def __init__(self, cell, voltage_cost=1e-5):
    self._voltage_cost = voltage_cost
    self._cell = cell

def __call__(self, voltages):
    voltage_32 = (tf.cast(voltages, tf.float32) - self._cell.voltage_offset) / self._cell.voltage_scale
    v_pos = tf.square(tf.nn.relu(voltage_32 - 1.))
    v_neg = tf.square(tf.nn.relu(-voltage_32 + 1.))
    voltage_loss = tf.reduce_mean(tf.reduce_sum(v_pos + v_neg, -1)) * self._voltage_cost
    return voltage_loss
```

### TF 权重缩放 (models.py:226-236)
```python
# 循环权重
weights = weights / voltage_scale[self._node_type_ids[indices[:, 0] // self._n_receptors]]

# 输入权重 - 关键！
input_weights = input_population['weights'].astype(np.float32)
input_weights = input_weights / voltage_scale[self._node_type_ids[input_indices[:, 0] // self._n_receptors]]
```

### JAX sparse_layer.py 权重缩放 (358-399)
```python
def scale_recurrent_weights(
    ...
    voltage_scale: Optional[np.ndarray] = None,
    ...
):
    if voltage_scale is not None and node_type_ids is not None:
        target_types = node_type_ids[target_indices]
        new_weights = new_weights / voltage_scale[target_types]
```
- 已有recurrent权重缩放
- **需要检查input权重是否有相同处理**

---

## Git状态
```
Current branch: main
Modified files:
- scripts/train.py
- src/v1_jax/models/v1_network.py
- src/v1_jax/data/stim_generator.py
- src/v1_jax/training/trainer.py
- src/v1_jax/training/regularizers.py
- uv.lock
```

---

## 额外发现（2026-03-06 16:30后）

### 输入权重缩放已存在
经过检查，JAX代码**已经**在 `prepare_input_connectivity` 中进行了voltage_scale缩放：
- 文件: `src/v1_jax/nn/sparse_layer.py:431-434`
```python
if voltage_scale is not None and node_type_ids is not None:
    target_neurons = indices[:, 0] // n_receptors
    target_types = node_type_ids[target_neurons]
    new_weights = new_weights / voltage_scale[target_types]
```
- 调用位置: `v1_network.py:196-204`
- 传入了正确的 `voltage_scale_types` 和 `node_type_ids`

### 形状处理正确
- V1Network.__call__ 在 line 317-320 正确处理了形状转换
- inputs: (seq_len, batch, n_inputs) → transpose → (batch, seq_len, n_inputs)
- 处理后再 transpose 回来

### 可能的真正原因
1. **合成刺激与真实LGN数据的差异**
   - 合成数据: 简单的sin波模式，空间均匀
   - 真实数据: 有空间结构，匹配LGN receptive fields

2. **firing_rate_scale 设置**
   - 当前: 5.0 → 平均~2.5 Hz
   - 真实LGN: 平均~5.4 Hz，但有空间稀疏性

3. **背景噪声(bkg_weights)处理**
   - 需要验证是否正确缩放

### 下一步
1. 直接加载真实LGN数据测试
2. 比较输入电流的数值范围
3. 检查背景噪声设置

---

## 总结

**核心问题**: JAX的firing rate比TF高40-150倍，导致所有loss都不正确

**排除的原因**:
- ~~输入权重未除以voltage_scale~~ (已验证存在)
- ~~形状不匹配~~ (已验证正确)

**待查原因**:
1. 合成刺激数据分布与真实LGN数据差异
2. 背景噪声设置
3. 需要直接使用真实pkl文件测试
