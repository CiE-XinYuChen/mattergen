# MatterGen 模型微调指南

## 概述

MatterGen 的微调功能允许您在预训练的基础模型上添加特定材料属性的条件生成能力。这使得模型能够生成具有特定物理或化学属性的晶体结构。

## 目录
- [微调原理](#微调原理)
- [支持的属性](#支持的属性)
- [快速开始](#快速开始)
- [单属性微调](#单属性微调)
- [多属性微调](#多属性微调)
- [自定义属性微调](#自定义属性微调)
- [微调配置详解](#微调配置详解)
- [最佳实践](#最佳实践)
- [故障排除](#故障排除)

## 微调原理

### 工作机制

MatterGen 的微调基于适配器模式：

1. **预训练基础**: 加载无条件生成的预训练模型
2. **模型替换**: 将标准 GemNet 替换为支持条件输入的 GemNetTCtrl
3. **属性嵌入**: 为新属性添加专门的嵌入层
4. **条件训练**: 在属性条件下重新训练模型

### 技术架构

```
预训练模型 (mattergen_base)
    ↓
GemNetT → GemNetTCtrl (添加条件输入)
    ↓
添加属性嵌入层 (PropertyEmbedding)
    ↓
微调训练 (低学习率、少轮数)
    ↓
条件生成模型
```

### 优势特点

- **保留基础能力**: 继承预训练模型的结构生成能力
- **快速收敛**: 相比从头训练，微调只需数百轮即可收敛
- **灵活扩展**: 可同时添加多个属性条件
- **资源高效**: 训练时间和计算资源需求显著降低

## 支持的属性

### 预定义属性列表

| 属性名称 | 类型 | 描述 | 单位 | 数值范围 |
|---------|------|------|------|----------|
| `dft_mag_density` | 浮点 | DFT计算的磁密度 | μB/原子 | 0-10 |
| `dft_band_gap` | 浮点 | DFT计算的带隙 | eV | 0-10 |
| `dft_bulk_modulus` | 浮点 | DFT计算的体积模量 | GPa | 0-500 |
| `ml_bulk_modulus` | 浮点 | ML预测的体积模量 | GPa | 0-500 |
| `energy_above_hull` | 浮点 | 凸包上方能量 | eV/原子 | 0-1 |
| `hhi_score` | 浮点 | 稀缺性指数 | - | 0-1 |
| `space_group` | 分类 | 晶体空间群 | - | 1-230 |
| `chemical_system` | 分类 | 化学体系 | - | "Li-O", "Na-Cl" 等 |

### 属性类型说明

#### 浮点型属性
- **嵌入方式**: NoiseLevelEncoding (噪声级别编码)
- **预处理**: StandardScalerTorch (标准化)
- **适用范围**: 连续数值型材料属性

#### 分类型属性
- **嵌入方式**: 自定义嵌入类 (如 ChemicalSystemMultiHotEmbedding)
- **预处理**: Identity (无需预处理)
- **适用范围**: 离散类别型材料属性

## 快速开始

### 环境准备

确保已完成基础安装和数据预处理：

```bash
# 激活环境
source .venv/bin/activate

# 验证数据集
ls datasets/cache/alex_mp_20/  # 应显示 train/ 和 val/

# 检查预训练模型
python -c "
from mattergen.adapter import GemNetTAdapter
print('适配器模块可用')
"
```

### 基础微调示例

以磁密度属性为例：

```bash
# 设置属性变量
export PROPERTY=dft_mag_density

# 执行微调
mattergen-finetune \
  adapter.pretrained_name=mattergen_base \
  data_module=alex_mp_20 \
  +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.$PROPERTY=$PROPERTY \
  ~trainer.logger \
  data_module.properties=["$PROPERTY"]
```

### 验证微调结果

```bash
# 检查微调输出
ls outputs/singlerun/$(date +%Y-%m-%d)/  # 查看今日训练输出

# 测试生成
export MODEL_PATH=outputs/singlerun/2025-XX-XX/XX-XX-XX/checkpoints/last.ckpt
export RESULTS_PATH=results/finetune_test/

mattergen-generate $RESULTS_PATH \
  --model_path=$MODEL_PATH \
  --batch_size=4 \
  --num_batches=1 \
  --properties_to_condition_on="{'dft_mag_density': 0.15}" \
  --diffusion_guidance_factor=2.0
```

## 单属性微调

### 浮点型属性微调

#### 磁密度微调
```bash
export PROPERTY=dft_mag_density

mattergen-finetune \
  adapter.pretrained_name=mattergen_base \
  data_module=alex_mp_20 \
  +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.$PROPERTY=$PROPERTY \
  ~trainer.logger \
  data_module.properties=["$PROPERTY"] \
  trainer.max_epochs=100  # 可调整训练轮数
```

#### 带隙微调
```bash
export PROPERTY=dft_band_gap

mattergen-finetune \
  adapter.pretrained_name=mattergen_base \
  data_module=alex_mp_20 \
  +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.$PROPERTY=$PROPERTY \
  ~trainer.logger \
  data_module.properties=["$PROPERTY"]
```

#### 体积模量微调
```bash
export PROPERTY=ml_bulk_modulus

mattergen-finetune \
  adapter.pretrained_name=mattergen_base \
  data_module=alex_mp_20 \
  +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.$PROPERTY=$PROPERTY \
  ~trainer.logger \
  data_module.properties=["$PROPERTY"]
```

### 分类型属性微调

#### 空间群微调
```bash
export PROPERTY=space_group

mattergen-finetune \
  adapter.pretrained_name=mattergen_base \
  data_module=alex_mp_20 \
  +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.$PROPERTY=$PROPERTY \
  ~trainer.logger \
  data_module.properties=["$PROPERTY"]
```

#### 化学体系微调
```bash
export PROPERTY=chemical_system

mattergen-finetune \
  adapter.pretrained_name=mattergen_base \
  data_module=alex_mp_20 \
  +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.$PROPERTY=$PROPERTY \
  ~trainer.logger \
  data_module.properties=["$PROPERTY"]
```

### 微调参数优化

#### 学习率调整
```bash
# 降低学习率 (更稳定收敛)
mattergen-finetune \
  adapter.pretrained_name=mattergen_base \
  data_module=alex_mp_20 \
  +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.dft_mag_density=dft_mag_density \
  data_module.properties=["dft_mag_density"] \
  lightning_module.optimizer_partial.lr=1e-6

# 提高学习率 (更快收敛)
mattergen-finetune \
  adapter.pretrained_name=mattergen_base \
  data_module=alex_mp_20 \
  +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.dft_mag_density=dft_mag_density \
  data_module.properties=["dft_mag_density"] \
  lightning_module.optimizer_partial.lr=1e-5
```

#### 训练轮数调整
```bash
# 短期微调 (快速原型)
mattergen-finetune \
  adapter.pretrained_name=mattergen_base \
  data_module=alex_mp_20 \
  +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.dft_mag_density=dft_mag_density \
  data_module.properties=["dft_mag_density"] \
  trainer.max_epochs=50

# 长期微调 (精细优化)
mattergen-finetune \
  adapter.pretrained_name=mattergen_base \
  data_module=alex_mp_20 \
  +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.dft_mag_density=dft_mag_density \
  data_module.properties=["dft_mag_density"] \
  trainer.max_epochs=500
```

## 多属性微调

### 双属性微调

#### 磁密度 + 带隙
```bash
export PROPERTY1=dft_mag_density
export PROPERTY2=dft_band_gap

mattergen-finetune \
  adapter.pretrained_name=mattergen_base \
  data_module=alex_mp_20 \
  +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.$PROPERTY1=$PROPERTY1 \
  +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.$PROPERTY2=$PROPERTY2 \
  ~trainer.logger \
  data_module.properties=["$PROPERTY1","$PROPERTY2"]
```

#### 化学体系 + 凸包能量
```bash
export PROPERTY1=chemical_system
export PROPERTY2=energy_above_hull

mattergen-finetune \
  adapter.pretrained_name=mattergen_base \
  data_module=alex_mp_20 \
  +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.$PROPERTY1=$PROPERTY1 \
  +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.$PROPERTY2=$PROPERTY2 \
  ~trainer.logger \
  data_module.properties=["$PROPERTY1","$PROPERTY2"]
```

### 三属性微调

```bash
export PROPERTY1=dft_mag_density
export PROPERTY2=dft_band_gap
export PROPERTY3=ml_bulk_modulus

mattergen-finetune \
  adapter.pretrained_name=mattergen_base \
  data_module=alex_mp_20 \
  +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.$PROPERTY1=$PROPERTY1 \
  +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.$PROPERTY2=$PROPERTY2 \
  +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.$PROPERTY3=$PROPERTY3 \
  ~trainer.logger \
  data_module.properties=["$PROPERTY1","$PROPERTY2","$PROPERTY3"]
```

### 多属性生成示例

微调完成后，可以进行多属性条件生成：

```bash
# 多属性条件生成
export MODEL_PATH=path/to/multi_property_model.ckpt
export RESULTS_PATH=results/multi_property/

mattergen-generate $RESULTS_PATH \
  --model_path=$MODEL_PATH \
  --batch_size=16 \
  --properties_to_condition_on="{'dft_mag_density': 0.15, 'dft_band_gap': 2.0, 'ml_bulk_modulus': 100}" \
  --diffusion_guidance_factor=2.0
```

## 自定义属性微调

### 添加新属性的步骤

#### 1. 注册属性名称

编辑 `mattergen/common/utils/globals.py`:

```python
PROPERTY_SOURCE_IDS = [
    "dft_mag_density",
    "dft_bulk_modulus",
    "dft_band_gap",
    "ml_bulk_modulus",
    "energy_above_hull",
    "hhi_score",
    "space_group",
    "chemical_system",
    # 添加新属性
    "my_custom_property",
    "thermal_conductivity",
    "hardness",
]
```

#### 2. 准备训练数据

在数据集CSV文件中添加新属性列：

```csv
structure_id,dft_mag_density,dft_band_gap,my_custom_property
mp-1,0.15,2.3,45.6
mp-2,0.0,0.0,12.3
...
```

#### 3. 重新预处理数据

```bash
# 重新处理数据集以包含新属性
csv-to-dataset \
  --csv-folder datasets/alex_mp_20/ \
  --dataset-name alex_mp_20 \
  --cache-folder datasets/cache \
  --force-reprocess
```

#### 4. 创建属性配置文件

##### 浮点型属性配置

创建 `mattergen/conf/lightning_module/diffusion_module/model/property_embeddings/my_custom_property.yaml`:

```yaml
_target_: mattergen.property_embeddings.PropertyEmbedding
name: my_custom_property

unconditional_embedding_module:
  _target_: mattergen.property_embeddings.EmbeddingVector
  hidden_dim: ${lightning_module.diffusion_module.model.hidden_dim}

conditional_embedding_module:
  _target_: mattergen.diffusion.model_utils.NoiseLevelEncoding
  d_model: ${lightning_module.diffusion_module.model.hidden_dim}

scaler:
  _target_: mattergen.common.utils.data_utils.StandardScalerTorch
```

##### 分类型属性配置

对于分类型属性，需要自定义嵌入类：

```python
# mattergen/property_embeddings.py 中添加
class MyCustomCategoricalEmbedding(nn.Module):
    def __init__(self, hidden_dim: int, num_categories: int = 100):
        super().__init__()
        self.embedding = nn.Embedding(num_categories, hidden_dim)
        
    def forward(self, x):
        # 实现自定义嵌入逻辑
        return self.embedding(x.long())
```

配置文件：
```yaml
_target_: mattergen.property_embeddings.PropertyEmbedding
name: my_custom_categorical_property

unconditional_embedding_module:
  _target_: mattergen.property_embeddings.EmbeddingVector
  hidden_dim: ${lightning_module.diffusion_module.model.hidden_dim}

conditional_embedding_module:
  _target_: mattergen.property_embeddings.MyCustomCategoricalEmbedding
  hidden_dim: ${lightning_module.diffusion_module.model.hidden_dim}
  num_categories: 50

scaler:
  _target_: torch.nn.Identity
```

#### 5. 执行微调

```bash
export PROPERTY=my_custom_property

mattergen-finetune \
  adapter.pretrained_name=mattergen_base \
  data_module=alex_mp_20 \
  +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.$PROPERTY=$PROPERTY \
  ~trainer.logger \
  data_module.properties=["$PROPERTY"]
```

### 自定义属性示例

#### 热导率属性
```yaml
# thermal_conductivity.yaml
_target_: mattergen.property_embeddings.PropertyEmbedding
name: thermal_conductivity

unconditional_embedding_module:
  _target_: mattergen.property_embeddings.EmbeddingVector
  hidden_dim: ${lightning_module.diffusion_module.model.hidden_dim}

conditional_embedding_module:
  _target_: mattergen.diffusion.model_utils.NoiseLevelEncoding
  d_model: ${lightning_module.diffusion_module.model.hidden_dim}

scaler:
  _target_: mattergen.common.utils.data_utils.StandardScalerTorch
```

#### 硬度属性
```yaml
# hardness.yaml
_target_: mattergen.property_embeddings.PropertyEmbedding
name: hardness

unconditional_embedding_module:
  _target_: mattergen.property_embeddings.EmbeddingVector
  hidden_dim: ${lightning_module.diffusion_module.model.hidden_dim}

conditional_embedding_module:
  _target_: mattergen.diffusion.model_utils.NoiseLevelEncoding
  d_model: ${lightning_module.diffusion_module.model.hidden_dim}

scaler:
  _target_: mattergen.common.utils.data_utils.LogScalerTorch  # 对数缩放
```

## 微调配置详解

### 适配器配置

#### 核心配置文件: `mattergen/conf/adapter/default.yaml`

```yaml
# 预训练模型设置
pretrained_name: mattergen_base    # 预训练模型名称
model_path: null                   # 或使用本地路径
load_epoch: last                   # 加载的epoch: last/best/数字

# 微调策略
full_finetuning: true              # true: 全参数微调, false: 仅微调新参数

# 适配器实例
adapter:
  _target_: mattergen.adapter.GemNetTAdapter
  property_embeddings_adapt: {}    # 运行时动态填充
```

#### 微调专用设置: `mattergen/conf/finetune.yaml`

```yaml
defaults:
  - data_module: mp_20
  - trainer: default
  - lightning_module: default
  - adapter: default              # 包含适配器配置

# 训练器配置
trainer:
  max_epochs: 200                 # 微调轮数 (远少于基础训练的200,000)
  logger:
    job_type: train_finetune     # 标记为微调任务

# Lightning模块配置
lightning_module:
  optimizer_partial:
    lr: 5e-6                     # 微调学习率 (远低于基础训练的1e-4)
```

### 属性嵌入配置

#### 配置组件说明

1. **PropertyEmbedding**: 主嵌入类
   - 管理条件和无条件嵌入
   - 协调缩放器和嵌入模块

2. **EmbeddingVector**: 无条件嵌入
   - 用于无条件生成
   - 简单的可学习向量

3. **NoiseLevelEncoding**: 浮点属性嵌入
   - 基于位置编码的连续值嵌入
   - 适用于所有浮点型属性

4. **自定义嵌入**: 分类属性嵌入
   - ChemicalSystemMultiHotEmbedding
   - SpaceGroupEmbedding
   - 可扩展自定义类

5. **缩放器**: 数据预处理
   - StandardScalerTorch: 标准化
   - LogScalerTorch: 对数缩放
   - Identity: 无缩放

### Hydra 配置语法解析

微调命令中的复杂语法解析：

```bash
+lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.$PROPERTY=$PROPERTY
```

#### 语法分解:
1. `+`: 添加新配置项
2. `lightning_module/diffusion_module/model/property_embeddings/`: 配置文件路径
3. `@adapter.adapter.property_embeddings_adapt.$PROPERTY`: 目标配置位置
4. `=$PROPERTY`: 使用的配置文件名

#### 等价的配置结构:
```yaml
adapter:
  adapter:
    property_embeddings_adapt:
      dft_mag_density:  # $PROPERTY 的值
        _target_: mattergen.property_embeddings.PropertyEmbedding
        name: dft_mag_density
        # ... 其余配置
```

## 最佳实践

### 数据质量保证

#### 数据完整性检查
```python
# 检查属性数据覆盖率
import pandas as pd
import numpy as np

# 加载训练数据
train_df = pd.read_csv('datasets/alex_mp_20/train.csv')

# 检查属性覆盖率
property_name = 'dft_mag_density'
coverage = (~train_df[property_name].isna()).mean()
print(f'{property_name} 数据覆盖率: {coverage:.2%}')

# 检查数值分布
valid_values = train_df[property_name].dropna()
print(f'数值范围: {valid_values.min():.3f} - {valid_values.max():.3f}')
print(f'平均值: {valid_values.mean():.3f}')
print(f'标准差: {valid_values.std():.3f}')
```

#### 数据质量过滤
```python
# 过滤异常值
def filter_outliers(df, column, n_std=3):
    mean_val = df[column].mean()
    std_val = df[column].std()
    lower_bound = mean_val - n_std * std_val
    upper_bound = mean_val + n_std * std_val
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# 应用过滤
filtered_df = filter_outliers(train_df, 'dft_mag_density')
print(f'过滤前: {len(train_df)} 样本')
print(f'过滤后: {len(filtered_df)} 样本')
```

### 训练策略优化

#### 学习率调度
```bash
# 使用余弦退火学习率
mattergen-finetune \
  adapter.pretrained_name=mattergen_base \
  data_module=alex_mp_20 \
  +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.dft_mag_density=dft_mag_density \
  data_module.properties=["dft_mag_density"] \
  lightning_module.scheduler_partial._target_=torch.optim.lr_scheduler.CosineAnnealingLR \
  lightning_module.scheduler_partial.T_max=200
```

#### 早停策略
```bash
# 添加早停回调
mattergen-finetune \
  adapter.pretrained_name=mattergen_base \
  data_module=alex_mp_20 \
  +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.dft_mag_density=dft_mag_density \
  data_module.properties=["dft_mag_density"] \
  +trainer.callbacks.early_stopping._target_=pytorch_lightning.callbacks.EarlyStopping \
  +trainer.callbacks.early_stopping.monitor=loss_val \
  +trainer.callbacks.early_stopping.patience=20
```

#### 梯度裁剪
```bash
# 防止梯度爆炸
mattergen-finetune \
  adapter.pretrained_name=mattergen_base \
  data_module=alex_mp_20 \
  +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.dft_mag_density=dft_mag_density \
  data_module.properties=["dft_mag_density"] \
  trainer.gradient_clip_val=1.0
```

### 模型验证

#### 收敛性检查
```python
# 监控训练损失
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def plot_training_curve(log_dir):
    ea = EventAccumulator(log_dir)
    ea.Reload()
    
    # 提取损失数据
    train_loss = ea.Scalars('train_loss')
    val_loss = ea.Scalars('loss_val')
    
    steps = [x.step for x in train_loss]
    train_values = [x.value for x in train_loss]
    val_values = [x.value for x in val_loss]
    
    plt.figure(figsize=(10, 6))
    plt.plot(steps, train_values, label='Training Loss')
    plt.plot(steps, val_values, label='Validation Loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Finetuning Loss Curves')
    plt.show()

# 使用示例
plot_training_curve('outputs/singlerun/2025-XX-XX/XX-XX-XX/lightning_logs/version_0')
```

#### 生成质量验证
```bash
# 生成测试样本
export MODEL_PATH=path/to/finetuned/model.ckpt
export RESULTS_PATH=results/validation/

# 不同引导强度测试
for guidance in 0.0 1.0 2.0 5.0; do
  mattergen-generate $RESULTS_PATH/guidance_$guidance \
    --model_path=$MODEL_PATH \
    --batch_size=16 \
    --num_batches=1 \
    --properties_to_condition_on="{'dft_mag_density': 0.15}" \
    --diffusion_guidance_factor=$guidance
done

# 评估生成质量
for guidance in 0.0 1.0 2.0 5.0; do
  mattergen-evaluate \
    --structures_path=$RESULTS_PATH/guidance_$guidance \
    --relax=False \
    --save_as="$RESULTS_PATH/guidance_$guidance/metrics.json"
done
```

### 性能优化

#### 内存优化
```bash
# 减少批次大小
mattergen-finetune \
  adapter.pretrained_name=mattergen_base \
  data_module=alex_mp_20 \
  +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.dft_mag_density=dft_mag_density \
  data_module.properties=["dft_mag_density"] \
  trainer.accumulate_grad_batches=8

# 启用梯度检查点
mattergen-finetune \
  adapter.pretrained_name=mattergen_base \
  data_module=alex_mp_20 \
  +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.dft_mag_density=dft_mag_density \
  data_module.properties=["dft_mag_density"] \
  lightning_module.diffusion_module.model.gradient_checkpointing=True
```

#### 加速训练
```bash
# 混合精度训练
mattergen-finetune \
  adapter.pretrained_name=mattergen_base \
  data_module=alex_mp_20 \
  +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.dft_mag_density=dft_mag_density \
  data_module.properties=["dft_mag_density"] \
  trainer.precision=16

# 编译模式 (PyTorch 2.0+)
mattergen-finetune \
  adapter.pretrained_name=mattergen_base \
  data_module=alex_mp_20 \
  +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.dft_mag_density=dft_mag_density \
  data_module.properties=["dft_mag_density"] \
  lightning_module.diffusion_module.model.compile=True
```

## 故障排除

### 常见错误及解决方案

#### 1. 属性配置错误
**错误**: `ConfigAttributeError: Key 'property_embeddings_adapt' not found`

**原因**: 属性名称未在 `PROPERTY_SOURCE_IDS` 中注册

**解决方案**:
```python
# 检查 mattergen/common/utils/globals.py
PROPERTY_SOURCE_IDS = [
    # ... 确保包含您的属性名称
    "your_property_name",
]
```

#### 2. 数据集属性缺失
**错误**: `KeyError: 'dft_mag_density' not found in dataset`

**原因**: 数据集中不包含指定属性

**解决方案**:
```bash
# 检查数据集属性
python -c "
import json
with open('datasets/cache/alex_mp_20/train/dft_mag_density.json') as f:
    data = json.load(f)
    print(f'属性数据长度: {len(data)}')
    print(f'前5个值: {data[:5]}')
"

# 如果文件不存在，重新预处理数据
csv-to-dataset --csv-folder datasets/alex_mp_20/ --dataset-name alex_mp_20 --cache-folder datasets/cache --force-reprocess
```

#### 3. 模型加载失败
**错误**: `RuntimeError: Missing key(s) in state_dict`

**原因**: 预训练模型与当前配置不匹配

**解决方案**:
```bash
# 使用非严格加载
mattergen-finetune \
  adapter.pretrained_name=mattergen_base \
  data_module=alex_mp_20 \
  +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.dft_mag_density=dft_mag_density \
  data_module.properties=["dft_mag_density"] \
  adapter.strict_loading=False
```

#### 4. 内存不足
**错误**: `CUDA out of memory during fine-tuning`

**解决方案**:
```bash
# 方案1: 减少批次大小
mattergen-finetune \
  adapter.pretrained_name=mattergen_base \
  data_module=alex_mp_20 \
  +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.dft_mag_density=dft_mag_density \
  data_module.properties=["dft_mag_density"] \
  trainer.accumulate_grad_batches=16

# 方案2: 启用梯度检查点
mattergen-finetune \
  adapter.pretrained_name=mattergen_base \
  data_module=alex_mp_20 \
  +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.dft_mag_density=dft_mag_density \
  data_module.properties=["dft_mag_density"] \
  lightning_module.diffusion_module.model.gradient_checkpointing=True
```

#### 5. 收敛问题
**问题**: 微调损失不收敛或震荡

**解决方案**:
```bash
# 降低学习率
mattergen-finetune \
  adapter.pretrained_name=mattergen_base \
  data_module=alex_mp_20 \
  +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.dft_mag_density=dft_mag_density \
  data_module.properties=["dft_mag_density"] \
  lightning_module.optimizer_partial.lr=1e-6

# 增加预热步数
mattergen-finetune \
  adapter.pretrained_name=mattergen_base \
  data_module=alex_mp_20 \
  +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.dft_mag_density=dft_mag_density \
  data_module.properties=["dft_mag_density"] \
  lightning_module.scheduler_partial.warmup_steps=100
```

### 调试技巧

#### 1. 配置验证
```bash
# 检查解析后的配置
mattergen-finetune \
  adapter.pretrained_name=mattergen_base \
  data_module=alex_mp_20 \
  +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.dft_mag_density=dft_mag_density \
  data_module.properties=["dft_mag_density"] \
  --cfg job
```

#### 2. 小规模测试
```bash
# 快速验证配置
mattergen-finetune \
  adapter.pretrained_name=mattergen_base \
  data_module=alex_mp_20 \
  +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.dft_mag_density=dft_mag_density \
  data_module.properties=["dft_mag_density"] \
  trainer.max_epochs=1 \
  trainer.limit_train_batches=2 \
  trainer.limit_val_batches=1
```

#### 3. 详细日志
```bash
# 启用详细日志
export MATTERGEN_LOG_LEVEL=DEBUG

mattergen-finetune \
  adapter.pretrained_name=mattergen_base \
  data_module=alex_mp_20 \
  +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.dft_mag_density=dft_mag_density \
  data_module.properties=["dft_mag_density"] \
  trainer.log_every_n_steps=1
```

---

## 总结

MatterGen 的微调功能为材料设计提供了强大的属性条件生成能力。通过遵循本指南的最佳实践，您可以：

1. **快速上手**: 使用预定义属性进行微调
2. **扩展功能**: 添加自定义材料属性
3. **优化性能**: 调整训练参数获得最佳效果
4. **确保质量**: 验证微调模型的生成能力

微调是 MatterGen 的核心优势之一，它将预训练的强大基础能力与特定应用需求相结合，为材料发现开辟了新的可能性。