# MatterGen 中文使用手册

<p align="center">
    <img src="assets/MatterGenlogo_.png" alt="MatterGen logo" width="600"/>
</p>

[![DOI](https://img.shields.io/badge/DOI-10.1038%2Fs41586--025--08628--5-blue)](https://www.nature.com/articles/s41586-025-08628-5)
[![arXiv](https://img.shields.io/badge/arXiv-2312.03687-blue.svg?logo=arxiv&logoColor=white.svg)](https://arxiv.org/abs/2312.03687)
[![Requires Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg?logo=python&logoColor=white)](https://python.org/downloads)

MatterGen 是一个用于无机材料设计的生成模型，能够在元素周期表范围内生成晶体结构。该模型可以进行无条件生成、属性条件生成，支持从头训练和针对特定属性的微调。

## 目录
- [安装指南](#安装指南)
- [核心架构](#核心架构)
- [配置系统详解](#配置系统详解)
- [数据预处理](#数据预处理)
- [模型训练](#模型训练)
- [微调系统](#微调系统)
- [结构生成](#结构生成)
- [模型评估](#模型评估)
- [工具脚本](#工具脚本)
- [代码质量控制](#代码质量控制)
- [故障排除](#故障排除)

## 安装指南

### 环境要求
- Python 3.10+
- CUDA GPU (推荐)
- Git LFS

### 快速安装
推荐使用 [uv](https://docs.astral.sh/uv/) 进行环境管理：

```bash
# 安装 uv
pip install uv

# 创建虚拟环境
uv venv .venv --python 3.10 
source .venv/bin/activate

# 安装 MatterGen
uv pip install -e .
```

### Git LFS 设置
检查是否已安装 Git LFS：
```bash
git lfs --version
```

如果未安装，请执行：
```bash
sudo apt install git-lfs
git lfs install
```

### Apple Silicon 支持 (实验性)
> ⚠️ **警告**: Apple Silicon 支持处于实验阶段，使用需谨慎。

在训练或生成前需要设置环境变量：
```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

训练时添加参数：
```bash
~trainer.strategy trainer.accelerator=mps
```

## 核心架构

### 主要组件

#### 1. 扩散模块 (`mattergen/diffusion/`)
核心扩散模型实现，包含：

- **`diffusion_module.py`**: 主要的扩散 Lightning 模块
- **`corruption/`**: 扩散过程的噪声腐蚀策略
  - `corruption.py`: 基础腐蚀类
  - `d3pm_corruption.py`: D3PM 离散扩散腐蚀
  - `multi_corruption.py`: 多变量腐蚀处理
  - `sde_lib.py`: 随机微分方程库
- **`sampling/`**: 采样算法和预测器/校正器
  - `pc_sampler.py`: 预测器-校正器采样器
  - `predictors.py`: 预测器实现
  - `classifier_free_guidance.py`: 无分类器引导
- **`d3pm/`**: 离散去噪扩散处理分类变量
- **`training/`**: 训练相关工具
  - `field_loss.py`: 场损失函数
  - `metrics.py`: 训练指标
- **`wrapped/`**: 包装的扩散组件
  - `wrapped_normal_loss.py`: 包装的正态损失
  - `wrapped_sde.py`: 包装的 SDE

#### 2. GemNet 骨干网络 (`mattergen/common/gemnet/`)
用于晶体表示的图神经网络：

- **`gemnet.py`**: 主要的 GemNet 模型
- **`gemnet_ctrl.py`**: 带控制的 GemNet 模型 (用于微调)
- **`layers/`**: 原子交互的神经网络层
  - `atom_update_block.py`: 原子更新块
  - `interaction_block.py`: 交互块
  - `embedding_block.py`: 嵌入块
  - `radial_basis.py`: 径向基函数
  - `spherical_basis.py`: 球面基函数

#### 3. 数据流水线 (`mattergen/common/data/`)
数据加载和处理组件：

- **`datamodule.py`**: PyTorch Lightning 数据模块
- **`dataset.py`**: 晶体数据集实现
- **`transform.py`**: 数据变换
- **`condition_factory.py`**: 条件数据工厂
- **`collate.py`**: 数据整理函数
- **`chemgraph.py`**: 化学图表示

#### 4. 属性嵌入 (`mattergen/property_embeddings.py`)
处理材料属性条件生成：

- **PropertyEmbedding**: 基础属性嵌入类
- **EmbeddingVector**: 向量嵌入
- **ChemicalSystemMultiHotEmbedding**: 化学系统多热编码
- **NoiseLevelEncoding**: 噪声级别编码

#### 5. 适配器系统 (`mattergen/adapter.py`)
微调时的模型适配：

- **GemNetTAdapter**: 在预训练模型基础上添加新的属性条件

## 配置系统详解

MatterGen 使用 Hydra 进行分层配置管理。配置系统包含以下几个层次：

### 主配置文件

#### `mattergen/conf/default.yaml` - 基础训练配置
```yaml
hydra:
  run:
    dir: ${oc.env:OUTPUT_DIR,outputs/singlerun/${now:%Y-%m-%d}/${now:%H-%M-%S}}

auto_resume: True

defaults:
  - data_module: mp_20           # 数据模块配置
  - trainer: default             # 训练器配置  
  - lightning_module: default    # Lightning 模块配置
  - lightning_module/diffusion_module: default
  - lightning_module/diffusion_module/model: mattergen
  - lightning_module/diffusion_module/corruption: default
```

#### `mattergen/conf/finetune.yaml` - 微调配置
```yaml
defaults:
  - data_module: mp_20
  - trainer: default
  - lightning_module: default
  - adapter: default            # 适配器配置

trainer:
  max_epochs: 200              # 微调轮数较少
  logger:
    job_type: train_finetune

lightning_module:
  optimizer_partial:
    lr: 5e-6                   # 微调学习率较低
```

### 数据模块配置

#### `mattergen/conf/data_module/alex_mp_20.yaml`
```yaml
_target_: mattergen.common.data.datamodule.CrystDataModule
_recursive_: true

# 支持的属性列表
properties: [
  "dft_bulk_modulus",      # DFT 体积模量
  "dft_band_gap",          # DFT 带隙
  "dft_mag_density",       # DFT 磁密度
  "ml_bulk_modulus",       # ML 预测体积模量
  "hhi_score",             # HHI 评分
  "space_group",           # 空间群
  "energy_above_hull"      # 凸包上方能量
]

# 数据变换
dataset_transforms: 
  - _target_: mattergen.common.data.dataset_transform.filter_sparse_properties
    _partial_: true

transforms:
- _target_: mattergen.common.data.transform.symmetrize_lattice
  _partial_: true
- _target_: mattergen.common.data.transform.set_chemical_system_string
  _partial_: true

# 平均密度 (原子/埃³)
average_density: 0.05771451654022283
root_dir: ${oc.env:PROJECT_ROOT}/../datasets/cache/alex_mp_20

# 批次大小计算 (考虑设备数量和梯度累积)
batch_size:
  train: ${eval:'(4096 // ${trainer.accumulate_grad_batches}) // (${trainer.devices} * ${trainer.num_nodes})'}
  val: ${eval:'(512 // ${trainer.accumulate_grad_batches}) // (${trainer.devices} * ${trainer.num_nodes})'}

max_epochs: 200000
```

### 适配器配置

#### `mattergen/conf/adapter/default.yaml`
```yaml
pretrained_name: mattergen_base    # 预训练模型名称
model_path: null                   # 或者使用本地路径
load_epoch: last                   # 加载的 epoch
full_finetuning: true              # 是否全量微调

adapter:
  _target_: mattergen.adapter.GemNetTAdapter
  property_embeddings_adapt: {}    # 新增的属性嵌入
```

### 属性嵌入配置

#### 浮点型属性配置 (`dft_mag_density.yaml`)
```yaml
_target_: mattergen.property_embeddings.PropertyEmbedding
name: dft_mag_density

unconditional_embedding_module:
  _target_: mattergen.property_embeddings.EmbeddingVector
  hidden_dim: ${lightning_module.diffusion_module.model.hidden_dim}

conditional_embedding_module:
  _target_: mattergen.diffusion.model_utils.NoiseLevelEncoding  # 浮点型属性使用噪声级别编码
  d_model: ${lightning_module.diffusion_module.model.hidden_dim}

scaler:
  _target_: mattergen.common.utils.data_utils.StandardScalerTorch  # 标准化缩放
```

#### 分类型属性配置 (`chemical_system.yaml`)
```yaml
_target_: mattergen.property_embeddings.PropertyEmbedding
name: chemical_system

unconditional_embedding_module:
  _target_: mattergen.property_embeddings.EmbeddingVector
  hidden_dim: ${lightning_module.diffusion_module.model.hidden_dim}

conditional_embedding_module:
  _target_: mattergen.property_embeddings.ChemicalSystemMultiHotEmbedding  # 分类型属性使用自定义嵌入
  hidden_dim: ${lightning_module.diffusion_module.model.hidden_dim}

scaler:
  _target_: torch.nn.Identity  # 分类型属性不需要缩放
```

### 采样配置

#### `sampling_conf/default.yaml` - 默认采样配置
```yaml
sampler_partial:
  _target_: mattergen.diffusion.sampling.classifier_free_guidance.GuidedPredictorCorrector.from_pl_module
  N: 1000                      # 扩散步数
  eps_t: ${eval:'1/${.N}'}     # 时间步长
  guidance_scale: 0.0          # 引导强度
  
  predictor_partials:
    pos:                       # 位置预测器
      _target_: mattergen.diffusion.wrapped.wrapped_predictors_correctors.WrappedAncestralSamplingPredictor
    cell:                      # 晶胞预测器
      _target_: mattergen.common.diffusion.predictors_correctors.LatticeAncestralSamplingPredictor
    atomic_numbers:            # 原子类型预测器
      _target_: mattergen.diffusion.d3pm.d3pm_predictors_correctors.D3PMAncestralSamplingPredictor
      predict_x0: True

  corrector_partials:
    pos:                       # 位置校正器
      _target_: mattergen.diffusion.wrapped.wrapped_predictors_correctors.WrappedLangevinCorrector
      max_step_size: 1e6
      snr: 0.4
    cell:                      # 晶胞校正器
      _target_: mattergen.common.diffusion.predictors_correctors.LatticeLangevinDiffCorrector
      max_step_size: 1e6
      snr: 0.2

  n_steps_corrector: 1         # 校正器步数

condition_loader_partial:
  _target_: mattergen.common.data.condition_factory.get_number_of_atoms_condition_loader
```

#### `sampling_conf/csp.yaml` - 晶体结构预测配置
与默认配置的主要区别是：
- 移除了 `atomic_numbers` 预测器 (原子类型固定)
- 使用组分数据加载器而非原子数量加载器

## 数据预处理

### 支持的数据集

1. **MP-20 数据集**
   - 包含 45,000 个通用无机材料
   - 单胞原子数不超过 20
   - 包含大多数实验已知材料

2. **Alex-MP-20 数据集**
   - 包含约 600,000 个结构
   - 来自 MP-20 和 Alexandria 数据库
   - 处理时间约 1 小时

### 数据预处理命令

#### MP-20 数据集
```bash
# 下载数据
git lfs pull -I data-release/mp-20/ --exclude=""

# 解压数据
unzip data-release/mp-20/mp_20.zip -d datasets

# 预处理数据
csv-to-dataset --csv-folder datasets/mp_20/ --dataset-name mp_20 --cache-folder datasets/cache
```

#### Alex-MP-20 数据集
```bash
# 下载数据 (较大文件)
git lfs pull -I data-release/alex-mp/alex_mp_20.zip --exclude=""

# 解压数据
unzip data-release/alex-mp/alex_mp_20.zip -d datasets

# 预处理数据 (约需 1 小时)
csv-to-dataset --csv-folder datasets/alex_mp_20/ --dataset-name alex_mp_20 --cache-folder datasets/cache
```

### 数据格式

预处理后的数据存储在 `datasets/cache/<dataset_name>/` 目录下，包含：
- `train/` 和 `val/` 子目录
- `.npy` 格式的数值数据 (原子坐标、晶胞参数等)
- `.json` 格式的属性数据 (带隙、磁密度等)

## 模型训练

### 基础模型训练

#### MP-20 数据集训练
```bash
mattergen-train data_module=mp_20 ~trainer.logger
```

#### Alex-MP-20 数据集训练
```bash
mattergen-train data_module=alex_mp_20 ~trainer.logger trainer.accumulate_grad_batches=4
```

> **说明**: 
> - `~trainer.logger` 禁用 W&B 日志记录
> - `trainer.accumulate_grad_batches=4` 用于大数据集的内存管理

### 晶体结构预测 (CSP) 模式训练
CSP 模式不对原子类型进行去噪，适用于已知化学式的结构预测：

```bash
mattergen-train --config-name=csp data_module=mp_20 ~trainer.logger
```

### 训练输出
训练输出保存在 `outputs/singlerun/${日期}/${时间}/` 目录下，包含：
- `checkpoints/`: 模型检查点
- `lightning_logs/`: 训练日志
- `config.yaml`: 训练配置

### 训练监控
- 验证损失 (`loss_val`) 应在约 80,000 步后达到 0.4
- 使用 TensorBoard 查看训练曲线：
```bash
tensorboard --logdir outputs/singlerun/
```

## 微调系统

微调是 MatterGen 的核心功能之一，允许在预训练模型基础上添加特定属性的条件生成能力。

### 支持的属性

在 `mattergen/common/utils/globals.py` 中定义的 `PROPERTY_SOURCE_IDS`:

| 属性名称 | 类型 | 描述 |
|---------|------|------|
| `dft_mag_density` | 浮点 | DFT 计算的磁密度 |
| `dft_bulk_modulus` | 浮点 | DFT 计算的体积模量 |
| `dft_shear_modulus` | 浮点 | DFT 计算的剪切模量 |
| `dft_band_gap` | 浮点 | DFT 计算的带隙 |
| `ml_bulk_modulus` | 浮点 | ML 预测的体积模量 |
| `energy_above_hull` | 浮点 | 凸包上方能量 |
| `formation_energy_per_atom` | 浮点 | 每原子形成能 |
| `hhi_score` | 浮点 | Herfindahl-Hirschman 指数 |
| `space_group` | 分类 | 晶体空间群 |
| `chemical_system` | 分类 | 化学体系 |

### 单属性微调

#### 基本语法
```bash
export PROPERTY=dft_mag_density
mattergen-finetune \
  adapter.pretrained_name=mattergen_base \
  data_module=mp_20 \
  +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.$PROPERTY=$PROPERTY \
  ~trainer.logger \
  data_module.properties=["$PROPERTY"]
```

#### 配置解析

1. **`adapter.pretrained_name=mattergen_base`**
   - 指定预训练模型名称
   - 也可使用 `adapter.model_path=/path/to/model` 指定本地路径

2. **`+lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.$PROPERTY=$PROPERTY`**
   - 这是 Hydra 的高级语法，解析如下：
   - `+`: 添加新的配置项
   - `lightning_module/diffusion_module/model/property_embeddings/`: 配置文件路径
   - `@adapter.adapter.property_embeddings_adapt.$PROPERTY`: 目标位置
   - `=$PROPERTY`: 配置文件名

3. **`data_module.properties=["$PROPERTY"]`**
   - 指定数据模块加载的属性
   - 确保数据集包含该属性

#### 实际示例
```bash
# 微调磁密度属性
export PROPERTY=dft_mag_density
mattergen-finetune \
  adapter.pretrained_name=mattergen_base \
  data_module=alex_mp_20 \
  +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.dft_mag_density=dft_mag_density \
  ~trainer.logger \
  data_module.properties=["dft_mag_density"]
```

### 多属性微调

#### 双属性微调示例
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

#### 多属性配置要点
- 每个属性需要一个单独的 `+lightning_module/...` 行
- `data_module.properties` 需要包含所有属性
- 确保数据集包含所有指定属性

### 微调配置详解

#### 适配器工作原理
1. **加载预训练模型**: 从检查点加载基础 GemNet 模型
2. **替换为控制模型**: 将 `GemNetT` 替换为 `GemNetTCtrl`
3. **添加属性嵌入**: 为新属性创建嵌入层
4. **冻结基础权重**: (可选) 仅训练新增的属性嵌入层

#### 关键配置参数

**`mattergen/conf/adapter/default.yaml`**:
```yaml
pretrained_name: mattergen_base    # 预训练模型
model_path: null                   # 本地模型路径 (与 pretrained_name 二选一)
load_epoch: last                   # 加载 epoch (last/best/数字)
full_finetuning: true              # 全量微调 vs 仅微调新参数

adapter:
  _target_: mattergen.adapter.GemNetTAdapter
  property_embeddings_adapt: {}    # 运行时填充
```

**微调特定设置**:
```yaml
trainer:
  max_epochs: 200                  # 微调轮数较少
  
lightning_module:
  optimizer_partial:
    lr: 5e-6                      # 学习率较低 (基础训练为 1e-4)
```

### 添加自定义属性

#### 步骤 1: 注册属性名称
在 `mattergen/common/utils/globals.py` 的 `PROPERTY_SOURCE_IDS` 列表中添加属性名称：

```python
PROPERTY_SOURCE_IDS = [
    "dft_mag_density",
    "dft_bulk_modulus",
    # ... 其他属性
    "my_custom_property",  # 添加新属性
]
```

#### 步骤 2: 准备数据
在数据集 CSV 文件中添加新列：
- `datasets/alex_mp_20/train.csv`
- `datasets/alex_mp_20/val.csv`

#### 步骤 3: 重新预处理数据
```bash
csv-to-dataset --csv-folder datasets/alex_mp_20/ --dataset-name alex_mp_20 --cache-folder datasets/cache
```

#### 步骤 4: 创建属性配置
创建 `mattergen/conf/lightning_module/diffusion_module/model/property_embeddings/my_custom_property.yaml`:

**浮点型属性**:
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

**分类型属性** (需要自定义嵌入类):
```yaml
_target_: mattergen.property_embeddings.PropertyEmbedding
name: my_custom_property
unconditional_embedding_module:
  _target_: mattergen.property_embeddings.EmbeddingVector
  hidden_dim: ${lightning_module.diffusion_module.model.hidden_dim}
conditional_embedding_module:
  _target_: mattergen.property_embeddings.MyCustomEmbedding  # 需要实现
  hidden_dim: ${lightning_module.diffusion_module.model.hidden_dim}
scaler:
  _target_: torch.nn.Identity
```

#### 步骤 5: 执行微调
```bash
export PROPERTY=my_custom_property
mattergen-finetune \
  adapter.pretrained_name=mattergen_base \
  data_module=alex_mp_20 \
  +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.$PROPERTY=$PROPERTY \
  ~trainer.logger \
  data_module.properties=["$PROPERTY"]
```

### 微调最佳实践

1. **数据质量**: 确保属性数据完整且准确
2. **学习率**: 微调使用较低学习率 (5e-6 vs 1e-4)
3. **训练轮数**: 微调轮数较少 (200 vs 200,000)
4. **属性规范化**: 浮点型属性建议使用标准化
5. **验证**: 定期检查微调收敛性和生成质量

## 结构生成

### 预训练模型列表

| 模型名称 | 描述 |
|---------|------|
| `mattergen_base` | 无条件基础模型 |
| `chemical_system` | 化学体系条件模型 |
| `space_group` | 空间群条件模型 |
| `dft_mag_density` | 磁密度条件模型 |
| `dft_band_gap` | 带隙条件模型 |
| `ml_bulk_modulus` | 体积模量条件模型 |
| `dft_mag_density_hhi_score` | 磁密度+HHI评分联合条件模型 |
| `chemical_system_energy_above_hull` | 化学体系+凸包能量联合条件模型 |

### 无条件生成

```bash
export MODEL_NAME=mattergen_base
export RESULTS_PATH=results/

# 生成结构 (batch_size * num_batches 个结构)
mattergen-generate $RESULTS_PATH \
  --pretrained-name=$MODEL_NAME \
  --batch_size=16 \
  --num_batches=1
```

#### 参数说明
- `--batch_size`: 批次大小，受 GPU 内存限制
- `--num_batches`: 批次数量
- `--record-trajectories`: 是否记录去噪轨迹 (默认 True)

#### 输出文件
生成的文件保存在 `$RESULTS_PATH` 目录下：
- `generated_crystals_cif.zip`: 每个结构的单独 CIF 文件
- `generated_crystals.extxyz`: 所有结构的单一文件
- `generated_trajectories.zip`: 完整去噪轨迹 (可选)

### 单属性条件生成

```bash
export MODEL_NAME=dft_mag_density
export RESULTS_PATH="results/$MODEL_NAME/"

# 生成磁密度为 0.15 的结构
mattergen-generate $RESULTS_PATH \
  --pretrained-name=$MODEL_NAME \
  --batch_size=16 \
  --properties_to_condition_on="{'dft_mag_density': 0.15}" \
  --diffusion_guidance_factor=2.0
```

#### 引导强度参数
`--diffusion_guidance_factor` (对应论文中的 γ 参数)：
- `0.0`: 无条件生成
- `1.0-5.0`: 典型条件生成范围
- 更高值: 更强的属性约束，但可能降低多样性

### 多属性条件生成

```bash
export MODEL_NAME=chemical_system_energy_above_hull
export RESULTS_PATH="results/$MODEL_NAME/"

# 生成 Li-O 体系且凸包能量为 0.05 的结构
mattergen-generate $RESULTS_PATH \
  --pretrained-name=$MODEL_NAME \
  --batch_size=16 \
  --properties_to_condition_on="{'energy_above_hull': 0.05, 'chemical_system': 'Li-O'}" \
  --diffusion_guidance_factor=2.0
```

### 晶体结构预测 (CSP) 模式

CSP 模式用于已知化学式的结构预测：

```bash
# 训练 CSP 模型
mattergen-train --config-name=csp data_module=mp_20 ~trainer.logger

# 使用 CSP 模型生成
mattergen-generate $RESULTS_PATH \
  --model_path=$CSP_MODEL_PATH \
  --target_compositions='[{"Na": 1, "Cl": 1}]' \
  --sampling-config-name=csp
```

### 生成配置优化

#### 采样配置覆盖
```bash
mattergen-generate $RESULTS_PATH \
  --pretrained-name=$MODEL_NAME \
  --sampling_config_overrides='sampler_partial.N=500' \  # 减少扩散步数
  --sampling_config_overrides='sampler_partial.n_steps_corrector=0'  # 禁用校正器
```

#### 批次大小优化
```bash
# 最大化 GPU 利用率
mattergen-generate $RESULTS_PATH \
  --pretrained-name=$MODEL_NAME \
  --batch_size=64 \  # 根据 GPU 内存调整
  --num_batches=10
```

### 高级生成选项

#### 原子数量分布
```bash
# 使用不同的原子数量分布
mattergen-generate $RESULTS_PATH \
  --pretrained-name=$MODEL_NAME \
  --sampling_config_overrides='condition_loader_partial.num_atoms_distribution=MP_20'
```

支持的分布：
- `ALEX_MP_20`: 默认分布
- `MP_20`: MP-20 数据集分布

#### 禁用轨迹记录
```bash
# 节省存储空间
mattergen-generate $RESULTS_PATH \
  --pretrained-name=$MODEL_NAME \
  --record-trajectories=False
```

## 模型评估

### 基础评估

#### 使用 MatterSim 弛豫和评估
```bash
# 下载参考数据集
git lfs pull -I data-release/alex-mp/reference_MP2020correction.gz --exclude=""

# 执行完整评估
mattergen-evaluate \
  --structures_path=$RESULTS_PATH \
  --relax=True \
  --structure_matcher='disordered' \
  --save_as="$RESULTS_PATH/metrics.json"
```

#### 使用预计算能量评估
```bash
# 使用外部计算的能量 (如 DFT)
mattergen-evaluate \
  --structures_path=$RESULTS_PATH \
  --energies_path='energies.npy' \
  --relax=False \
  --structure_matcher='disordered' \
  --save_as='metrics'
```

### 评估配置选项

#### 结构匹配器
- `'disordered'`: 无序结构匹配器 (默认，更宽松)
- `'ordered'`: 有序结构匹配器 (更严格)

#### MatterSim 模型选择
```bash
# 使用更大的 MatterSim 模型
mattergen-evaluate \
  --structures_path=$RESULTS_PATH \
  --potential_load_path="MatterSim-v1.0.0-5M.pth" \
  --relax=True
```

#### 保存弛豫结构
```bash
mattergen-evaluate \
  --structures_path=$RESULTS_PATH \
  --relax=True \
  --structures_output_path="relaxed_structures.extxyz"
```

### 评估指标

评估脚本计算以下指标：

1. **新颖性 (Novelty)**: 生成结构与参考数据集的不重复度
2. **唯一性 (Uniqueness)**: 生成结构内部的去重度  
3. **稳定性 (Stability)**: 基于能量的热力学稳定性
4. **有效性 (Validity)**: 结构的物理合理性
5. **RMSD**: 与最近邻参考结构的距离

### 自定义参考数据集

#### 创建自定义参考数据集
```python
from mattergen.evaluation.reference.reference_dataset import ReferenceDataset
from mattergen.evaluation.reference.reference_dataset_serializer import LMDBGZSerializer
from mattergen.evaluation.utils.vasprunlike import VasprunLike
from pymatgen.entries.compatibility import MaterialsProject2020Compatibility

# 准备结构-能量对
entries = []
for structure, energy in zip(structures, energies):
    vasprun_like = VasprunLike(structure=structure, energy=energy)
    entries.append(vasprun_like.get_computed_entry(
        inc_structure=True, 
        energy_correction_scheme=MaterialsProject2020Compatibility()
    ))

# 创建并序列化参考数据集
reference_dataset = ReferenceDataset.from_entries(
    name="my_reference_dataset", 
    entries=entries
)
LMDBGZSerializer().serialize(reference_dataset, "my_reference.gz")
```

#### 使用自定义参考数据集
```bash
mattergen-evaluate \
  --structures_path=$RESULTS_PATH \
  --reference_dataset_path="my_reference.gz" \
  --relax=True
```

### 基准测试

项目提供了基准测试工具，位于 `benchmark/` 目录：

#### Jupyter 笔记本
`benchmark/plot_benchmark_results.ipynb` 用于生成论文中的图表

#### 基准指标
`benchmark/metrics/` 目录包含各种基准模型的结果：
- `mattergen.json`: MatterGen 结果
- `cdvae.json`: CDVAE 基准
- `diffcsp_mp_20.json`: DiffCSP 基准
- 等等

#### 添加自己的结果
将评估生成的 `metrics.json` 文件复制到 `benchmark/metrics/` 目录即可参与比较。

## 工具脚本

### 主要命令行工具

#### 1. `mattergen-train` - 模型训练
```bash
# 基础用法
mattergen-train data_module=alex_mp_20

# 禁用日志
mattergen-train data_module=alex_mp_20 ~trainer.logger

# 梯度累积
mattergen-train data_module=alex_mp_20 trainer.accumulate_grad_batches=4

# CSP 模式
mattergen-train --config-name=csp data_module=mp_20
```

#### 2. `mattergen-finetune` - 模型微调
```bash
# 单属性微调
mattergen-finetune \
  adapter.pretrained_name=mattergen_base \
  data_module=alex_mp_20 \
  +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.dft_mag_density=dft_mag_density \
  data_module.properties=["dft_mag_density"]

# 多属性微调
mattergen-finetune \
  adapter.pretrained_name=mattergen_base \
  data_module=alex_mp_20 \
  +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.dft_mag_density=dft_mag_density \
  +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.dft_band_gap=dft_band_gap \
  data_module.properties=["dft_mag_density","dft_band_gap"]
```

#### 3. `mattergen-generate` - 结构生成
```bash
# 无条件生成
mattergen-generate results/ --pretrained-name=mattergen_base --batch_size=16

# 条件生成  
mattergen-generate results/ \
  --pretrained-name=dft_mag_density \
  --properties_to_condition_on="{'dft_mag_density': 0.15}" \
  --diffusion_guidance_factor=2.0

# CSP 生成
mattergen-generate results/ \
  --model_path=/path/to/csp/model \
  --target_compositions='[{"Na": 1, "Cl": 1}]' \
  --sampling-config-name=csp
```

#### 4. `mattergen-evaluate` - 模型评估
```bash
# 基础评估
mattergen-evaluate --structures_path=results/ --relax=True

# 指定参考数据集
mattergen-evaluate \
  --structures_path=results/ \
  --reference_dataset_path=my_reference.gz \
  --relax=True

# 使用预计算能量
mattergen-evaluate \
  --structures_path=results/ \
  --energies_path=energies.npy \
  --relax=False
```

#### 5. `csv-to-dataset` - 数据预处理
```bash
# 基础用法
csv-to-dataset \
  --csv-folder datasets/alex_mp_20/ \
  --dataset-name alex_mp_20 \
  --cache-folder datasets/cache

# 强制重新处理
csv-to-dataset \
  --csv-folder datasets/alex_mp_20/ \
  --dataset-name alex_mp_20 \
  --cache-folder datasets/cache \
  --force-reprocess
```

### Fire CLI 参数语法

MatterGen 使用 [Python Fire](https://github.com/google/python-fire) 进行 CLI 参数解析：

#### 字典参数
```bash
# 正确: 无空格
--properties_to_condition_on="{'dft_mag_density':0.15,'hhi_score':0.5}"

# 错误: 有空格
--properties_to_condition_on="{'dft_mag_density': 0.15, 'hhi_score': 0.5}"
```

#### 列表参数
```bash
# 字符串列表
--properties='["dft_mag_density","dft_band_gap"]'

# 字典列表
--target_compositions='[{"Na":1,"Cl":1},{"Li":2,"O":1}]'
```

#### 布尔参数
```bash
# 布尔值
--relax=True
--record_trajectories=False
```

### 脚本位置
所有脚本位于 `mattergen/scripts/` 目录：
- `generate.py`: 结构生成脚本
- `finetune.py`: 微调脚本  
- `run.py`: 训练脚本
- `evaluate.py`: 评估脚本
- `csv_to_dataset.py`: 数据预处理脚本

## 代码质量控制

### 代码格式化

#### Black 格式化
```bash
# 格式化所有 Python 文件
black mattergen/ --line-length 100

# 检查格式 (不修改)
black mattergen/ --line-length 100 --check

# 显示差异
black mattergen/ --line-length 100 --diff
```

#### isort 导入排序
```bash
# 排序导入语句
isort mattergen/ --profile black --line-length 100

# 检查排序
isort mattergen/ --profile black --line-length 100 --check-only

# 显示差异
isort mattergen/ --profile black --line-length 100 --diff
```

### 代码检查

#### pylint 静态分析
```bash
# 分析所有代码
pylint mattergen/

# 分析特定模块
pylint mattergen/generator.py

# 生成报告
pylint mattergen/ --output-format=json > pylint_report.json
```

### 测试

#### 运行测试
```bash
# 运行所有测试
pytest mattergen/tests/
pytest mattergen/common/tests/
pytest mattergen/diffusion/tests/

# 运行特定测试
pytest mattergen/tests/test_generator.py
pytest mattergen/common/tests/gemnet_test.py

# 详细输出
pytest mattergen/tests/ -v

# 覆盖率报告
pytest mattergen/tests/ --cov=mattergen --cov-report=html
```

#### 测试文件结构
```
mattergen/
├── tests/                    # 主要测试
│   ├── test_generator.py
│   ├── test_mattergen_denoiser.py
│   └── test_structure_matcher.py
├── common/tests/             # 通用组件测试
│   ├── data_utils_test.py
│   └── gemnet_test.py
└── diffusion/tests/          # 扩散模型测试
    ├── test_d3pm.py
    ├── test_losses.py
    └── test_sampling.py
```

### 预提交钩子

#### 设置 pre-commit
```bash
# 安装 pre-commit
pip install pre-commit

# 安装钩子
pre-commit install

# 手动运行所有钩子
pre-commit run --all-files
```

#### 配置文件 `.pre-commit-config.yaml`
```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 22.3.0
    hooks:
      - id: black
        args: [--line-length=100]
        
  - repo: https://github.com/pycqa/isort
    rev: 5.10.1
    hooks:
      - id: isort
        args: [--profile=black, --line-length=100]
        
  - repo: https://github.com/pycqa/pylint
    rev: v2.13.9
    hooks:
      - id: pylint
```

### 配置文件

#### pyproject.toml 配置
```toml
[tool.black]
line-length = 100
include = '\.pyi?$'

[tool.isort]
profile = "black"
line_length = 100
known_first_party = ["mattergen"]

[tool.pytest.ini_options]
testpaths = ["mattergen/tests", "mattergen/common/tests", "mattergen/diffusion/tests"]
python_files = ["test_*.py"]
addopts = "-v --tb=short"
```

## 故障排除

### 常见错误及解决方案

#### 1. CUDA 内存不足
**错误**: `RuntimeError: CUDA out of memory`

**解决方案**:
```bash
# 减少批次大小
mattergen-train data_module=alex_mp_20 trainer.accumulate_grad_batches=8

# 启用梯度检查点
mattergen-train data_module=alex_mp_20 lightning_module.diffusion_module.model.gradient_checkpointing=True

# 使用混合精度
mattergen-train data_module=alex_mp_20 trainer.precision=16
```

#### 2. 配置文件错误
**错误**: `ConfigAttributeError: Key 'property_embeddings_adapt' not found`

**解决方案**:
- 检查属性名称是否在 `PROPERTY_SOURCE_IDS` 中
- 确认配置文件路径正确
- 验证 Hydra 语法

#### 3. 数据集错误
**错误**: `FileNotFoundError: datasets/cache/alex_mp_20/train not found`

**解决方案**:
```bash
# 重新预处理数据
csv-to-dataset --csv-folder datasets/alex_mp_20/ --dataset-name alex_mp_20 --cache-folder datasets/cache

# 检查数据完整性
ls -la datasets/cache/alex_mp_20/
```

#### 4. 模型加载错误
**错误**: `RuntimeError: Error(s) in loading state_dict`

**解决方案**:
```bash
# 使用非严格加载
mattergen-generate results/ --model_path=/path/to/model --strict_checkpoint_loading=False

# 检查模型兼容性
python -c "import torch; print(torch.load('/path/to/model.ckpt').keys())"
```

#### 5. Apple Silicon 问题
**错误**: `RuntimeError: MPS backend out of memory`

**解决方案**:
```bash
# 设置回退
export PYTORCH_ENABLE_MPS_FALLBACK=1

# 使用 CPU
mattergen-train data_module=mp_20 trainer.accelerator=cpu

# 减少批次大小
mattergen-train data_module=mp_20 trainer.accelerator=mps data_module.batch_size.train=1
```

#### 6. Git LFS 问题
**错误**: `Git LFS: smudge filter lfs failed`

**解决方案**:
```bash
# 重新安装 Git LFS
git lfs install --force

# 手动拉取特定文件
git lfs pull -I "checkpoints/mattergen_base" --exclude=""

# 检查 LFS 状态
git lfs status
```

### 性能优化建议

#### 1. 训练性能优化
```bash
# 启用编译模式 (PyTorch 2.0+)
mattergen-train data_module=alex_mp_20 lightning_module.diffusion_module.model.compile=True

# 使用数据加载器多进程
mattergen-train data_module=alex_mp_20 data_module.num_workers.train=4

# 启用 Tensor Core
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

#### 2. 生成性能优化
```bash
# 减少扩散步数
mattergen-generate results/ \
  --pretrained-name=mattergen_base \
  --sampling_config_overrides='sampler_partial.N=500'

# 禁用校正器
mattergen-generate results/ \
  --pretrained-name=mattergen_base \
  --sampling_config_overrides='sampler_partial.n_steps_corrector=0'

# 最大化批次大小
mattergen-generate results/ \
  --pretrained-name=mattergen_base \
  --batch_size=128
```

#### 3. 内存使用优化
```bash
# 清理缓存
python -c "import torch; torch.cuda.empty_cache()"

# 监控内存使用
nvidia-smi -l 1

# 使用内存映射数据集
mattergen-train data_module=alex_mp_20 data_module.pin_memory=False
```

### 调试技巧

#### 1. 启用详细日志
```bash
# 设置日志级别
export PYTHONPATH=/path/to/mattergen:$PYTHONPATH
export MATTERGEN_LOG_LEVEL=DEBUG

# 训练时详细输出
mattergen-train data_module=mp_20 trainer.log_every_n_steps=1
```

#### 2. 配置检查
```bash
# 打印解析后的配置
mattergen-train data_module=mp_20 --cfg job

# 检查 Hydra 配置
python -m hydra.main config_path=mattergen/conf config_name=default --cfg job
```

#### 3. 小规模测试
```bash
# 快速训练测试
mattergen-train data_module=mp_20 trainer.max_epochs=1 trainer.limit_train_batches=2

# 小批次生成测试
mattergen-generate results/ --pretrained-name=mattergen_base --batch_size=1 --num_batches=1
```

### 获取帮助

#### 官方资源
- **GitHub Issues**: [https://github.com/microsoft/mattergen/issues](https://github.com/microsoft/mattergen/issues)
- **Discussions**: [https://github.com/microsoft/mattergen/discussions](https://github.com/microsoft/mattergen/discussions)
- **论文**: [Nature 2025](https://www.nature.com/articles/s41586-025-08628-5)

#### 社区支持
- 在 GitHub Discussions 的 Q&A 部分提问
- 查看已有的 Issues 和解决方案
- 提交 Bug 报告时请包含完整的错误信息和重现步骤

---

## 许可证和引用

### 许可证
本项目基于 MIT 许可证开源。

### 引用
如果您使用了 MatterGen 的代码、模型、数据或评估流程，请引用：

```bibtex
@article{MatterGen2025,
  author  = {Zeni, Claudio and Pinsler, Robert and Z{\"u}gner, Daniel and Fowler, Andrew and Horton, Matthew and Fu, Xiang and Wang, Zilong and Shysheya, Aliaksandra and Crabb{\'e}, Jonathan and Ueda, Shoko and Sordillo, Roberto and Sun, Lixin and Smith, Jake and Nguyen, Bichlien and Schulz, Hannes and Lewis, Sarah and Huang, Chin-Wei and Lu, Ziheng and Zhou, Yichi and Yang, Han and Hao, Hongxia and Li, Jielan and Yang, Chunlei and Li, Wenjie and Tomioka, Ryota and Xie, Tian},
  journal = {Nature},
  title   = {A generative model for inorganic materials design},
  year    = {2025},
  doi     = {10.1038/s41586-025-08628-5},
}
```

### 商标声明
本项目可能包含 Microsoft 或其他项目、产品或服务的商标或标识。Microsoft 商标或标识的授权使用须遵循 [Microsoft 商标和品牌指南](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general)。

---

*该中文手册基于 MatterGen v1.0 编写，涵盖了代码库的核心功能和使用方法。如有疑问或需要技术支持，请访问项目的 GitHub 页面。*