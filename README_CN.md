# MatterGen 中文使用手册

<p align="center">
    <img src="assets/MatterGenlogo_.png" alt="MatterGen logo" width="600"/>
</p>

[![DOI](https://img.shields.io/badge/DOI-10.1038%2Fs41586--025--08628--5-blue)](https://www.nature.com/articles/s41586-025-08628-5)
[![arXiv](https://img.shields.io/badge/arXiv-2312.03687-blue.svg?logo=arxiv&logoColor=white.svg)](https://arxiv.org/abs/2312.03687)
[![Requires Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg?logo=python&logoColor=white)](https://python.org/downloads)

MatterGen 是一个用于无机材料设计的生成模型，能够在元素周期表范围内生成晶体结构。该模型支持无条件生成、属性条件生成、从头训练和针对特定属性的微调。

## 快速导航

- [🚀 快速开始](#-快速开始)
- [⚙️ 安装指南](#-安装指南)  
- [📊 数据预处理](#-数据预处理)
- [🎯 模型训练](#-模型训练)
- [🔮 结构生成](#-结构生成)
- [📈 模型评估](#-模型评估)
- [🔧 代码质量](#-代码质量)
- [❓ 故障排除](#-故障排除)

## 🚀 快速开始

### 环境准备
```bash
# 安装包管理器
pip install uv

# 创建虚拟环境 
uv venv .venv --python 3.10 
source .venv/bin/activate

# 安装 MatterGen
uv pip install -e .
```

### 数据准备
```bash
# 下载并预处理 MP-20 数据集
git lfs pull -I data-release/mp-20/ --exclude=""
unzip data-release/mp-20/mp_20.zip -d datasets
csv-to-dataset --csv-folder datasets/mp_20/ --dataset-name mp_20 --cache-folder datasets/cache
```

### 模型训练
```bash
# 训练无条件基础模型
mattergen-train data_module=mp_20 ~trainer.logger
```

### 生成结构
```bash
# 无条件生成
export MODEL_NAME=mattergen_base
export RESULTS_PATH=results/
mattergen-generate $RESULTS_PATH --pretrained-name=$MODEL_NAME --batch_size=16 --num_batches=1
```

### 评估结果
```bash
# 使用 MatterSim 进行评估
git lfs pull -I data-release/alex-mp/reference_MP2020correction.gz --exclude=""
mattergen-evaluate --structures_path=$RESULTS_PATH --relax=True --save_as="$RESULTS_PATH/metrics.json"
```

## ⚙️ 安装指南

### 系统要求
- **Python**: 3.10+
- **GPU**: CUDA 兼容GPU (推荐)
- **存储**: 50GB+ 可用空间
- **内存**: 16GB RAM (推荐 32GB)

### 安装步骤

#### 1. 克隆仓库
```bash
git clone https://github.com/microsoft/mattergen.git
cd mattergen
```

#### 2. 设置环境
```bash
# 使用 uv (推荐)
uv venv .venv --python 3.10 
source .venv/bin/activate
uv pip install -e .

# 或使用 conda
conda create -n mattergen python=3.10
conda activate mattergen
pip install -e .
```

#### 3. 验证安装
```bash
python -c "import mattergen; print('✅ 安装成功!')"
mattergen-train --help
```

#### 4. Git LFS 设置
```bash
# Ubuntu/Debian
sudo apt install git-lfs
git lfs install

# CentOS/RHEL  
sudo yum install git-lfs
git lfs install

# macOS
brew install git-lfs
git lfs install
```

## 📊 数据预处理

### 数据集对比

| 数据集 | 结构数量 | 处理时间 | 适用场景 |
|--------|----------|----------|----------|
| **MP-20** | ~45,000 | ~10分钟 | 快速原型、测试 |
| **Alex-MP-20** | ~600,000 | ~1小时 | 完整训练、生产 |

### MP-20 数据集 (快速开始)
```bash
# 1. 下载数据
git lfs pull -I data-release/mp-20/ --exclude=""

# 2. 解压并预处理
unzip data-release/mp-20/mp_20.zip -d datasets
csv-to-dataset --csv-folder datasets/mp_20/ --dataset-name mp_20 --cache-folder datasets/cache

# 3. 验证
ls datasets/cache/mp_20/  # 应显示 train/ 和 val/ 目录
```

### Alex-MP-20 数据集 (完整版)
```bash
# 1. 下载数据 (较大文件)
git lfs pull -I data-release/alex-mp/alex_mp_20.zip --exclude=""

# 2. 解压并预处理 
unzip data-release/alex-mp/alex_mp_20.zip -d datasets
csv-to-dataset --csv-folder datasets/alex_mp_20/ --dataset-name alex_mp_20 --cache-folder datasets/cache

# 3. 验证数据完整性
python -c "
import numpy as np
print('✅ 训练集:', np.load('datasets/cache/alex_mp_20/train/atomic_numbers.npy').shape)
print('✅ 验证集:', np.load('datasets/cache/alex_mp_20/val/atomic_numbers.npy').shape)
"
```

### 数据格式结构
```
datasets/cache/{dataset_name}/
├── train/
│   ├── atomic_numbers.npy     # 原子类型  
│   ├── pos.npy                # 原子坐标
│   ├── cell.npy               # 晶胞参数
│   ├── num_atoms.npy          # 原子数量
│   └── *.json                 # 材料属性
└── val/ (相同结构)
```

## 🎯 模型训练

### 基础训练

#### MP-20 训练 (适合开发测试)
```bash
# 基础无条件模型
mattergen-train data_module=mp_20 ~trainer.logger

# 启用日志监控
mattergen-train data_module=mp_20
```

#### Alex-MP-20 训练 (生产级别)
```bash
# 大数据集训练 (需要梯度累积)
mattergen-train data_module=alex_mp_20 ~trainer.logger trainer.accumulate_grad_batches=4

# 根据GPU内存调整
mattergen-train data_module=alex_mp_20 ~trainer.logger trainer.accumulate_grad_batches=8
```

### 晶体结构预测 (CSP) 模式
```bash
# CSP 模式训练 (已知组分的结构预测)
mattergen-train --config-name=csp data_module=mp_20 ~trainer.logger
```

### 性能优化配置

#### GPU 内存优化
```bash
# 梯度检查点
mattergen-train data_module=alex_mp_20 lightning_module.diffusion_module.model.gradient_checkpointing=True

# 混合精度
mattergen-train data_module=alex_mp_20 trainer.precision=16

# 减少批次大小
mattergen-train data_module=alex_mp_20 trainer.accumulate_grad_batches=16
```

#### 多GPU 训练
```bash
# 数据并行
mattergen-train data_module=alex_mp_20 trainer.devices=4 trainer.strategy=ddp

# 模型并行 (大模型)
mattergen-train data_module=alex_mp_20 trainer.devices=4 trainer.strategy=deepspeed_stage_2
```

### 训练监控

#### 输出结构
```
outputs/singlerun/{date}/{time}/
├── checkpoints/               # 模型检查点
├── lightning_logs/            # 训练日志  
└── config.yaml               # 运行配置
```

#### TensorBoard 监控
```bash
tensorboard --logdir outputs/singlerun/
# 访问: http://localhost:6006
```

#### 训练指标参考
- **验证损失**: ~0.4 (80,000步后)
- **训练时间**: MP-20 约12-24小时 (单GPU)
- **收敛标志**: 验证损失稳定，生成质量稳定

## 🔮 结构生成

### 预训练模型总览

| 模型名称 | 功能描述 | 条件属性 |
|---------|----------|----------|
| `mattergen_base` | 无条件生成 | 无 |
| `chemical_system` | 化学体系条件生成 | 化学组分 |
| `space_group` | 空间群条件生成 | 晶体对称性 |
| `dft_mag_density` | 磁密度条件生成 | DFT磁密度 |
| `dft_band_gap` | 带隙条件生成 | DFT带隙 |
| `ml_bulk_modulus` | 体积模量条件生成 | ML体积模量 |

### 无条件生成
```bash
export MODEL_NAME=mattergen_base
export RESULTS_PATH=results/unconditional/

mattergen-generate $RESULTS_PATH \
  --pretrained-name=$MODEL_NAME \
  --batch_size=16 \
  --num_batches=10
```

### 属性条件生成

#### 单属性条件
```bash
# 特定磁密度材料
export MODEL_NAME=dft_mag_density
mattergen-generate results/mag_density/ \
  --pretrained-name=$MODEL_NAME \
  --batch_size=16 \
  --properties_to_condition_on="{'dft_mag_density': 0.15}" \
  --diffusion_guidance_factor=2.0
```

#### 多属性条件
```bash
# 特定化学体系和能量
export MODEL_NAME=chemical_system_energy_above_hull
mattergen-generate results/li_o_stable/ \
  --pretrained-name=$MODEL_NAME \
  --batch_size=16 \
  --properties_to_condition_on="{'chemical_system': 'Li-O', 'energy_above_hull': 0.05}" \
  --diffusion_guidance_factor=2.0
```

### 引导强度控制

| `diffusion_guidance_factor` | 效果 | 适用场景 |
|----------------------------|------|----------|
| 0.0 | 无约束 | 最大多样性 |
| 1.0-2.0 | 温和约束 | 平衡约束与多样性 |
| 3.0-5.0 | 强约束 | 精确属性控制 |
| >5.0 | 过强约束 | 可能降低质量 |

### 生成优化

#### 采样配置
```bash
# 快速生成 (减少扩散步数)
mattergen-generate $RESULTS_PATH \
  --pretrained-name=$MODEL_NAME \
  --sampling_config_overrides='sampler_partial.N=500'

# 高质量生成 (增加校正器步数)
mattergen-generate $RESULTS_PATH \
  --pretrained-name=$MODEL_NAME \
  --sampling_config_overrides='sampler_partial.n_steps_corrector=3'
```

#### 批量生成
```bash
# 大规模生成
mattergen-generate $RESULTS_PATH \
  --pretrained-name=$MODEL_NAME \
  --batch_size=64 \
  --num_batches=100 \
  --record-trajectories=False  # 节省存储
```

### 输出文件说明
```
{RESULTS_PATH}/
├── generated_crystals_cif.zip     # 标准CIF文件
├── generated_crystals.extxyz      # 扩展XYZ格式
└── generated_trajectories.zip     # 去噪轨迹 (可选)
```

## 📈 模型评估

### 评估流程

#### 完整评估 (推荐)
```bash
# 下载参考数据
git lfs pull -I data-release/alex-mp/reference_MP2020correction.gz --exclude=""

# 执行评估 (包含DFT弛豫)
mattergen-evaluate \
  --structures_path=$RESULTS_PATH \
  --relax=True \
  --structure_matcher='disordered' \
  --save_as="$RESULTS_PATH/metrics.json"
```

#### 快速评估
```bash
# 仅结构分析 (不弛豫)
mattergen-evaluate \
  --structures_path=$RESULTS_PATH \
  --relax=False \
  --structure_matcher='disordered' \
  --save_as="$RESULTS_PATH/metrics_quick.json"
```

### 评估指标解读

| 指标 | 定义 | 理想值 | 说明 |
|------|------|--------|------|
| **新颖性** | 与已知结构的不重复程度 | ~1.0 | 接近1表示完全新颖 |
| **唯一性** | 生成结构间的去重程度 | ~1.0 | 接近1表示无重复 |
| **稳定性** | 基于能量的热力学稳定性 | ~1.0 | 接近1表示高稳定性 |
| **有效性** | 结构的物理化学合理性 | 1.0 | 1.0表示完全有效 |
| **RMSD** | 与最近邻结构的距离 | 较低 | 低值表示与已知结构相似 |

### 基准测试
```bash
# 查看已有基准结果
ls benchmark/metrics/

# 查看对比图表
jupyter notebook benchmark/plot_benchmark_results.ipynb

# 添加自己的结果
cp $RESULTS_PATH/metrics.json benchmark/metrics/my_method.json
```

## 🔧 代码质量

### 测试
```bash
# 运行完整测试套件
pytest mattergen/tests/ mattergen/common/tests/ mattergen/diffusion/tests/

# 运行特定测试
pytest mattergen/tests/test_generator.py -v

# 生成覆盖率报告
pytest mattergen/tests/ --cov=mattergen --cov-report=html
```

### 代码格式化
```bash
# 格式化代码
black mattergen/ --line-length 100

# 整理导入
isort mattergen/ --profile black --line-length 100

# 代码质量检查
pylint mattergen/
```

## ❓ 故障排除

### 常见问题

#### CUDA 内存不足
```bash
# 减少批次大小
mattergen-train data_module=alex_mp_20 trainer.accumulate_grad_batches=8

# 启用梯度检查点
mattergen-train data_module=alex_mp_20 lightning_module.diffusion_module.model.gradient_checkpointing=True

# 混合精度
mattergen-train data_module=alex_mp_20 trainer.precision=16
```

#### 数据集文件未找到
```bash
# 检查数据
ls datasets/alex_mp_20/

# 重新预处理
csv-to-dataset --csv-folder datasets/alex_mp_20/ --dataset-name alex_mp_20 --cache-folder datasets/cache --force-reprocess
```

#### Git LFS 问题
```bash
# 重新安装
git lfs install --force

# 手动拉取
git lfs pull -I "data-release/mp-20/" --exclude=""

# 检查状态
git lfs status
```

### 性能优化

#### 训练优化
```bash
# 使用更多数据加载进程
mattergen-train data_module=alex_mp_20 data_module.num_workers.train=8

# 优化内存分配
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

#### 生成优化
```bash
# 减少扩散步数
mattergen-generate results/ \
  --pretrained-name=mattergen_base \
  --sampling_config_overrides='sampler_partial.N=500'

# 最大化批次大小
mattergen-generate results/ \
  --pretrained-name=mattergen_base \
  --batch_size=128
```

### 调试技巧
```bash
# 启用详细日志
export MATTERGEN_LOG_LEVEL=DEBUG

# 小规模测试
mattergen-train data_module=mp_20 trainer.max_epochs=1 trainer.limit_train_batches=2

# 检查配置
mattergen-train data_module=mp_20 --cfg job
```

## 📚 相关文档

- [📖 微调指南](README_FINETUNE_CN.md) - 详细的模型微调教程
- [🏗️ 架构解析](README_ARCHITECTURE_CN.md) - 模型架构深度解析  
- [🔄 标准流程](README_WORKFLOW_CN.md) - 端到端工作流程指南

## 📞 获取帮助

- **GitHub Issues**: [https://github.com/microsoft/mattergen/issues](https://github.com/microsoft/mattergen/issues)
- **GitHub Discussions**: [https://github.com/microsoft/mattergen/discussions](https://github.com/microsoft/mattergen/discussions)
- **论文参考**: [Nature 2025](https://www.nature.com/articles/s41586-025-08628-5)

## 📜 许可证和引用

### 许可证
本项目基于 MIT 许可证开源。详见 [LICENSE](LICENSE) 文件。

### 引用
如果使用了 MatterGen，请引用：

```bibtex
@article{MatterGen2025,
  author  = {Zeni, Claudio and Pinsler, Robert and Z{\"u}gner, Daniel and Fowler, Andrew and Horton, Matthew and Fu, Xiang and Wang, Zilong and Shysheya, Aliaksandra and Crabb{\'e}, Jonathan and Ueda, Shoko and Sordillo, Roberto and Sun, Lixin and Smith, Jake and Nguyen, Bichlien and Schulz, Hannes and Lewis, Sarah and Huang, Chin-Wei and Lu, Ziheng and Zhou, Yichi and Yang, Han and Hao, Hongxia and Li, Jielan and Yang, Chunlei and Li, Wenjie and Tomioka, Ryota and Xie, Tian},
  journal = {Nature},
  title   = {A generative model for inorganic materials design},
  year    = {2025},
  doi     = {10.1038/s41586-025-08628-5},
}
```

---

*基于 MatterGen v1.0 编写 | 更新日期: 2025年*