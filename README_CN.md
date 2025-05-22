# MatterGen 中文使用手册

<p align="center">
    <img src="assets/MatterGenlogo_.png" alt="MatterGen logo" width="600"/>
</p>

[![DOI](https://img.shields.io/badge/DOI-10.1038%2Fs41586--025--08628--5-blue)](https://www.nature.com/articles/s41586-025-08628-5)
[![arXiv](https://img.shields.io/badge/arXiv-2312.03687-blue.svg?logo=arxiv&logoColor=white.svg)](https://arxiv.org/abs/2312.03687)
[![Requires Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg?logo=python&logoColor=white)](https://python.org/downloads)

MatterGen 是一个用于无机材料设计的生成模型，能够在元素周期表范围内生成晶体结构。该模型可以进行无条件生成、属性条件生成，支持从头训练和针对特定属性的微调。

## 目录
- [快速开始](#快速开始)
- [安装指南](#安装指南)
- [数据预处理](#数据预处理)
- [模型训练](#模型训练)
- [结构生成](#结构生成)
- [模型评估](#模型评估)
- [代码质量控制](#代码质量控制)
- [故障排除](#故障排除)

## 快速开始

### 1. 环境准备
```bash
# 安装 uv 包管理器
pip install uv

# 创建虚拟环境
uv venv .venv --python 3.10 
source .venv/bin/activate

# 安装 MatterGen
uv pip install -e .
```

### 2. 数据准备
```bash
# 下载并预处理 MP-20 数据集
git lfs pull -I data-release/mp-20/ --exclude=""
unzip data-release/mp-20/mp_20.zip -d datasets
csv-to-dataset --csv-folder datasets/mp_20/ --dataset-name mp_20 --cache-folder datasets/cache
```

### 3. 训练基础模型
```bash
# 训练无条件基础模型
mattergen-train data_module=mp_20 ~trainer.logger
```

### 4. 生成晶体结构
```bash
# 无条件生成16个结构
export MODEL_NAME=mattergen_base
export RESULTS_PATH=results/
mattergen-generate $RESULTS_PATH --pretrained-name=$MODEL_NAME --batch_size=16 --num_batches=1
```

### 5. 评估生成结果
```bash
# 使用 MatterSim 弛豫和评估
git lfs pull -I data-release/alex-mp/reference_MP2020correction.gz --exclude=""
mattergen-evaluate --structures_path=$RESULTS_PATH --relax=True --save_as="$RESULTS_PATH/metrics.json"
```

## 安装指南

### 系统要求
- **Python**: 3.10 或更高版本
- **GPU**: CUDA 兼容GPU (推荐)
- **存储**: 至少 50GB 可用空间
- **内存**: 16GB RAM (推荐 32GB)

### 详细安装步骤

#### 1. 安装包管理器
```bash
# 安装 uv (推荐的包管理器)
pip install uv

# 或使用 pip + conda
conda create -n mattergen python=3.10
conda activate mattergen
```

#### 2. 克隆仓库
```bash
git clone https://github.com/microsoft/mattergen.git
cd mattergen
```

#### 3. 安装依赖
```bash
# 使用 uv (推荐)
uv venv .venv --python 3.10 
source .venv/bin/activate
uv pip install -e .

# 或使用 pip
pip install -e .
```

#### 4. 验证安装
```bash
# 检查安装是否成功
python -c "import mattergen; print('安装成功!')"

# 检查命令行工具
mattergen-train --help
mattergen-generate --help
```

### Git LFS 设置
```bash
# 检查 Git LFS 是否已安装
git lfs --version

# 如果未安装 (Ubuntu/Debian)
sudo apt install git-lfs
git lfs install

# 如果未安装 (CentOS/RHEL)
sudo yum install git-lfs
git lfs install

# 如果未安装 (macOS)
brew install git-lfs
git lfs install
```

## 数据预处理

### 数据集概览

MatterGen 支持两个主要数据集：

| 数据集 | 结构数量 | 描述 | 处理时间 |
|--------|----------|------|----------|
| MP-20 | ~45,000 | 通用无机材料，单胞原子数≤20 | ~10分钟 |
| Alex-MP-20 | ~600,000 | MP-20 + Alexandria 数据库 | ~1小时 |

### MP-20 数据集处理
适合快速原型开发和测试：

```bash
# 1. 下载数据
git lfs pull -I data-release/mp-20/ --exclude=""

# 2. 解压数据
unzip data-release/mp-20/mp_20.zip -d datasets

# 3. 预处理 (约10分钟)
csv-to-dataset --csv-folder datasets/mp_20/ --dataset-name mp_20 --cache-folder datasets/cache

# 4. 验证数据
ls datasets/cache/mp_20/  # 应显示 train/ 和 val/ 目录
```

### Alex-MP-20 数据集处理
适合完整模型训练：

```bash
# 1. 下载数据 (文件较大，需要时间)
git lfs pull -I data-release/alex-mp/alex_mp_20.zip --exclude=""

# 2. 解压数据
unzip data-release/alex-mp/alex_mp_20.zip -d datasets

# 3. 预处理 (约1小时)
csv-to-dataset --csv-folder datasets/alex_mp_20/ --dataset-name alex_mp_20 --cache-folder datasets/cache

# 4. 验证数据完整性
python -c "
import numpy as np
print('训练集原子数量:', np.load('datasets/cache/alex_mp_20/train/atomic_numbers.npy').shape)
print('验证集原子数量:', np.load('datasets/cache/alex_mp_20/val/atomic_numbers.npy').shape)
"
```

### 数据格式说明

预处理后的数据存储结构：
```
datasets/cache/{dataset_name}/
├── train/
│   ├── atomic_numbers.npy     # 原子类型
│   ├── pos.npy                # 原子坐标
│   ├── cell.npy               # 晶胞参数
│   ├── num_atoms.npy          # 原子数量
│   └── *.json                 # 材料属性
└── val/
    └── (相同结构)
```

## 模型训练

### 基础模型训练

#### 在 MP-20 上训练
```bash
# 基础训练 (无条件生成模型)
mattergen-train data_module=mp_20 ~trainer.logger

# 启用日志记录 (可选)
mattergen-train data_module=mp_20

# 自定义输出目录
export OUTPUT_DIR=custom_output
mattergen-train data_module=mp_20 ~trainer.logger
```

#### 在 Alex-MP-20 上训练
```bash
# 大数据集训练 (需要梯度累积)
mattergen-train data_module=alex_mp_20 ~trainer.logger trainer.accumulate_grad_batches=4

# 调整批次大小 (根据GPU内存)
mattergen-train data_module=alex_mp_20 ~trainer.logger trainer.accumulate_grad_batches=8
```

### 晶体结构预测 (CSP) 模式

CSP 模式适用于已知化学组分的结构预测：

```bash
# CSP 模式训练
mattergen-train --config-name=csp data_module=mp_20 ~trainer.logger

# CSP 模式特点:
# - 不对原子类型进行去噪
# - 适合结构优化任务
# - 输入固定的化学组分
```

### 训练监控

#### 训练输出结构
```
outputs/singlerun/{date}/{time}/
├── checkpoints/               # 模型检查点
│   ├── epoch=N-step=M.ckpt
│   └── last.ckpt
├── lightning_logs/            # 训练日志
│   └── version_0/
│       ├── events.out.tfevents.*
│       ├── hparams.yaml
│       └── checkpoints/
└── config.yaml               # 运行配置
```

#### 使用 TensorBoard 监控
```bash
# 启动 TensorBoard
tensorboard --logdir outputs/singlerun/

# 在浏览器中访问 http://localhost:6006
```

#### 训练指标参考
- **验证损失**: 应在 80,000 步后达到 ~0.4
- **训练时间**: MP-20 约需 12-24 小时 (单GPU)
- **收敛标志**: 验证损失平稳，生成质量稳定

### 训练配置优化

#### GPU 内存优化
```bash
# 启用梯度检查点
mattergen-train data_module=alex_mp_20 lightning_module.diffusion_module.model.gradient_checkpointing=True

# 使用混合精度
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

## 结构生成

### 预训练模型

MatterGen 提供多个预训练模型：

| 模型名称 | 功能 | 支持属性 |
|---------|------|----------|
| `mattergen_base` | 无条件生成 | 无 |
| `chemical_system` | 化学体系条件生成 | 化学组分 |
| `space_group` | 空间群条件生成 | 晶体对称性 |
| `dft_mag_density` | 磁密度条件生成 | DFT磁密度 |
| `dft_band_gap` | 带隙条件生成 | DFT带隙 |
| `ml_bulk_modulus` | 体积模量条件生成 | ML体积模量 |

### 无条件生成

```bash
# 基础无条件生成
export MODEL_NAME=mattergen_base
export RESULTS_PATH=results/unconditional/

mattergen-generate $RESULTS_PATH \
  --pretrained-name=$MODEL_NAME \
  --batch_size=16 \
  --num_batches=10
```

#### 生成参数说明
- `--batch_size`: 每批次生成的结构数量 (受GPU内存限制)
- `--num_batches`: 批次数量 (总结构数 = batch_size × num_batches)
- `--record-trajectories`: 是否保存去噪轨迹 (默认True)

### 属性条件生成

#### 单属性条件生成
```bash
# 生成特定磁密度的材料
export MODEL_NAME=dft_mag_density
export RESULTS_PATH=results/mag_density/

mattergen-generate $RESULTS_PATH \
  --pretrained-name=$MODEL_NAME \
  --batch_size=16 \
  --properties_to_condition_on="{'dft_mag_density': 0.15}" \
  --diffusion_guidance_factor=2.0
```

#### 多属性条件生成
```bash
# 生成特定化学体系和能量的材料
export MODEL_NAME=chemical_system_energy_above_hull
export RESULTS_PATH=results/li_o_stable/

mattergen-generate $RESULTS_PATH \
  --pretrained-name=$MODEL_NAME \
  --batch_size=16 \
  --properties_to_condition_on="{'chemical_system': 'Li-O', 'energy_above_hull': 0.05}" \
  --diffusion_guidance_factor=2.0
```

### 引导强度调节

`--diffusion_guidance_factor` 参数控制属性约束强度：

| 数值 | 效果 | 适用场景 |
|------|------|----------|
| 0.0 | 无约束 (等价于无条件生成) | 最大多样性 |
| 1.0-2.0 | 温和约束 | 平衡约束与多样性 |
| 3.0-5.0 | 强约束 | 精确属性控制 |
| >5.0 | 过强约束 | 可能降低质量 |

### 晶体结构预测 (CSP)

```bash
# 使用 CSP 模型预测 NaCl 结构
export CSP_MODEL_PATH=path/to/csp/model.ckpt
export RESULTS_PATH=results/nacl_csp/

mattergen-generate $RESULTS_PATH \
  --model_path=$CSP_MODEL_PATH \
  --target_compositions='[{"Na": 1, "Cl": 1}]' \
  --sampling-config-name=csp \
  --batch_size=32
```

### 生成优化

#### 采样配置调整
```bash
# 快速生成 (减少扩散步数)
mattergen-generate $RESULTS_PATH \
  --pretrained-name=$MODEL_NAME \
  --sampling_config_overrides='sampler_partial.N=500'

# 高质量生成 (增加校正器步数)
mattergen-generate $RESULTS_PATH \
  --pretrained-name=$MODEL_NAME \
  --sampling_config_overrides='sampler_partial.n_steps_corrector=3'

# 禁用校正器 (最快生成)
mattergen-generate $RESULTS_PATH \
  --pretrained-name=$MODEL_NAME \
  --sampling_config_overrides='sampler_partial.n_steps_corrector=0'
```

#### 批量生成
```bash
# 大规模生成
mattergen-generate $RESULTS_PATH \
  --pretrained-name=$MODEL_NAME \
  --batch_size=64 \
  --num_batches=100 \
  --record-trajectories=False  # 节省存储空间
```

### 输出文件格式

生成完成后，结果保存在指定目录：

```
{RESULTS_PATH}/
├── generated_crystals_cif.zip     # 单独的CIF文件
├── generated_crystals.extxyz      # 统一的扩展XYZ格式
└── generated_trajectories.zip     # 去噪轨迹 (可选)
```

#### 文件格式说明
- **CIF格式**: 标准晶体学信息文件，可用于大多数材料分析软件
- **ExtXYZ格式**: 扩展XYZ格式，包含晶胞参数和属性信息
- **轨迹文件**: 完整的去噪过程，用于分析生成机制

## 模型评估

### 基础评估流程

#### 完整评估 (推荐)
```bash
# 下载参考数据集
git lfs pull -I data-release/alex-mp/reference_MP2020correction.gz --exclude=""

# 执行完整评估 (包含DFT弛豫)
mattergen-evaluate \
  --structures_path=$RESULTS_PATH \
  --relax=True \
  --structure_matcher='disordered' \
  --save_as="$RESULTS_PATH/metrics.json"
```

#### 快速评估 (不弛豫)
```bash
# 仅结构分析，不进行DFT弛豫
mattergen-evaluate \
  --structures_path=$RESULTS_PATH \
  --relax=False \
  --structure_matcher='disordered' \
  --save_as="$RESULTS_PATH/metrics_quick.json"
```

#### 使用预计算能量
```bash
# 使用外部计算的能量数据
mattergen-evaluate \
  --structures_path=$RESULTS_PATH \
  --energies_path="precomputed_energies.npy" \
  --relax=False \
  --save_as="$RESULTS_PATH/metrics_external.json"
```

### 评估配置选项

#### 结构匹配器选择
```bash
# 无序匹配器 (默认，更宽松)
--structure_matcher='disordered'

# 有序匹配器 (更严格)
--structure_matcher='ordered'
```

#### MatterSim 模型选择
```bash
# 使用不同大小的MatterSim模型
--potential_load_path="MatterSim-v1.0.0-1M.pth"   # 小模型，快速
--potential_load_path="MatterSim-v1.0.0-5M.pth"   # 大模型，精确
```

#### 保存弛豫结构
```bash
# 保存弛豫后的结构用于进一步分析
mattergen-evaluate \
  --structures_path=$RESULTS_PATH \
  --relax=True \
  --structures_output_path="$RESULTS_PATH/relaxed_structures.extxyz"
```

### 评估指标解读

评估脚本计算以下关键指标：

#### 1. 新颖性 (Novelty)
- **定义**: 生成结构与参考数据集的不重复程度
- **计算**: 使用结构匹配算法比较生成结构与已知结构
- **理想值**: 接近1.0 (完全新颖)

#### 2. 唯一性 (Uniqueness)
- **定义**: 生成结构内部的去重程度
- **计算**: 统计生成结构中的重复结构比例
- **理想值**: 接近1.0 (无重复)

#### 3. 稳定性 (Stability)
- **定义**: 基于能量的热力学稳定性
- **计算**: 凸包分析，计算能量上方距离
- **理想值**: 接近1.0 (高稳定性)

#### 4. 有效性 (Validity)
- **定义**: 结构的物理化学合理性
- **计算**: 检查键长、配位数等几何参数
- **理想值**: 1.0 (完全有效)

#### 5. RMSD
- **定义**: 与最近邻参考结构的距离
- **计算**: 均方根偏差
- **理想值**: 低RMSD表示与已知结构相似

### 自定义评估

#### 创建自定义参考数据集
```python
# create_reference.py
from mattergen.evaluation.reference.reference_dataset import ReferenceDataset
from mattergen.evaluation.reference.reference_dataset_serializer import LMDBGZSerializer
from mattergen.evaluation.utils.vasprunlike import VasprunLike
from pymatgen.entries.compatibility import MaterialsProject2020Compatibility

# 准备结构和能量数据
structures = [...]  # pymatgen Structure对象列表
energies = [...]    # 对应的能量列表

# 创建计算条目
entries = []
for structure, energy in zip(structures, energies):
    vasprun_like = VasprunLike(structure=structure, energy=energy)
    entry = vasprun_like.get_computed_entry(
        inc_structure=True,
        energy_correction_scheme=MaterialsProject2020Compatibility()
    )
    entries.append(entry)

# 创建并保存参考数据集
reference_dataset = ReferenceDataset.from_entries(
    name="custom_reference",
    entries=entries
)
LMDBGZSerializer().serialize(reference_dataset, "custom_reference.gz")
```

#### 使用自定义参考数据集
```bash
mattergen-evaluate \
  --structures_path=$RESULTS_PATH \
  --reference_dataset_path="custom_reference.gz" \
  --relax=True \
  --save_as="custom_evaluation.json"
```

### 基准测试和比较

#### 查看基准结果
```bash
# 查看已有基准测试结果
ls benchmark/metrics/
# mattergen.json  cdvae.json  diffcsp_mp_20.json  ...

# 使用Jupyter查看对比图表
jupyter notebook benchmark/plot_benchmark_results.ipynb
```

#### 添加自己的结果到基准测试
```bash
# 将评估结果复制到基准目录
cp $RESULTS_PATH/metrics.json benchmark/metrics/my_method.json

# 重新运行基准分析
jupyter notebook benchmark/plot_benchmark_results.ipynb
```

## 代码质量控制

### 测试

#### 运行所有测试
```bash
# 运行完整测试套件
pytest mattergen/tests/ mattergen/common/tests/ mattergen/diffusion/tests/

# 显示详细输出
pytest mattergen/tests/ -v

# 生成覆盖率报告
pytest mattergen/tests/ --cov=mattergen --cov-report=html
```

#### 运行特定测试
```bash
# 测试生成器
pytest mattergen/tests/test_generator.py

# 测试GemNet模型
pytest mattergen/common/tests/gemnet_test.py

# 测试扩散模块
pytest mattergen/diffusion/tests/test_d3pm.py
```

### 代码格式化

#### 使用 Black 格式化
```bash
# 格式化所有代码
black mattergen/ --line-length 100

# 仅检查格式
black mattergen/ --line-length 100 --check

# 显示将要修改的内容
black mattergen/ --line-length 100 --diff
```

#### 使用 isort 整理导入
```bash
# 排序所有导入语句
isort mattergen/ --profile black --line-length 100

# 仅检查排序
isort mattergen/ --profile black --line-length 100 --check-only
```

#### 使用 pylint 检查代码质量
```bash
# 分析代码质量
pylint mattergen/

# 生成详细报告
pylint mattergen/ --output-format=text > pylint_report.txt
```

## 故障排除

### 常见问题解决方案

#### 1. CUDA 内存不足
**症状**: `RuntimeError: CUDA out of memory`

**解决方案**:
```bash
# 方案1: 减少批次大小
mattergen-train data_module=alex_mp_20 trainer.accumulate_grad_batches=8

# 方案2: 启用梯度检查点
mattergen-train data_module=alex_mp_20 lightning_module.diffusion_module.model.gradient_checkpointing=True

# 方案3: 使用混合精度
mattergen-train data_module=alex_mp_20 trainer.precision=16

# 方案4: 清理GPU缓存
python -c "import torch; torch.cuda.empty_cache()"
```

#### 2. 数据集文件未找到
**症状**: `FileNotFoundError: datasets/cache/alex_mp_20/train not found`

**解决方案**:
```bash
# 检查数据是否已下载
ls datasets/alex_mp_20/

# 重新预处理数据
csv-to-dataset --csv-folder datasets/alex_mp_20/ --dataset-name alex_mp_20 --cache-folder datasets/cache

# 强制重新处理
csv-to-dataset --csv-folder datasets/alex_mp_20/ --dataset-name alex_mp_20 --cache-folder datasets/cache --force-reprocess
```

#### 3. Git LFS 问题
**症状**: `Git LFS: smudge filter lfs failed`

**解决方案**:
```bash
# 重新安装Git LFS
git lfs install --force

# 手动拉取特定文件
git lfs pull -I "data-release/mp-20/" --exclude=""

# 检查LFS状态
git lfs status
```

#### 4. 模型加载错误
**症状**: `RuntimeError: Error(s) in loading state_dict`

**解决方案**:
```bash
# 使用非严格加载
mattergen-generate results/ --model_path=/path/to/model --strict_checkpoint_loading=False

# 检查模型文件完整性
python -c "
import torch
checkpoint = torch.load('/path/to/model.ckpt')
print('检查点键:', list(checkpoint.keys()))
"
```

### 性能优化建议

#### 训练性能优化
```bash
# 启用编译模式 (PyTorch 2.0+)
mattergen-train data_module=alex_mp_20 lightning_module.diffusion_module.model.compile=True

# 使用更多数据加载进程
mattergen-train data_module=alex_mp_20 data_module.num_workers.train=8

# 优化内存分配
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

#### 生成性能优化
```bash
# 减少扩散步数 (快速生成)
mattergen-generate results/ \
  --pretrained-name=mattergen_base \
  --sampling_config_overrides='sampler_partial.N=500'

# 最大化批次大小
mattergen-generate results/ \
  --pretrained-name=mattergen_base \
  --batch_size=128
```

### 调试技巧

#### 启用详细日志
```bash
# 设置环境变量
export MATTERGEN_LOG_LEVEL=DEBUG
export PYTHONPATH=/path/to/mattergen:$PYTHONPATH

# 详细训练输出
mattergen-train data_module=mp_20 trainer.log_every_n_steps=1
```

#### 小规模测试
```bash
# 快速训练测试
mattergen-train data_module=mp_20 trainer.max_epochs=1 trainer.limit_train_batches=2

# 小批次生成测试
mattergen-generate results/ --pretrained-name=mattergen_base --batch_size=1 --num_batches=1
```

#### 配置检查
```bash
# 查看解析后的配置
mattergen-train data_module=mp_20 --cfg job

# 检查Hydra配置
python -m hydra.main config_path=mattergen/conf config_name=default --cfg job
```

### 获取帮助

- **官方文档**: 查看项目README和代码注释
- **GitHub Issues**: [https://github.com/microsoft/mattergen/issues](https://github.com/microsoft/mattergen/issues)
- **GitHub Discussions**: [https://github.com/microsoft/mattergen/discussions](https://github.com/microsoft/mattergen/discussions)
- **论文参考**: [Nature 2025](https://www.nature.com/articles/s41586-025-08628-5)

---

## 许可证和引用

### 许可证
本项目基于 MIT 许可证开源。详见 [LICENSE](LICENSE) 文件。

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

---

*该中文手册基于 MatterGen v1.0 编写。如有疑问，请访问项目的 GitHub 页面获取技术支持。*