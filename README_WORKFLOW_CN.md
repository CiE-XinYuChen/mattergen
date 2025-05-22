# MatterGen 标准工作流程

<p align="center">
    <img src="assets/MatterGenlogo_.png" alt="MatterGen logo" width="400"/>
</p>

本文档提供 MatterGen 的标准端到端工作流程，涵盖从环境配置到结果分析的完整材料设计流程。

## 📋 目录

- [🚀 快速开始流程](#-快速开始流程)
- [🏗️ 完整工作流程](#-完整工作流程)
- [🎯 应用场景](#-应用场景)
- [📊 数据管理](#-数据管理)
- [🔄 实验管理](#-实验管理)
- [📈 结果分析](#-结果分析)
- [🛠️ 自动化脚本](#-自动化脚本)
- [📚 最佳实践](#-最佳实践)

## 🚀 快速开始流程

### 最小可行工作流程 (30分钟)

适合快速验证和原型开发：

#### 步骤 1: 环境准备 (5分钟)
```bash
# 克隆项目
git clone https://github.com/microsoft/mattergen.git
cd mattergen

# 创建环境
pip install uv
uv venv .venv --python 3.10
source .venv/bin/activate
uv pip install -e .

# 验证安装
python -c "import mattergen; print('✅ 安装成功')"
```

#### 步骤 2: 数据准备 (10分钟)
```bash
# 下载小数据集 (MP-20)
git lfs pull -I data-release/mp-20/ --exclude=""
unzip data-release/mp-20/mp_20.zip -d datasets

# 快速预处理
csv-to-dataset --csv-folder datasets/mp_20/ --dataset-name mp_20 --cache-folder datasets/cache
```

#### 步骤 3: 快速生成 (5分钟)
```bash
# 使用预训练模型生成
export MODEL_NAME=mattergen_base
export RESULTS_PATH=results/quick_test/

mattergen-generate $RESULTS_PATH \
  --pretrained-name=$MODEL_NAME \
  --batch_size=4 \
  --num_batches=1
```

#### 步骤 4: 结果查看 (5分钟)
```bash
# 查看生成结果
ls $RESULTS_PATH
unzip $RESULTS_PATH/generated_crystals_cif.zip -d $RESULTS_PATH/cifs/
ls $RESULTS_PATH/cifs/  # 查看生成的 CIF 文件

# 快速统计
python -c "
import glob
cif_files = glob.glob('$RESULTS_PATH/cifs/*.cif')
print(f'生成了 {len(cif_files)} 个晶体结构')
"
```

#### 步骤 5: 简单评估 (5分钟)
```bash
# 不弛豫的快速评估
mattergen-evaluate \
  --structures_path=$RESULTS_PATH \
  --relax=False \
  --save_as="$RESULTS_PATH/quick_metrics.json"

# 查看评估结果
cat $RESULTS_PATH/quick_metrics.json | jq '.'
```

## 🏗️ 完整工作流程

### 端到端材料设计流程

#### 第一阶段: 项目规划 (1-2天)

##### 1.1 需求分析
```markdown
## 项目需求模板

### 目标材料类型
- [ ] 无机晶体材料
- [ ] 特定化学体系: ________________
- [ ] 原子数量范围: ___ 到 ___ 个原子

### 目标属性
- [ ] 无条件生成 (探索新结构)
- [ ] 磁性材料 (磁密度: ___ μB/Å³)
- [ ] 半导体材料 (带隙: ___ eV)
- [ ] 硬质材料 (体积模量: ___ GPa)
- [ ] 稳定材料 (凸包上方能量: < ___ eV/atom)
- [ ] 其他: ________________

### 生成数量和质量要求
- 目标生成数量: _______ 个结构
- 质量要求: 新颖性 > ___%, 稳定性 > ___%
- 后续验证方法: DFT计算 / 实验合成 / 其他

### 计算资源
- 可用GPU: _______ (型号: _______)
- 内存限制: _______ GB
- 时间预算: _______ 天
```

##### 1.2 技术路线选择
```bash
# 评估不同数据集和模型的适用性
python scripts/evaluate_datasets.py --requirements requirements.yaml
```

#### 第二阶段: 环境搭建 (半天)

##### 2.1 完整环境配置
```bash
# 完整安装脚本
#!/bin/bash
set -e

echo "🚀 开始安装 MatterGen 完整环境..."

# 1. 基础环境
echo "📦 安装基础依赖..."
pip install uv git-lfs
git lfs install

# 2. 项目环境
echo "🔧 创建项目环境..."
uv venv .venv --python 3.10
source .venv/bin/activate

# 3. 安装包
echo "📚 安装 MatterGen..."
uv pip install -e .

# 4. 验证安装
echo "✅ 验证安装..."
python -c "
import torch
import mattergen
print(f'✅ PyTorch: {torch.__version__}')
print(f'✅ CUDA: {torch.cuda.is_available()}')
print(f'✅ MatterGen: 安装成功')
"

# 5. 测试工具
echo "🧪 测试命令行工具..."
mattergen-train --help > /dev/null && echo "✅ mattergen-train"
mattergen-generate --help > /dev/null && echo "✅ mattergen-generate"
mattergen-evaluate --help > /dev/null && echo "✅ mattergen-evaluate"

echo "🎉 安装完成！"
```

##### 2.2 数据集准备
```bash
# 数据集选择和准备脚本
#!/bin/bash

echo "📊 准备数据集..."

# 根据项目需求选择数据集
if [ "$DATASET_SIZE" = "small" ]; then
    echo "📦 下载 MP-20 数据集 (约45k结构)..."
    git lfs pull -I data-release/mp-20/ --exclude=""
    unzip data-release/mp-20/mp_20.zip -d datasets
    DATASET_NAME="mp_20"
elif [ "$DATASET_SIZE" = "large" ]; then
    echo "📦 下载 Alex-MP-20 数据集 (约600k结构)..."
    git lfs pull -I data-release/alex-mp/alex_mp_20.zip --exclude=""
    unzip data-release/alex-mp/alex_mp_20.zip -d datasets
    DATASET_NAME="alex_mp_20"
fi

# 数据预处理
echo "⚙️ 预处理数据集..."
csv-to-dataset \
  --csv-folder datasets/$DATASET_NAME/ \
  --dataset-name $DATASET_NAME \
  --cache-folder datasets/cache

# 数据验证
echo "✅ 验证数据完整性..."
python scripts/validate_dataset.py --dataset $DATASET_NAME

echo "🎉 数据准备完成！"
```

#### 第三阶段: 模型准备 (1-3天)

##### 3.1 基础模型验证
```bash
# 验证预训练模型
echo "🧪 测试预训练模型..."

export MODEL_NAME=mattergen_base
export TEST_RESULTS=results/model_test/

# 小规模测试生成
mattergen-generate $TEST_RESULTS \
  --pretrained-name=$MODEL_NAME \
  --batch_size=4 \
  --num_batches=1

# 检查生成质量
mattergen-evaluate \
  --structures_path=$TEST_RESULTS \
  --relax=False \
  --save_as="$TEST_RESULTS/test_metrics.json"

echo "✅ 基础模型验证完成"
```

##### 3.2 模型微调 (可选)
```bash
# 根据项目需求进行微调
if [ "$NEED_FINETUNING" = "true" ]; then
    echo "🎯 开始模型微调..."
    
    export PROPERTY=$TARGET_PROPERTY
    export FINETUNE_OUTPUT="outputs/finetune_${PROPERTY}"
    
    # 微调训练
    mattergen-finetune \
      adapter.pretrained_name=mattergen_base \
      data_module=alex_mp_20 \
      +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.$PROPERTY=$PROPERTY \
      ~trainer.logger \
      data_module.properties=["$PROPERTY"] \
      trainer.max_epochs=100
    
    # 验证微调效果
    LATEST_RUN=$(ls -t outputs/singlerun/ | head -1)
    export FINETUNED_MODEL="outputs/singlerun/$LATEST_RUN/checkpoints/last.ckpt"
    
    echo "✅ 模型微调完成"
fi
```

#### 第四阶段: 大规模生成 (1-2天)

##### 4.1 生成参数优化
```bash
# 批量测试不同参数组合
echo "🔧 优化生成参数..."

# 测试不同引导强度
for guidance in 0.0 1.0 2.0 3.0 5.0; do
    echo "测试引导强度: $guidance"
    
    mattergen-generate results/param_test/guidance_$guidance \
      --pretrained-name=$MODEL_NAME \
      --batch_size=16 \
      --num_batches=2 \
      --properties_to_condition_on="{'$TARGET_PROPERTY': $TARGET_VALUE}" \
      --diffusion_guidance_factor=$guidance
    
    # 快速评估
    mattergen-evaluate \
      --structures_path=results/param_test/guidance_$guidance \
      --relax=False \
      --save_as="results/param_test/guidance_$guidance/metrics.json"
done

# 分析最优参数
python scripts/analyze_parameters.py --results_dir results/param_test/
```

##### 4.2 大规模生成
```bash
# 大规模生成脚本
#!/bin/bash

echo "🚀 开始大规模生成..."

export MODEL_NAME=${FINAL_MODEL_NAME}
export RESULTS_PATH=results/production/
export TOTAL_STRUCTURES=${TARGET_STRUCTURE_COUNT}
export BATCH_SIZE=${OPTIMIZED_BATCH_SIZE}
export GUIDANCE_FACTOR=${OPTIMIZED_GUIDANCE}

# 计算需要的批次数
NUM_BATCHES=$((TOTAL_STRUCTURES / BATCH_SIZE))

echo "📊 生成参数:"
echo "  - 模型: $MODEL_NAME"
echo "  - 总结构数: $TOTAL_STRUCTURES"
echo "  - 批次大小: $BATCH_SIZE"
echo "  - 批次数量: $NUM_BATCHES"
echo "  - 引导强度: $GUIDANCE_FACTOR"

# 分批生成，避免单次运行过长
BATCH_PER_RUN=10
RUNS=$((NUM_BATCHES / BATCH_PER_RUN))

for run in $(seq 1 $RUNS); do
    echo "🔄 执行生成轮次 $run/$RUNS..."
    
    OUTPUT_DIR="${RESULTS_PATH}/run_${run}"
    
    mattergen-generate $OUTPUT_DIR \
      --pretrained-name=$MODEL_NAME \
      --batch_size=$BATCH_SIZE \
      --num_batches=$BATCH_PER_RUN \
      --properties_to_condition_on="{'$TARGET_PROPERTY': $TARGET_VALUE}" \
      --diffusion_guidance_factor=$GUIDANCE_FACTOR
    
    echo "✅ 轮次 $run 完成"
done

echo "🎉 大规模生成完成！"
```

#### 第五阶段: 质量评估 (1-2天)

##### 5.1 结构质量评估
```bash
# 全面质量评估
echo "📈 开始质量评估..."

# 合并所有生成结果
python scripts/merge_results.py \
  --input_dirs results/production/run_* \
  --output_dir results/production/merged/

# 执行完整评估
echo "🔬 执行完整结构评估..."
git lfs pull -I data-release/alex-mp/reference_MP2020correction.gz --exclude=""

mattergen-evaluate \
  --structures_path=results/production/merged/ \
  --relax=True \
  --structure_matcher='disordered' \
  --save_as="results/production/merged/full_metrics.json" \
  --structures_output_path="results/production/merged/relaxed_structures.extxyz"

echo "✅ 结构评估完成"
```

##### 5.2 属性分析
```bash
# 属性统计和分析
echo "📊 分析生成结构的属性分布..."

python scripts/property_analysis.py \
  --structures_path results/production/merged/ \
  --target_property $TARGET_PROPERTY \
  --target_value $TARGET_VALUE \
  --output_dir results/production/analysis/

# 生成分析报告
python scripts/generate_report.py \
  --metrics_file results/production/merged/full_metrics.json \
  --analysis_dir results/production/analysis/ \
  --output_file results/production/final_report.html
```

#### 第六阶段: 结果筛选 (半天)

##### 6.1 多条件筛选
```bash
# 高质量候选结构筛选
echo "🎯 筛选高质量候选结构..."

python scripts/filter_candidates.py \
  --structures_path results/production/merged/ \
  --metrics_file results/production/merged/full_metrics.json \
  --criteria "
    novelty > 0.95 and
    stability > 0.8 and
    validity == 1.0 and
    target_property_error < 0.1
  " \
  --output_dir results/production/candidates/ \
  --max_candidates 100

echo "✅ 筛选出 $(ls results/production/candidates/*.cif | wc -l) 个候选结构"
```

##### 6.2 候选结构排序
```bash
# 根据综合评分排序
python scripts/rank_candidates.py \
  --candidates_dir results/production/candidates/ \
  --ranking_strategy "weighted_score" \
  --weights "novelty:0.3,stability:0.4,target_property:0.3" \
  --output_file results/production/ranked_candidates.csv

# 输出前10名候选
head -11 results/production/ranked_candidates.csv
```

#### 第七阶段: 验证准备 (半天)

##### 7.1 DFT计算准备
```bash
# 为DFT计算准备输入文件
echo "⚙️ 准备DFT计算输入..."

python scripts/prepare_dft_inputs.py \
  --candidates_file results/production/ranked_candidates.csv \
  --top_n 20 \
  --dft_software VASP \
  --output_dir results/production/dft_inputs/

echo "✅ DFT输入文件准备完成"
```

##### 7.2 实验验证信息
```bash
# 生成实验验证指南
python scripts/generate_synthesis_guide.py \
  --candidates_file results/production/ranked_candidates.csv \
  --top_n 10 \
  --output_file results/production/synthesis_guide.md

echo "📋 实验合成指南已生成"
```

## 🎯 应用场景

### 场景1: 磁性材料设计

#### 目标设定
```bash
# 设计高磁矩密度材料
export TARGET_PROPERTY="dft_mag_density"
export TARGET_VALUE="1.5"  # μB/Å³
export CHEMICAL_CONSTRAINTS="Fe,Co,Ni"  # 限制在铁磁性元素
```

#### 专用工作流程
```bash
# 1. 数据筛选
python scripts/filter_training_data.py \
  --property $TARGET_PROPERTY \
  --min_value 1.0 \
  --chemical_elements $CHEMICAL_CONSTRAINTS \
  --output_dir datasets/magnetic_focused/

# 2. 专用微调
mattergen-finetune \
  adapter.pretrained_name=mattergen_base \
  data_module.data_path=datasets/magnetic_focused/ \
  +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.$TARGET_PROPERTY=$TARGET_PROPERTY \
  data_module.properties=["$TARGET_PROPERTY"] \
  trainer.max_epochs=200

# 3. 磁性材料生成
mattergen-generate results/magnetic_materials/ \
  --model_path=$FINETUNED_MODEL \
  --batch_size=32 \
  --num_batches=50 \
  --properties_to_condition_on="{'$TARGET_PROPERTY': $TARGET_VALUE}" \
  --diffusion_guidance_factor=3.0

# 4. 磁性能评估
python scripts/magnetic_analysis.py \
  --structures_path results/magnetic_materials/ \
  --output_dir results/magnetic_analysis/
```

### 场景2: 超硬材料发现

#### 目标设定
```bash
# 设计高体积模量材料
export TARGET_PROPERTY="ml_bulk_modulus"
export TARGET_VALUE="400"  # GPa
export HARDNESS_THRESHOLD="10"  # GPa (Vickers硬度)
```

#### 专用工作流程
```bash
# 1. 多属性优化
mattergen-generate results/superhard_materials/ \
  --pretrained-name=ml_bulk_modulus \
  --batch_size=32 \
  --num_batches=100 \
  --properties_to_condition_on="{'$TARGET_PROPERTY': $TARGET_VALUE}" \
  --diffusion_guidance_factor=2.5

# 2. 硬度预测和筛选
python scripts/predict_hardness.py \
  --structures_path results/superhard_materials/ \
  --model_path models/hardness_predictor.pkl \
  --threshold $HARDNESS_THRESHOLD \
  --output_dir results/superhard_candidates/

# 3. 力学性能分析
python scripts/mechanical_analysis.py \
  --candidates_dir results/superhard_candidates/ \
  --analysis_type "elastic_constants" \
  --output_dir results/mechanical_analysis/
```

### 场景3: 稳定新化合物设计

#### 目标设定
```bash
# 设计低凸包上方能量的新化合物
export TARGET_PROPERTY="energy_above_hull"
export TARGET_VALUE="0.02"  # eV/atom
export NOVELTY_THRESHOLD="0.98"
```

#### 专用工作流程
```bash
# 1. 稳定性约束生成
mattergen-generate results/stable_compounds/ \
  --pretrained-name=chemical_system_energy_above_hull \
  --batch_size=64 \
  --num_batches=200 \
  --properties_to_condition_on="{'$TARGET_PROPERTY': $TARGET_VALUE}" \
  --diffusion_guidance_factor=4.0

# 2. 新颖性筛选
python scripts/novelty_filter.py \
  --structures_path results/stable_compounds/ \
  --reference_databases "MP,OQMD,AFLOW" \
  --novelty_threshold $NOVELTY_THRESHOLD \
  --output_dir results/novel_stable_compounds/

# 3. 热力学稳定性验证
python scripts/stability_analysis.py \
  --candidates_dir results/novel_stable_compounds/ \
  --analysis_methods "convex_hull,phase_diagram" \
  --output_dir results/stability_analysis/
```

## 📊 数据管理

### 数据组织结构

#### 推荐目录结构
```
project_materials_design/
├── data/                          # 原始数据
│   ├── datasets/
│   │   ├── mp_20/                # MP-20 数据集
│   │   ├── alex_mp_20/           # Alex-MP-20 数据集
│   │   └── custom/               # 自定义数据集
│   └── cache/                    # 预处理缓存
├── models/                       # 模型相关
│   ├── pretrained/               # 预训练模型
│   ├── finetuned/               # 微调模型
│   └── checkpoints/             # 训练检查点
├── experiments/                  # 实验记录
│   ├── exp_001_baseline/
│   ├── exp_002_magnetic/
│   └── exp_003_superhard/
├── results/                      # 生成结果
│   ├── structures/               # 生成的结构
│   ├── analysis/                # 分析结果
│   └── reports/                 # 报告文档
├── scripts/                     # 自动化脚本
├── configs/                     # 配置文件
└── docs/                        # 项目文档
```

#### 数据版本控制
```bash
# 使用 DVC 进行数据版本控制
pip install dvc[gdrive]  # 或其他云存储

# 初始化DVC
dvc init

# 添加数据到版本控制
dvc add data/datasets/
dvc add results/

# 提交到Git
git add data/datasets/.dvc results/.dvc .dvcignore
git commit -m "添加数据版本控制"

# 推送数据到远程存储
dvc remote add -d gdrive gdrive://your-gdrive-folder-id
dvc push
```

### 实验记录模板

#### 实验配置文件
```yaml
# experiments/exp_001_baseline/config.yaml
experiment:
  name: "baseline_unconditional_generation"
  description: "基础无条件生成实验"
  date: "2025-01-XX"
  researcher: "研究员姓名"

model:
  name: "mattergen_base"
  type: "pretrained"
  
generation:
  batch_size: 32
  num_batches: 100
  total_structures: 3200
  
evaluation:
  relax: true
  structure_matcher: "disordered"
  
goals:
  - "验证基础生成质量"
  - "建立质量基线"
  - "测试计算资源需求"

expected_results:
  novelty: "> 0.90"
  validity: "> 0.95"
  uniqueness: "> 0.85"
```

#### 实验执行脚本
```bash
#!/bin/bash
# experiments/exp_001_baseline/run_experiment.sh

# 实验环境设置
export EXP_NAME="exp_001_baseline"
export EXP_DIR="experiments/$EXP_NAME"
export RESULTS_DIR="$EXP_DIR/results"

# 记录实验开始
echo "🚀 开始实验: $EXP_NAME" | tee $EXP_DIR/experiment.log
echo "开始时间: $(date)" | tee -a $EXP_DIR/experiment.log

# 生成结构
mattergen-generate $RESULTS_DIR \
  --pretrained-name=mattergen_base \
  --batch_size=32 \
  --num_batches=100 2>&1 | tee -a $EXP_DIR/experiment.log

# 评估结果
mattergen-evaluate \
  --structures_path=$RESULTS_DIR \
  --relax=True \
  --save_as="$RESULTS_DIR/metrics.json" 2>&1 | tee -a $EXP_DIR/experiment.log

# 生成报告
python scripts/generate_experiment_report.py \
  --config $EXP_DIR/config.yaml \
  --results $RESULTS_DIR/metrics.json \
  --output $EXP_DIR/report.html

echo "✅ 实验完成: $(date)" | tee -a $EXP_DIR/experiment.log
```

## 🔄 实验管理

### 实验跟踪系统

#### 使用 MLflow 跟踪实验
```python
# scripts/experiment_tracker.py
import mlflow
import mlflow.pytorch
from pathlib import Path

class ExperimentTracker:
    def __init__(self, experiment_name: str):
        mlflow.set_experiment(experiment_name)
        self.run = mlflow.start_run()
    
    def log_config(self, config: dict):
        """记录实验配置"""
        for key, value in config.items():
            mlflow.log_param(key, value)
    
    def log_metrics(self, metrics: dict):
        """记录评估指标"""
        for key, value in metrics.items():
            mlflow.log_metric(key, value)
    
    def log_model(self, model_path: str):
        """记录模型文件"""
        mlflow.log_artifact(model_path, "model")
    
    def log_results(self, results_dir: str):
        """记录生成结果"""
        mlflow.log_artifacts(results_dir, "results")
    
    def finish(self):
        """结束实验记录"""
        mlflow.end_run()

# 使用示例
tracker = ExperimentTracker("magnetic_materials_design")
tracker.log_config({
    "model": "dft_mag_density",
    "target_value": 1.5,
    "guidance_factor": 3.0
})
# ... 实验执行 ...
tracker.log_metrics(evaluation_results)
tracker.finish()
```

#### 自动化实验比较
```python
# scripts/compare_experiments.py
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def compare_experiments(experiment_dirs: list):
    """比较多个实验的结果"""
    
    results = []
    
    for exp_dir in experiment_dirs:
        # 读取实验配置和结果
        config_file = Path(exp_dir) / "config.yaml"
        metrics_file = Path(exp_dir) / "results" / "metrics.json"
        
        if config_file.exists() and metrics_file.exists():
            config = yaml.load(config_file.read_text())
            metrics = json.load(metrics_file.open())
            
            result = {
                "experiment": config["experiment"]["name"],
                "model": config["model"]["name"],
                **metrics
            }
            results.append(result)
    
    # 创建比较表格
    df = pd.DataFrame(results)
    
    # 生成比较图表
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    df.plot(x="experiment", y="novelty", kind="bar", ax=axes[0,0])
    df.plot(x="experiment", y="validity", kind="bar", ax=axes[0,1])
    df.plot(x="experiment", y="stability", kind="bar", ax=axes[1,0])
    df.plot(x="experiment", y="uniqueness", kind="bar", ax=axes[1,1])
    
    plt.tight_layout()
    plt.savefig("experiments/comparison.png")
    
    return df
```

## 📈 结果分析

### 生成质量分析

#### 综合质量评估脚本
```python
# scripts/quality_analysis.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ase.io import read
import json

class QualityAnalyzer:
    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        self.structures = self.load_structures()
        self.metrics = self.load_metrics()
    
    def load_structures(self):
        """加载生成的结构"""
        extxyz_file = self.results_dir / "generated_crystals.extxyz"
        if extxyz_file.exists():
            return read(str(extxyz_file), ":")
        return []
    
    def load_metrics(self):
        """加载评估指标"""
        metrics_file = self.results_dir / "metrics.json"
        if metrics_file.exists():
            return json.load(metrics_file.open())
        return {}
    
    def analyze_composition_diversity(self):
        """分析组分多样性"""
        compositions = []
        for structure in self.structures:
            symbols = structure.get_chemical_symbols()
            composition = "-".join(sorted(set(symbols)))
            compositions.append(composition)
        
        composition_counts = pd.Series(compositions).value_counts()
        
        plt.figure(figsize=(12, 6))
        composition_counts.head(20).plot(kind='bar')
        plt.title('Top 20 Chemical Compositions')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.results_dir / "composition_diversity.png")
        
        return composition_counts
    
    def analyze_structure_properties(self):
        """分析结构属性分布"""
        properties = {
            'num_atoms': [],
            'density': [],
            'volume': []
        }
        
        for structure in self.structures:
            properties['num_atoms'].append(len(structure))
            properties['density'].append(structure.get_density())
            properties['volume'].append(structure.get_volume())
        
        # 绘制分布图
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for i, (prop, values) in enumerate(properties.items()):
            axes[i].hist(values, bins=30, alpha=0.7)
            axes[i].set_title(f'{prop.replace("_", " ").title()} Distribution')
            axes[i].set_ylabel('Count')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / "structure_properties.png")
        
        return properties
    
    def generate_quality_report(self):
        """生成质量分析报告"""
        report = {
            'total_structures': len(self.structures),
            'metrics_summary': self.metrics,
            'composition_diversity': self.analyze_composition_diversity().to_dict(),
            'structure_properties': self.analyze_structure_properties()
        }
        
        # 保存报告
        with open(self.results_dir / "quality_report.json", 'w') as f:
            json.dump(report, f, indent=2)
        
        return report

# 使用示例
analyzer = QualityAnalyzer("results/production/merged/")
report = analyzer.generate_quality_report()
```

### 属性-结构关系分析

#### 属性关联分析
```python
# scripts/property_structure_analysis.py
class PropertyStructureAnalyzer:
    def __init__(self, structures_file: str, properties_file: str):
        self.structures = read(structures_file, ":")
        self.properties = pd.read_csv(properties_file)
    
    def extract_structural_features(self):
        """提取结构特征"""
        features = []
        
        for structure in self.structures:
            feature = {
                'num_atoms': len(structure),
                'density': structure.get_density(),
                'volume': structure.get_volume(),
                'avg_coordination': self.calc_avg_coordination(structure),
                'space_group': self.get_space_group(structure),
                'composition_complexity': len(set(structure.get_chemical_symbols()))
            }
            features.append(feature)
        
        return pd.DataFrame(features)
    
    def analyze_property_correlations(self, target_property: str):
        """分析属性与结构特征的关联"""
        structural_features = self.extract_structural_features()
        
        # 合并结构特征和属性数据
        data = pd.concat([structural_features, self.properties[target_property]], axis=1)
        
        # 计算相关系数
        correlations = data.corr()[target_property].sort_values(ascending=False)
        
        # 绘制相关性热图
        plt.figure(figsize=(10, 8))
        sns.heatmap(data.corr(), annot=True, cmap='coolwarm', center=0)
        plt.title(f'Property-Structure Correlations for {target_property}')
        plt.tight_layout()
        plt.savefig(f"analysis/{target_property}_correlations.png")
        
        return correlations
    
    def identify_design_rules(self, target_property: str, threshold: float):
        """识别设计规则"""
        data = self.extract_structural_features()
        data[target_property] = self.properties[target_property]
        
        # 筛选高性能样本
        high_performance = data[data[target_property] > threshold]
        
        # 分析高性能样本的共同特征
        rules = {}
        for feature in data.columns:
            if feature != target_property:
                rules[feature] = {
                    'mean': high_performance[feature].mean(),
                    'std': high_performance[feature].std(),
                    'range': (high_performance[feature].min(), high_performance[feature].max())
                }
        
        return rules
```

## 🛠️ 自动化脚本

### 端到端自动化脚本

#### 主控脚本
```python
#!/usr/bin/env python3
# scripts/automated_workflow.py

import argparse
import yaml
import subprocess
import logging
from pathlib import Path
from datetime import datetime

class AutomatedWorkflow:
    def __init__(self, config_file: str):
        self.config = yaml.load(Path(config_file).read_text())
        self.setup_logging()
    
    def setup_logging(self):
        """设置日志系统"""
        log_file = f"workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def run_stage(self, stage_name: str, commands: list):
        """执行工作流程阶段"""
        self.logger.info(f"开始执行阶段: {stage_name}")
        
        for i, command in enumerate(commands):
            self.logger.info(f"执行命令 {i+1}/{len(commands)}: {command}")
            
            try:
                result = subprocess.run(
                    command, 
                    shell=True, 
                    capture_output=True, 
                    text=True,
                    check=True
                )
                self.logger.info(f"命令执行成功")
            except subprocess.CalledProcessError as e:
                self.logger.error(f"命令执行失败: {e}")
                self.logger.error(f"错误输出: {e.stderr}")
                raise
        
        self.logger.info(f"阶段 {stage_name} 完成")
    
    def run_full_workflow(self):
        """执行完整工作流程"""
        workflow_config = self.config['workflow']
        
        for stage in workflow_config['stages']:
            stage_name = stage['name']
            
            if stage.get('enabled', True):
                commands = stage['commands']
                self.run_stage(stage_name, commands)
            else:
                self.logger.info(f"跳过阶段: {stage_name} (已禁用)")
        
        self.logger.info("完整工作流程执行完成！")

def main():
    parser = argparse.ArgumentParser(description='MatterGen 自动化工作流程')
    parser.add_argument('--config', required=True, help='工作流程配置文件')
    args = parser.parse_args()
    
    workflow = AutomatedWorkflow(args.config)
    workflow.run_full_workflow()

if __name__ == "__main__":
    main()
```

#### 工作流程配置模板
```yaml
# configs/workflow_template.yaml
workflow:
  name: "automated_materials_design"
  description: "自动化材料设计工作流程"
  
  variables:
    MODEL_NAME: "mattergen_base"
    TARGET_PROPERTY: "dft_mag_density"
    TARGET_VALUE: "1.5"
    BATCH_SIZE: "32"
    NUM_BATCHES: "50"
    RESULTS_PATH: "results/automated_run"
  
  stages:
    - name: "environment_setup"
      enabled: true
      commands:
        - "source .venv/bin/activate"
        - "python -c 'import mattergen; print(\"✅ Environment ready\")'"
    
    - name: "data_preparation"
      enabled: true
      commands:
        - "git lfs pull -I data-release/alex-mp/alex_mp_20.zip --exclude=''"
        - "unzip -o data-release/alex-mp/alex_mp_20.zip -d datasets"
        - "csv-to-dataset --csv-folder datasets/alex_mp_20/ --dataset-name alex_mp_20 --cache-folder datasets/cache"
    
    - name: "model_finetuning"
      enabled: false  # 可选阶段
      commands:
        - "mattergen-finetune adapter.pretrained_name=${MODEL_NAME} data_module=alex_mp_20 +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.${TARGET_PROPERTY}=${TARGET_PROPERTY} ~trainer.logger data_module.properties=[\"${TARGET_PROPERTY}\"]"
    
    - name: "structure_generation"
      enabled: true
      commands:
        - "mkdir -p ${RESULTS_PATH}"
        - "mattergen-generate ${RESULTS_PATH} --pretrained-name=${MODEL_NAME} --batch_size=${BATCH_SIZE} --num_batches=${NUM_BATCHES} --properties_to_condition_on='{\"${TARGET_PROPERTY}\": ${TARGET_VALUE}}' --diffusion_guidance_factor=2.0"
    
    - name: "quality_evaluation"
      enabled: true
      commands:
        - "git lfs pull -I data-release/alex-mp/reference_MP2020correction.gz --exclude=''"
        - "mattergen-evaluate --structures_path=${RESULTS_PATH} --relax=True --save_as='${RESULTS_PATH}/metrics.json'"
    
    - name: "result_analysis"
      enabled: true
      commands:
        - "python scripts/quality_analysis.py --results_dir ${RESULTS_PATH}"
        - "python scripts/generate_report.py --metrics_file ${RESULTS_PATH}/metrics.json --output_file ${RESULTS_PATH}/report.html"
```

### 批量实验管理

#### 参数扫描脚本
```python
# scripts/parameter_sweep.py
import itertools
import yaml
from pathlib import Path

class ParameterSweep:
    def __init__(self, base_config: str, sweep_config: str):
        self.base_config = yaml.load(Path(base_config).read_text())
        self.sweep_config = yaml.load(Path(sweep_config).read_text())
    
    def generate_experiments(self):
        """生成所有参数组合的实验配置"""
        
        sweep_params = self.sweep_config['parameters']
        param_names = list(sweep_params.keys())
        param_values = list(sweep_params.values())
        
        experiments = []
        
        for i, combination in enumerate(itertools.product(*param_values)):
            # 创建实验配置
            exp_config = self.base_config.copy()
            exp_name = f"sweep_exp_{i:03d}"
            
            # 应用参数组合
            for param_name, param_value in zip(param_names, combination):
                self.set_nested_param(exp_config, param_name, param_value)
            
            # 设置实验名称和输出目录
            exp_config['experiment']['name'] = exp_name
            exp_config['output_dir'] = f"experiments/{exp_name}"
            
            experiments.append((exp_name, exp_config))
        
        return experiments
    
    def set_nested_param(self, config: dict, param_path: str, value):
        """设置嵌套参数值"""
        keys = param_path.split('.')
        current = config
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value
    
    def run_sweep(self):
        """执行参数扫描"""
        experiments = self.generate_experiments()
        
        for exp_name, exp_config in experiments:
            print(f"🚀 执行实验: {exp_name}")
            
            # 保存实验配置
            exp_dir = Path(f"experiments/{exp_name}")
            exp_dir.mkdir(parents=True, exist_ok=True)
            
            config_file = exp_dir / "config.yaml"
            with open(config_file, 'w') as f:
                yaml.dump(exp_config, f, default_flow_style=False)
            
            # 执行实验
            workflow = AutomatedWorkflow(str(config_file))
            workflow.run_full_workflow()
            
            print(f"✅ 实验 {exp_name} 完成")

# 参数扫描配置示例
# configs/sweep_config.yaml
"""
parameters:
  generation.guidance_factor: [1.0, 2.0, 3.0, 5.0]
  generation.batch_size: [16, 32, 64]
  model.target_property_value: [1.0, 1.5, 2.0]
"""
```

## 📚 最佳实践

### 计算资源优化

#### GPU 内存管理
```python
# scripts/memory_optimizer.py
import torch
import psutil
from pathlib import Path

class MemoryOptimizer:
    def __init__(self):
        self.gpu_memory = self.get_gpu_memory()
        self.cpu_memory = self.get_cpu_memory()
    
    def get_gpu_memory(self):
        """获取GPU内存信息"""
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / 1e9
        return 0
    
    def get_cpu_memory(self):
        """获取CPU内存信息"""
        return psutil.virtual_memory().total / 1e9
    
    def recommend_batch_size(self, model_size: str = "base"):
        """推荐批次大小"""
        
        memory_requirements = {
            "base": {"gpu": 8, "cpu": 16},      # GB
            "large": {"gpu": 16, "cpu": 32},
            "xl": {"gpu": 32, "cpu": 64}
        }
        
        req = memory_requirements.get(model_size, memory_requirements["base"])
        
        if self.gpu_memory >= req["gpu"] and self.cpu_memory >= req["cpu"]:
            batch_sizes = {"base": 32, "large": 16, "xl": 8}
            return batch_sizes.get(model_size, 16)
        else:
            # 内存不足时的降级建议
            return max(1, int(self.gpu_memory / req["gpu"] * 16))
    
    def optimize_generation_config(self, target_structures: int):
        """优化生成配置"""
        recommended_batch_size = self.recommend_batch_size()
        num_batches = target_structures // recommended_batch_size
        
        config = {
            "batch_size": recommended_batch_size,
            "num_batches": num_batches,
            "gradient_checkpointing": self.gpu_memory < 16,
            "mixed_precision": True,
            "num_workers": min(8, psutil.cpu_count())
        }
        
        return config
```

#### 分布式计算配置
```bash
# scripts/distributed_setup.sh
#!/bin/bash

# 多GPU训练配置
export MASTER_ADDR=localhost
export MASTER_PORT=12355
export WORLD_SIZE=4  # GPU数量

# 启动分布式训练
torchrun --nproc_per_node=4 scripts/distributed_train.py \
  --config configs/distributed_config.yaml

# 多节点配置 (如果有多台机器)
# 主节点
torchrun --nnodes=2 --node_rank=0 --master_addr=MASTER_IP --master_port=12355 --nproc_per_node=4 scripts/distributed_train.py

# 从节点
torchrun --nnodes=2 --node_rank=1 --master_addr=MASTER_IP --master_port=12355 --nproc_per_node=4 scripts/distributed_train.py
```

### 质量控制检查点

#### 质量检查脚本
```python
# scripts/quality_checkpoints.py
class QualityCheckpoints:
    def __init__(self, config: dict):
        self.config = config
        self.checkpoints = [
            self.check_data_integrity,
            self.check_model_performance,
            self.check_generation_quality,
            self.check_evaluation_metrics
        ]
    
    def run_all_checks(self):
        """运行所有质量检查"""
        results = {}
        
        for checkpoint in self.checkpoints:
            check_name = checkpoint.__name__
            try:
                result = checkpoint()
                results[check_name] = {"status": "pass", "details": result}
                print(f"✅ {check_name}: PASS")
            except Exception as e:
                results[check_name] = {"status": "fail", "error": str(e)}
                print(f"❌ {check_name}: FAIL - {e}")
        
        return results
    
    def check_data_integrity(self):
        """检查数据完整性"""
        data_path = Path(self.config['data_path'])
        
        required_files = ['train/', 'val/']
        for file_path in required_files:
            if not (data_path / file_path).exists():
                raise FileNotFoundError(f"Missing required data: {file_path}")
        
        return {"data_files": "complete"}
    
    def check_model_performance(self):
        """检查模型性能"""
        # 运行小规模测试
        test_result = self.run_model_test()
        
        if test_result['loss'] > self.config['max_acceptable_loss']:
            raise ValueError(f"Model loss too high: {test_result['loss']}")
        
        return test_result
    
    def check_generation_quality(self):
        """检查生成质量"""
        # 生成小批量样本进行质量检查
        test_structures = self.generate_test_structures()
        
        validity_rate = self.calculate_validity_rate(test_structures)
        if validity_rate < self.config['min_validity_rate']:
            raise ValueError(f"Validity rate too low: {validity_rate}")
        
        return {"validity_rate": validity_rate}
    
    def check_evaluation_metrics(self):
        """检查评估指标"""
        metrics = self.load_evaluation_metrics()
        
        for metric, threshold in self.config['metric_thresholds'].items():
            if metrics.get(metric, 0) < threshold:
                raise ValueError(f"Metric {metric} below threshold: {metrics[metric]} < {threshold}")
        
        return metrics
```

### 错误处理和恢复

#### 自动重试机制
```python
# scripts/robust_executor.py
import time
import traceback
from functools import wraps

def retry_on_failure(max_retries=3, delay=60):
    """装饰器：失败时自动重试"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    print(f"❌ 尝试 {attempt + 1}/{max_retries} 失败: {e}")
                    
                    if attempt < max_retries - 1:
                        print(f"⏳ {delay}秒后重试...")
                        time.sleep(delay)
                    else:
                        print("❌ 所有重试均失败")
                        traceback.print_exc()
                        raise
            
        return wrapper
    return decorator

@retry_on_failure(max_retries=3, delay=120)
def robust_generation(config):
    """带重试机制的稳健生成"""
    return run_generation(config)

@retry_on_failure(max_retries=2, delay=60)
def robust_evaluation(structures_path):
    """带重试机制的稳健评估"""
    return run_evaluation(structures_path)
```

#### 检查点恢复
```python
# scripts/checkpoint_manager.py
class CheckpointManager:
    def __init__(self, experiment_dir: str):
        self.experiment_dir = Path(experiment_dir)
        self.checkpoint_file = self.experiment_dir / "progress.json"
    
    def save_progress(self, stage: str, status: str, data: dict = None):
        """保存进度检查点"""
        progress = self.load_progress()
        
        progress[stage] = {
            "status": status,
            "timestamp": datetime.now().isoformat(),
            "data": data or {}
        }
        
        with open(self.checkpoint_file, 'w') as f:
            json.dump(progress, f, indent=2)
    
    def load_progress(self):
        """加载进度检查点"""
        if self.checkpoint_file.exists():
            return json.load(self.checkpoint_file.open())
        return {}
    
    def get_next_stage(self, stages: list):
        """获取下一个需要执行的阶段"""
        progress = self.load_progress()
        
        for stage in stages:
            if stage not in progress or progress[stage]["status"] != "completed":
                return stage
        
        return None  # 所有阶段都已完成
    
    def resume_workflow(self, workflow_config):
        """从检查点恢复工作流程"""
        stages = workflow_config['stages']
        next_stage = self.get_next_stage([s['name'] for s in stages])
        
        if next_stage:
            print(f"🔄 从阶段 '{next_stage}' 恢复工作流程")
            return next_stage
        else:
            print("✅ 所有阶段已完成")
            return None
```

---

## 🎯 总结

### 工作流程优势
1. **标准化**: 提供可重复的标准流程
2. **自动化**: 减少人工干预和错误
3. **可追踪**: 完整的实验记录和版本控制
4. **可扩展**: 支持不同应用场景的定制

### 关键成功要素
1. **充分的前期规划**: 明确目标和约束条件
2. **合适的计算资源**: 根据需求配置硬件
3. **严格的质量控制**: 在每个阶段进行质量检查
4. **完善的文档记录**: 便于结果复现和分析

### 持续改进方向
1. **流程自动化**: 进一步减少手动操作
2. **智能调优**: 基于历史数据自动优化参数
3. **云端部署**: 支持大规模云计算环境
4. **社区集成**: 与材料数据库和工具生态集成

---

## 📞 获取帮助

- **主文档**: [README_CN.md](README_CN.md)
- **微调指南**: [README_FINETUNE_CN.md](README_FINETUNE_CN.md)
- **架构文档**: [README_ARCHITECTURE_CN.md](README_ARCHITECTURE_CN.md)
- **GitHub Issues**: [https://github.com/microsoft/mattergen/issues](https://github.com/microsoft/mattergen/issues)

*本工作流程指南基于 MatterGen v1.0 编写*