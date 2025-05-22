# MatterGen 微调指南

<p align="center">
    <img src="assets/MatterGenlogo_.png" alt="MatterGen logo" width="400"/>
</p>

本指南详细介绍如何对 MatterGen 模型进行微调，以支持特定的材料属性预测和条件生成。

## 📋 目录

- [🎯 微调概述](#-微调概述)
- [⚙️ 环境配置](#-环境配置)
- [📊 支持的属性](#-支持的属性)
- [🔧 单属性微调](#-单属性微调)
- [🎨 多属性微调](#-多属性微调)
- [📈 训练监控](#-训练监控)
- [🔮 微调模型使用](#-微调模型使用)
- [⚡ 性能优化](#-性能优化)
- [❓ 常见问题](#-常见问题)

## 🎯 微调概述

### 什么是微调？
微调是在预训练的基础模型上，针对特定材料属性进行进一步训练的过程。这使得模型能够：

- **属性条件生成**: 根据指定的材料属性生成满足条件的晶体结构
- **更好的属性预测**: 对特定属性有更精确的理解和预测能力
- **定制化应用**: 针对特定研究领域或应用场景优化

### 微调 vs 从头训练

| 方面 | 微调 | 从头训练 |
|------|------|----------|
| **训练时间** | 数小时到1天 | 1-3天 |
| **数据需求** | 较少 (基础模型已学习通用特征) | 更多 |
| **计算资源** | 较低 | 较高 |
| **性能** | 在特定属性上更优 | 通用性更强 |
| **推荐场景** | 特定属性应用 | 新的模型架构或数据 |

## ⚙️ 环境配置

### 前置条件
确保已完成基础安装 (参考 [README_CN.md](README_CN.md))：

```bash
# 激活环境
source .venv/bin/activate

# 验证安装
python -c "import mattergen; print('✅ MatterGen 已安装')"
mattergen-finetune --help
```

### 数据准备
微调建议使用 Alex-MP-20 数据集，因为它包含更多的材料属性：

```bash
# 下载 Alex-MP-20 数据集
git lfs pull -I data-release/alex-mp/alex_mp_20.zip --exclude=""
unzip data-release/alex-mp/alex_mp_20.zip -d datasets

# 预处理数据
csv-to-dataset --csv-folder datasets/alex_mp_20/ --dataset-name alex_mp_20 --cache-folder datasets/cache
```

## 📊 支持的属性

MatterGen 支持以下材料属性的微调：

### 连续数值属性

| 属性ID | 属性名称 | 单位 | 范围 | 描述 |
|--------|----------|------|------|------|
| `dft_mag_density` | DFT磁密度 | μB/Å³ | 0-2.0 | 每单位体积的磁矩 |
| `dft_band_gap` | DFT带隙 | eV | 0-10.0 | 导带和价带之间的能量差 |
| `dft_bulk_modulus` | DFT体积模量 | GPa | 0-500 | 材料的压缩阻力 |
| `ml_bulk_modulus` | ML体积模量 | GPa | 0-500 | 机器学习预测的体积模量 |
| `energy_above_hull` | 凸包上方能量 | eV/atom | 0-1.0 | 热力学稳定性指标 |
| `hhi_score` | HHI稀缺性得分 | - | 0-1.0 | 元素稀缺性评分 |

### 分类属性

| 属性ID | 属性名称 | 取值类型 | 示例 | 描述 |
|--------|----------|----------|------|------|
| `chemical_system` | 化学体系 | 字符串 | "Li-O", "Fe-Ni-Al" | 材料的化学组分 |
| `space_group` | 空间群 | 整数 | 1-230 | 晶体的对称性分类 |

### 查看属性分布

```bash
# 查看数据集中的属性分布
python -c "
import pandas as pd
import numpy as np

# 读取训练数据
data = pd.read_csv('datasets/alex_mp_20/train.csv')
print('📊 数据集统计:')
print(f'总样本数: {len(data)}')

# 显示各属性的统计信息
properties = ['dft_mag_density', 'dft_band_gap', 'energy_above_hull', 'chemical_system']
for prop in properties:
    if prop in data.columns:
        if prop == 'chemical_system':
            print(f'\n{prop}: {data[prop].nunique()} 种不同组分')
            print(data[prop].value_counts().head())
        else:
            print(f'\n{prop}:')
            print(f'  范围: {data[prop].min():.3f} - {data[prop].max():.3f}')
            print(f'  平均: {data[prop].mean():.3f}')
            print(f'  标准差: {data[prop].std():.3f}')
"
```

## 🔧 单属性微调

### 基础微调命令结构

```bash
export PROPERTY=<属性名称>
mattergen-finetune \
  adapter.pretrained_name=mattergen_base \
  data_module=alex_mp_20 \
  +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.$PROPERTY=$PROPERTY \
  ~trainer.logger \
  data_module.properties=["$PROPERTY"]
```

### 具体属性微调示例

#### 1. 磁密度微调
```bash
export PROPERTY=dft_mag_density
mattergen-finetune \
  adapter.pretrained_name=mattergen_base \
  data_module=alex_mp_20 \
  +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.$PROPERTY=$PROPERTY \
  ~trainer.logger \
  data_module.properties=["$PROPERTY"]
```

#### 2. 带隙微调
```bash
export PROPERTY=dft_band_gap
mattergen-finetune \
  adapter.pretrained_name=mattergen_base \
  data_module=alex_mp_20 \
  +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.$PROPERTY=$PROPERTY \
  ~trainer.logger \
  data_module.properties=["$PROPERTY"]
```

#### 3. 化学体系微调
```bash
export PROPERTY=chemical_system
mattergen-finetune \
  adapter.pretrained_name=mattergen_base \
  data_module=alex_mp_20 \
  +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.$PROPERTY=$PROPERTY \
  ~trainer.logger \
  data_module.properties=["$PROPERTY"]
```

#### 4. 空间群微调
```bash
export PROPERTY=space_group
mattergen-finetune \
  adapter.pretrained_name=mattergen_base \
  data_module=alex_mp_20 \
  +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.$PROPERTY=$PROPERTY \
  ~trainer.logger \
  data_module.properties=["$PROPERTY"]
```

## 🎨 多属性微调

### 双属性微调

#### 磁密度 + HHI得分
```bash
export PROPERTY1=dft_mag_density
export PROPERTY2=hhi_score

mattergen-finetune \
  adapter.pretrained_name=mattergen_base \
  data_module=alex_mp_20 \
  +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.$PROPERTY1=$PROPERTY1 \
  +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.$PROPERTY2=$PROPERTY2 \
  ~trainer.logger \
  data_module.properties=["$PROPERTY1","$PROPERTY2"]
```

#### 化学体系 + 凸包上方能量
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
export PROPERTY3=chemical_system

mattergen-finetune \
  adapter.pretrained_name=mattergen_base \
  data_module=alex_mp_20 \
  +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.$PROPERTY1=$PROPERTY1 \
  +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.$PROPERTY2=$PROPERTY2 \
  +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.$PROPERTY3=$PROPERTY3 \
  ~trainer.logger \
  data_module.properties=["$PROPERTY1","$PROPERTY2","$PROPERTY3"]
```

## 📈 训练监控

### 训练输出结构
```
outputs/singlerun/{date}/{time}/
├── checkpoints/
│   ├── epoch=N-step=M.ckpt    # 训练检查点
│   └── last.ckpt              # 最后一个检查点
├── lightning_logs/
│   └── version_0/
│       ├── events.out.tfevents.*  # TensorBoard日志
│       └── hparams.yaml           # 超参数配置
└── config.yaml                # 完整运行配置
```

### TensorBoard 监控
```bash
# 启动 TensorBoard
tensorboard --logdir outputs/singlerun/

# 在浏览器中访问
echo "📊 访问: http://localhost:6006"
```

### 关键训练指标

#### 损失函数监控
- **总损失 (total_loss)**: 综合损失，应持续下降
- **属性损失 (property_loss)**: 属性预测损失，应收敛到较低值
- **扩散损失 (diffusion_loss)**: 结构生成损失，应稳定

#### 验证指标
- **验证损失**: 应在训练损失附近，不应持续上升 (过拟合警告)
- **属性准确性**: 对于分类属性，查看分类准确率
- **属性MAE/MSE**: 对于连续属性，查看平均绝对误差

### 训练时间参考

| 属性数量 | 数据集 | 预期时间 (单GPU) | 内存需求 |
|----------|--------|------------------|----------|
| 1个属性 | Alex-MP-20 | 4-8小时 | 16GB+ |
| 2个属性 | Alex-MP-20 | 6-12小时 | 20GB+ |
| 3个属性 | Alex-MP-20 | 8-16小时 | 24GB+ |

### 早停和检查点
```bash
# 启用早停
mattergen-finetune \
  adapter.pretrained_name=mattergen_base \
  data_module=alex_mp_20 \
  +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.$PROPERTY=$PROPERTY \
  ~trainer.logger \
  data_module.properties=["$PROPERTY"] \
  trainer.enable_checkpointing=true \
  trainer.callbacks.early_stopping.patience=5
```

## 🔮 微调模型使用

### 保存微调模型
训练完成后，检查点自动保存在 `outputs/` 目录中。找到最佳检查点：

```bash
# 查找最新的训练输出
LATEST_RUN=$(ls -t outputs/singlerun/ | head -1)
CHECKPOINT_DIR="outputs/singlerun/$LATEST_RUN/checkpoints"

echo "📁 检查点目录: $CHECKPOINT_DIR"
ls -la $CHECKPOINT_DIR

# 通常使用 last.ckpt 或 epoch=*-step=*.ckpt
export FINETUNED_MODEL="$CHECKPOINT_DIR/last.ckpt"
```

### 使用微调模型生成

#### 单属性条件生成
```bash
# 使用微调的磁密度模型
export RESULTS_PATH=results/finetuned_mag_density/

mattergen-generate $RESULTS_PATH \
  --model_path=$FINETUNED_MODEL \
  --batch_size=16 \
  --properties_to_condition_on="{'dft_mag_density': 0.15}" \
  --diffusion_guidance_factor=2.0
```

#### 多属性条件生成
```bash
# 使用多属性微调模型
export RESULTS_PATH=results/multi_property/

mattergen-generate $RESULTS_PATH \
  --model_path=$FINETUNED_MODEL \
  --batch_size=16 \
  --properties_to_condition_on="{'dft_mag_density': 0.15, 'chemical_system': 'Fe-O'}" \
  --diffusion_guidance_factor=2.0
```

### 模型部署和分享

#### 创建可分享的模型包
```bash
# 创建模型包目录
mkdir -p model_package/$PROPERTY

# 复制检查点和配置
cp $FINETUNED_MODEL model_package/$PROPERTY/
cp outputs/singlerun/$LATEST_RUN/config.yaml model_package/$PROPERTY/

# 创建README
cat > model_package/$PROPERTY/README.md << EOF
# MatterGen 微调模型: $PROPERTY

## 模型信息
- 基础模型: mattergen_base
- 微调属性: $PROPERTY
- 训练数据: Alex-MP-20
- 训练时间: $(date)

## 使用方法
\`\`\`bash
mattergen-generate results/ \\
  --model_path=last.ckpt \\
  --properties_to_condition_on="{'$PROPERTY': <value>}" \\
  --diffusion_guidance_factor=2.0
\`\`\`
EOF

echo "📦 模型包创建完成: model_package/$PROPERTY/"
```

## ⚡ 性能优化

### 内存优化
```bash
# 启用梯度检查点 (节省内存)
mattergen-finetune \
  adapter.pretrained_name=mattergen_base \
  data_module=alex_mp_20 \
  +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.$PROPERTY=$PROPERTY \
  ~trainer.logger \
  data_module.properties=["$PROPERTY"] \
  lightning_module.diffusion_module.model.gradient_checkpointing=true

# 减少批次大小
mattergen-finetune \
  adapter.pretrained_name=mattergen_base \
  data_module=alex_mp_20 \
  +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.$PROPERTY=$PROPERTY \
  ~trainer.logger \
  data_module.properties=["$PROPERTY"] \
  trainer.accumulate_grad_batches=8

# 使用混合精度
mattergen-finetune \
  adapter.pretrained_name=mattergen_base \
  data_module=alex_mp_20 \
  +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.$PROPERTY=$PROPERTY \
  ~trainer.logger \
  data_module.properties=["$PROPERTY"] \
  trainer.precision=16
```

### 多GPU 微调
```bash
# 数据并行 (推荐)
mattergen-finetune \
  adapter.pretrained_name=mattergen_base \
  data_module=alex_mp_20 \
  +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.$PROPERTY=$PROPERTY \
  ~trainer.logger \
  data_module.properties=["$PROPERTY"] \
  trainer.devices=4 \
  trainer.strategy=ddp

# 模型并行 (大内存需求时)
mattergen-finetune \
  adapter.pretrained_name=mattergen_base \
  data_module=alex_mp_20 \
  +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.$PROPERTY=$PROPERTY \
  ~trainer.logger \
  data_module.properties=["$PROPERTY"] \
  trainer.devices=4 \
  trainer.strategy=deepspeed_stage_2
```

### 学习率调优
```bash
# 自定义学习率
mattergen-finetune \
  adapter.pretrained_name=mattergen_base \
  data_module=alex_mp_20 \
  +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.$PROPERTY=$PROPERTY \
  ~trainer.logger \
  data_module.properties=["$PROPERTY"] \
  lightning_module.lr=1e-5

# 学习率调度
mattergen-finetune \
  adapter.pretrained_name=mattergen_base \
  data_module=alex_mp_20 \
  +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.$PROPERTY=$PROPERTY \
  ~trainer.logger \
  data_module.properties=["$PROPERTY"] \
  lightning_module.lr_scheduler.factor=0.8 \
  lightning_module.lr_scheduler.patience=3
```

## ❓ 常见问题

### 配置相关

#### Q: 如何查看支持的属性列表？
```bash
# 查看支持的属性
python -c "
from mattergen.common.utils.globals import PROPERTY_SOURCE_IDS
print('支持的属性:')
for prop in PROPERTY_SOURCE_IDS:
    print(f'  - {prop}')
"
```

#### Q: 如何修改微调参数？
```bash
# 查看完整配置
mattergen-finetune \
  adapter.pretrained_name=mattergen_base \
  data_module=alex_mp_20 \
  +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.$PROPERTY=$PROPERTY \
  data_module.properties=["$PROPERTY"] \
  --cfg job
```

### 训练相关

#### Q: 微调损失不下降怎么办？
```bash
# 1. 检查学习率
mattergen-finetune ... lightning_module.lr=1e-6  # 降低学习率

# 2. 检查数据
python -c "
import pandas as pd
data = pd.read_csv('datasets/alex_mp_20/train.csv')
print(f'属性 {PROPERTY} 的有效样本: {data[PROPERTY].notna().sum()}')
print(f'属性分布: {data[PROPERTY].describe()}')
"

# 3. 增加训练步数
mattergen-finetune ... trainer.max_epochs=20
```

#### Q: 如何处理内存不足？
```bash
# 组合多种策略
mattergen-finetune \
  adapter.pretrained_name=mattergen_base \
  data_module=alex_mp_20 \
  +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.$PROPERTY=$PROPERTY \
  ~trainer.logger \
  data_module.properties=["$PROPERTY"] \
  trainer.precision=16 \
  trainer.accumulate_grad_batches=16 \
  lightning_module.diffusion_module.model.gradient_checkpointing=true \
  data_module.batch_size.train=4
```

### 使用相关

#### Q: 如何选择合适的引导强度？
建议的引导强度范围：

| 属性类型 | 推荐范围 | 备注 |
|----------|----------|------|
| 连续属性 | 1.0-3.0 | 从低开始尝试 |
| 分类属性 | 2.0-5.0 | 可以设置更高 |
| 多属性 | 1.5-2.5 | 避免属性间冲突 |

#### Q: 生成的结构不满足条件怎么办？
```bash
# 1. 增加引导强度
--diffusion_guidance_factor=3.0

# 2. 增加生成批次，筛选合适的结构
--num_batches=20

# 3. 检查属性值是否在训练数据范围内
python -c "
import pandas as pd
data = pd.read_csv('datasets/alex_mp_20/train.csv')
prop = '$PROPERTY'
target_value = 0.15  # 你的目标值
print(f'训练数据中 {prop} 的范围: {data[prop].min():.3f} - {data[prop].max():.3f}')
print(f'目标值 {target_value} 是否在范围内: {data[prop].min() <= target_value <= data[prop].max()}')
"
```

## 📚 进阶主题

### 自定义属性微调
如果需要对数据集中不存在的属性进行微调，需要：

1. **准备数据**: 添加新属性列到数据集
2. **配置属性**: 在 `globals.py` 中添加属性ID
3. **创建配置**: 为新属性创建embedding配置文件
4. **修改数据模块**: 确保数据模块能加载新属性

详细步骤请参考 [架构解析文档](README_ARCHITECTURE_CN.md)。

### 微调效果评估
```bash
# 生成测试样本
mattergen-generate test_results/ --model_path=$FINETUNED_MODEL --batch_size=32 --num_batches=10

# 评估生成质量
mattergen-evaluate --structures_path=test_results/ --relax=True --save_as=test_metrics.json

# 分析属性分布
python scripts/analyze_properties.py --structures=test_results/ --target_property=$PROPERTY
```

---

## 📞 获取帮助

- **主文档**: [README_CN.md](README_CN.md)
- **架构文档**: [README_ARCHITECTURE_CN.md](README_ARCHITECTURE_CN.md)
- **GitHub Issues**: [https://github.com/microsoft/mattergen/issues](https://github.com/microsoft/mattergen/issues)

*本微调指南基于 MatterGen v1.0 编写*