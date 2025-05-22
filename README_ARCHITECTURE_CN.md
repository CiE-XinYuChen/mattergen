# MatterGen 架构解析

<p align="center">
    <img src="assets/MatterGenlogo_.png" alt="MatterGen logo" width="400"/>
</p>

本文档深入解析 MatterGen 的模型架构、设计理念和核心组件，帮助开发者理解系统的内部工作机制。

## 📋 目录

- [🏗️ 整体架构](#-整体架构)
- [🧠 扩散模型核心](#-扩散模型核心)
- [🔬 GemNet 神经网络](#-gemnet-神经网络)
- [🎯 属性嵌入系统](#-属性嵌入系统)
- [🔄 数据流处理](#-数据流处理)
- [⚙️ 配置系统](#-配置系统)
- [🚀 微调机制](#-微调机制)
- [🔧 扩展开发](#-扩展开发)

## 🏗️ 整体架构

### 系统层级结构

MatterGen 采用模块化分层架构设计：

```
MatterGen 系统架构
├── 🎯 应用层 (Applications)
│   ├── 条件生成 (Conditional Generation)
│   ├── 结构预测 (Structure Prediction)  
│   └── 属性优化 (Property Optimization)
├── 🧠 模型层 (Model Layer)
│   ├── 扩散模块 (Diffusion Module)
│   ├── GemNet 骨干 (GemNet Backbone)
│   └── 属性嵌入 (Property Embeddings)
├── 📊 数据层 (Data Layer)
│   ├── 数据集管理 (Dataset Management)
│   ├── 预处理管道 (Preprocessing Pipeline)
│   └── 批处理系统 (Batching System)
└── ⚙️ 基础设施 (Infrastructure)
    ├── 配置管理 (Configuration Management)
    ├── 训练框架 (Training Framework)
    └── 评估系统 (Evaluation System)
```

### 核心设计理念

#### 1. 组合性 (Composability)
- **模块解耦**: 各组件独立开发和测试
- **接口标准化**: 统一的数据接口和调用约定
- **可插拔设计**: 支持不同组件的灵活组合

#### 2. 可扩展性 (Extensibility)
- **属性系统**: 支持新材料属性的动态添加
- **模型适配**: 支持不同神经网络骨干的集成
- **评估指标**: 支持自定义评估标准和指标

#### 3. 可重现性 (Reproducibility)
- **配置驱动**: 所有实验通过配置文件完全描述
- **种子管理**: 确保随机数生成的可重现性
- **版本控制**: 模型和数据的版本化管理

## 🧠 扩散模型核心

### 扩散过程原理

MatterGen 基于离散扩散模型 (Discrete Diffusion Models) 进行晶体结构生成：

#### 前向扩散 (Forward Process)
```python
# 伪代码示例
def forward_diffusion(x0, t, noise_schedule):
    """
    将干净数据 x0 在时间步 t 处加入噪声
    
    Args:
        x0: 原始干净数据 (晶体结构)
        t: 时间步 (0 到 T)
        noise_schedule: 噪声调度策略
    
    Returns:
        xt: 时间步 t 的噪声数据
        noise: 添加的噪声
    """
    noise = sample_noise(x0.shape)
    xt = add_noise(x0, noise, t, noise_schedule)
    return xt, noise
```

#### 反向去噪 (Reverse Process)
```python
def reverse_diffusion(xt, t, model, conditions=None):
    """
    从噪声数据 xt 预测原始数据 x0
    
    Args:
        xt: 噪声数据
        t: 当前时间步
        model: 去噪模型 (GemNet)
        conditions: 条件信息 (材料属性)
    
    Returns:
        x0_pred: 预测的原始数据
    """
    # 使用模型预测噪声或原始数据
    prediction = model(xt, t, conditions)
    x0_pred = denoise_prediction(xt, prediction, t)
    return x0_pred
```

### 扩散模块组件

#### 1. 损坏策略 (Corruption Strategies)
```
mattergen/diffusion/corruption/
├── corruption.py          # 基础损坏接口
├── d3pm_corruption.py     # D3PM 离散损坏
├── multi_corruption.py    # 多字段损坏组合
└── sde_lib.py            # 随机微分方程库
```

**D3PM 损坏** (用于原子类型):
- 原子类型从真实值逐渐转变为随机分布
- 支持不同的转移矩阵设计
- 保持周期表的化学约束

**连续损坏** (用于坐标和晶胞):
- 原子坐标加入高斯噪声
- 晶胞参数进行尺度扰动
- 保持晶体学约束

#### 2. 采样算法 (Sampling Algorithms)
```
mattergen/diffusion/sampling/
├── pc_sampler.py              # 主采样器
├── predictors.py              # 预测器算法
├── predictors_correctors.py   # 预测-校正算法
└── classifier_free_guidance.py # 无分类器引导
```

**预测-校正采样**:
```python
def pc_sampling(model, conditions, num_steps=1000):
    """预测-校正采样算法"""
    x = initialize_noise()
    
    for t in reversed(range(num_steps)):
        # 预测步骤
        x = predictor_step(x, t, model, conditions)
        
        # 校正步骤 (可选)
        x = corrector_step(x, t, model, conditions)
    
    return x
```

### 数据表示

#### 晶体结构编码
```python
class CrystalStructure:
    """晶体结构的内部表示"""
    
    def __init__(self):
        self.atomic_numbers: torch.Tensor  # [N] 原子类型
        self.pos: torch.Tensor            # [N, 3] 分数坐标
        self.cell: torch.Tensor           # [3, 3] 晶胞矩阵
        self.num_atoms: int               # 原子数量
        
    def to_graph(self):
        """转换为图神经网络输入格式"""
        # 构建原子间连接
        edges = build_crystal_graph(self.pos, self.cell)
        return AtomGraph(
            node_features=self.atomic_numbers,
            edge_indices=edges,
            positions=self.pos,
            cell=self.cell
        )
```

## 🔬 GemNet 神经网络

### 网络架构设计

GemNet (Geometric Message Passing Neural Network) 是 MatterGen 的核心神经网络骨干：

#### 整体结构
```
GemNet 架构
├── 📥 嵌入层 (Embedding Layers)
│   ├── 原子嵌入 (Atom Embedding)
│   ├── 边嵌入 (Edge Embedding)
│   └── 角度嵌入 (Angle Embedding)
├── 🔄 交互层 (Interaction Blocks)
│   ├── 消息传递 (Message Passing)
│   ├── 原子更新 (Atom Update)
│   └── 几何感知 (Geometric Awareness)
├── 🎯 输出层 (Output Layers)
│   ├── 结构预测 (Structure Prediction)
│   ├── 属性预测 (Property Prediction)
│   └── 噪声估计 (Noise Estimation)
└── 🔧 控制机制 (Control Mechanisms)
    ├── 条件输入 (Conditional Input)
    ├── 时间步编码 (Timestep Encoding)
    └── 注意力机制 (Attention Mechanisms)
```

#### 代码实现概览
```python
class GemNetT(nn.Module):
    """GemNet-T 主体架构"""
    
    def __init__(self, 
                 num_spherical: int = 7,
                 num_radial: int = 6,
                 num_blocks: int = 5,
                 hidden_dim: int = 256):
        super().__init__()
        
        # 嵌入层
        self.atom_embedding = AtomEmbedding(hidden_dim)
        self.edge_embedding = EdgeEmbedding(num_radial, hidden_dim)
        self.angle_embedding = AngleEmbedding(num_spherical, hidden_dim)
        
        # 交互块
        self.interaction_blocks = nn.ModuleList([
            InteractionBlock(hidden_dim, num_radial, num_spherical)
            for _ in range(num_blocks)
        ])
        
        # 输出层
        self.output_blocks = OutputBlock(hidden_dim)
    
    def forward(self, batch, conditions=None):
        # 构建图表示
        node_features, edge_features, angle_features = self.embed(batch)
        
        # 交互层处理
        for block in self.interaction_blocks:
            node_features = block(node_features, edge_features, angle_features)
        
        # 输出预测
        outputs = self.output_blocks(node_features, conditions)
        return outputs
```

### 几何感知机制

#### 1. 球谐函数 (Spherical Harmonics)
```python
class SphericalBasisLayer(nn.Module):
    """球谐函数基层，处理角度信息"""
    
    def __init__(self, num_spherical, num_radial):
        super().__init__()
        self.num_spherical = num_spherical
        self.num_radial = num_radial
        
    def forward(self, dist, angle, idx_kj):
        # 计算球谐函数
        rbf = self.radial_basis(dist)  # 径向基函数
        sbf = self.spherical_basis(angle)  # 球谐基函数
        return rbf, sbf
```

#### 2. 径向基函数 (Radial Basis Functions)
```python
class RadialBasis(nn.Module):
    """径向基函数，编码距离信息"""
    
    def __init__(self, num_radial, cutoff=6.0):
        super().__init__()
        self.cutoff = cutoff
        self.frequencies = nn.Parameter(torch.randn(num_radial))
        
    def forward(self, distances):
        # Bessel 函数作为径向基
        d_scaled = distances / self.cutoff
        bessel = spherical_bessel_fn(self.frequencies[:, None] * d_scaled[None, :])
        return bessel * cutoff_fn(distances, self.cutoff)
```

### 条件控制机制

#### GemNetTCtrl - 条件控制版本
```python
class GemNetTCtrl(GemNetT):
    """支持条件输入的 GemNet 变体"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # 添加条件处理模块
        self.condition_processor = ConditionProcessor(self.hidden_dim)
        self.cross_attention = CrossAttentionLayer(self.hidden_dim)
    
    def forward(self, batch, conditions=None, timestep=None):
        # 标准前向传播
        node_features = super().forward(batch)
        
        if conditions is not None:
            # 处理条件信息
            condition_features = self.condition_processor(conditions)
            
            # 交叉注意力融合
            node_features = self.cross_attention(
                query=node_features,
                key_value=condition_features
            )
        
        return node_features
```

## 🎯 属性嵌入系统

### 属性嵌入架构

属性嵌入系统负责将材料属性转换为神经网络可处理的向量表示：

#### 核心组件
```
mattergen/property_embeddings.py
├── PropertyEmbedding        # 主嵌入类
├── EmbeddingVector         # 无条件嵌入
├── NoiseLevelEncoding      # 连续属性编码
├── ChemicalSystemEmbedding # 化学体系嵌入
└── SpaceGroupEmbedding     # 空间群嵌入
```

#### 属性嵌入基类
```python
class PropertyEmbedding(nn.Module):
    """属性嵌入基类"""
    
    def __init__(self, 
                 name: str,
                 unconditional_embedding_module: nn.Module,
                 conditional_embedding_module: nn.Module,
                 scaler: nn.Module):
        super().__init__()
        self.name = name
        self.unconditional = unconditional_embedding_module
        self.conditional = conditional_embedding_module
        self.scaler = scaler
    
    def forward(self, values=None, unconditional_prob=0.1):
        if values is None or torch.rand(1) < unconditional_prob:
            # 无条件生成
            return self.unconditional()
        else:
            # 条件生成
            scaled_values = self.scaler(values)
            return self.conditional(scaled_values)
```

### 连续属性编码

#### 噪声级别编码 (Noise Level Encoding)
```python
class NoiseLevelEncoding(nn.Module):
    """基于位置编码的连续属性嵌入"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        
        # 预计算位置编码
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                            -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # 将连续值映射到位置编码空间
        x_scaled = (x * 1000).long().clamp(0, self.pe.size(0) - 1)
        return self.pe[x_scaled]
```

### 分类属性编码

#### 化学体系嵌入
```python
class ChemicalSystemMultiHotEmbedding(nn.Module):
    """化学体系的多热编码嵌入"""
    
    def __init__(self, hidden_dim: int, max_elements: int = 118):
        super().__init__()
        self.element_embedding = nn.Embedding(max_elements + 1, hidden_dim)
        self.aggregation = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, chemical_systems):
        """
        Args:
            chemical_systems: List of chemical system strings, e.g., ["Li-O", "Fe-Ni-Al"]
        """
        embeddings = []
        
        for system in chemical_systems:
            # 解析化学体系字符串
            elements = parse_chemical_system(system)
            element_ids = [ELEMENT_TO_ID.get(elem, 0) for elem in elements]
            
            # 嵌入和聚合
            elem_embeds = self.element_embedding(torch.tensor(element_ids))
            system_embed = torch.mean(elem_embeds, dim=0)
            embeddings.append(system_embed)
        
        batch_embeds = torch.stack(embeddings)
        return self.aggregation(batch_embeds)
```

#### 空间群嵌入
```python
class SpaceGroupEmbedding(nn.Module):
    """空间群的类别嵌入"""
    
    def __init__(self, hidden_dim: int, num_space_groups: int = 230):
        super().__init__()
        self.embedding = nn.Embedding(num_space_groups + 1, hidden_dim)
        
    def forward(self, space_groups):
        """
        Args:
            space_groups: Tensor of space group numbers (1-230)
        """
        # 空间群编号从1开始，调整为从0开始的索引
        sg_indices = space_groups.long()
        return self.embedding(sg_indices)
```

### 数据预处理

#### 标准化器
```python
class StandardScalerTorch(nn.Module):
    """PyTorch 版本的标准化器"""
    
    def __init__(self):
        super().__init__()
        self.register_buffer('mean', torch.tensor(0.0))
        self.register_buffer('std', torch.tensor(1.0))
        self.fitted = False
    
    def fit(self, data):
        """根据数据拟合标准化参数"""
        self.mean = torch.mean(data)
        self.std = torch.std(data)
        self.fitted = True
    
    def forward(self, x):
        if not self.fitted:
            return x
        return (x - self.mean) / (self.std + 1e-8)
```

## 🔄 数据流处理

### 数据管道架构

#### 数据模块设计
```
mattergen/common/data/
├── datamodule.py      # Lightning 数据模块
├── dataset.py         # 数据集类
├── collate.py         # 批处理整理
├── transform.py       # 数据变换
└── types.py          # 数据类型定义
```

#### 数据加载流程
```python
class CrystalDataModule(pl.LightningDataModule):
    """晶体数据的 Lightning 数据模块"""
    
    def __init__(self, 
                 data_path: str,
                 batch_size: int = 32,
                 num_workers: int = 4,
                 properties: List[str] = None):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.properties = properties or []
    
    def setup(self, stage=None):
        # 加载预处理数据
        self.train_dataset = CrystalDataset(
            f"{self.data_path}/train",
            properties=self.properties,
            transform=self.get_transform('train')
        )
        
        self.val_dataset = CrystalDataset(
            f"{self.data_path}/val",
            properties=self.properties,
            transform=self.get_transform('val')
        )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn
        )
```

#### 批处理整理
```python
class CrystalBatchCollator:
    """晶体数据的批处理整理器"""
    
    def __call__(self, batch_list):
        """
        将批次数据整理为统一格式
        
        Args:
            batch_list: List of individual crystal data
            
        Returns:
            batched_data: Batched crystal structures
        """
        # 分离不同类型的数据
        structures = [item['structure'] for item in batch_list]
        properties = {key: [] for key in batch_list[0].get('properties', {})}
        
        # 整理属性数据
        for item in batch_list:
            for key, value in item.get('properties', {}).items():
                properties[key].append(value)
        
        # 构建图表示
        batched_graph = self.batch_structures(structures)
        
        # 整理属性张量
        batched_properties = {}
        for key, values in properties.items():
            if values:
                batched_properties[key] = torch.tensor(values)
        
        return {
            'graph': batched_graph,
            'properties': batched_properties,
            'num_structures': len(structures)
        }
```

### 图构建算法

#### 周期性边构建
```python
def build_crystal_graph(positions, cell, cutoff=6.0):
    """构建晶体的周期性图表示"""
    
    # 考虑周期性边界条件
    extended_positions = []
    extended_indices = []
    
    # 生成周期性副本
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            for dz in [-1, 0, 1]:
                shift = dx * cell[0] + dy * cell[1] + dz * cell[2]
                shifted_pos = positions + shift
                extended_positions.append(shifted_pos)
                extended_indices.extend(range(len(positions)))
    
    # 计算距离矩阵
    distances = compute_distances(extended_positions)
    
    # 构建边
    edge_indices = []
    edge_distances = []
    
    for i in range(len(positions)):
        for j, dist in enumerate(distances[i]):
            if 0 < dist < cutoff:
                edge_indices.append([i, extended_indices[j]])
                edge_distances.append(dist)
    
    return torch.tensor(edge_indices).T, torch.tensor(edge_distances)
```

## ⚙️ 配置系统

### Hydra 配置框架

MatterGen 使用 Hydra 进行层次化配置管理：

#### 配置目录结构
```
mattergen/conf/
├── default.yaml              # 默认配置
├── finetune.yaml            # 微调配置
├── csp.yaml                 # CSP 模式配置
├── adapter/
│   └── default.yaml         # 适配器配置
├── data_module/
│   ├── mp_20.yaml          # MP-20 数据配置
│   └── alex_mp_20.yaml     # Alex-MP-20 数据配置
├── lightning_module/
│   └── diffusion_module/
│       ├── default.yaml    # 扩散模块配置
│       └── model/
│           ├── mattergen.yaml           # 模型配置
│           └── property_embeddings/     # 属性嵌入配置
│               ├── dft_mag_density.yaml
│               ├── chemical_system.yaml
│               └── ...
└── trainer/
    └── default.yaml         # 训练器配置
```

#### 配置组合机制
```yaml
# default.yaml - 主配置文件
defaults:
  - data_module: mp_20              # 选择数据模块
  - trainer: default               # 选择训练器
  - lightning_module: default      # 选择 Lightning 模块
  - _self_                        # 当前文件优先级最高

# 运行时参数
seed: 42
experiment_name: "mattergen_base"
output_dir: "outputs/"

# 可以被命令行覆盖的参数
hydra:
  run:
    dir: ${output_dir}/singlerun/${now:%Y-%m-%d}/${now:%H-%M-%S}
```

#### 动态配置注入
```python
# Hydra 装饰器用法
@hydra.main(config_path="conf", config_name="default", version_base=None)
def train(cfg: DictConfig):
    """训练主函数"""
    
    # 实例化组件
    datamodule = hydra.utils.instantiate(cfg.data_module)
    model = hydra.utils.instantiate(cfg.lightning_module)
    trainer = hydra.utils.instantiate(cfg.trainer)
    
    # 开始训练
    trainer.fit(model, datamodule)
```

### 配置模板系统

#### 属性嵌入配置模板
```yaml
# property_embeddings/template.yaml
_target_: mattergen.property_embeddings.PropertyEmbedding
name: ${property_name}

# 无条件嵌入 (用于无条件生成)
unconditional_embedding_module:
  _target_: mattergen.property_embeddings.EmbeddingVector
  hidden_dim: ${lightning_module.diffusion_module.model.hidden_dim}

# 条件嵌入 (用于条件生成)
conditional_embedding_module:
  _target_: ${conditional_embedding_class}
  hidden_dim: ${lightning_module.diffusion_module.model.hidden_dim}

# 数据预处理
scaler:
  _target_: ${scaler_class}
```

#### 特定属性配置
```yaml
# dft_mag_density.yaml
defaults:
  - template

property_name: "dft_mag_density"
conditional_embedding_class: "mattergen.diffusion.model_utils.NoiseLevelEncoding"
scaler_class: "mattergen.common.utils.data_utils.StandardScalerTorch"

# 特定参数覆盖
conditional_embedding_module:
  d_model: ${lightning_module.diffusion_module.model.hidden_dim}
  max_len: 5000
```

## 🚀 微调机制

### 适配器模式

MatterGen 的微调基于适配器模式 (Adapter Pattern) 实现：

#### 适配器类设计
```python
class GemNetTAdapter(nn.Module):
    """GemNet 的适配器包装器"""
    
    def __init__(self,
                 pretrained_name: str = None,
                 model_path: str = None,
                 property_embeddings_adapt: Dict = None,
                 full_finetuning: bool = True):
        super().__init__()
        
        # 加载预训练模型
        self.base_model = self.load_pretrained_model(pretrained_name, model_path)
        
        # 替换模型骨干为支持条件的版本
        self.base_model.diffusion_module.model = self.adapt_backbone(
            self.base_model.diffusion_module.model
        )
        
        # 添加属性嵌入适配
        self.property_embeddings_adapt = nn.ModuleDict()
        if property_embeddings_adapt:
            for name, config in property_embeddings_adapt.items():
                self.property_embeddings_adapt[name] = hydra.utils.instantiate(config)
        
        # 设置参数训练状态
        self.setup_parameter_training(full_finetuning)
    
    def adapt_backbone(self, original_model):
        """将标准 GemNet 适配为条件控制版本"""
        if isinstance(original_model, GemNetT):
            # 创建新的条件控制模型
            adapted_model = GemNetTCtrl(
                num_spherical=original_model.num_spherical,
                num_radial=original_model.num_radial,
                num_blocks=original_model.num_blocks,
                hidden_dim=original_model.hidden_dim
            )
            
            # 复制预训练权重
            adapted_model.load_state_dict(
                original_model.state_dict(), 
                strict=False
            )
            
            return adapted_model
        
        return original_model
```

#### 参数冻结策略
```python
def setup_parameter_training(self, full_finetuning: bool):
    """设置参数训练策略"""
    
    if full_finetuning:
        # 全参数微调：所有参数可训练
        for param in self.parameters():
            param.requires_grad = True
    else:
        # 仅微调新增参数
        # 冻结预训练参数
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # 解冻新增的属性嵌入参数
        for param in self.property_embeddings_adapt.parameters():
            param.requires_grad = True
        
        # 解冻适配层参数（如果有）
        if hasattr(self.base_model.diffusion_module.model, 'condition_processor'):
            for param in self.base_model.diffusion_module.model.condition_processor.parameters():
                param.requires_grad = True
```

### 条件注入机制

#### 前向传播适配
```python
def forward(self, batch, conditioning_info=None):
    """适配器的前向传播"""
    
    # 处理条件信息
    processed_conditions = None
    if conditioning_info:
        processed_conditions = self.process_conditions(conditioning_info)
    
    # 调用基础模型
    outputs = self.base_model(batch, conditions=processed_conditions)
    
    return outputs

def process_conditions(self, conditioning_info):
    """处理条件信息"""
    condition_embeddings = {}
    
    for property_name, property_value in conditioning_info.items():
        if property_name in self.property_embeddings_adapt:
            # 使用对应的属性嵌入处理
            embedding = self.property_embeddings_adapt[property_name](property_value)
            condition_embeddings[property_name] = embedding
    
    # 聚合多个属性嵌入
    if condition_embeddings:
        # 简单平均聚合（可以使用更复杂的策略）
        aggregated = torch.stack(list(condition_embeddings.values())).mean(dim=0)
        return aggregated
    
    return None
```

## 🔧 扩展开发

### 添加新的材料属性

#### 1. 定义属性标识
```python
# mattergen/common/utils/globals.py
PROPERTY_SOURCE_IDS = [
    # 现有属性
    "dft_mag_density",
    "dft_band_gap",
    # ... 其他属性
    
    # 新增属性
    "thermal_conductivity",    # 热导率
    "elastic_modulus",         # 弹性模量
    "hardness",               # 硬度
]
```

#### 2. 创建属性嵌入类
```python
# mattergen/property_embeddings.py
class ThermalConductivityEmbedding(nn.Module):
    """热导率的专用嵌入"""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, thermal_conductivity_values):
        # 对热导率进行对数变换（由于数值范围大）
        log_values = torch.log(thermal_conductivity_values + 1e-6)
        return self.projection(log_values.unsqueeze(-1))
```

#### 3. 配置文件定义
```yaml
# mattergen/conf/lightning_module/diffusion_module/model/property_embeddings/thermal_conductivity.yaml
_target_: mattergen.property_embeddings.PropertyEmbedding
name: thermal_conductivity

unconditional_embedding_module:
  _target_: mattergen.property_embeddings.EmbeddingVector
  hidden_dim: ${lightning_module.diffusion_module.model.hidden_dim}

conditional_embedding_module:
  _target_: mattergen.property_embeddings.ThermalConductivityEmbedding
  hidden_dim: ${lightning_module.diffusion_module.model.hidden_dim}

scaler:
  _target_: mattergen.common.utils.data_utils.LogScalerTorch  # 对数缩放器
```

#### 4. 数据预处理扩展
```python
# 数据预处理脚本中添加
def process_thermal_conductivity(df):
    """处理热导率数据"""
    
    # 数据清洗
    df['thermal_conductivity'] = df['thermal_conductivity'].fillna(df['thermal_conductivity'].median())
    
    # 异常值处理
    Q1 = df['thermal_conductivity'].quantile(0.25)
    Q3 = df['thermal_conductivity'].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    df['thermal_conductivity'] = df['thermal_conductivity'].clip(lower_bound, upper_bound)
    
    return df
```

### 自定义评估指标

#### 评估器扩展
```python
# mattergen/evaluation/metrics/custom_metrics.py
class ThermalStabilityEvaluator:
    """热稳定性评估器"""
    
    def __init__(self, temperature_range=(300, 1000)):
        self.temperature_range = temperature_range
    
    def evaluate(self, structures, properties):
        """评估结构的热稳定性"""
        
        stability_scores = []
        
        for structure, props in zip(structures, properties):
            # 基于多个属性计算热稳定性得分
            thermal_conductivity = props.get('thermal_conductivity', 0)
            elastic_modulus = props.get('elastic_modulus', 0)
            
            # 简化的稳定性评分计算
            stability = self.calculate_thermal_stability(
                thermal_conductivity, 
                elastic_modulus
            )
            
            stability_scores.append(stability)
        
        return {
            'thermal_stability_mean': np.mean(stability_scores),
            'thermal_stability_std': np.std(stability_scores),
            'thermal_stability_scores': stability_scores
        }
    
    def calculate_thermal_stability(self, tc, em):
        """基于物理公式计算热稳定性"""
        # 这里使用简化的经验公式
        # 实际应用中应该使用更精确的物理模型
        return (tc * 0.3 + em * 0.7) / (tc + em + 1e-6)
```

### 新的神经网络骨干

#### 骨干网络接口
```python
class BackboneInterface(nn.Module):
    """神经网络骨干的标准接口"""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
    
    def forward(self, batch, conditions=None, timestep=None):
        """
        标准前向传播接口
        
        Args:
            batch: 批处理的图数据
            conditions: 条件信息 (可选)
            timestep: 扩散时间步 (可选)
            
        Returns:
            output: 模型输出
        """
        raise NotImplementedError
    
    def get_embedding_dim(self):
        """返回嵌入维度"""
        return self.hidden_dim
```

#### 自定义骨干实现
```python
class CustomTransformerBackbone(BackboneInterface):
    """基于 Transformer 的自定义骨干网络"""
    
    def __init__(self, 
                 hidden_dim: int = 256,
                 num_heads: int = 8,
                 num_layers: int = 6):
        super().__init__(hidden_dim)
        
        self.atom_embedding = nn.Embedding(119, hidden_dim)  # 元素嵌入
        self.pos_encoding = PositionalEncoding(hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, batch, conditions=None, timestep=None):
        # 处理原子特征
        node_features = self.atom_embedding(batch.atomic_numbers)
        
        # 添加位置编码
        node_features = self.pos_encoding(node_features, batch.pos)
        
        # 条件注入
        if conditions is not None:
            node_features = node_features + conditions.unsqueeze(1)
        
        # Transformer 处理
        output = self.transformer(node_features)
        
        return self.output_projection(output)
```

---

## 📊 性能分析

### 计算复杂度

#### 时间复杂度分析
- **GemNet 前向传播**: O(N²) - N 为原子数量
- **扩散采样**: O(T × N²) - T 为时间步数
- **批处理**: O(B × N²) - B 为批次大小

#### 内存使用优化
```python
# 梯度检查点示例
class MemoryEfficientGemNet(GemNetT):
    """内存高效的 GemNet 实现"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_checkpoint = kwargs.get('gradient_checkpointing', False)
    
    def forward(self, batch, conditions=None):
        if self.use_checkpoint and self.training:
            # 使用梯度检查点减少内存使用
            return checkpoint(super().forward, batch, conditions)
        else:
            return super().forward(batch, conditions)
```

### 扩展性考虑

#### 分布式训练支持
```python
# 分布式训练配置
class DistributedGemNet(GemNetT):
    """支持分布式训练的 GemNet"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # 同步批量归一化
        self.sync_bn = kwargs.get('sync_batchnorm', False)
        if self.sync_bn:
            self = nn.SyncBatchNorm.convert_sync_batchnorm(self)
    
    def configure_optimizers(self):
        """配置分布式优化器"""
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4)
        
        # 分布式时调整学习率
        if torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
            optimizer.param_groups[0]['lr'] *= world_size
        
        return optimizer
```

---

## 📚 总结

MatterGen 的架构设计体现了以下核心理念：

### 🎯 设计优势
1. **模块化**: 清晰的组件边界，便于开发和维护
2. **可扩展**: 支持新属性、新模型、新评估指标
3. **可配置**: 基于 Hydra 的灵活配置系统
4. **高效**: 优化的数据流和计算图

### 🚀 扩展方向
1. **新的物理约束**: 集成更多材料科学约束
2. **多尺度建模**: 支持从原子到宏观的多尺度建模
3. **主动学习**: 结合主动学习策略优化数据效率
4. **物理信息**: 集成物理定律作为归纳偏置

### 🔬 研究前沿
1. **几何深度学习**: 探索新的几何表示方法
2. **因果推理**: 将因果关系引入材料设计
3. **迁移学习**: 跨材料体系的知识迁移
4. **可解释性**: 提高模型决策的可解释性

---

## 📞 获取帮助

- **主文档**: [README_CN.md](README_CN.md)
- **微调指南**: [README_FINETUNE_CN.md](README_FINETUNE_CN.md)
- **GitHub Issues**: [https://github.com/microsoft/mattergen/issues](https://github.com/microsoft/mattergen/issues)

*本架构文档基于 MatterGen v1.0 编写*