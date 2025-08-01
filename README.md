# 腾讯推荐系统项目

本项目是一个基于深度学习的推荐系统，采用Transformer架构进行序列建模，支持多模态特征融合和RQ-VAE特征压缩技术。

## 项目概述

该推荐系统主要特点：
- 🚀 基于Transformer的序列推荐模型
- 🎯 支持用户行为序列建模
- 🌟 多模态特征融合（文本、图像等embedding）
- 📊 RQ-VAE特征压缩和语义ID生成
- ⚡ Flash Attention优化的注意力机制
- 🔄 支持正负样本对比学习

## 文件结构说明

### 1. run.sh - 项目启动脚本
```bash
#!/bin/bash
# 简单的启动脚本，用于运行主程序
python -u main.py
```

**功能说明**：
- 项目的入口启动脚本
- 自动切换到运行目录并执行main.py
- 使用`-u`参数确保输出不被缓冲

### 2. main.py - 主程序入口 (6.4KB)
**核心功能**：
- **训练流程控制**：完整的模型训练和验证流程
- **参数解析**：支持命令行参数配置，包括学习率、批次大小、模型结构等
- **数据加载**：集成MyDataset进行训练和验证数据的加载
- **模型初始化**：BaselineModel的初始化和参数配置
- **损失计算**：使用BCEWithLogitsLoss进行二分类损失计算
- **模型保存**：定期保存模型检查点和训练日志

**关键参数**：
```python
--batch_size: 批次大小 (默认: 128)
--lr: 学习率 (默认: 0.001) 
--maxlen: 序列最大长度 (默认: 101)
--hidden_units: 隐藏层维度 (默认: 32)
--num_blocks: Transformer层数 (默认: 1)
--num_heads: 注意力头数 (默认: 1)
--dropout_rate: Dropout率 (默认: 0.2)
--mm_emb_id: 多模态特征ID (默认: ['81'])
```

**训练流程**：
1. 数据集加载和划分（9:1训练验证比例）
2. 模型初始化和参数初始化
3. 正负样本对比学习训练
4. 验证集评估和模型保存

### 3. dataset.py - 数据处理模块 (17KB)
**核心类**：

#### MyDataset - 训练数据集
**主要功能**：
- **用户序列加载**：从jsonl文件动态加载用户行为序列
- **特征处理**：支持稀疏特征、数组特征、连续特征和多模态embedding特征
- **序列填充**：left-padding方式填充序列到固定长度
- **负采样**：训练时随机生成负样本
- **特征对齐**：处理缺失特征并填充默认值

**数据格式**：
```python
# 序列数据格式: [(user_id, item_id, user_feat, item_feat, action_type, timestamp)]
# 返回格式: (seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat)
```

**特征类型支持**：
- **用户稀疏特征**: ['103', '104', '105', '109'] 
- **物品稀疏特征**: ['100', '117', '111', '118', '101', '102', '119', '120', '114', '112', '121', '115', '122', '116']
- **用户数组特征**: ['106', '107', '108', '110']
- **物品多模态特征**: 支持81-86号特征ID的embedding

#### MyTestDataset - 测试数据集
**主要功能**：
- 继承MyDataset的基础功能
- 处理冷启动问题（训练时未见过的特征值）
- 支持预测场景的数据格式

**工具函数**：
- `load_mm_emb()`: 加载多模态特征embedding
- `save_emb()`: 保存embedding为二进制格式

### 4. model.py - 主模型实现 (19KB)
**核心组件**：

#### FlashMultiHeadAttention - 优化的注意力机制
**特性**：
- 支持PyTorch 2.0+的Flash Attention
- 降级兼容标准注意力机制
- 支持因果掩码和填充掩码

#### PointWiseFeedForward - 前馈网络
**实现**：
- 使用1D卷积实现点对点前馈
- 包含ReLU激活和Dropout正则化

#### BaselineModel - 主推荐模型
**模型架构**：
```
输入层: 用户/物品ID + 多种特征
    ↓
特征融合层: 稀疏embedding + 数组embedding + 多模态embedding
    ↓  
Transformer层: Multi-Head Attention + Feed Forward (x num_blocks)
    ↓
输出层: 序列表示用于正负样本对比
```

**关键功能**：
- **特征处理**：`feat2emb()` - 将各类特征转换为embedding
- **序列建模**：`log2feats()` - Transformer编码用户行为序列
- **对比学习**：`forward()` - 计算正负样本的相似度
- **预测推理**：`predict()` - 生成用户表示用于检索
- **候选生成**：`save_item_emb()` - 批量生成物品embedding用于检索

**特征类型处理**：
- 稀疏特征 → Embedding查表
- 数组特征 → Embedding求和池化  
- 连续特征 → 直接使用数值
- 多模态特征 → 线性变换到统一维度

### 5. model_rqvae.py - RQ-VAE特征压缩 (14KB)
**目标**：将高维多模态embedding压缩为离散的语义ID

#### 核心组件

**VQEmbedding - 向量量化模块**：
- 使用K-means或平衡K-means初始化码本
- 支持余弦距离和L2距离的相似度计算
- 将连续向量映射为离散语义ID

**RQ - 残差量化器**：
- 多层级残差量化，逐步减少量化误差
- 支持共享或独立码本
- 生成多层级语义ID组合

**RQVAE - 完整的RQ-VAE模型**：
```
原始Embedding → Encoder → 潜在表示 → RQ量化 → Decoder → 重构Embedding
                                    ↓
                             语义ID (用作新特征)
```

**训练目标**：
- 重构损失：MSE(原始embedding, 重构embedding)
- 量化损失：向量量化的commitment loss

**应用流程**：
1. 使用MmEmbDataset读取多模态embedding数据
2. 训练RQ-VAE模型学习压缩表示
3. 将embedding转换为语义ID
4. 将语义ID作为新的稀疏特征加入主模型训练

## 技术特点

### 1. 多模态特征融合
- 支持文本、图像等多种模态的预训练embedding
- 统一的特征处理框架，自动处理不同维度的特征
- 特征缺失时的默认值填充机制

### 2. 高效的序列建模
- Flash Attention优化，提升训练和推理效率
- 因果掩码确保序列建模的时序性
- 位置编码增强序列理解能力

### 3. 对比学习训练
- 正负样本对比的训练方式
- BCEWithLogitsLoss计算相似度损失
- 支持不同action_type的加权训练

### 4. 特征压缩技术
- RQ-VAE将高维embedding压缩为语义ID
- 多层级量化减少信息损失
- 可作为新特征提升模型效果

### 5. 工程优化
- 批处理优化的特征转换
- 内存高效的动态数据加载
- 支持GPU加速训练

## 使用方法

### 环境要求
```bash
torch >= 1.12.0
numpy
sklearn
tqdm
tensorboard
```

### 数据准备
确保以下数据文件存在：
- `seq.jsonl`: 用户行为序列数据
- `item_feat_dict.json`: 物品特征字典
- `indexer.pkl`: ID映射索引
- `creative_emb/`: 多模态embedding文件夹

### 训练命令
```bash
# 设置环境变量
export TRAIN_DATA_PATH=/path/to/data
export TRAIN_LOG_PATH=/path/to/logs  
export TRAIN_TF_EVENTS_PATH=/path/to/tensorboard
export TRAIN_CKPT_PATH=/path/to/checkpoints

# 运行训练
bash run.sh
# 或直接运行
python main.py --batch_size 256 --lr 0.001 --num_epochs 10
```

### RQ-VAE使用
```python
# 1. 准备多模态embedding数据
dataset = MmEmbDataset(data_dir, feature_id='82')

# 2. 训练RQ-VAE模型  
model = RQVAE(input_dim=1024, hidden_channels=[512, 256], ...)

# 3. 生成语义ID
semantic_ids = model._get_codebook(embeddings)

# 4. 作为新特征加入主模型训练
```

## 模型效果

该模型通过以下方式提升推荐效果：
- **序列建模**：捕获用户行为的时序依赖
- **多模态融合**：利用丰富的内容特征
- **特征压缩**：RQ-VAE提供更紧凑的特征表示
- **对比学习**：提升用户和物品的表示质量

## 扩展性

项目具有良好的扩展性：
- 可轻松添加新的特征类型
- 支持不同的Transformer变体
- 可集成其他序列建模技术
- 支持分布式训练扩展# Tencent_rec
# Tencent_rec
