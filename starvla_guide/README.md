# STAR-VLA 入门指南
## 专为有 LeRobot 和 OpenPI 经验的开发者

欢迎来到 STAR-VLA 的世界！既然你已经熟悉 LeRobot 和 OpenPI，我会重点讲解 STAR-VLA 与它们的关联和区别，帮助你快速上手。

---

## 🎯 1. STAR-VLA 是什么？

**STAR-VLA** (Star Vision-Language-Action) 是一个**模块化的视觉-语言-动作模型开发框架**。

### 核心特点
- **乐高式架构**: 每个组件都可以插拔替换
- **统一框架**: 支持多种VLA模型变体
- **生产就绪**: 从训练到部署的完整流程
- **社区驱动**: 活跃的开源社区和持续更新

### 与 LeRobot 和 OpenPI 的关系
```
LeRobot (数据) → STAR-VLA (训练) → OpenPI (部署)
```

---

## 🔄 2. 与你已有经验的对比

### 2.1 数据格式对比

| 框架 | 数据格式 | 你的经验 |
|------|----------|----------|
| **LeRobot** | LeRobot v2.1 格式 | ✅ 你已经处理过 |
| **OpenPI** | 自定义格式 | ✅ 你熟悉 |
| **STAR-VLA** | **兼容 LeRobot 格式** | 🎯 可以直接使用 |

**关键点**: STAR-VLA 可以直接读取 LeRobot 格式的数据集！

### 2.2 训练流程对比

| 步骤 | LeRobot | OpenPI | STAR-VLA |
|------|---------|--------|----------|
| 数据准备 | ✅ 数据清洗 | ✅ 数据转换 | ✅ **直接使用 LeRobot 数据** |
| 模型定义 | 自定义模型 | PI0/PI0.5 | **多种架构可选** (ACT, DiT, FAST等) |
| 训练脚本 | 自定义 | 官方脚本 | **标准化训练流程** |
| 评估 | 自定义 | 官方评估 | **多基准测试支持** |

### 2.3 部署对比

| 框架 | 服务器 | 你的经验 |
|------|--------|----------|
| **OpenPI** | 自定义服务器 | ✅ 你写过 server |
| **STAR-VLA** | **标准化 WebSocket 服务器** | 🎯 更简单 |

---

## 🏗️ 3. STAR-VLA 核心架构

### 3.1 整体架构
```
┌─────────────────────────────────────────────┐
│              STAR-VLA 框架                   │
├─────────────────────────────────────────────┤
│  数据层 (LeRobot 兼容)                       │
│  ├── 数据加载器                              │
│  ├── 数据预处理                              │
│  └── 数据增强                                │
├─────────────────────────────────────────────┤
│  模型层 (可插拔)                             │
│  ├── 视觉编码器 (ResNet, ViT, CLIP)         │
│  ├── 语言编码器 (BERT, Qwen)                │
│  ├── 动作头 (ACT, DiT, FAST, π)             │
│  └── 融合层                                 │
├─────────────────────────────────────────────┤
│  训练层                                     │
│  ├── 标准训练循环                            │
│  ├── 分布式训练                              │
│  └── 检查点管理                              │
├─────────────────────────────────────────────┤
│  部署层                                     │
│  ├── WebSocket 服务器                        │
│  ├── REST API                               │
│  └── 实时推理                                │
└─────────────────────────────────────────────┘
```

### 3.2 关键组件详解

#### 1. **基础框架** (`baseframework`)
```python
# 类似于 OpenPI 的模型包装器
from starVLA.model.framework.base_framework import baseframework

# 加载预训练模型
model = baseframework.from_pretrained("path/to/checkpoint")

# 推理
actions = model.predict(images, language_instruction)
```

#### 2. **动作头** (Action Heads)
STAR-VLA 支持多种动作预测架构：

| 动作头 | 特点 | 适用场景 |
|--------|------|----------|
| **ACT** | Action Chunking with Transformers | 你的模型！适合连续动作 |
| **DiT** | Diffusion Transformer | 复杂动作分布 |
| **FAST** | Fast Action Sampling | 实时性要求高 |
| **π** | Flow Matching | 平滑动作生成 |

#### 3. **数据加载器** (`lerobot_datasets`)
```python
# 直接使用 LeRobot 数据集
from starVLA.dataloader import build_dataloader

dataloader = build_dataloader(
    cfg=config,
    dataset_py="lerobot_datasets"  # 直接兼容 LeRobot
)
```

---

## 📊 4. 数据流详解

### 4.1 从 LeRobot 到 STAR-VLA

```
LeRobot 数据集 (你的数据)
    ↓
STAR-VLA 数据加载器 (自动识别)
    ↓
预处理 (图像缩放、归一化)
    ↓
模型输入 (图像 + 语言指令)
    ↓
动作预测 (20维动作序列)
    ↓
反归一化 (恢复原始动作空间)
```

### 4.2 数据格式要求

你的 LeRobot 数据集需要包含：
```
dataset/
├── images/           # 图像序列
├── actions.npy       # 动作序列
├── states.npy        # 状态序列 (可选)
└── metadata.json     # 元数据
```

STAR-VLA 会自动处理：
- 图像缩放到 160x160
- 动作归一化
- 语言指令编码

---

## 🚀 5. 快速上手指南

### 5.1 环境准备

```bash
# 1. 克隆 STAR-VLA
git clone https://github.com/starVLA/starVLA.git
cd starVLA

# 2. 创建虚拟环境
conda create -n starvla python=3.10
conda activate starvla

# 3. 安装依赖
pip install -e .
pip install lerobot  # 如果需要 LeRobot 数据支持
```

### 5.2 使用你的 LeRobot 数据

```yaml
# config.yaml
datasets:
  vla_data:
    dataset_py: lerobot_datasets  # 关键：使用 LeRobot 数据加载器
    data_root_dir: /path/to/your/lerobot/dataset
    data_mix: your_task_name
    per_device_batch_size: 2
    image_size: [160, 160]
```

### 5.3 训练 ACT 模型 (你的场景)

```bash
# 使用你的配置文件
accelerate launch starVLA/training/train_starvla.py \
    --config_yaml examples/realRobots/G1ThreeCam/train_files/train_g1_act.yaml \
    --framework.name ACT \
    --datasets.vla_data.data_root_dir /path/to/your/data
```

### 5.4 启动服务器 (你有经验的部分)

```bash
# 类似 OpenPI 的服务器，但更简单
python deployment/model_server/server_policy.py \
    --ckpt_path output/starvla_act_g1/g1_act_task_04/final_model \
    --port 5000
```

---

## 🔧 6. 与 OpenPI 服务器的对比

### 6.1 相似点
- 都是 **WebSocket 服务器**
- 都支持 **实时推理**
- 都有 **健康检查端点**

### 6.2 STAR-VLA 服务器的优势

| 特性 | OpenPI 服务器 | STAR-VLA 服务器 |
|------|---------------|-----------------|
| **标准化** | 自定义实现 | 框架内置 |
| **多模型支持** | 单一模型 | 多种架构 |
| **配置管理** | 手动配置 | 自动从检查点读取 |
| **动作归一化** | 客户端处理 | 服务器端处理 |

### 6.3 服务器 API 对比

```python
# OpenPI 风格
client = OpenPIPolicyClient(host="localhost", port=5000)
actions = client.predict(images, instruction)

# STAR-VLA 风格 (几乎相同)
from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy
client = WebsocketClientPolicy(host="localhost", port=5000)
actions = client.predict(images, instruction)
```

---

## 📈 7. 训练监控

### 7.1 日志格式
STAR-VLA 使用标准的日志格式：
```
Step 10000, Loss: {'action_dit_loss': 0.147, 'mse_score': 0.00348}
```

### 7.2 与你的经验对比
- **LeRobot**: 你可能用 WandB 或 TensorBoard
- **STAR-VLA**: **原生支持 WandB**，配置更简单

```yaml
# 启用 WandB
wandb_project: starVLA_training
wandb_entity: your_username
```

---

## 🎯 8. 你的下一步行动

### 8.1 立即可以做的事
1. **检查你的 LeRobot 数据**：确保格式正确
2. **尝试 STAR-VLA 训练**：使用你的数据训练 ACT 模型
3. **部署服务器**：用你熟悉的方式部署

### 8.2 学习路径建议

```
第1周：熟悉架构
├── 阅读本文档
├── 运行示例代码
└── 理解数据流

第2周：动手实践
├── 使用你的数据训练
├── 调整超参数
└── 评估性能

第3周：高级特性
├── 尝试不同的动作头
├── 多任务训练
└── 部署优化
```

### 8.3 推荐学习资源

1. **官方文档**: https://starvla.github.io
2. **示例代码**: `examples/` 目录
3. **训练配置**: `examples/realRobots/G1ThreeCam/`
4. **部署指南**: `deployment/` 目录

---

## 💡 9. 常见问题解答

### Q1: STAR-VLA 和 LeRobot 的关系？
**A**: STAR-VLA **兼容** LeRobot 数据格式，可以无缝迁移你的数据集。

### Q2: 我需要重写数据加载器吗？
**A**: **不需要**！STAR-VLA 内置了 LeRobot 数据加载器。

### Q3: 服务器部署和 OpenPI 一样吗？
**A**: **几乎一样**，但 STAR-VLA 的服务器更标准化，配置更简单。

### Q4: 我可以复用 OpenPI 的客户端代码吗？
**A**: **可以**，WebSocket 协议兼容，只需修改连接参数。

---

## 🎉 总结

既然你有 LeRobot 和 OpenPI 经验，学习 STAR-VLA 会非常快：

1. **数据**: 直接使用你的 LeRobot 数据集
2. **训练**: 标准化的训练流程，配置更简单
3. **部署**: 类似 OpenPI 的服务器，但更强大
4. **扩展**: 模块化设计，易于定制

**你的优势**：
- ✅ 熟悉数据格式
- ✅ 了解训练流程
- ✅ 有服务器部署经验

**STAR-VLA 给你的**：
- 🎯 更强大的模型架构
- 🎯 更标准化的流程
- 🎯 更活跃的社区支持

开始你的 STAR-VLA 之旅吧！🚀
