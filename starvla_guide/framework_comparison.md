# STARVLA 框架分类详解：VM4A、VLM4A、WM4A

## 🎯 一句话总结

STARVLA 将模型框架分为三大类，基于**感知骨干网络**的不同：

- **VM4A**: 视觉-运动-动作 (Vision-Motor-Action)
- **VLM4A**: 视觉-语言-运动-动作 (Vision-Language-Motor-Action)  
- **WM4A**: 世界模型-运动-动作 (World-Model-Motor-Action)

---

## 📊 框架分类图

```
STARVLA 模型框架
├── VM4A (视觉-运动-动作)
│   ├── ACT          # Action Chunking with Transformers
│   └── DiffusionPolicy  # 扩散策略
│
├── VLM4A (视觉-语言-运动-动作)
│   ├── QwenPI       # Qwen + Flow-matching
│   ├── QwenGR00T    # Qwen + GR00T
│   ├── QwenFast     # Qwen + FAST
│   ├── QwenOFT      # Qwen + OFT
│   └── ... (其他变体)
│
└── WM4A (世界模型-运动-动作)
    ├── CosmoPredict2PI   # Cosmos-Predict2 + PI
    ├── CosmoPredict2GR00T # Cosmos-Predict2 + GR00T
    ├── CosmoPredict2OFT  # Cosmos-Predict2 + OFT
    ├── WanPI             # Wan2.2 + PI
    ├── WanGR00T          # Wan2.2 + GR00T
    └── WanOFT            # Wan2.2 + OFT
```

---

## 🔍 详细对比

### 1. VM4A (Vision-Motor-Action)

**定义**: 直接从视觉输入预测动作，不使用语言或世界模型。

**核心特点**:
- ✅ **最简单**: 视觉 → 动作
- ✅ **最快速**: 计算量最小
- ✅ **最轻量**: 模型参数最少
- ❌ **无语言理解**: 不能理解自然语言指令
- ❌ **无物理先验**: 不理解物理规律

**代表模型**:
```python
# ACT (Action Chunking with Transformers)
# 文件: starVLA/model/framework/VM4A/ACT.py

@FRAMEWORK_REGISTRY.register("ACT")
class ACT(baseframework):
    def __init__(self, config):
        # 直接使用 LeRobot 的 ACTPolicy
        self.action_model = ACTPolicy(act_config)
    
    def forward(self, examples):
        # 视觉 → 动作
        batch = self._examples_to_act_batch(examples)
        output = self.action_model.forward(batch)
        return {"action_loss": output[0]}
```

**架构**:
```
输入图像 → ResNet/ViT → 特征提取 → 动作预测头 → 动作输出
```

---

### 2. VLM4A (Vision-Language-Motor-Action)

**定义**: 使用视觉语言模型 (VLM) 作为感知骨干，结合语言理解和视觉感知。

**核心特点**:
- ✅ **语言理解**: 能理解自然语言指令
- ✅ **多模态**: 融合视觉和语言信息
- ✅ **预训练知识**: 利用大规模预训练模型
- ❌ **计算量大**: 需要运行大型语言模型
- ❌ **无物理先验**: 不理解物理规律

**代表模型**:
```python
# QwenPI (Qwen + Flow-matching)
# 文件: starVLA/model/framework/VLM4A/QwenPI.py

@FRAMEWORK_REGISTRY.register("QwenPI")
class QwenPI(baseframework):
    def __init__(self, config):
        # 1. 加载 Qwen-VL 模型
        self.qwen_vl = get_vlm_model("qwen2.5-vl")
        
        # 2. 创建动作头 (Flow-matching)
        self.action_head = LayerwiseFlowmatchingActionHead(...)
    
    def forward(self, examples):
        # 视觉 + 语言 → 特征 → 动作
        images = examples["images"]
        text = examples["instruction"]
        
        # Qwen-VL 编码
        vl_features = self.qwen_vl(images, text)
        
        # 动作预测
        actions = self.action_head(vl_features)
        return {"action_loss": actions}
```

**架构**:
```
输入图像 + 语言指令 → Qwen-VL → 多模态特征 → 动作预测头 → 动作输出
```

---

### 3. WM4A (World-Model-Motor-Action)

**定义**: 使用视频生成世界模型作为感知骨干，学习物理规律和时空表示。

**核心特点**:
- ✅ **物理理解**: 学习物理规律和时空关系
- ✅ **丰富表示**: 从视频预测任务中学习
- ✅ **泛化能力强**: 理解物体运动和交互
- ❌ **计算量最大**: 需要运行视频生成模型
- ❌ **模型最大**: 参数量最多

**代表模型**:
```python
# CosmoPredict2PI (Cosmos-Predict2 + PI)
# 文件: starVLA/model/framework/WM4A/CosmoPredict2PI.py

@FRAMEWORK_REGISTRY.register("CosmoPredict2PI")
class CosmoPredict2PI(baseframework):
    def __init__(self, config):
        # 1. 加载 Cosmos-Predict2 世界模型
        self.world_model = get_world_model("cosmos-predict2")
        
        # 2. 创建动作头 (Layer-wise Flow-matching)
        self.action_head = LayerwiseFlowmatchingActionHead(...)
    
    def forward(self, examples):
        # 图像 → 世界模型编码 → 物理特征 → 动作
        images = examples["images"]
        
        # 世界模型编码 (学习物理规律)
        physics_features = self.world_model.encode(images)
        
        # 动作预测
        actions = self.action_head(physics_features)
        return {"action_loss": actions}
```

**架构**:
```
输入图像 → Cosmos-Predict2/Wan2.2 → 物理特征 → 动作预测头 → 动作输出
```

---

## 🧩 动作头变体

每种框架都可以搭配不同的动作头：

| 动作头 | 特点 | 速度 | 精度 |
|--------|------|------|------|
| **OFT** | MLP 回归，只使用最后一层 | ⚡ 最快 | ⭐⭐ |
| **GR00T** | 单层 Flow-matching | ⚡ 中等 | ⭐⭐⭐ |
| **PI** | 多层交叉注意力 DiT | 🐢 最慢 | ⭐⭐⭐⭐ |

### 组合示例

```python
# VLM4A + PI 动作头
"QwenPI"  # Qwen-VL + Layer-wise DiT

# WM4A + GR00T 动作头
"CosmoPredict2GR00T"  # Cosmos-Predict2 + 单层 Flow-matching

# VM4A (无动作头选择，使用 LeRobot 内置)
"ACT"  # LeRobot ACTPolicy
```

---

## 📈 性能对比

| 框架类型 | 语言理解 | 物理理解 | 计算量 | 模型大小 | 适用场景 |
|----------|----------|----------|--------|----------|----------|
| **VM4A** | ❌ 无 | ❌ 无 | ⚡ 小 | 📦 小 | 简单任务，实时性要求高 |
| **VLM4A** | ✅ 有 | ❌ 无 | ⚡ 中 | 📦 中 | 需要语言指令的任务 |
| **WM4A** | ❌ 无 | ✅ 有 | ⚡ 大 | 📦 大 | 复杂物理交互任务 |

---

## 🔧 技术实现细节

### 1. 感知骨干网络

#### VM4A (视觉编码器)
```python
# 使用标准视觉编码器
self.visual_encoder = torchvision.models.resnet18(pretrained=True)
# 或
self.visual_encoder = VisionTransformer(...)
```

#### VLM4A (视觉语言模型)
```python
# 使用预训练的 VLM
from transformers import AutoModelForCausalLM
self.vlm = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-VL-7B")
```

#### WM4A (世界模型)
```python
# 使用视频生成模型
from diffusers import DiTTransformer2DModel
self.world_model = DiTTransformer2DModel.from_pretrained("nvidia/Cosmos-Predict2-2B")
```

### 2. 特征提取

#### VM4A
```python
# 提取视觉特征
visual_features = self.visual_encoder(images)  # [B, hidden_dim]
```

#### VLM4A
```python
# 提取多模态特征
inputs = self.processor(text=instructions, images=images, return_tensors="pt")
outputs = self.vlm(**inputs, output_hidden_states=True)
vl_features = outputs.hidden_states[-1]  # [B, seq_len, hidden_dim]
```

#### WM4A
```python
# 提取物理特征
latent = self.vae.encode(images).latent_dist.sample()
physics_features = self.world_model(latent, timestep).sample  # [B, N_tokens, hidden_dim]
```

### 3. 动作预测

#### OFT (简单 MLP)
```python
class OFTActionHead(nn.Module):
    def __init__(self, input_dim, action_dim):
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )
    
    def forward(self, features):
        # 只使用最后一层特征
        last_features = features[:, -1, :]
        return self.mlp(last_features)
```

#### GR00T (单层 Flow-matching)
```python
class GR00TActionHead(nn.Module):
    def __init__(self, input_dim, action_dim):
        self.flow_net = FlowNetwork(input_dim, action_dim)
    
    def forward(self, features, t):
        # 单层特征 + 时间步
        last_features = features[:, -1, :]
        return self.flow_net(last_features, t)
```

#### PI (多层 DiT)
```python
class PIActionHead(nn.Module):
    def __init__(self, input_dim, action_dim, num_layers):
        self.dit = DiTTransformer(num_layers, input_dim, action_dim)
    
    def forward(self, features, t):
        # 多层特征 + 时间步
        return self.dit(features, t)  # 使用所有层的特征
```

---

## 🎯 如何选择框架

### 选择 VM4A (ACT) 如果：
- ✅ 任务简单，不需要语言指令
- ✅ 需要实时推理
- ✅ 计算资源有限
- ✅ 使用 LeRobot 数据集

### 选择 VLM4A (QwenPI) 如果：
- ✅ 需要理解自然语言指令
- ✅ 任务需要多模态理解
- ✅ 有中等计算资源
- ✅ 需要预训练知识

### 选择 WM4A (CosmoPredict2PI) 如果：
- ✅ 任务涉及复杂物理交互
- ✅ 需要理解物体运动规律
- ✅ 有充足计算资源
- ✅ 需要强泛化能力

---

## 🚀 快速开始示例

### 1. 使用 VM4A (ACT)
```yaml
# config.yaml
framework:
  name: ACT
  chunk_size: 50
  action_dim: 20
```

### 2. 使用 VLM4A (QwenPI)
```yaml
# config.yaml
framework:
  name: QwenPI
  qwenvl:
    base_vlm: "Qwen/Qwen2.5-VL-7B"
  action_model:
    action_model_type: "LayerwiseFM"
    action_dim: 20
```

### 3. 使用 WM4A (CosmoPredict2PI)
```yaml
# config.yaml
framework:
  name: CosmoPredict2PI
  world_model:
    base_wm: "nvidia/Cosmos-Predict2-2B"
  action_model:
    action_model_type: "LayerwiseFM"
    action_dim: 20
```

---

## 📊 模型大小对比

| 框架 | 骨干网络 | 参数量 | 显存需求 |
|------|----------|--------|----------|
| **VM4A (ACT)** | ResNet-18 | ~11M | ~2GB |
| **VLM4A (QwenPI)** | Qwen2.5-VL-7B | ~7B | ~14GB |
| **WM4A (CosmoPredict2PI)** | Cosmos-Predict2-2B | ~2B | ~4GB |

---

## 🎉 总结

### 三大框架的核心区别

| 框架 | 感知骨干 | 核心能力 | 计算需求 |
|------|----------|----------|----------|
| **VM4A** | 视觉编码器 | 视觉感知 | ⚡ 低 |
| **VLM4A** | 视觉语言模型 | 视觉 + 语言 | ⚡ 中 |
| **WM4A** | 世界模型 | 视觉 + 物理 | ⚡ 高 |

### 你的选择建议

既然你有 LeRobot 经验，建议：

1. **从 VM4A (ACT) 开始**: 最简单，直接使用 LeRobot
2. **尝试 VLM4A**: 如果需要语言指令理解
3. **探索 WM4A**: 如果需要物理理解

**STARVLA 的模块化设计让你可以轻松切换不同框架！** 🚀
