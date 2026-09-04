# STARVLA 模型搭建源码详解

## 🎯 核心思想：模块化乐高式架构

STARVLA 采用 **"乐高积木"式** 的模块化设计，每个组件都可以独立替换和组合。

---

## 📁 目录结构

```
starVLA/model/
├── framework/                    # 框架层 (顶层组装)
│   ├── base_framework.py        # 基础框架抽象
│   ├── share_tools.py           # 共享工具
│   ├── VM4A/                    # 视觉-运动-动作模型
│   │   └── ACT.py               # ACT 架构实现 ⭐
│   └── VLM4A/                   # 视觉-语言-运动-动作模型
│       ├── QwenPI.py            # Qwen + Flow-matching
│       ├── QwenGR00T.py         # Qwen + GR00T
│       └── ...                  # 其他变体
├── modules/                     # 可复用模块
│   ├── action_model/            # 动作预测头
│   ├── vlm/                     # 视觉语言模型
│   └── world_model/             # 世界模型
└── tools.py                     # 工具函数
```

---

## 🔧 模型搭建流程详解

### 1. 入口点：`build_framework()`

```python
# 文件位置: external/starVLA/starVLA/model/framework/base_framework.py

def build_framework(cfg):
    """
    构建框架模型的单一入口点
    Args:
        cfg: 配置对象，包含 cfg.framework.name
    Returns:
        nn.Module: 实例化的框架模型
    """
    # 1. 自动导入所有框架模块
    _auto_import_framework_modules()
    
    # 2. 从注册表获取框架类
    framework_id = cfg.framework.name  # 例如 "ACT", "QwenPI"
    model_class = FRAMEWORK_REGISTRY[framework_id]
    
    # 3. 实例化模型
    return model_class(cfg)
```

### 2. 框架注册机制

```python
# 使用装饰器注册框架
@FRAMEWORK_REGISTRY.register("ACT")
class ACT(baseframework):
    def __init__(self, config, **kwargs):
        super().__init__()
        # ... 初始化代码
```

### 3. 基础框架类：`baseframework`

```python
class baseframework(PreTrainedModel):
    """
    高层 VLA 模型组装的轻量级基类
    子类需要:
      - 接受结构化配置
      - 在 __init__ 中注册组件
      - 使用提供的辅助函数处理动作归一化
    """
    
    def __init__(self, hf_config=None):
        if hf_config is None:
            hf_config = PretrainedConfig()
        super().__init__(hf_config)
    
    # 关键方法
    def forward(self, examples, **kwargs):
        """训练前向传播"""
        raise NotImplementedError
    
    def predict_action(self, examples, **kwargs):
        """推理预测动作"""
        raise NotImplementedError
```

---

## 🏗️ ACT 模型搭建详解

### 1. ACT 架构概述

```python
# 文件位置: external/starVLA/starVLA/model/framework/VM4A/ACT.py

@FRAMEWORK_REGISTRY.register("ACT")
class ACT(baseframework):
    """
    ACT (Action Chunking with Transformers) 框架
    使用 LeRobot 的 ACTPolicy 实现
    """
```

### 2. 配置数据类

```python
@dataclass
class ACTDefaultConfig:
    """ACT 默认配置"""
    name: str = "ACT"
    chunk_size: int = 50           # 动作块大小
    action_dim: int = 8            # 动作维度
    state_dim: int = 8             # 状态维度
    n_obs_steps: int = 1           # 观察步数
    image_keys: tuple = ("cam0_rgb", "cam1_rgb")  # 相机键名
    image_size: tuple = (224, 224) # 图像尺寸
    pretrained_backbone_weights: str = "ResNet18_Weights.IMAGENET1K_V1"
```

### 3. 初始化过程

```python
def __init__(self, config, **kwargs):
    super().__init__()
    
    # 1. 合并配置
    self.config = merge_framework_config(ACTDefaultConfig, config)
    framework_config = self.config.framework
    
    # 2. 创建 ACT 配置
    act_config = ACTConfig(
        n_obs_steps=int(framework_config.n_obs_steps),
        chunk_size=int(framework_config.chunk_size),
        n_action_steps=int(framework_config.chunk_size),
        input_features=_build_input_features(framework_config),
        output_features=_build_output_features(framework_config),
        pretrained_backbone_weights=framework_config.get(
            "pretrained_backbone_weights", None
        ),
    )
    
    # 3. 创建 ACTPolicy 模型
    self.action_model = ACTPolicy(act_config)
    
    # 4. 设置归一化器为恒等映射
    self._set_identity_normalizers()
```

### 4. 特征构建

```python
def _build_input_features(config):
    """构建输入特征"""
    image_keys = config.get("image_keys", ACTDefaultConfig.image_keys)
    image_height, image_width = _get_image_size(config)
    
    features = {
        # 图像特征
        f"observation.images.{image_key}": PolicyFeature(
            type=FeatureType.VISUAL,
            shape=(3, image_height, image_width)
        )
        for image_key in image_keys
    }
    
    # 状态特征
    features[OBS_STATE] = PolicyFeature(
        type=FeatureType.STATE,
        shape=(int(config.state_dim),)
    )
    
    return features

def _build_output_features(config):
    """构建输出特征"""
    return {
        ACTION: PolicyFeature(
            type=FeatureType.ACTION,
            shape=(int(config.action_dim),)
        )
    }
```

### 5. 数据转换

```python
def _examples_to_act_batch(self, examples, require_action=True):
    """将示例数据转换为 ACT 批次格式"""
    
    framework_config = self.config.framework
    image_keys = list(framework_config.get("image_keys", ACTDefaultConfig.image_keys))
    action_dim = int(framework_config.action_dim)
    state_dim = int(framework_config.state_dim)
    image_height, image_width = _get_image_size(framework_config)
    target_image_size = (image_width, image_height)
    
    batch = {}
    
    # 1. 处理图像
    for view_idx, image_key in enumerate(image_keys):
        view_frames = [example["image"][view_idx] for example in examples]
        
        # 快速路径：如果图像尺寸匹配，直接转换
        if isinstance(view_frames[0], np.ndarray) and all(
            isinstance(f, np.ndarray) and f.shape[:2] == (image_height, image_width)
            for f in view_frames
        ):
            arr = np.stack(view_frames, axis=0)
            t = torch.from_numpy(arr).to(device, non_blocking=True)
            t = t.permute(0, 3, 1, 2).to(torch.float32).div_(255.0).to(model_dtype)
            batch[f"observation.images.{image_key}"] = t
            continue
        
        # 慢速路径：需要调整大小
        images = []
        for example in examples:
            image = example["image"][view_idx]
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            image = image.convert("RGB")
            if image.size != target_image_size:
                image = image.resize(target_image_size)
            image_array = np.asarray(image, dtype=np.float32) / 255.0
            image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)
            images.append(image_tensor)
        batch[f"observation.images.{image_key}"] = torch.stack(images, dim=0)
    
    # 2. 处理状态
    states = []
    for example in examples:
        state = example.get("state", np.zeros(state_dim, dtype=np.float32))
        state_tensor = torch.as_tensor(state, dtype=model_dtype, device=device)
        states.append(state_tensor)
    batch[OBS_STATE] = torch.stack(states, dim=0)
    
    # 3. 处理动作 (如果需要)
    if require_action:
        actions = []
        for example in examples:
            action_tensor = torch.as_tensor(example["action"], dtype=model_dtype, device=device)
            actions.append(action_tensor[-chunk_size:])
        batch[ACTION] = torch.stack(actions, dim=0)
        batch["action_is_pad"] = torch.zeros(len(examples), chunk_size, dtype=torch.bool)
    
    return batch
```

### 6. 前向传播

```python
def forward(self, examples, **kwargs):
    """训练前向传播"""
    batch = self._examples_to_act_batch(examples)
    output = self.action_model.forward(batch)
    loss = output[0] if isinstance(output, (tuple, list)) else output
    return {"action_loss": loss}

@torch.inference_mode()
def predict_action(self, examples, **kwargs):
    """推理预测动作"""
    batch = self._examples_to_act_batch(examples, require_action=False)
    
    # 自动混合精度
    model_param = next(self.action_model.parameters())
    autocast_enabled = model_param.dtype in (torch.float16, torch.bfloat16)
    
    if autocast_enabled:
        with torch.autocast(model_param.device.type, dtype=model_param.dtype):
            action_chunk = self.action_model.predict_action_chunk(batch)
    else:
        action_chunk = self.action_model.predict_action_chunk(batch)
    
    normalized_actions = action_chunk.detach().float().cpu().numpy()
    return {"normalized_actions": normalized_actions}
```

---

## 🧩 模块化设计详解

### 1. 视觉编码器

```python
# 在 ACT 中，视觉编码器由 LeRobot 的 ACTPolicy 提供
# 默认使用 ResNet-18

# 如果需要自定义，可以替换：
class CustomVisualEncoder(nn.Module):
    def __init__(self, backbone="resnet18"):
        super().__init__()
        if backbone == "resnet18":
            self.encoder = torchvision.models.resnet18(pretrained=True)
        elif backbone == "vit":
            self.encoder = VisionTransformer()
```

### 2. 语言编码器

```python
# 在 VLM4A 框架中，语言编码器由 Qwen 提供
class QwenLanguageEncoder:
    def __init__(self, model_path):
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
    
    def encode(self, text):
        return self.model.encode(text)
```

### 3. 动作预测头

```python
# STARVLA 支持多种动作头
# 1. ACT (Action Chunking with Transformers)
# 2. DiT (Diffusion Transformer)
# 3. FAST (Fast Action Sampling)
# 4. Flow-matching (π 风格)

# 示例：Flow-matching 动作头
class FlowMatchingActionHead(nn.Module):
    def __init__(self, action_dim, hidden_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(action_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, x, t):
        # t 是时间步
        return self.network(torch.cat([x, t], dim=-1))
```

---

## 🔄 模型组装流程

### 1. 配置驱动

```yaml
# 配置文件示例
framework:
  name: "ACT"  # 指定框架类型
  chunk_size: 50
  action_dim: 20
  state_dim: 20
  image_keys: [cam_high_rgb, cam_left_wrist_rgb, cam_right_wrist_rgb]
  image_size: [160, 160]
```

### 2. 自动组装

```python
# 1. 加载配置
cfg = OmegaConf.load("config.yaml")

# 2. 构建框架
model = build_framework(cfg)

# 3. 模型结构
print(model)
# ACT(
#   (action_model): ACTPolicy(
#     (encoder): ResNet18(...)
#     (decoder): TransformerDecoder(...)
#   )
# )
```

### 3. 权重加载

```python
# 从检查点加载权重
model = baseframework.from_pretrained("path/to/checkpoint")

# 内部流程：
# 1. 读取配置和归一化统计
model_config, norm_stats = read_mode_config(checkpoint_path)

# 2. 构建模型
model = build_framework(cfg=model_config)

# 3. 加载权重
model.load_state_dict(state_dict, strict=True)

# 4. 设置归一化统计
model.norm_stats = norm_stats
```

---

## 🎯 关键设计模式

### 1. 注册表模式

```python
# 框架注册表
FRAMEWORK_REGISTRY = Registry()

# 注册新框架
@FRAMEWORK_REGISTRY.register("MyFramework")
class MyFramework(baseframework):
    pass

# 使用框架
model_class = FRAMEWORK_REGISTRY["MyFramework"]
model = model_class(config)
```

### 2. 配置合并

```python
def merge_framework_config(DefaultConfig, user_config):
    """合并默认配置和用户配置"""
    default_dict = dataclasses.asdict(DefaultConfig())
    merged = OmegaConf.merge(default_dict, user_config)
    return merged
```

### 3. 特征抽象

```python
@dataclass
class PolicyFeature:
    type: FeatureType  # VISUAL, STATE, ACTION
    shape: tuple       # 特征形状
    # ... 其他属性
```

---

## 🚀 创建自定义模型

### 1. 步骤 1: 定义配置

```python
@dataclass
class MyModelDefaultConfig:
    name: str = "MyModel"
    action_dim: int = 7
    hidden_dim: int = 512
    # ... 其他参数
```

### 2. 步骤 2: 注册框架

```python
@FRAMEWORK_REGISTRY.register("MyModel")
class MyModel(baseframework):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = merge_framework_config(MyModelDefaultConfig, config)
        
        # 创建组件
        self.visual_encoder = create_visual_encoder()
        self.language_encoder = create_language_encoder()
        self.action_head = create_action_head()
```

### 3. 步骤 3: 实现前向传播

```python
    def forward(self, examples, **kwargs):
        # 1. 数据预处理
        batch = self._preprocess(examples)
        
        # 2. 特征提取
        visual_features = self.visual_encoder(batch["images"])
        language_features = self.language_encoder(batch["text"])
        
        # 3. 特征融合
        fused_features = self.fusion(visual_features, language_features)
        
        # 4. 动作预测
        actions = self.action_head(fused_features)
        
        # 5. 计算损失
        loss = self.compute_loss(actions, batch["target_actions"])
        
        return {"action_loss": loss}
```

---

## 📊 与 LeRobot 的集成

### 1. 数据格式兼容

```python
# STARVLA 直接使用 LeRobot 数据格式
from starVLA.dataloader.lerobot_datasets import get_vla_dataset

dataset = get_vla_dataset({
    "dataset_py": "lerobot_datasets",
    "data_root_dir": "/path/to/lerobot/data"
})
```

### 2. ACT 策略复用

```python
# ACT 框架直接使用 LeRobot 的 ACTPolicy
from lerobot.policies.act.modeling_act import ACTPolicy

class ACT(baseframework):
    def __init__(self, config):
        # 直接使用 LeRobot 的 ACTPolicy
        self.action_model = ACTPolicy(act_config)
```

---

## 🎉 总结

### STARVLA 模型搭建的关键点

1. **模块化设计**: 每个组件都可以独立替换
2. **配置驱动**: 通过 YAML 配置控制模型结构
3. **注册表模式**: 方便扩展新架构
4. **特征抽象**: 统一的输入输出接口
5. **LeRobot 集成**: 直接使用 LeRobot 组件

### 你的 ACT 模型结构

```
ACT 框架
├── 视觉编码器: ResNet-18 (来自 LeRobot)
├── 状态编码器: MLP
├── 动作解码器: Transformer (来自 LeRobot)
└── 损失函数: MSE + 平滑损失
```

### 扩展性

要创建新的模型架构，只需：
1. 继承 `baseframework`
2. 注册到 `FRAMEWORK_REGISTRY`
3. 实现 `forward()` 和 `predict_action()`

**STARVLA 的模块化设计让模型搭建像搭积木一样简单！** 🚀
