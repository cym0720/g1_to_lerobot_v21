# ACT 模型搭建详解：结合源码的完整讲解

## 🎯 ACT 是什么？

**ACT (Action Chunking with Transformers)** 是一个用于机器人动作预测的 Transformer 模型，由 Tony Z. Zhao 等人在论文 "Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware" 中提出。

---

## 📁 源码结构

```
ACT 模型源码位置:
├── STARVLA 框架层
│   └── external/starVLA/starVLA/model/framework/VM4A/ACT.py
│
└── LeRobot 实现层
    └── D:\Anaconda\Anaconda\Lib\site-packages\lerobot\policies\act\
        ├── configuration_act.py  # 配置类
        ├── modeling_act.py       # 模型实现
        └── processor_act.py      # 数据处理器
```

---

## 🏗️ ACT 模型架构

### 整体架构图

```
输入数据
├── 图像 (多相机) ──→ ResNet 骨干网络 ──→ 图像特征
├── 机器人状态 ──→ 线性投影 ──→ 状态特征
└── 动作序列 (训练时) ──→ VAE 编码器 ──→ 潜在变量
        │
        ▼
    Transformer 编码器
        │
        ▼
    Transformer 解码器
        │
        ▼
    动作预测头
        │
        ▼
    输出动作序列
```

---

## 🔧 源码详解

### 1. 配置类：`ACTConfig`

```python
# 文件: lerobot/policies/act/configuration_act.py

@dataclass
class ACTConfig:
    """ACT 模型配置"""
    
    # 基础配置
    chunk_size: int = 50           # 动作块大小
    n_action_steps: int = 50       # 动作执行步数
    n_obs_steps: int = 1           # 观察步数
    
    # 模型维度
    dim_model: int = 512           # Transformer 隐藏维度
    n_heads: int = 8               # 注意力头数
    n_encoder_layers: int = 4      # 编码器层数
    n_decoder_layers: int = 7      # 解码器层数
    dim_feedforward: int = 3200    # 前馈网络维度
    dropout: float = 0.1           # Dropout 概率
    
    # VAE 配置
    use_vae: bool = True           # 是否使用 VAE
    latent_dim: int = 32           # 潜在维度
    n_vae_encoder_layers: int = 4  # VAE 编码器层数
    kl_weight: float = 10.0        # KL 散度权重
    
    # 视觉配置
    vision_backbone: str = "resnet18"  # 视觉骨干网络
    pretrained_backbone_weights: str = "ResNet18_Weights.IMAGENET1K_V1"
    
    # 特征配置
    image_features: dict = None    # 图像特征配置
    robot_state_feature: dict = None  # 机器人状态特征
    action_feature: dict = None    # 动作特征配置
```

### 2. 模型类：`ACT`

```python
# 文件: lerobot/policies/act/modeling_act.py

class ACT(nn.Module):
    """Action Chunking Transformer: ACTPolicy 的底层神经网络"""
    
    def __init__(self, config: ACTConfig):
        super().__init__()
        self.config = config
        
        # 1. VAE 编码器 (可选)
        if self.config.use_vae:
            self.vae_encoder = ACTEncoder(config, is_vae_encoder=True)
            self.vae_encoder_cls_embed = nn.Embedding(1, config.dim_model)
            
            # 状态投影层
            if self.config.robot_state_feature:
                self.vae_encoder_robot_state_input_proj = nn.Linear(
                    self.config.robot_state_feature.shape[0], config.dim_model
                )
            
            # 动作投影层
            self.vae_encoder_action_input_proj = nn.Linear(
                self.config.action_feature.shape[0], config.dim_model
            )
            
            # 潜在分布投影层
            self.vae_encoder_latent_output_proj = nn.Linear(
                config.dim_model, config.latent_dim * 2
            )
        
        # 2. 视觉骨干网络 (ResNet)
        if self.config.image_features:
            backbone_model = getattr(torchvision.models, config.vision_backbone)(
                replace_stride_with_dilation=[False, False, config.replace_final_stride_with_dilation],
                weights=config.pretrained_backbone_weights,
                norm_layer=FrozenBatchNorm2d,
            )
            # 提取 ResNet 的中间层特征
            self.backbone = IntermediateLayerGetter(
                backbone_model, return_layers={"layer4": "feature_map"}
            )
        
        # 3. 编码器输入投影层
        # 潜在变量投影
        self.encoder_latent_input_proj = nn.Linear(config.latent_dim, config.dim_model)
        
        # 状态投影
        if self.config.robot_state_feature:
            self.encoder_robot_state_input_proj = nn.Linear(
                self.config.robot_state_feature.shape[0], config.dim_model
            )
        
        # 环境状态投影
        if self.config.env_state_feature:
            self.encoder_env_state_input_proj = nn.Linear(
                self.config.env_state_feature.shape[0], config.dim_model
            )
        
        # 图像特征投影
        if self.config.image_features:
            self.encoder_img_feat_input_proj = nn.Conv2d(
                backbone_model.fc.in_features, config.dim_model, kernel_size=1
            )
        
        # 4. 位置编码
        # 1D 位置编码 (用于潜在变量、状态等)
        self.encoder_1d_feature_pos_embed = nn.Embedding(
            self._get_encoder_1d_feature_count(), config.dim_model
        )
        
        # 2D 位置编码 (用于图像特征)
        if self.config.image_features:
            self.encoder_cam_feat_pos_embed = ACTSinusoidalPositionEmbedding2d(
                config.dim_model // 2
            )
        
        # 5. Transformer 编码器和解码器
        self.encoder = ACTEncoder(config)
        self.decoder = ACTDecoder(config)
        
        # 6. 解码器位置编码
        self.decoder_pos_embed = nn.Embedding(config.chunk_size, config.dim_model)
        
        # 7. 动作预测头
        self.action_head = nn.Linear(config.dim_model, self.config.action_feature.shape[0])
        
        # 8. 参数初始化
        self._reset_parameters()
```

### 3. 前向传播：`forward()`

```python
def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, tuple[Tensor, Tensor] | tuple[None, None]]:
    """ACT 的前向传播"""
    
    # 1. 准备潜在变量
    if self.config.use_vae and ACTION in batch and self.training:
        # 训练时使用 VAE 编码器
        # 准备 VAE 编码器输入: [cls, robot_state, action_sequence]
        cls_embed = einops.repeat(
            self.vae_encoder_cls_embed.weight, "1 d -> b 1 d", b=batch_size
        )
        
        if self.config.robot_state_feature:
            robot_state_embed = self.vae_encoder_robot_state_input_proj(batch[OBS_STATE])
            robot_state_embed = robot_state_embed.unsqueeze(1)
        
        action_embed = self.vae_encoder_action_input_proj(batch[ACTION])
        
        # 拼接输入
        if self.config.robot_state_feature:
            vae_encoder_input = [cls_embed, robot_state_embed, action_embed]
        else:
            vae_encoder_input = [cls_embed, action_embed]
        vae_encoder_input = torch.cat(vae_encoder_input, axis=1)
        
        # 位置编码
        pos_embed = self.vae_encoder_pos_enc.clone().detach()
        
        # 创建填充掩码
        cls_joint_is_pad = torch.full(
            (batch_size, 2 if self.config.robot_state_feature else 1),
            False,
            device=batch[OBS_STATE].device,
        )
        key_padding_mask = torch.cat([cls_joint_is_pad, batch["action_is_pad"]], axis=1)
        
        # VAE 编码器前向传播
        cls_token_out = self.vae_encoder(
            vae_encoder_input.permute(1, 0, 2),
            pos_embed=pos_embed.permute(1, 0, 2),
            key_padding_mask=key_padding_mask,
        )[0]
        
        # 获取潜在分布参数
        latent_pdf_params = self.vae_encoder_latent_output_proj(cls_token_out)
        mu = latent_pdf_params[:, :self.config.latent_dim]
        log_sigma_x2 = latent_pdf_params[:, self.config.latent_dim:]
        
        # 重参数化技巧采样
        latent_sample = mu + log_sigma_x2.div(2).exp() * torch.randn_like(mu)
    else:
        # 推理时使用零潜在变量
        mu = log_sigma_x2 = None
        latent_sample = torch.zeros([batch_size, self.config.latent_dim], dtype=torch.float32).to(
            batch[OBS_STATE].device
        )
    
    # 2. 准备 Transformer 编码器输入
    encoder_in_tokens = [self.encoder_latent_input_proj(latent_sample)]
    encoder_in_pos_embed = list(self.encoder_1d_feature_pos_embed.weight.unsqueeze(1))
    
    # 添加机器人状态 token
    if self.config.robot_state_feature:
        encoder_in_tokens.append(self.encoder_robot_state_input_proj(batch[OBS_STATE]))
    
    # 添加环境状态 token
    if self.config.env_state_feature:
        encoder_in_tokens.append(self.encoder_env_state_input_proj(batch[OBS_ENV_STATE]))
    
    # 3. 处理图像特征
    if self.config.image_features:
        for img in batch[OBS_IMAGES]:
            # 提取图像特征
            cam_features = self.backbone(img)["feature_map"]
            
            # 计算 2D 位置编码
            cam_pos_embed = self.encoder_cam_feat_pos_embed(cam_features).to(dtype=cam_features.dtype)
            
            # 投影到模型维度
            cam_features = self.encoder_img_feat_input_proj(cam_features)
            
            # 重排为序列格式: (H*W, B, C)
            cam_features = einops.rearrange(cam_features, "b c h w -> (h w) b c")
            cam_pos_embed = einops.rearrange(cam_pos_embed, "b c h w -> (h w) b c")
            
            # 添加到编码器输入
            encoder_in_tokens.extend(list(cam_features))
            encoder_in_pos_embed.extend(list(cam_pos_embed))
    
    # 4. 堆叠所有 token
    encoder_in_tokens = torch.stack(encoder_in_tokens, axis=0)
    encoder_in_pos_embed = torch.stack(encoder_in_pos_embed, axis=0)
    
    # 5. Transformer 编码器前向传播
    encoder_out = self.encoder(encoder_in_tokens, pos_embed=encoder_in_pos_embed)
    
    # 6. 准备 Transformer 解码器输入
    decoder_in = torch.zeros(
        (self.config.chunk_size, batch_size, self.config.dim_model),
        dtype=encoder_in_pos_embed.dtype,
        device=encoder_in_pos_embed.device,
    )
    
    # 7. Transformer 解码器前向传播
    decoder_out = self.decoder(
        decoder_in,
        encoder_out,
        encoder_pos_embed=encoder_in_pos_embed,
        decoder_pos_embed=self.decoder_pos_embed.weight.unsqueeze(1),
    )
    
    # 8. 重排回批次格式: (B, S, C)
    decoder_out = decoder_out.transpose(0, 1)
    
    # 9. 动作预测
    actions = self.action_head(decoder_out)
    
    return actions, (mu, log_sigma_x2)
```

---

## 🧩 核心组件详解

### 1. 视觉骨干网络 (ResNet)

```python
# 使用 torchvision 的 ResNet
backbone_model = getattr(torchvision.models, config.vision_backbone)(
    replace_stride_with_dilation=[False, False, config.replace_final_stride_with_dilation],
    weights=config.pretrained_backbone_weights,
    norm_layer=FrozenBatchNorm2d,
)

# 提取中间层特征
self.backbone = IntermediateLayerGetter(
    backbone_model, return_layers={"layer4": "feature_map"}
)

# 使用方式
cam_features = self.backbone(img)["feature_map"]  # (B, C, H, W)
```

### 2. VAE 编码器

```python
class ACTEncoder(nn.Module):
    """Transformer 编码器"""
    
    def __init__(self, config: ACTConfig, is_vae_encoder: bool = False):
        super().__init__()
        self.is_vae_encoder = is_vae_encoder
        num_layers = config.n_vae_encoder_layers if self.is_vae_encoder else config.n_encoder_layers
        self.layers = nn.ModuleList([ACTEncoderLayer(config) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(config.dim_model) if config.pre_norm else nn.Identity()
    
    def forward(self, x, pos_embed=None, key_padding_mask=None):
        for layer in self.layers:
            x = layer(x, pos_embed=pos_embed, key_padding_mask=key_padding_mask)
        x = self.norm(x)
        return x
```

### 3. Transformer 编码器层

```python
class ACTEncoderLayer(nn.Module):
    """Transformer 编码器层"""
    
    def __init__(self, config: ACTConfig):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(config.dim_model, config.n_heads, dropout=config.dropout)
        
        # 前馈网络
        self.linear1 = nn.Linear(config.dim_model, config.dim_feedforward)
        self.dropout = nn.Dropout(config.dropout)
        self.linear2 = nn.Linear(config.dim_feedforward, config.dim_model)
        
        # 层归一化
        self.norm1 = nn.LayerNorm(config.dim_model)
        self.norm2 = nn.LayerNorm(config.dim_model)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)
        
        self.activation = get_activation_fn(config.feedforward_activation)
        self.pre_norm = config.pre_norm
    
    def forward(self, x, pos_embed=None, key_padding_mask=None):
        # 残差连接
        skip = x
        
        # 预归一化
        if self.pre_norm:
            x = self.norm1(x)
        
        # 自注意力
        q = k = x if pos_embed is None else x + pos_embed
        x = self.self_attn(q, k, value=x, key_padding_mask=key_padding_mask)
        x = x[0]  # 只取输出，不取注意力权重
        x = skip + self.dropout1(x)
        
        # 前馈网络
        if self.pre_norm:
            skip = x
            x = self.norm2(x)
        else:
            x = self.norm1(x)
            skip = x
        
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = skip + self.dropout2(x)
        
        if not self.pre_norm:
            x = self.norm2(x)
        
        return x
```

### 4. Transformer 解码器

```python
class ACTDecoder(nn.Module):
    """Transformer 解码器"""
    
    def __init__(self, config: ACTConfig):
        super().__init__()
        self.layers = nn.ModuleList([ACTDecoderLayer(config) for _ in range(config.n_decoder_layers)])
        self.norm = nn.LayerNorm(config.dim_model)
    
    def forward(self, x, encoder_out, decoder_pos_embed=None, encoder_pos_embed=None):
        for layer in self.layers:
            x = layer(x, encoder_out, decoder_pos_embed=decoder_pos_embed, encoder_pos_embed=encoder_pos_embed)
        if self.norm is not None:
            x = self.norm(x)
        return x
```

### 5. Transformer 解码器层

```python
class ACTDecoderLayer(nn.Module):
    """Transformer 解码器层"""
    
    def __init__(self, config: ACTConfig):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(config.dim_model, config.n_heads, dropout=config.dropout)
        self.multihead_attn = nn.MultiheadAttention(config.dim_model, config.n_heads, dropout=config.dropout)
        
        # 前馈网络
        self.linear1 = nn.Linear(config.dim_model, config.dim_feedforward)
        self.dropout = nn.Dropout(config.dropout)
        self.linear2 = nn.Linear(config.dim_feedforward, config.dim_model)
        
        # 层归一化
        self.norm1 = nn.LayerNorm(config.dim_model)
        self.norm2 = nn.LayerNorm(config.dim_model)
        self.norm3 = nn.LayerNorm(config.dim_model)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)
        self.dropout3 = nn.Dropout(config.dropout)
        
        self.activation = get_activation_fn(config.feedforward_activation)
        self.pre_norm = config.pre_norm
    
    def forward(self, x, encoder_out, decoder_pos_embed=None, encoder_pos_embed=None):
        # 残差连接
        skip = x
        
        # 预归一化
        if self.pre_norm:
            x = self.norm1(x)
        
        # 自注意力
        q = k = self.maybe_add_pos_embed(x, decoder_pos_embed)
        x = self.self_attn(q, k, value=x)[0]
        x = skip + self.dropout1(x)
        
        # 交叉注意力
        if self.pre_norm:
            skip = x
            x = self.norm2(x)
        else:
            x = self.norm1(x)
            skip = x
        
        x = self.multihead_attn(
            query=self.maybe_add_pos_embed(x, decoder_pos_embed),
            key=self.maybe_add_pos_embed(encoder_out, encoder_pos_embed),
            value=encoder_out,
        )[0]
        x = skip + self.dropout2(x)
        
        # 前馈网络
        if self.pre_norm:
            skip = x
            x = self.norm3(x)
        else:
            x = self.norm2(x)
            skip = x
        
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = skip + self.dropout3(x)
        
        if not self.pre_norm:
            x = self.norm3(x)
        
        return x
```

---

## 🔄 数据流详解

### 1. 输入数据格式

```python
batch = {
    # 图像数据: (B, n_cameras, C, H, W)
    "observation.images.cam_high_rgb": torch.Tensor,
    "observation.images.cam_left_wrist_rgb": torch.Tensor,
    "observation.images.cam_right_wrist_rgb": torch.Tensor,
    
    # 机器人状态: (B, state_dim)
    "observation.state": torch.Tensor,
    
    # 动作序列 (训练时): (B, chunk_size, action_dim)
    "action": torch.Tensor,
    
    # 动作填充掩码: (B, chunk_size)
    "action_is_pad": torch.Tensor,
}
```

### 2. 数据流图

```
输入批次
    │
    ├── 图像: (B, 3, 3, 160, 160)
    │   │
    │   ▼
    │   ResNet-18 骨干网络
    │   │
    │   ▼
    │   特征图: (B, 512, 5, 5)
    │   │
    │   ▼
    │   2D 位置编码
    │   │
    │   ▼
    │   卷积投影: (B, 512, 5, 5) → (B, 512, 5, 5)
    │   │
    │   ▼
    │   重排序列: (25, B, 512)
    │
    ├── 状态: (B, 20)
    │   │
    │   ▼
    │   线性投影: (B, 512)
    │
    ├── 潜在变量: (B, 32)
    │   │
    │   ▼
    │   线性投影: (B, 512)
    │
    ▼
堆叠所有 token: (27, B, 512)  # 25(图像) + 1(状态) + 1(潜在)
    │
    ▼
Transformer 编码器 (4层)
    │
    ▼
编码器输出: (27, B, 512)
    │
    ▼
零初始化解码器输入: (50, B, 512)  # chunk_size=50
    │
    ▼
Transformer 解码器 (7层)
    │
    ▼
解码器输出: (50, B, 512)
    │
    ▼
动作预测头: (50, B, 512) → (50, B, 20)
    │
    ▼
输出动作序列: (B, 50, 20)
```

---

## 🎯 关键设计点

### 1. 动作分块 (Action Chunking)

```python
# ACT 预测一个动作序列，而不是单个动作
# chunk_size = 50 表示一次预测 50 个时间步的动作

# 优势:
# 1. 捕获动作的时间相关性
# 2. 提高动作平滑性
# 3. 减少推理频率
```

### 2. 变分自编码器 (VAE)

```python
# VAE 用于建模动作序列的分布
# 训练时: 使用 VAE 编码器学习潜在表示
# 推理时: 使用标准正态分布采样

# 优势:
# 1. 处理动作的多模态性
# 2. 生成多样化的动作
# 3. 提高泛化能力
```

### 3. 位置编码

```python
# 1D 位置编码: 用于潜在变量、状态等
self.encoder_1d_feature_pos_embed = nn.Embedding(...)

# 2D 位置编码: 用于图像特征
self.encoder_cam_feat_pos_embed = ACTSinusoidalPositionEmbedding2d(...)

# 优势:
# 1. 保持序列顺序信息
# 2. 处理不同长度的输入
# 3. 保持空间关系
```

---

## 🚀 STARVLA 集成

### 1. STARVLA 的 ACT 包装器

```python
# 文件: external/starVLA/starVLA/model/framework/VM4A/ACT.py

@FRAMEWORK_REGISTRY.register("ACT")
class ACT(baseframework):
    def __init__(self, config, **kwargs):
        super().__init__()
        
        # 合并配置
        self.config = merge_framework_config(ACTDefaultConfig, config)
        framework_config = self.config.framework
        
        # 创建 ACT 配置
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
        
        # 创建 ACTPolicy 模型
        self.action_model = ACTPolicy(act_config)
        
        # 设置归一化器为恒等映射
        self._set_identity_normalizers()
```

### 2. 数据转换

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

---

## 📊 模型参数统计

### 典型配置下的参数量

```
组件参数量:
├── ResNet-18 骨干网络: ~11M
├── VAE 编码器: ~2M
├── Transformer 编码器: ~4M
├── Transformer 解码器: ~7M
├── 投影层: ~1M
└── 动作头: ~0.01M

总参数量: ~25M
```

---

## 🎉 总结

### ACT 模型的关键特点

1. **动作分块**: 一次预测多个时间步的动作
2. **VAE 架构**: 处理动作的多模态性
3. **Transformer**: 捕获长程依赖关系
4. **多相机输入**: 支持多个相机视角
5. **端到端训练**: 从图像直接到动作

### STARVLA 的 ACT 实现

- ✅ **直接兼容 LeRobot**: 使用 LeRobot 的 ACTPolicy
- ✅ **模块化设计**: 可以替换骨干网络或动作头
- ✅ **标准化接口**: 统一的训练和推理接口
- ✅ **配置驱动**: 通过 YAML 配置控制模型结构

### 你的 ACT 模型

既然你使用 STARVLA 的 ACT 框架，你的模型结构是:

```
ACT 框架 (STARVLA)
└── ACTPolicy (LeRobot)
    ├── ResNet-18 视觉编码器
    ├── VAE 编码器 (可选)
    ├── Transformer 编码器 (4层)
    ├── Transformer 解码器 (7层)
    └── 动作预测头
```

**STARVLA + LeRobot = 强大的 ACT 实现！** 🚀
