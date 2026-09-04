# 实际操作示例：从 LeRobot 到 STAR-VLA

## 📋 场景描述
你已经有 LeRobot 格式的数据集，想要用 STAR-VLA 训练 ACT 模型。

---

## 🎯 步骤 1: 检查你的 LeRobot 数据

```bash
# 查看数据集结构
ls -la /path/to/your/lerobot/dataset/

# 应该看到类似这样的结构：
# dataset/
# ├── images/
# │   ├── cam_high_rgb/
# │   ├── cam_left_wrist_rgb/
# │   └── cam_right_wrist_rgb/
# ├── actions.npy
# ├── states.npy
# └── metadata.json
```

---

## 🎯 步骤 2: 创建 STAR-VLA 配置文件

```yaml
# train_my_act.yaml
run_id: my_act_training
run_root_dir: output/
seed: 42

framework:
  name: ACT  # 使用 ACT 架构
  chunk_size: 50  # 动作块大小
  action_dim: 20  # 你的动作维度
  state_dim: 20   # 状态维度
  n_obs_steps: 1
  image_size: [160, 160]  # 图像尺寸
  image_keys: [cam_high_rgb, cam_left_wrist_rgb, cam_right_wrist_rgb]
  pretrained_backbone_weights: null

datasets:
  vla_data:
    dataset_py: lerobot_datasets  # 关键：使用 LeRobot 数据加载器
    include_state: true
    data_root_dir: /path/to/your/lerobot/dataset  # 你的数据路径
    data_mix: your_task_name
    default_prompt: "perform the requested task"
    per_device_batch_size: 2
    load_all_data_for_training: true
    obs_image_size: [160, 160]
    video_backend: torchvision_av

trainer:
  max_train_steps: 10000
  num_warmup_steps: 500
  save_interval: 1000
  eval_interval: 1000
  learning_rate:
    base: 2.5e-05
    action_model: 1.0e-04
  lr_scheduler_type: cosine_with_min_lr
  scheduler_specific_kwargs:
    min_lr: 1.0e-06
  freeze_modules: ""
  loss_scale:
    vla: 1.0
    vlm: 0.0
  max_grad_norm: 1.0
  weight_decay: 0.0
  logging_frequency: 20
  gradient_clipping: 1.0
  gradient_accumulation_steps: 4
  gradient_checkpointing: true
  optimizer:
    name: AdamW
    betas: [0.9, 0.95]
    eps: 1.0e-08
    weight_decay: 1.0e-08
```

---

## 🎯 步骤 3: 启动训练

```bash
# 设置环境变量
export PYTHONPATH=$PYTHONPATH:$(pwd)/external/starVLA

# 启动训练
accelerate launch --num_processes 1 \
    starVLA/training/train_starvla.py \
    --config_yaml train_my_act.yaml \
    --framework.name ACT \
    --datasets.vla_data.data_root_dir /path/to/your/lerobot/dataset \
    --datasets.vla_data.data_mix your_task_name
```

---

## 🎯 步骤 4: 监控训练

### 查看日志
```bash
# 实时查看训练日志
tail -f output/my_act_training/logs/training.log

# 查看关键指标
grep "Step.*Loss:" output/my_act_training/logs/training.log | tail -10
```

### WandB 监控 (可选)
```yaml
# 在配置文件中添加
wandb_project: my_starvla_training
wandb_entity: your_username
```

---

## 🎯 步骤 5: 评估模型

```bash
# 评估模型性能
python starVLA/evaluation/eval_starvla.py \
    --ckpt_path output/my_act_training/final_model \
    --eval_task your_task_name
```

---

## 🎯 步骤 6: 部署服务器

### 启动服务器
```bash
# 使用 STAR-VLA 标准服务器
python deployment/model_server/server_policy.py \
    --ckpt_path output/my_act_training/final_model \
    --port 5000
```

### 客户端连接
```python
from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy

# 连接到服务器
client = WebsocketClientPolicy(host="localhost", port=5000)

# 获取模型元数据
metadata = client.metadata
print(f"动作块大小: {metadata['action_chunk_size']}")
print(f"动作维度: {metadata['action_dim']}")

# 准备数据
images = {
    "cam_high_rgb": your_high_camera_image,
    "cam_left_wrist_rgb": your_left_camera_image,
    "cam_right_wrist_rgb": your_right_camera_image
}

# 预测动作
actions = client.predict(images)
print(f"预测动作形状: {actions.shape}")
```

---

## 🔄 与 OpenPI 的对比

### 相似点
```python
# OpenPI 风格
from openpi_client import OpenPIPolicyClient
client = OpenPIPolicyClient(host="localhost", port=5000)
actions = client.predict(images, instruction)

# STAR-VLA 风格 (几乎相同)
from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy
client = WebsocketClientPolicy(host="localhost", port=5000)
actions = client.predict(images, instruction)
```

### 主要区别
| 方面 | OpenPI | STAR-VLA |
|------|--------|----------|
| **数据格式** | 自定义 | **直接使用 LeRobot** |
| **服务器** | 自定义实现 | **标准化框架** |
| **配置** | 手动配置 | **从检查点自动读取** |
| **动作归一化** | 客户端处理 | **服务器端处理** |

---

## 🚨 常见问题解决

### 问题 1: 数据加载失败
```bash
ModuleNotFoundError: No module named 'lerobot'
```
**解决**: 
```bash
pip install lerobot
```

### 问题 2: 图像尺寸不匹配
```bash
RuntimeError: Expected input images to have shape [B, 3, 160, 160]
```
**解决**: 
- 检查你的图像尺寸是否为 160x160
- 或在配置中修改 `image_size`

### 问题 3: 动作维度不匹配
```bash
RuntimeError: Expected action_dim=20, got 7
```
**解决**: 
- 检查你的动作维度
- 在配置中修改 `action_dim`

### 问题 4: 服务器连接失败
```bash
ConnectionRefusedError: [Errno 111] Connection refused
```
**解决**:
1. 检查服务器是否启动
2. 检查端口是否正确
3. 检查防火墙设置

---

## 📊 性能优化建议

### 1. 批量大小调整
```yaml
# 根据你的 GPU 内存调整
per_device_batch_size: 4  # 如果内存足够，可以增大
gradient_accumulation_steps: 2  # 或使用梯度累积
```

### 2. 混合精度训练
```bash
# 启用 bf16 混合精度
accelerate launch --mixed_precision bf16 ...
```

### 3. 多 GPU 训练
```bash
# 使用多个 GPU
accelerate launch --num_processes 4 ...
```

---

## 🎉 总结

从 LeRobot 到 STAR-VLA 的迁移非常简单：

1. **数据**: 直接使用，无需转换
2. **配置**: 创建 YAML 配置文件
3. **训练**: 使用标准化训练脚本
4. **部署**: 使用标准化服务器

**你的优势**：
- ✅ 熟悉 LeRobot 数据格式
- ✅ 了解训练流程
- ✅ 有服务器部署经验

**STAR-VLA 给你的**：
- 🎯 更强大的模型架构
- 🎯 更标准化的流程
- 🎯 更活跃的社区支持

开始你的 STAR-VLA 之旅吧！🚀
