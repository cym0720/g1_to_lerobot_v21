# STAR-VLA 快速参考卡
## 专为有 LeRobot 和 OpenPI 经验的开发者

---

## 🔄 数据格式转换

### LeRobot → STAR-VLA
```python
# 直接使用，无需转换！
# 只需在配置中指定 dataset_py: lerobot_datasets
```

### OpenPI → STAR-VLA
```python
# 如果需要从 OpenPI 迁移，只需调整数据格式
# STAR-VLA 支持多种数据格式
```

---

## 📁 目录结构速查

```
your_project/
├── data/                    # LeRobot 数据集
│   ├── task_01/
│   ├── task_02/
│   └── ...
├── config/                  # 训练配置
│   └── train_config.yaml
├── output/                  # 模型输出
│   └── starvla_act_g1/
├── external/                # STAR-VLA 框架
│   └── starVLA/
└── scripts/                 # 自定义脚本
```

---

## ⚡ 常用命令

### 训练
```bash
# 单任务训练
accelerate launch starVLA/training/train_starvla.py \
    --config_yaml your_config.yaml \
    --framework.name ACT

# 多任务训练
accelerate launch starVLA/training/train_starvla.py \
    --config_yaml your_config.yaml \
    --datasets.vla_data.data_mix "task1+task2+task3"
```

### 评估
```bash
# 评估模型
python starVLA/evaluation/eval_starvla.py \
    --ckpt_path output/your_model/final_model \
    --eval_task your_task
```

### 部署
```bash
# 启动服务器
python deployment/model_server/server_policy.py \
    --ckpt_path output/your_model/final_model \
    --port 5000

# 客户端连接
from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy
client = WebsocketClientPolicy(host="localhost", port=5000)
```

---

## 🔧 配置模板

### 基础训练配置
```yaml
# train_config.yaml
run_id: my_training
run_root_dir: output/
seed: 42

framework:
  name: ACT
  chunk_size: 50
  action_dim: 20
  state_dim: 20
  image_size: [160, 160]
  image_keys: [cam_high_rgb, cam_left_wrist_rgb, cam_right_wrist_rgb]

datasets:
  vla_data:
    dataset_py: lerobot_datasets  # 关键：使用 LeRobot 数据加载器
    data_root_dir: /path/to/your/data
    data_mix: your_task
    per_device_batch_size: 2

trainer:
  max_train_steps: 10000
  learning_rate:
    base: 2.5e-05
    action_model: 1.0e-04
  save_interval: 1000
```

---

## 📊 监控指标

### 关键指标
- **action_dit_loss**: 0.1-0.3 正常 (扩散模型损失)
- **mse_score**: <0.01 优秀 (均方误差)
- **learning_rate**: 按调度器变化

### WandB 集成
```yaml
wandb_project: my_starvla_project
wandb_entity: your_username
```

---

## 🚨 常见问题

### 1. 数据加载失败
**问题**: `ModuleNotFoundError: No module named 'lerobot'`
**解决**: `pip install lerobot`

### 2. CUDA 内存不足
**问题**: `RuntimeError: CUDA out of memory`
**解决**: 
- 减小 `per_device_batch_size`
- 使用 `gradient_checkpointing: true`
- 使用 `--use_bf16` 半精度

### 3. 图像尺寸不匹配
**问题**: 图像尺寸与配置不一致
**解决**: 确保 `image_size` 与你的数据一致

### 4. 服务器连接失败
**问题**: 无法连接到 WebSocket 服务器
**解决**:
- 检查服务器是否启动
- 检查端口是否被占用
- 检查防火墙设置

---

## 🔗 有用链接

- **官方文档**: https://starvla.github.io
- **GitHub**: https://github.com/starVLA/starVLA
- **HuggingFace**: https://huggingface.co/StarVLA
- **论文**: https://arxiv.org/abs/2604.05014

---

## 📞 获取帮助

1. **查看示例**: `examples/` 目录
2. **阅读文档**: `docs/` 目录
3. **社区讨论**: GitHub Issues
4. **微信群**: 扫描 README 中的二维码

---

## 🎯 你的优势

既然你有 LeRobot 和 OpenPI 经验：

✅ **数据**: 直接使用 LeRobot 格式  
✅ **训练**: 熟悉的训练流程  
✅ **部署**: 类似 OpenPI 的服务器  
✅ **调试**: 相似的调试方法  

**开始吧！** 🚀
