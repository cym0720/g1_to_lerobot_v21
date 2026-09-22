# STAR-VLA ACT 训练脚本

本目录包含用于训练和部署 STAR-VLA 模型的三个脚本。

## 文件说明

### 1. `model_starvla.py` - 模型定义
STAR-VLA 模型架构，包含：
- 视觉编码器 (ResNet-18)
- 语言编码器 (BERT)
- 多模态融合层
- ACT 风格的动作解码器

### 2. `train_starvla_act.py` - 训练脚本
用于训练 STAR-VLA 模型的完整训练流程，支持：
- 自定义数据集加载
- 多模态训练
- WandB 集成
- 检查点保存

### 3. `server_starvla.py` - 推理服务器
提供 REST API 用于动作预测，包含：
- `/predict` - 单个预测
- `/predict_batch` - 批量预测
- `/health` - 健康检查
- `/model_info` - 模型信息

## 环境要求

```bash
pip install torch torchvision transformers flask pillow numpy tqdm wandb
```

## 使用方法

### 训练模型

```bash
python train_starvla_act.py \
    --data_dir /path/to/your/dataset \
    --save_dir ./checkpoints \
    --batch_size 32 \
    --num_epochs 100 \
    --learning_rate 1e-4 \
    --use_wandb
```

### 启动推理服务器

```bash
python server_starvla.py \
    --model_path ./checkpoints/best_model.pth \
    --host 0.0.0.0 \
    --port 5000 \
    --device cuda
```

### API 使用示例

#### 单个预测
```bash
curl -X POST http://localhost:5000/predict \
    -H "Content-Type: application/json" \
    -d '{
        "image": "base64_encoded_image_string",
        "instruction": "Pick up the red block"
    }'
```

#### 批量预测
```bash
curl -X POST http://localhost:5000/predict_batch \
    -H "Content-Type: application/json" \
    -d '{
        "requests": [
            {
                "image": "base64_encoded_image_string_1",
                "instruction": "Pick up the red block"
            },
            {
                "image": "base64_encoded_image_string_2",
                "instruction": "Move to the left"
            }
        ]
    }'
```

## 数据格式

训练数据目录结构：
```
data_dir/
├── episode_001/
│   ├── images/
│   │   ├── 0000.png
│   │   ├── 0001.png
│   │   └── ...
│   ├── actions.npy
│   └── instruction.txt
├── episode_002/
│   └── ...
└── ...
```

- `images/`: 包含 RGB 图像序列
- `actions.npy`: NumPy 数组，形状为 `[seq_len, action_dim]`
- `instruction.txt`: 自然语言指令

## 模型架构

STAR-VLA 结合了：
1. **视觉编码**: ResNet-18 提取图像特征
2. **语言编码**: BERT 处理自然语言指令
3. **多模态融合**: 连接视觉和语言特征
4. **动作解码**: 生成动作块 (Action Chunk)

## 注意事项

1. 确保 GPU 内存足够，特别是使用 BERT 时
2. 调整 `action_chunk_size` 以适应你的任务需求
3. 可以使用 WandB 监控训练过程
4. 服务器支持批量预测以提高效率
