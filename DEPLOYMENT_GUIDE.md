# STAR-VLA ACT 模型部署指南

本指南帮助你在新电脑上部署和使用 STAR-VLA ACT 模型。

## 📦 打包内容

本次打包包含以下内容：

### 1. 脚本文件 (`starvla_act_scripts/`)
- `train_g1_act.sh` - 训练启动脚本
- `inference_starvla_act.py` - 推理脚本
- `server_starvla_act.py` - 推理服务器
- `README.md` - 使用说明

### 2. 模型文件 (`output/starvla_act_g1/`)
- `g1_act_task_04/final_model/` - 任务 04 模型
- `g1_act_task_05/final_model/` - 任务 05 模型
- `g1_act_task_06/final_model/` - 任务 06 模型

### 3. 框架代码 (`external/starVLA/`)
- STAR-VLA 框架源代码
- 部署工具和示例

## 🖥️ 环境要求

### 硬件要求
- **GPU**: NVIDIA GPU，显存 ≥ 8GB (推荐 16GB+)
- **内存**: ≥ 16GB
- **存储**: ≥ 2GB 可用空间

### 软件要求
- **操作系统**: Linux (推荐 Ubuntu 20.04/22.04)
- **Python**: 3.8-3.11
- **CUDA**: 11.7 或 11.8
- **PyTorch**: 2.0+

## 🚀 快速部署步骤

### 步骤 1: 解压文件
```bash
unzip starvla_act_deployment.zip
cd starvla_act_deployment
```

### 步骤 2: 安装依赖
```bash
# 创建虚拟环境 (推荐)
python -m venv venv
source venv/bin/activate

# 安装 PyTorch (根据你的 CUDA 版本)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装其他依赖
pip install flask pillow numpy tqdm wandb omegaconf accelerate
```

### 步骤 3: 验证模型
```bash
# 检查模型文件是否存在
ls -la output/starvla_act_g1/g1_act_task_04/final_model/

# 测试推理脚本
python starvla_act_scripts/inference_starvla_act.py \
    --ckpt_path output/starvla_act_g1/g1_act_task_04/final_model \
    --image_dir /path/to/test/images \
    --output_file test_actions.npy
```

### 步骤 4: 启动服务器
```bash
python starvla_act_scripts/server_starvla_act.py \
    --ckpt_path output/starvla_act_g1/g1_act_task_04/final_model \
    --host 0.0.0.0 \
    --port 5000
```

## 📋 详细使用说明

### 训练新模型
```bash
# 设置环境变量
export DATA_ROOT_DIR=/path/to/your/dataset
export RUN_ROOT_DIR=./output
export LOG_DIR=./logs

# 训练任务 04
./starvla_act_scripts/train_g1_act.sh 04
```

### 推理测试
```bash
# 单张图像推理
python starvla_act_scripts/inference_starvla_act.py \
    --ckpt_path output/starvla_act_g1/g1_act_task_04/final_model \
    --image_dir /path/to/images \
    --output_file predicted_actions.npy

# 查看结果
python -c "import numpy as np; actions = np.load('predicted_actions.npy'); print(f'动作形状: {actions.shape}')"
```

### 服务器 API

#### 健康检查
```bash
curl http://localhost:5000/health
```

#### 模型信息
```bash
curl http://localhost:5000/model_info
```

#### 多相机预测
```bash
curl -X POST http://localhost:5000/predict \
    -H "Content-Type: application/json" \
    -d '{
        "images": {
            "cam_high_rgb": "base64_encoded_image_1",
            "cam_left_wrist_rgb": "base64_encoded_image_2",
            "cam_right_wrist_rgb": "base64_encoded_image_3"
        }
    }'
```

#### 单相机预测
```bash
curl -X POST http://localhost:5000/predict_single \
    -H "Content-Type: application/json" \
    -d '{
        "image": "base64_encoded_image",
        "cam_name": "cam_high_rgb"
    }'
```

## 🔧 配置说明

### 训练配置
训练配置文件位于 `external/starVLA/examples/realRobots/G1ThreeCam/train_files/train_g1_act.yaml`

关键参数：
- `framework.action_dim`: 动作维度 (默认 20)
- `framework.chunk_size`: 动作块大小 (默认 50)
- `framework.image_size`: 图像尺寸 (默认 [160, 160])
- `trainer.max_train_steps`: 最大训练步数 (默认 10000)

### 服务器配置
服务器启动参数：
- `--ckpt_path`: 模型检查点路径
- `--host`: 服务器地址 (默认 0.0.0.0)
- `--port`: 服务器端口 (默认 5000)
- `--device`: 设备 (cuda/cpu)
- `--use_bf16`: 使用 bfloat16 精度

## 🐛 故障排除

### 1. 模型加载失败
**错误**: `ModuleNotFoundError: No module named 'starVLA'`

**解决**: 确保在正确的目录下运行脚本，或设置 PYTHONPATH：
```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/external/starVLA
```

### 2. CUDA 内存不足
**错误**: `RuntimeError: CUDA out of memory`

**解决**:
- 减小批量大小
- 使用 `--use_bf16` 启用半精度
- 使用更小的模型

### 3. 图像尺寸不匹配
**错误**: 图像尺寸与训练时不一致

**解决**: 确保输入图像尺寸为 160x160，或使用脚本中的预处理函数

### 4. 服务器连接失败
**错误**: 无法连接到服务器

**解决**:
- 检查服务器是否启动
- 检查防火墙设置
- 确认端口是否被占用

## 📊 性能优化

### 1. 使用 TensorRT 加速
```bash
# 安装 TensorRT
pip install tensorrt

# 导出为 TensorRT 格式
python -c "
import torch
from starVLA.model.framework.base_framework import baseframework
model = baseframework.from_pretrained('output/starvla_act_g1/g1_act_task_04/final_model')
# ... TensorRT 转换代码
"
```

### 2. 使用 ONNX Runtime
```bash
pip install onnxruntime-gpu
# 导出为 ONNX 格式
```

### 3. 批量推理优化
修改服务器脚本，支持批量推理以提高吞吐量。

## 📚 相关资源

- STAR-VLA 官方文档
- LeRobot 数据集格式
- ACT 论文: "Action Chunking with Transformers"

## 📞 技术支持

如有问题，请提供以下信息：
1. 错误日志
2. 系统环境信息
3. 模型检查点路径
4. 输入数据格式

## 📝 更新日志

### v1.0 (2026-09-02)
- 初始版本
- 支持 G1 机器人 ACT 模型训练和推理
- 提供 REST API 服务器
- 完整的部署文档
