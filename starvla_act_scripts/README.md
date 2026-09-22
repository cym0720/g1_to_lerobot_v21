# STAR-VLA ACT 脚本

本目录包含用于训练和部署 STAR-VLA ACT 模型的三个脚本。

## 文件说明

### 1. `train_g1_act.sh` - 训练启动脚本
用于启动 G1 机器人的 ACT 模型训练。基于 STAR-VLA 框架。

**使用方法：**
```bash
# 训练任务 04
./train_g1_act.sh 04

# 训练任务 05
./train_g1_act.sh 05

# 训练任务 06
./train_g1_act.sh 06
```

**环境变量：**
- `DATA_ROOT_DIR`: 数据集根目录
- `RUN_ROOT_DIR`: 模型保存根目录
- `LOG_DIR`: 日志目录
- `STARVLA_PYTHON`: Python 解释器路径

### 2. `inference_starvla_act.py` - 推理脚本
用于加载训练好的 ACT 模型并进行推理。

**使用方法：**
```bash
python inference_starvla_act.py \
    --ckpt_path output/starvla_act_g1/g1_act_task_04/final_model \
    --image_dir /path/to/images \
    --output_file predicted_actions.npy
```

**参数说明：**
- `--ckpt_path`: 模型检查点路径
- `--image_dir`: 图像目录路径
- `--device`: 设备 (cuda/cpu)
- `--use_bf16`: 使用 bfloat16 精度
- `--output_file`: 输出文件路径

### 3. `server_starvla_act.py` - 推理服务器
提供 REST API 用于动作预测。

**使用方法：**
```bash
python server_starvla_act.py \
    --ckpt_path output/starvla_act_g1/g1_act_task_04/final_model \
    --host 0.0.0.0 \
    --port 5000
```

**API 端点：**
- `POST /predict`: 多相机预测
- `POST /predict_single`: 单相机预测
- `GET /health`: 健康检查
- `GET /model_info`: 模型信息

## 目录结构

```
starvla_act_scripts/
├── train_g1_act.sh           # 训练启动脚本
├── inference_starvla_act.py  # 推理脚本
├── server_starvla_act.py     # 推理服务器
└── README.md                 # 本文件
```

## 前提条件

1. 已安装 STAR-VLA 框架
2. 已训练 ACT 模型 (检查点)
3. 已安装依赖：`torch`, `flask`, `pillow`, `numpy`

## 模型检查点位置

训练好的模型位于：
```
output/starvla_act_g1/
├── g1_act_task_04/
│   ├── checkpoints/
│   └── final_model/
├── g1_act_task_05/
│   ├── checkpoints/
│   └── final_model/
└── g1_act_task_06/
    ├── checkpoints/
    └── final_model/
```

## 注意事项

1. 训练脚本需要在 Linux 环境下运行
2. 推理服务器支持多线程，适合生产环境部署
3. 图像输入格式应与训练时一致
4. 动作输出为归一化后的值，需要根据训练配置进行反归一化
