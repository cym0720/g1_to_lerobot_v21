"""
STAR-VLA ACT 推理服务器
提供 REST API 用于动作预测
"""
import argparse
import logging
import numpy as np
import torch
from flask import Flask, request, jsonify
from pathlib import Path
import base64
import io
from PIL import Image

# 添加 starVLA 到 Python 路径
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "external" / "starVLA"))

from deployment.model_server.policy_wrapper import PolicyServerWrapper


class STARVLAServer:
    """STAR-VLA 推理服务器"""
    
    def __init__(self, ckpt_path, device='cuda', use_bf16=False):
        """
        初始化服务器
        Args:
            ckpt_path: 模型检查点路径
            device: 设备 (cuda/cpu)
            use_bf16: 是否使用 bfloat16 精度
        """
        self.device = device
        self.ckpt_path = ckpt_path
        
        # 加载模型包装器
        logging.info(f"Loading model from {ckpt_path}")
        self.wrapper = PolicyServerWrapper(
            ckpt_path=ckpt_path,
            device=device,
            use_bf16=use_bf16
        )
        
        # 获取模型元数据
        self.metadata = self.wrapper.metadata
        logging.info(f"Model metadata: {self.metadata}")
        
        # 获取动作块大小
        self.action_chunk_size = self.metadata['action_chunk_size']
        logging.info(f"Action chunk size: {self.action_chunk_size}")
        
        # 图像预处理参数
        self.image_size = self.metadata.get('training_obs_image_size', [160, 160])
    
    def preprocess_image(self, image_data):
        """
        预处理图像数据
        Args:
            image_data: 图像数据 (base64 字符串、字节或 PIL 图像)
        Returns:
            image: numpy 数组，形状为 (H, W, 3)
        """
        if isinstance(image_data, str):
            # Base64 编码的图像
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        elif isinstance(image_data, bytes):
            # 原始字节
            image = Image.open(io.BytesIO(image_data)).convert('RGB')
        else:
            # PIL 图像
            image = image_data
        
        # 调整大小
        image = image.resize((self.image_size[1], self.image_size[0]))
        
        # 转换为 numpy 数组
        image_np = np.array(image)
        
        return image_np
    
    def predict(self, images, state=None, unnorm_key=None):
        """
        预测动作
        Args:
            images: 图像字典，格式为 {'cam_high_rgb': img1, 'cam_left_wrist_rgb': img2, 'cam_right_wrist_rgb': img3}
            state: 状态向量 (可选)
            unnorm_key: 反归一化键 (可选)
        Returns:
            actions: 预测的动作序列
        """
        # 准备输入数据
        examples = {
            'images': images,
            'state': state
        }
        
        # 调用模型包装器进行预测
        result = self.wrapper.predict_action(examples, unnorm_key=unnorm_key)
        
        # 提取动作
        actions = result['actions'][0]  # 取第一个批次
        
        return actions


# 创建 Flask 应用
app = Flask(__name__)

# 全局服务器实例
server = None


@app.route('/predict', methods=['POST'])
def predict():
    """预测动作的 API 端点"""
    try:
        # 获取请求数据
        data = request.json
        
        if not data:
            return jsonify({'error': '没有提供数据'}), 400
        
        # 获取图像
        images_data = data.get('images', {})
        if not images_data:
            return jsonify({'error': '没有提供图像'}), 400
        
        # 预处理图像
        processed_images = {}
        for cam_name, image_data in images_data.items():
            processed_images[cam_name] = server.preprocess_image(image_data)
        
        # 获取状态 (可选)
        state = data.get('state')
        if state is not None:
            state = np.array(state)
        
        # 获取反归一化键 (可选)
        unnorm_key = data.get('unnorm_key')
        
        # 预测动作
        actions = server.predict(processed_images, state=state, unnorm_key=unnorm_key)
        
        # 转换为列表以便 JSON 序列化
        actions_list = actions.tolist()
        
        return jsonify({
            'success': True,
            'actions': actions_list,
            'action_dim': actions.shape[-1],
            'chunk_size': actions.shape[0]
        })
    
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/predict_single', methods=['POST'])
def predict_single():
    """单个图像预测的 API 端点"""
    try:
        # 获取请求数据
        data = request.json
        
        if not data:
            return jsonify({'error': '没有提供数据'}), 400
        
        # 获取图像
        image_data = data.get('image')
        if not image_data:
            return jsonify({'error': '没有提供图像'}), 400
        
        # 获取相机名称
        cam_name = data.get('cam_name', 'cam_high_rgb')
        
        # 预处理图像
        processed_image = server.preprocess_image(image_data)
        
        # 创建图像字典
        images = {cam_name: processed_image}
        
        # 获取状态 (可选)
        state = data.get('state')
        if state is not None:
            state = np.array(state)
        
        # 预测动作
        actions = server.predict(images, state=state)
        
        # 转换为列表以便 JSON 序列化
        actions_list = actions.tolist()
        
        return jsonify({
            'success': True,
            'actions': actions_list,
            'action_dim': actions.shape[-1],
            'chunk_size': actions.shape[0]
        })
    
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    """健康检查端点"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': server is not None,
        'device': str(server.device) if server else 'unknown',
        'action_chunk_size': server.action_chunk_size if server else 0
    })


@app.route('/model_info', methods=['GET'])
def model_info():
    """模型信息端点"""
    if server is None:
        return jsonify({'error': '模型未加载'}), 500
    
    return jsonify({
        'metadata': server.metadata,
        'action_chunk_size': server.action_chunk_size,
        'device': str(server.device)
    })


def main():
    parser = argparse.ArgumentParser(description="STAR-VLA ACT 推理服务器")
    parser.add_argument('--ckpt_path', type=str, required=True, help='模型检查点路径')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='服务器主机')
    parser.add_argument('--port', type=int, default=5000, help='服务器端口')
    parser.add_argument('--device', type=str, default='cuda', help='设备')
    parser.add_argument('--use_bf16', action='store_true', help='使用 bfloat16')
    parser.add_argument('--debug', action='store_true', help='调试模式')
    
    args = parser.parse_args()
    
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    
    # 初始化服务器
    global server
    server = STARVLAServer(
        ckpt_path=args.ckpt_path,
        device=args.device,
        use_bf16=args.use_bf16
    )
    
    print(f"启动 STAR-VLA ACT 推理服务器...")
    print(f"模型路径: {args.ckpt_path}")
    print(f"服务器地址: {args.host}:{args.port}")
    print(f"设备: {server.device}")
    print(f"动作块大小: {server.action_chunk_size}")
    
    # 启动 Flask 应用
    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug,
        threaded=True
    )


if __name__ == '__main__':
    main()
