"""
STAR-VLA 推理服务器
提供 REST API 用于动作预测
"""
import os
import argparse
import torch
import torch.nn as nn
from flask import Flask, request, jsonify
from PIL import Image
import io
import base64
import numpy as np
from pathlib import Path
import json
from transformers import BertTokenizer
import torchvision.transforms as transforms

from model_starvla import STARVLA


class STARVLAServer:
    """STAR-VLA 推理服务器"""
    
    def __init__(self, model_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 加载模型
        self.model = self._load_model(model_path)
        self.model.eval()
        
        # 初始化 tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def _load_model(self, model_path):
        """加载训练好的模型"""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # 从检查点获取参数
        args = checkpoint.get('args', {})
        action_dim = args.get('action_dim', 7)
        action_chunk_size = args.get('action_chunk_size', 10)
        hidden_dim = args.get('hidden_dim', 512)
        
        # 创建模型
        model = STARVLA(
            action_dim=action_dim,
            action_chunk_size=action_chunk_size,
            hidden_dim=hidden_dim
        ).to(self.device)
        
        # 加载权重
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return model
    
    def preprocess_image(self, image_data):
        """预处理图像"""
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
        
        # 应用变换
        image_tensor = self.transform(image).unsqueeze(0)  # 添加批次维度
        return image_tensor.to(self.device)
    
    def preprocess_language(self, instruction):
        """预处理语言指令"""
        tokens = self.tokenizer(
            instruction,
            padding='max_length',
            max_length=32,
            truncation=True,
            return_tensors='pt'
        )
        return tokens['input_ids'].to(self.device), tokens['attention_mask'].to(self.device)
    
    def predict(self, image, instruction):
        """
        预测动作
        Args:
            image: 图像数据 (base64 字符串、字节或 PIL 图像)
            instruction: 语言指令
        Returns:
            action_chunk: 预测的动作块
        """
        with torch.no_grad():
            # 预处理
            image_tensor = self.preprocess_image(image)
            language_tokens, attention_mask = self.preprocess_language(instruction)
            
            # 推理
            action_chunk = self.model(image_tensor, language_tokens, attention_mask)
            
            # 转换为 numpy 数组
            action_chunk = action_chunk.cpu().numpy()
            
            return action_chunk


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
        image_data = data.get('image')
        if not image_data:
            return jsonify({'error': '没有提供图像'}), 400
        
        # 获取指令
        instruction = data.get('instruction', 'Pick up the object')
        
        # 预测动作
        action_chunk = server.predict(image_data, instruction)
        
        # 转换为列表以便 JSON 序列化
        action_list = action_chunk.tolist()
        
        return jsonify({
            'success': True,
            'action_chunk': action_list,
            'action_dim': action_chunk.shape[-1],
            'chunk_size': action_chunk.shape[1]
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """批量预测动作的 API 端点"""
    try:
        # 获取请求数据
        data = request.json
        
        if not data or 'requests' not in data:
            return jsonify({'error': '没有提供请求数据'}), 400
        
        requests_list = data['requests']
        results = []
        
        for req in requests_list:
            image_data = req.get('image')
            instruction = req.get('instruction', 'Pick up the object')
            
            if not image_data:
                results.append({'error': '没有提供图像'})
                continue
            
            try:
                action_chunk = server.predict(image_data, instruction)
                results.append({
                    'success': True,
                    'action_chunk': action_chunk.tolist(),
                    'action_dim': action_chunk.shape[-1],
                    'chunk_size': action_chunk.shape[1]
                })
            except Exception as e:
                results.append({'error': str(e)})
        
        return jsonify({
            'success': True,
            'results': results
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    """健康检查端点"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': server is not None,
        'device': str(server.device) if server else 'unknown'
    })


@app.route('/model_info', methods=['GET'])
def model_info():
    """模型信息端点"""
    if server is None:
        return jsonify({'error': '模型未加载'}), 500
    
    # 获取模型参数
    model = server.model
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return jsonify({
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'action_dim': model.action_dim,
        'action_chunk_size': model.action_chunk_size,
        'hidden_dim': model.fusion_layer[0].out_features,
        'device': str(server.device)
    })


def main():
    parser = argparse.ArgumentParser(description="STAR-VLA 推理服务器")
    parser.add_argument('--model_path', type=str, required=True, help='模型文件路径')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='服务器主机')
    parser.add_argument('--port', type=int, default=5000, help='服务器端口')
    parser.add_argument('--device', type=str, default='cuda', help='设备 (cuda/cpu)')
    parser.add_argument('--debug', action='store_true', help='调试模式')
    
    args = parser.parse_args()
    
    # 初始化服务器
    global server
    server = STARVLAServer(args.model_path, args.device)
    
    print(f"启动 STAR-VLA 推理服务器...")
    print(f"模型路径: {args.model_path}")
    print(f"服务器地址: {args.host}:{args.port}")
    print(f"设备: {server.device}")
    
    # 启动 Flask 应用
    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug,
        threaded=True
    )


if __name__ == '__main__':
    main()
