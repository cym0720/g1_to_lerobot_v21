"""
STAR-VLA ACT 推理脚本
用于加载训练好的 ACT 模型并进行推理
"""
import argparse
import logging
import numpy as np
import torch
from pathlib import Path

# 添加 starVLA 到 Python 路径
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "external" / "starVLA"))

from deployment.model_server.policy_wrapper import PolicyServerWrapper


class STARVLAInference:
    """STAR-VLA 推理类"""
    
    def __init__(self, ckpt_path, device='cuda', use_bf16=False):
        """
        初始化推理器
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
    
    def predict(self, images, state=None, unnorm_key=None):
        """
        预测动作
        Args:
            images: 图像字典，格式为 {'cam_high_rgb': img1, 'cam_left_wrist_rgb': img2, 'cam_right_wrist_rgb': img3}
                    每个图像应为 numpy 数组，形状为 (H, W, 3)，值范围 [0, 255]
            state: 状态向量 (可选)，形状为 (state_dim,)
            unnorm_key: 反归一化键 (可选)
        Returns:
            actions: 预测的动作序列，形状为 (action_chunk_size, action_dim)
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
    
    def preprocess_image(self, image_path):
        """
        预处理图像文件
        Args:
            image_path: 图像文件路径
        Returns:
            image: numpy 数组，形状为 (H, W, 3)
        """
        from PIL import Image
        import torchvision.transforms as transforms
        
        # 加载图像
        image = Image.open(image_path).convert('RGB')
        
        # 调整大小 (根据训练配置)
        image_size = self.metadata.get('training_obs_image_size', [160, 160])
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor()
        ])
        
        # 转换为 numpy 数组
        image_tensor = transform(image)
        image_np = image_tensor.permute(1, 2, 0).numpy()  # (H, W, C)
        
        return image_np
    
    def load_images_from_directory(self, image_dir):
        """
        从目录加载图像
        Args:
            image_dir: 图像目录路径
        Returns:
            images: 图像字典
        """
        from pathlib import Path
        
        image_dir = Path(image_dir)
        images = {}
        
        # 查找图像文件
        for cam_name in ['cam_high_rgb', 'cam_left_wrist_rgb', 'cam_right_wrist_rgb']:
            # 尝试不同的图像格式
            for ext in ['.png', '.jpg', '.jpeg']:
                image_path = image_dir / f"{cam_name}{ext}"
                if image_path.exists():
                    images[cam_name] = self.preprocess_image(image_path)
                    break
        
        return images


def main():
    parser = argparse.ArgumentParser(description="STAR-VLA ACT 推理脚本")
    parser.add_argument('--ckpt_path', type=str, required=True, help='模型检查点路径')
    parser.add_argument('--image_dir', type=str, required=True, help='图像目录路径')
    parser.add_argument('--device', type=str, default='cuda', help='设备')
    parser.add_argument('--use_bf16', action='store_true', help='使用 bfloat16')
    parser.add_argument('--output_file', type=str, default='predicted_actions.npy', help='输出文件路径')
    
    args = parser.parse_args()
    
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    
    # 创建推理器
    inferencer = STARVLAInference(
        ckpt_path=args.ckpt_path,
        device=args.device,
        use_bf16=args.use_bf16
    )
    
    # 加载图像
    images = inferencer.load_images_from_directory(args.image_dir)
    
    if not images:
        logging.error(f"No images found in {args.image_dir}")
        return
    
    logging.info(f"Loaded {len(images)} camera views")
    
    # 预测动作
    actions = inferencer.predict(images)
    
    # 保存动作
    np.save(args.output_file, actions)
    logging.info(f"Saved predicted actions to {args.output_file}")
    logging.info(f"Action shape: {actions.shape}")


if __name__ == '__main__':
    main()
