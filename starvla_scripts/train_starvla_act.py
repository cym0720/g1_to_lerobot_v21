"""
STAR-VLA ACT 训练脚本
用于训练 STAR-VLA 模型
"""
import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import wandb

from model_starvla import STARVLA, STARVLALoss


class RobotDataset(Dataset):
    """
    机器人数据集
    加载图像、语言指令和动作序列
    """
    
    def __init__(self, data_dir, tokenizer, action_chunk_size=10, image_size=224):
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.action_chunk_size = action_chunk_size
        self.image_size = image_size
        
        # 加载数据索引
        self.samples = self._load_samples()
    
    def _load_samples(self):
        """加载数据样本列表"""
        samples = []
        # 假设数据目录结构为: data_dir/episode_xxx/
        for episode_dir in sorted(self.data_dir.glob("episode_*")):
            if episode_dir.is_dir():
                # 检查必要的文件是否存在
                if (episode_dir / "images").exists() and (episode_dir / "actions.npy").exists():
                    samples.append(episode_dir)
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        episode_dir = self.samples[idx]
        
        # 加载图像 (取第一帧)
        image_files = sorted((episode_dir / "images").glob("*.png"))
        if len(image_files) == 0:
            image_files = sorted((episode_dir / "images").glob("*.jpg"))
        
        if len(image_files) > 0:
            # 简化处理：加载第一张图像
            image = self._load_image(image_files[0])
        else:
            # 创建随机图像作为占位符
            image = torch.randn(3, self.image_size, self.image_size)
        
        # 加载动作序列
        actions = np.load(episode_dir / "actions.npy")
        
        # 确保动作序列长度足够
        if len(actions) < self.action_chunk_size:
            # 填充动作序列
            padding = np.zeros((self.action_chunk_size - len(actions), actions.shape[1]))
            actions = np.concatenate([actions, padding], axis=0)
        else:
            # 截取指定长度
            actions = actions[:self.action_chunk_size]
        
        # 加载语言指令
        instruction_file = episode_dir / "instruction.txt"
        if instruction_file.exists():
            with open(instruction_file, 'r', encoding='utf-8') as f:
                instruction = f.read().strip()
        else:
            instruction = "Pick up the object"
        
        # 编码语言指令
        language_tokens = self.tokenizer(
            instruction,
            padding='max_length',
            max_length=32,
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'image': image,
            'language_tokens': language_tokens['input_ids'].squeeze(0),
            'attention_mask': language_tokens['attention_mask'].squeeze(0),
            'actions': torch.FloatTensor(actions)
        }
    
    def _load_image(self, image_path):
        """加载并预处理图像"""
        # 简化实现：返回随机张量
        # 实际应用中应使用 PIL 或 torchvision 加载
        return torch.randn(3, self.image_size, self.image_size)


def train(args):
    """训练函数"""
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 初始化 wandb (可选)
    if args.use_wandb:
        wandb.init(project="starvla-training", config=args)
    
    # 初始化 tokenizer
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # 创建数据集
    train_dataset = RobotDataset(
        data_dir=args.data_dir,
        tokenizer=tokenizer,
        action_chunk_size=args.action_chunk_size
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # 创建模型
    model = STARVLA(
        action_dim=args.action_dim,
        action_chunk_size=args.action_chunk_size,
        hidden_dim=args.hidden_dim
    ).to(device)
    
    # 创建损失函数
    criterion = STARVLALoss(lambda_smooth=args.lambda_smooth)
    
    # 创建优化器
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # 学习率调度器
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.num_epochs,
        eta_min=args.learning_rate * 0.01
    )
    
    # 训练循环
    best_loss = float('inf')
    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0.0
        total_mse = 0.0
        total_smooth = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # 移动数据到设备
            images = batch['image'].to(device)
            language_tokens = batch['language_tokens'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            target_actions = batch['actions'].to(device)
            
            # 前向传播
            predicted_actions = model(images, language_tokens, attention_mask)
            
            # 计算损失
            loss, mse_loss, smooth_loss = criterion(predicted_actions, target_actions)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            
            optimizer.step()
            
            # 更新统计
            total_loss += loss.item()
            total_mse += mse_loss.item()
            total_smooth += smooth_loss if isinstance(smooth_loss, float) else smooth_loss.item()
            
            # 更新进度条
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'mse': f"{mse_loss.item():.4f}",
                'smooth': f"{smooth_loss if isinstance(smooth_loss, float) else smooth_loss.item():.4f}"
            })
            
            # 记录到 wandb
            if args.use_wandb:
                wandb.log({
                    'train/loss': loss.item(),
                    'train/mse_loss': mse_loss.item(),
                    'train/smooth_loss': smooth_loss if isinstance(smooth_loss, float) else smooth_loss.item(),
                    'train/learning_rate': optimizer.param_groups[0]['lr']
                })
        
        # 更新学习率
        scheduler.step()
        
        # 计算平均损失
        avg_loss = total_loss / len(train_loader)
        avg_mse = total_mse / len(train_loader)
        avg_smooth = total_smooth / len(train_loader)
        
        print(f"Epoch {epoch+1}/{args.num_epochs}:")
        print(f"  平均损失: {avg_loss:.4f}")
        print(f"  MSE 损失: {avg_mse:.4f}")
        print(f"  平滑损失: {avg_smooth:.4f}")
        
        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_dir = Path(args.save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': best_loss,
                'args': vars(args)
            }, save_dir / 'best_model.pth')
            
            print(f"  保存最佳模型到: {save_dir / 'best_model.pth'}")
        
        # 定期保存检查点
        if (epoch + 1) % args.save_every == 0:
            save_dir = Path(args.save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_loss,
                'args': vars(args)
            }, save_dir / f'checkpoint_epoch_{epoch+1}.pth')
    
    # 保存最终模型
    save_dir = Path(args.save_dir)
    torch.save({
        'epoch': args.num_epochs - 1,
        'model_state_dict': model.state_dict(),
        'args': vars(args)
    }, save_dir / 'final_model.pth')
    
    print("训练完成！")
    
    if args.use_wandb:
        wandb.finish()


def main():
    parser = argparse.ArgumentParser(description="STAR-VLA ACT 训练脚本")
    
    # 数据参数
    parser.add_argument('--data_dir', type=str, required=True, help='数据目录')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='模型保存目录')
    
    # 模型参数
    parser.add_argument('--action_dim', type=int, default=7, help='动作维度')
    parser.add_argument('--action_chunk_size', type=int, default=10, help='动作块大小')
    parser.add_argument('--hidden_dim', type=int, default=512, help='隐藏层维度')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=32, help='批量大小')
    parser.add_argument('--num_epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='权重衰减')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='梯度裁剪范数')
    parser.add_argument('--lambda_smooth', type=float, default=0.1, help='平滑损失权重')
    
    # 其他参数
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载线程数')
    parser.add_argument('--save_every', type=int, default=10, help='每 N 个 epoch 保存检查点')
    parser.add_argument('--use_wandb', action='store_true', help='使用 wandb 记录训练')
    
    args = parser.parse_args()
    
    # 训练
    train(args)


if __name__ == '__main__':
    main()
