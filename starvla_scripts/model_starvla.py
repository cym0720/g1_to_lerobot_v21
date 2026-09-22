"""
STAR-VLA 模型定义
基于 ACT (Action Chunking with Transformers) 训练框架
"""
import torch
import torch.nn as nn
import torchvision.models as models
from transformers import BertModel, BertTokenizer


class STARVLA(nn.Module):
    """
    STAR-VLA: Vision-Language-Action 模型
    结合视觉编码器、语言编码器和动作解码器
    """
    
    def __init__(self, action_dim=7, action_chunk_size=10, hidden_dim=512):
        super().__init__()
        self.action_dim = action_dim
        self.action_chunk_size = action_chunk_size
        
        # 视觉编码器 (ResNet-18)
        self.visual_encoder = models.resnet18(pretrained=True)
        self.visual_encoder.fc = nn.Linear(self.visual_encoder.fc.in_features, hidden_dim)
        
        # 语言编码器 (BERT)
        self.language_encoder = BertModel.from_pretrained('bert-base-uncased')
        self.language_projection = nn.Linear(768, hidden_dim)
        
        # 多模态融合层
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 动作解码器 (ACT style)
        self.action_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_chunk_size * action_dim)
        )
        
        # 位置编码
        self.positional_encoding = nn.Parameter(
            torch.randn(1, action_chunk_size, action_dim)
        )
    
    def forward(self, images, language_tokens, attention_mask=None):
        """
        前向传播
        Args:
            images: [batch_size, 3, H, W] 视觉输入
            language_tokens: [batch_size, seq_len] 语言 token
            attention_mask: [batch_size, seq_len] 注意力掩码
        Returns:
            action_chunk: [batch_size, action_chunk_size, action_dim] 动作块
        """
        batch_size = images.shape[0]
        
        # 视觉特征提取
        visual_features = self.visual_encoder(images)  # [batch_size, hidden_dim]
        
        # 语言特征提取
        language_output = self.language_encoder(
            input_ids=language_tokens,
            attention_mask=attention_mask
        )
        language_features = language_output.last_hidden_state[:, 0, :]  # [batch_size, 768]
        language_features = self.language_projection(language_features)  # [batch_size, hidden_dim]
        
        # 多模态融合
        combined_features = torch.cat([visual_features, language_features], dim=1)
        fused_features = self.fusion_layer(combined_features)  # [batch_size, hidden_dim]
        
        # 动作解码
        action_chunk = self.action_decoder(fused_features)  # [batch_size, action_chunk_size * action_dim]
        action_chunk = action_chunk.view(batch_size, self.action_chunk_size, self.action_dim)
        
        # 添加位置编码
        action_chunk = action_chunk + self.positional_encoding
        
        return action_chunk


class STARVLALoss(nn.Module):
    """
    STAR-VLA 损失函数
    结合 MSE 损失和动作平滑损失
    """
    
    def __init__(self, lambda_smooth=0.1):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.lambda_smooth = lambda_smooth
    
    def forward(self, predicted_actions, target_actions):
        """
        计算损失
        Args:
            predicted_actions: [batch_size, action_chunk_size, action_dim]
            target_actions: [batch_size, action_chunk_size, action_dim]
        """
        # MSE 损失
        mse = self.mse_loss(predicted_actions, target_actions)
        
        # 动作平滑损失 (鼓励连续动作之间的平滑性)
        if predicted_actions.shape[1] > 1:
            action_diff = predicted_actions[:, 1:, :] - predicted_actions[:, :-1, :]
            smooth_loss = torch.mean(torch.norm(action_diff, dim=-1))
        else:
            smooth_loss = 0.0
        
        total_loss = mse + self.lambda_smooth * smooth_loss
        return total_loss, mse, smooth_loss
