import torch
import torch.nn as nn
from torchvision.models import alexnet

class FogVisibilityModel(nn.Module):
    """雾天能见度二分类模型有雾/无雾基于EfficientNet-B1"""
    def __init__(self, num_classes=3, pretrained=False):
        super(FogVisibilityModel, self).__init__()
        # 加载预训练的EfficientNet-B1骨干网络
        self.backbone = alexnet(pretrained=pretrained)

        # 光照感知分支：输出多维度光照特征（优化晨昏场景识别）
        self.lightness_branch = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 4),  # 输出4维光照特征：亮度/色温/光源类型/对比度
        )

        # 获取骨干网络特征输出维度（不变）
        in_features = self.backbone.classifier[1].in_features

        # 替换骨干网络的分类层为恒等映射（仅保留特征提取部分）
        self.backbone.classifier = nn.Identity()

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, num_classes)  #输出维度改为num_classes
        )

        # 注意力机制：基于光照特征动态调整骨干特征权重
        self.attention = nn.Sequential(
            nn.Linear(4, 256),  # 4维光照特征→隐藏层
            nn.ReLU(),
            nn.Linear(256, in_features),  # 映射到特征权重
            nn.Sigmoid()
        )

    def forward(self, x):
        # 提取光照特征
        light_features = self.lightness_branch(x)  

        # 提取骨干网络特征
        backbone_features = self.backbone(x) 

        # 生成注意力权重并调整特征
        attention_weights = self.attention(light_features)  
        weighted_features = backbone_features * attention_weights  # 特征加权

        # 分类预测
        logits = self.classifier(weighted_features)  
        return logits, light_features