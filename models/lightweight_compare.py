"""
轻量级对比网络 - 用于对比 CoordAtt 和 CoordCrossAtt

设计原则:
- 使用小输入尺寸 (64x64)
- 使用更少的通道数
- 简化网络结构
- 在 CPU 上快速训练
"""
import torch
import torch.nn as nn
from modules.att_visualize import CoordAttWithVisualization, CoordCrossAttWithVisualization
from modules.conv import Conv


class LightweightCoordAttNet(nn.Module):
    """轻量级 CoordAtt 网络"""

    def __init__(self, num_classes=10, img_size=64):
        super().__init__()
        self.num_classes = num_classes

        # 轻量级 backbone
        self.conv1 = Conv(3, 16, k=3, s=1, p=1)   # 64x64
        self.conv2 = Conv(16, 32, k=3, s=2, p=1)  # 32x32
        self.coord_att1 = CoordAttWithVisualization(inp=32, oup=32, reduction=4)

        self.conv3 = Conv(32, 64, k=3, s=2, p=1)  # 16x16
        self.coord_att2 = CoordAttWithVisualization(inp=64, oup=64, reduction=4)

        self.conv4 = Conv(64, 128, k=3, s=2, p=1) # 8x8
        self.coord_att3 = CoordAttWithVisualization(inp=128, oup=128, reduction=8)

        # 全局池化 + 分类头
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, num_classes)

        self.coord_att_layers = nn.ModuleList([
            self.coord_att1,
            self.coord_att2,
            self.coord_att3
        ])

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.coord_att1(x)

        x = self.conv3(x)
        x = self.coord_att2(x)

        x = self.conv4(x)
        x = self.coord_att3(x)

        x = self.pool(x).flatten(1)
        return self.fc(x)

    def forward_with_attention(self, x, layer_idx=0):
        """前向传播并返回指定层的注意力权重"""
        x = self.conv1(x)
        x = self.conv2(x)

        if layer_idx == 0:
            x, a_h, a_w = self.coord_att_layers[0].forward_with_attention(x)
        else:
            x = self.coord_att_layers[0](x)

        x = self.conv3(x)

        if layer_idx == 1:
            x, a_h, a_w = self.coord_att_layers[1].forward_with_attention(x)
        else:
            x = self.coord_att_layers[1](x)

        x = self.conv4(x)

        if layer_idx == 2:
            x, a_h, a_w = self.coord_att_layers[2].forward_with_attention(x)
        else:
            x = self.coord_att_layers[2](x)

        x = self.pool(x).flatten(1)
        logits = self.fc(x)

        if layer_idx >= 0:
            return logits, a_h, a_w
        return logits, None, None


class LightweightCoordCrossAttNet(nn.Module):
    """轻量级 CoordCrossAtt 网络"""

    def __init__(self, num_classes=10, img_size=64, num_heads=1):
        super().__init__()
        self.num_classes = num_classes

        # 轻量级 backbone
        self.conv1 = Conv(3, 16, k=3, s=1, p=1)   # 64x64
        self.conv2 = Conv(16, 32, k=3, s=2, p=1)  # 32x32
        self.coord_cross_att1 = CoordCrossAttWithVisualization(
            inp=32, oup=32, reduction=4, num_heads=num_heads
        )

        self.conv3 = Conv(32, 64, k=3, s=2, p=1)  # 16x16
        self.coord_cross_att2 = CoordCrossAttWithVisualization(
            inp=64, oup=64, reduction=4, num_heads=num_heads
        )

        self.conv4 = Conv(64, 128, k=3, s=2, p=1) # 8x8
        self.coord_cross_att3 = CoordCrossAttWithVisualization(
            inp=128, oup=128, reduction=8, num_heads=num_heads
        )

        # 全局池化 + 分类头
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, num_classes)

        self.coord_att_layers = nn.ModuleList([
            self.coord_cross_att1,
            self.coord_cross_att2,
            self.coord_cross_att3
        ])

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.coord_cross_att1(x)

        x = self.conv3(x)
        x = self.coord_cross_att2(x)

        x = self.conv4(x)
        x = self.coord_cross_att3(x)

        x = self.pool(x).flatten(1)
        return self.fc(x)

    def forward_with_attention(self, x, layer_idx=0):
        """前向传播并返回指定层的注意力权重"""
        x = self.conv1(x)
        x = self.conv2(x)

        if layer_idx == 0:
            x, attn, y_att = self.coord_att_layers[0].forward_with_attention(x)
        else:
            x = self.coord_att_layers[0](x)

        x = self.conv3(x)

        if layer_idx == 1:
            x, attn, y_att = self.coord_att_layers[1].forward_with_attention(x)
        else:
            x = self.coord_att_layers[1](x)

        x = self.conv4(x)

        if layer_idx == 2:
            x, attn, y_att = self.coord_att_layers[2].forward_with_attention(x)
        else:
            x = self.coord_att_layers[2](x)

        x = self.pool(x).flatten(1)
        logits = self.fc(x)

        if layer_idx >= 0:
            return logits, attn, y_att
        return logits, None, None
