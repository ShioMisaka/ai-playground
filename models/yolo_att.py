"""
带注意力机制的 YOLO 检测器

包含使用 CoordAtt 和 CoordCrossAtt 的 YOLO 检测器实现。
"""
import torch
import torch.nn as nn
from .conv import Conv
from .head import Detect
from .att_visualize import CoordAttWithVisualization, CoordCrossAttWithVisualization


class YOLOCoordAttDetector(nn.Module):
    """
    带 Coordinate Attention 的 YOLO 检测器

    使用多个 CoordAtt 层增强特征表达，输出 3 个尺度的特征图用于检测。

    Args:
        nc: 类别数量
        anchors: anchor boxes (默认使用 YOLOv3 anchors)
    """

    def __init__(self, nc=1, anchors=None):
        super().__init__()
        self.nc = nc

        # Default YOLOv3 anchors (P3/8, P4/16, P5/32)
        if anchors is None:
            anchors = [
                [10, 13, 16, 30, 33, 23],      # P3/8
                [30, 61, 62, 45, 59, 119],     # P4/16
                [116, 90, 156, 198, 373, 326]  # P5/32
            ]

        # ===== Backbone with CoordAtt =====
        self.conv0 = Conv(3, 32, k=3, s=1, p=1)
        self.conv1 = Conv(32, 64, k=3, s=2, p=1)

        self.conv2 = Conv(64, 128, k=3, s=2, p=1)
        self.coord_att1 = CoordAttWithVisualization(inp=128, oup=128, reduction=4)

        self.conv3 = Conv(128, 256, k=3, s=2, p=1)
        self.coord_att2 = CoordAttWithVisualization(inp=256, oup=256, reduction=4)

        self.conv4 = Conv(256, 512, k=3, s=2, p=1)
        self.coord_att3 = CoordAttWithVisualization(inp=512, oup=512, reduction=8)

        self.conv5 = Conv(512, 1024, k=3, s=2, p=1)
        self.coord_att4 = CoordAttWithVisualization(inp=1024, oup=1024, reduction=8)

        # ===== Neck: FPN =====
        self.up_conv1 = Conv(1024, 512, k=1, s=1)
        self.upsample1 = nn.Upsample(None, 2, mode='nearest')

        self.up_conv2 = Conv(512, 256, k=1, s=1)
        self.upsample2 = nn.Upsample(None, 2, mode='nearest')

        self.up_conv3 = Conv(256, 128, k=1, s=1)
        self.upsample3 = nn.Upsample(None, 2, mode='nearest')

        # ===== Detection Head =====
        self.detect = Detect(nc=nc, anchors=anchors, ch=(256, 512, 1024))

        # 存储所有的 CoordAtt 层用于可视化
        self.coord_att_layers = nn.ModuleList([
            self.coord_att1,
            self.coord_att2,
            self.coord_att3,
            self.coord_att4
        ])

        # 导入 loss 函数
        from .yolo_loss import YOLOLoss
        self.loss_fn = YOLOLoss(
            num_classes=nc,
            anchors=anchors,
            img_size=640
        )

    def forward(self, x, targets=None):
        """前向传播"""
        # ===== Backbone =====
        x = self.conv0(x)
        x = self.conv1(x)

        x = self.conv2(x)
        p3_backbone = self.coord_att1(x)

        x = self.conv3(p3_backbone)
        p4_backbone = self.coord_att2(x)

        x = self.conv4(p4_backbone)
        p5_backbone = self.coord_att3(x)

        x = self.conv5(p5_backbone)
        p6_backbone = self.coord_att4(x)

        # ===== Neck: FPN =====
        x = self.up_conv1(p6_backbone)
        x = self.upsample1(x)
        p5 = torch.cat([x, p5_backbone], dim=1)
        p5 = Conv(1024, 512, k=1, s=1)(p5)

        x = self.up_conv2(p5)
        x = self.upsample2(x)
        p4 = torch.cat([x, p4_backbone], dim=1)
        p4 = Conv(512, 256, k=1, s=1)(p4)

        x = self.up_conv3(p4)
        x = self.upsample3(x)
        p3 = torch.cat([x, p3_backbone], dim=1)
        p3 = Conv(256, 128, k=1, s=1)(p3)

        # ===== Detection =====
        predictions = self.detect([p4, p5, p6_backbone])

        if targets is not None:
            loss = self.loss_fn(predictions, targets)
            return {'predictions': predictions, 'loss': loss}

        return predictions

    def forward_with_attention(self, x, layer_idx=-1):
        """
        前向传播并返回指定层的注意力权重

        Args:
            x: 输入图像 [batch, 3, H, W]
            layer_idx: 要返回哪一层的注意力 (0-3, -1 表示不返回)

        Returns:
            如果 layer_idx >= 0: (predictions, a_h, a_w)
            否则: (predictions, None, None)
        """
        x = self.conv0(x)
        x = self.conv1(x)

        x = self.conv2(x)
        if layer_idx == 0:
            feat, a_h, a_w = self.coord_att_layers[0].forward_with_attention(x)
            p3_backbone = feat
        else:
            p3_backbone = self.coord_att_layers[0](x)

        x = self.conv3(p3_backbone)
        if layer_idx == 1:
            feat, a_h, a_w = self.coord_att_layers[1].forward_with_attention(x)
            p4_backbone = feat
        else:
            p4_backbone = self.coord_att_layers[1](x)

        x = self.conv4(p4_backbone)
        if layer_idx == 2:
            feat, a_h, a_w = self.coord_att_layers[2].forward_with_attention(x)
            p5_backbone = feat
        else:
            p5_backbone = self.coord_att_layers[2](x)

        x = self.conv5(p5_backbone)
        if layer_idx == 3:
            feat, a_h, a_w = self.coord_att_layers[3].forward_with_attention(x)
            p6_backbone = feat
        else:
            p6_backbone = self.coord_att_layers[3](x)

        # ===== Neck =====
        x = self.up_conv1(p6_backbone)
        x = self.upsample1(x)
        p5 = torch.cat([x, p5_backbone], dim=1)
        p5 = Conv(1024, 512, k=1, s=1)(p5)

        x = self.up_conv2(p5)
        x = self.upsample2(x)
        p4 = torch.cat([x, p4_backbone], dim=1)
        p4 = Conv(512, 256, k=1, s=1)(p4)

        x = self.up_conv3(p4)
        x = self.upsample3(x)
        p3 = torch.cat([x, p3_backbone], dim=1)
        p3 = Conv(256, 128, k=1, s=1)(p3)

        predictions = self.detect([p4, p5, p6_backbone])

        if layer_idx >= 0:
            return predictions, a_h, a_w
        return predictions, None, None


class YOLOCoordCrossAttDetector(nn.Module):
    """
    带 Coordinate Cross Attention 的 YOLO 检测器

    使用多个 CoordCrossAtt 层增强特征表达，输出 3 个尺度的特征图用于检测。

    Args:
        nc: 类别数量
        anchors: anchor boxes (默认使用 YOLOv3 anchors)
        num_heads: CoordCrossAtt 的注意力头数
    """

    def __init__(self, nc=1, anchors=None, num_heads=1):
        super().__init__()
        self.nc = nc
        self.num_heads = num_heads

        # Default YOLOv3 anchors (P3/8, P4/16, P5/32)
        if anchors is None:
            anchors = [
                [10, 13, 16, 30, 33, 23],      # P3/8
                [30, 61, 62, 45, 59, 119],     # P4/16
                [116, 90, 156, 198, 373, 326]  # P5/32
            ]

        # ===== Backbone with CoordCrossAtt =====
        self.conv0 = Conv(3, 32, k=3, s=1, p=1)
        self.conv1 = Conv(32, 64, k=3, s=2, p=1)

        self.conv2 = Conv(64, 128, k=3, s=2, p=1)
        self.coord_cross_att1 = CoordCrossAttWithVisualization(inp=128, oup=128, reduction=4, num_heads=num_heads)

        self.conv3 = Conv(128, 256, k=3, s=2, p=1)
        self.coord_cross_att2 = CoordCrossAttWithVisualization(inp=256, oup=256, reduction=4, num_heads=num_heads)

        self.conv4 = Conv(256, 512, k=3, s=2, p=1)
        self.coord_cross_att3 = CoordCrossAttWithVisualization(inp=512, oup=512, reduction=8, num_heads=num_heads)

        self.conv5 = Conv(512, 1024, k=3, s=2, p=1)
        self.coord_cross_att4 = CoordCrossAttWithVisualization(inp=1024, oup=1024, reduction=8, num_heads=num_heads)

        # ===== Neck: FPN =====
        self.up_conv1 = Conv(1024, 512, k=1, s=1)
        self.upsample1 = nn.Upsample(None, 2, mode='nearest')

        self.up_conv2 = Conv(512, 256, k=1, s=1)
        self.upsample2 = nn.Upsample(None, 2, mode='nearest')

        self.up_conv3 = Conv(256, 128, k=1, s=1)
        self.upsample3 = nn.Upsample(None, 2, mode='nearest')

        # ===== Detection Head =====
        self.detect = Detect(nc=nc, anchors=anchors, ch=(256, 512, 1024))

        # 存储所有的 CoordCrossAtt 层用于可视化
        self.coord_att_layers = nn.ModuleList([
            self.coord_cross_att1,
            self.coord_cross_att2,
            self.coord_cross_att3,
            self.coord_cross_att4
        ])

        # 导入 loss 函数
        from .yolo_loss import YOLOLoss
        self.loss_fn = YOLOLoss(
            num_classes=nc,
            anchors=anchors,
            img_size=640
        )

    def forward(self, x, targets=None):
        """前向传播"""
        # ===== Backbone =====
        x = self.conv0(x)
        x = self.conv1(x)

        x = self.conv2(x)
        p3_backbone = self.coord_cross_att1(x)

        x = self.conv3(p3_backbone)
        p4_backbone = self.coord_cross_att2(x)

        x = self.conv4(p4_backbone)
        p5_backbone = self.coord_cross_att3(x)

        x = self.conv5(p5_backbone)
        p6_backbone = self.coord_cross_att4(x)

        # ===== Neck: FPN =====
        x = self.up_conv1(p6_backbone)
        x = self.upsample1(x)
        p5 = torch.cat([x, p5_backbone], dim=1)
        p5 = Conv(1024, 512, k=1, s=1)(p5)

        x = self.up_conv2(p5)
        x = self.upsample2(x)
        p4 = torch.cat([x, p4_backbone], dim=1)
        p4 = Conv(512, 256, k=1, s=1)(p4)

        x = self.up_conv3(p4)
        x = self.upsample3(x)
        p3 = torch.cat([x, p3_backbone], dim=1)
        p3 = Conv(256, 128, k=1, s=1)(p3)

        # ===== Detection =====
        predictions = self.detect([p4, p5, p6_backbone])

        if targets is not None:
            loss = self.loss_fn(predictions, targets)
            return {'predictions': predictions, 'loss': loss}

        return predictions

    def forward_with_attention(self, x, layer_idx=-1):
        """
        前向传播并返回指定层的注意力权重

        Args:
            x: 输入图像 [batch, 3, H, W]
            layer_idx: 要返回哪一层的注意力 (0-3, -1 表示不返回)

        Returns:
            如果 layer_idx >= 0: (predictions, attn, y_att)
            否则: (predictions, None, None)
        """
        x = self.conv0(x)
        x = self.conv1(x)

        x = self.conv2(x)
        if layer_idx == 0:
            feat, attn, y_att = self.coord_att_layers[0].forward_with_attention(x)
            p3_backbone = feat
        else:
            p3_backbone = self.coord_att_layers[0](x)

        x = self.conv3(p3_backbone)
        if layer_idx == 1:
            feat, attn, y_att = self.coord_att_layers[1].forward_with_attention(x)
            p4_backbone = feat
        else:
            p4_backbone = self.coord_att_layers[1](x)

        x = self.conv4(p4_backbone)
        if layer_idx == 2:
            feat, attn, y_att = self.coord_att_layers[2].forward_with_attention(x)
            p5_backbone = feat
        else:
            p5_backbone = self.coord_att_layers[2](x)

        x = self.conv5(p5_backbone)
        if layer_idx == 3:
            feat, attn, y_att = self.coord_att_layers[3].forward_with_attention(x)
            p6_backbone = feat
        else:
            p6_backbone = self.coord_att_layers[3](x)

        # ===== Neck =====
        x = self.up_conv1(p6_backbone)
        x = self.upsample1(x)
        p5 = torch.cat([x, p5_backbone], dim=1)
        p5 = Conv(1024, 512, k=1, s=1)(p5)

        x = self.up_conv2(p5)
        x = self.upsample2(x)
        p4 = torch.cat([x, p4_backbone], dim=1)
        p4 = Conv(512, 256, k=1, s=1)(p4)

        x = self.up_conv3(p4)
        x = self.upsample3(x)
        p3 = torch.cat([x, p3_backbone], dim=1)
        p3 = Conv(256, 128, k=1, s=1)(p3)

        predictions = self.detect([p4, p5, p6_backbone])

        if layer_idx >= 0:
            return predictions, attn, y_att
        return predictions, None, None
