"""
带注意力机制的 YOLO 检测器

使用模板方法模式，将公共的 YOLO 结构提取到基类，
子类只需指定使用哪种注意力模块。

设计模式：
- 模板方法模式：YOLOBaseDetector 定义算法骨架，子类实现具体步骤
- 策略模式：通过 _create_attention_layer() 抽象方法切换注意力策略
"""
import torch
import torch.nn as nn
from .conv import Conv
from .head import Detect


class YOLOBaseDetector(nn.Module):
    """
    YOLO 检测器基类（模板方法模式）

    定义了 YOLO 检测器的整体结构，子类通过实现
    _create_attention_layer() 来选择不同的注意力机制。

    Args:
        nc: 类别数量
        anchors: anchor boxes
        num_heads: 注意力头数（仅 CoordCrossAtt 使用）
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

        # ===== 构建网络 =====
        self._build_backbone()
        self._build_neck()
        self._build_head(anchors)
        self._build_loss(anchors)

    def _build_backbone(self):
        """构建 Backbone（模板方法）"""
        # Stage 0-1: 初始卷积
        self.conv0 = Conv(3, 32, k=3, s=1, p=1)
        self.conv1 = Conv(32, 64, k=3, s=2, p=1)

        # Stage 2: 128 通道
        self.conv2 = Conv(64, 128, k=3, s=2, p=1)
        self.coord_att1 = self._create_attention_layer(128, 128, 4)

        # Stage 3: 256 通道
        self.conv3 = Conv(128, 256, k=3, s=2, p=1)
        self.coord_att2 = self._create_attention_layer(256, 256, 4)

        # Stage 4: 512 通道
        self.conv4 = Conv(256, 512, k=3, s=2, p=1)
        self.coord_att3 = self._create_attention_layer(512, 512, 8)

        # Stage 5: 1024 通道
        self.conv5 = Conv(512, 1024, k=3, s=2, p=1)
        self.coord_att4 = self._create_attention_layer(1024, 1024, 8)

        # 存储所有的注意力层用于可视化
        self.coord_att_layers = nn.ModuleList([
            self.coord_att1,
            self.coord_att2,
            self.coord_att3,
            self.coord_att4
        ])

    def _build_neck(self):
        """构建 FPN Neck（模板方法）"""
        self.up_conv1 = Conv(1024, 512, k=1, s=1)
        self.upsample1 = nn.Upsample(None, 2, mode='nearest')

        self.up_conv2 = Conv(512, 256, k=1, s=1)
        self.upsample2 = nn.Upsample(None, 2, mode='nearest')

        self.up_conv3 = Conv(256, 128, k=1, s=1)
        self.upsample3 = nn.Upsample(None, 2, mode='nearest')

    def _build_head(self, anchors):
        """构建检测头（模板方法）"""
        self.detect = Detect(nc=self.nc, anchors=anchors, ch=(256, 512, 1024))

    def _build_loss(self, anchors):
        """构建损失函数（模板方法）"""
        from .yolo_loss import YOLOLoss
        self.loss_fn = YOLOLoss(
            num_classes=self.nc,
            anchors=anchors,
            img_size=640
        )

    def _create_attention_layer(self, inp, oup, reduction):
        """
        创建注意力层（抽象方法 - 策略模式）

        子类通过实现此方法来选择使用哪种注意力机制。

        Args:
            inp: 输入通道数
            oup: 输出通道数
            reduction: 缩减比例

        Returns:
            注意力模块实例
        """
        raise NotImplementedError("子类必须实现 _create_attention_layer()")

    def _forward_attention_layer(self, layer, x, layer_idx, current_idx):
        """
        前向传播注意力层（辅助方法）

        Args:
            layer: 注意力层
            x: 输入特征
            layer_idx: 用户指定要获取注意力的层索引
            current_idx: 当前层索引

        Returns:
            (output_feature, attention_weights) 或 (output_feature, None, None)
        """
        if layer_idx == current_idx:
            return layer.forward_with_attention(x)
        return layer(x), None, None

    def forward(self, x, targets=None):
        """前向传播（模板方法）"""
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
        前向传播并返回指定层的注意力权重（模板方法）

        Args:
            x: 输入图像 [batch, 3, H, W]
            layer_idx: 要返回哪一层的注意力 (0-3, -1 表示不返回)

        Returns:
            如果 layer_idx >= 0: (predictions, attention_weights...)
            否则: (predictions, None, None)
        """
        x = self.conv0(x)
        x = self.conv1(x)

        # 处理各层注意力
        x = self.conv2(x)
        p3_backbone, att1_1, att1_2 = self._forward_attention_layer(
            self.coord_att_layers[0], x, layer_idx, 0
        )

        x = self.conv3(p3_backbone)
        p4_backbone, att2_1, att2_2 = self._forward_attention_layer(
            self.coord_att_layers[1], x, layer_idx, 1
        )

        x = self.conv4(p4_backbone)
        p5_backbone, att3_1, att3_2 = self._forward_attention_layer(
            self.coord_att_layers[2], x, layer_idx, 2
        )

        x = self.conv5(p5_backbone)
        p6_backbone, att4_1, att4_2 = self._forward_attention_layer(
            self.coord_att_layers[3], x, layer_idx, 3
        )

        # 提取注意力权重
        if layer_idx >= 0:
            att_weights = (att4_1, att4_2) if layer_idx == 3 else \
                         (att3_1, att3_2) if layer_idx == 2 else \
                         (att2_1, att2_2) if layer_idx == 1 else \
                         (att1_1, att1_2)
        else:
            att_weights = (None, None)

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
            return (predictions, *att_weights)
        return predictions, None, None


class YOLOCoordAttDetector(YOLOBaseDetector):
    """
    带 Coordinate Attention 的 YOLO 检测器

    实现策略：使用 CoordAtt 作为注意力模块
    """

    def __init__(self, nc=1, anchors=None):
        super().__init__(nc=nc, anchors=anchors, num_heads=1)

    def _create_attention_layer(self, inp, oup, reduction):
        """创建 CoordAtt 注意力层"""
        from .att_visualize import CoordAttWithVisualization
        return CoordAttWithVisualization(inp=inp, oup=oup, reduction=reduction)


class YOLOCoordCrossAttDetector(YOLOBaseDetector):
    """
    带 Coordinate Cross Attention 的 YOLO 检测器

    实现策略：使用 CoordCrossAtt 作为注意力模块

    Args:
        nc: 类别数量
        anchors: anchor boxes
        num_heads: CoordCrossAtt 的注意力头数
    """

    def __init__(self, nc=1, anchors=None, num_heads=1):
        super().__init__(nc=nc, anchors=anchors, num_heads=num_heads)

    def _create_attention_layer(self, inp, oup, reduction):
        """创建 CoordCrossAtt 注意力层"""
        from .att_visualize import CoordCrossAttWithVisualization
        return CoordCrossAttWithVisualization(
            inp=inp, oup=oup, reduction=reduction, num_heads=self.num_heads
        )
