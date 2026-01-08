import torch
import torch.nn as nn
from .conv import Conv
from .head import Detect

class CoordAtt(nn.Module):
    def __init__(self, inp: int, oup: int, reduction: int = 32):
        """
        Coordinate Attention 模块 (标准版)
        :param inp: 输入通道数
        :param oup: 输出通道数
        :param reduction: 缩减比例，用于构建 Bottleneck 结构
        """
        super().__init__()
        # 1. 空间池化：分别聚合水平和垂直信息
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        # 计算中间层的压缩通道数 (Bottleneck)
        mip = max(8, inp // reduction)

        # 2. 核心融合层：将 H 和 W 信息拼接后进行通道压缩和非线性激活
        # 这里使用 Ultralytics 的 Conv，默认包含 BatchNorm2d 和 SiLU
        self.cv1 = Conv(inp, mip, k=1, s=1, p=0)

        # 3. 恢复层：将压缩的通道 mip 恢复到输出通道 oup
        # 注意：这里不需要 SiLU，后面手动跟 Sigmoid
        self.cv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.cv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

        # 4. 如果输入输出通道不一致，需要一个 shortcut 变换
        self.identity = nn.Conv2d(inp, oup, 1) if inp != oup else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.size()

        # --- 信息嵌入 (Embedding) ---
        x_h = self.pool_h(x) # [N, C, H, 1]
        x_w = self.pool_w(x).permute(0, 1, 3, 2) # [N, C, 1, W] -> [N, C, W, 1]

        # --- 协同注意力生成 (Generation) ---
        # 拼接后的尺寸: [N, C, H+W, 1]
        y = self.cv1(torch.cat([x_h, x_w], dim=2)) # 压缩到 [N, mip, H+W, 1]

        # 拆分回两个方向
        x_h, x_w = torch.split(y, [h, w], dim=2) # 分别为 [N, mip, H, 1] 和 [N, mip, W, 1]
        x_w = x_w.permute(0, 1, 3, 2) # 转回 [N, mip, 1, W]

        # 生成 H 和 W 方向的权重 (Sigmoid 激活)
        a_h = self.cv_h(x_h).sigmoid() # [N, oup, H, 1]
        a_w = self.cv_w(x_w).sigmoid() # [N, oup, 1, W]

        # --- 重加权 (Reweight) ---
        # 如果 inp != oup，先转换 x，否则直接相乘
        return self.identity(x) * a_h * a_w


class CoordAttWithVisualization(CoordAtt):
    """带可视化功能的 Coordinate Attention，返回注意力权重"""

    def forward_with_attention(self, x):
        """返回输出和注意力权重"""
        n, c, h, w = x.size()

        # --- 信息嵌入 (Embedding) ---
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        # --- 协同注意力生成 (Generation) ---
        y = self.cv1(torch.cat([x_h, x_w], dim=2))

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        # 生成 H 和 W 方向的权重
        a_h = self.cv_h(x_h).sigmoid()
        a_w = self.cv_w(x_w).sigmoid()

        # --- 重加权 (Reweight) ---
        output = self.identity(x) * a_h * a_w

        return output, a_h, a_w


class YOLOCoordAttDetector(nn.Module):
    """
    带 Coordinate Attention 的 YOLO 检测器

    使用多个 CoordAtt 层增强特征表达，输出 3 个尺度的特征图用于检测
    """

    def __init__(self, nc=1, anchors=None):
        """
        Args:
            nc: 类别数量
            anchors: anchor boxes (默认使用 YOLOv3 anchors)
        """
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
        # Stage 1: 初始卷积 (stride 1)
        self.conv0 = Conv(3, 32, k=3, s=1, p=1)

        # Stage 2: P2 (stride 2)
        self.conv1 = Conv(32, 64, k=3, s=2, p=1)

        # Stage 3: P3 (stride 4) - 第一层 CoordAtt
        self.conv2 = Conv(64, 128, k=3, s=2, p=1)
        self.coord_att1 = CoordAttWithVisualization(inp=128, oup=128, reduction=4)

        # Stage 4: P4 (stride 8) - 第二层 CoordAtt
        self.conv3 = Conv(128, 256, k=3, s=2, p=1)
        self.coord_att2 = CoordAttWithVisualization(inp=256, oup=256, reduction=4)

        # Stage 5: P5 (stride 16) - 第三层 CoordAtt
        self.conv4 = Conv(256, 512, k=3, s=2, p=1)
        self.coord_att3 = CoordAttWithVisualization(inp=512, oup=512, reduction=8)

        # Stage 6: P6 (stride 32) - 第四层 CoordAtt
        self.conv5 = Conv(512, 1024, k=3, s=2, p=1)
        self.coord_att4 = CoordAttWithVisualization(inp=1024, oup=1024, reduction=8)

        # ===== Neck: FPN (只做特征融合，不改变stride) =====
        # 上采样路径
        self.up_conv1 = Conv(1024, 512, k=1, s=1)
        self.upsample1 = nn.Upsample(None, 2, mode='nearest')

        self.up_conv2 = Conv(512, 256, k=1, s=1)
        self.upsample2 = nn.Upsample(None, 2, mode='nearest')

        self.up_conv3 = Conv(256, 128, k=1, s=1)
        self.upsample3 = nn.Upsample(None, 2, mode='nearest')

        # ===== Detection Head =====
        # 输出 3 个尺度的特征图
        # p4 (stride 8): 256 channels, p5 (stride 16): 512 channels, p6 (stride 32): 1024 channels
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
        """
        前向传播

        Args:
            x: 输入图像 [batch, 3, H, W]
            targets: ground truth [num_boxes, 6] - (batch_idx, class_id, x, y, w, h)

        Returns:
            如果 targets 为 None: 返回预测结果
            如果提供 targets: 返回 {'predictions': ..., 'loss': ...}
        """
        # ===== Backbone with CoordAtt =====
        # Stage 1: 初始卷积
        x = self.conv0(x)

        # Stage 2: P2 (stride 2)
        x = self.conv1(x)

        # Stage 3: P3 (stride 4)
        x = self.conv2(x)
        p3_backbone = self.coord_att1(x)

        # Stage 4: P4 (stride 8)
        x = self.conv3(p3_backbone)
        p4_backbone = self.coord_att2(x)

        # Stage 5: P5 (stride 16)
        x = self.conv4(p4_backbone)
        p5_backbone = self.coord_att3(x)

        # Stage 6: P6 (stride 32)
        x = self.conv5(p5_backbone)
        p6_backbone = self.coord_att4(x)

        # ===== Neck: FPN 上采样融合 =====
        # P6 上采样到 P5 尺度并融合
        x = self.up_conv1(p6_backbone)
        x = self.upsample1(x)  # stride 32 -> 16
        p5 = torch.cat([x, p5_backbone], dim=1)  # [512+512=1024]
        p5 = Conv(1024, 512, k=1, s=1)(p5)

        # P5 上采样到 P4 尺度并融合
        x = self.up_conv2(p5)
        x = self.upsample2(x)  # stride 16 -> 8
        p4 = torch.cat([x, p4_backbone], dim=1)  # [256+256=512]
        p4 = Conv(512, 256, k=1, s=1)(p4)

        # P4 上采样到 P3 尺度并融合
        x = self.up_conv3(p4)
        x = self.upsample3(x)  # stride 8 -> 4
        p3 = torch.cat([x, p3_backbone], dim=1)  # [128+128=256]
        p3 = Conv(256, 128, k=1, s=1)(p3)

        # ===== Detection =====
        # 直接使用 backbone 的 P5, P6 和上采样得到的 P3
        # 但需要确保 stride 正确：P3/8, P4/16, P5/32
        # 当前：p3 是 stride 4，p4 是 stride 8，p5 是 stride 16
        # 所以需要使用 p4, p5, p6 作为检测输入

        predictions = self.detect([p4, p5, p6_backbone])

        # 如果提供了 targets，计算 loss
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
        # ===== Backbone with CoordAtt =====
        x = self.conv0(x)
        x = self.conv1(x)

        # Stage 3: P3 (stride 4)
        x = self.conv2(x)
        if layer_idx == 0:
            feat, a_h, a_w = self.coord_att_layers[0].forward_with_attention(x)
            p3_backbone = feat
        else:
            p3_backbone = self.coord_att_layers[0](x)

        # Stage 4: P4 (stride 8)
        x = self.conv3(p3_backbone)
        if layer_idx == 1:
            feat, a_h, a_w = self.coord_att_layers[1].forward_with_attention(x)
            p4_backbone = feat
        else:
            p4_backbone = self.coord_att_layers[1](x)

        # Stage 5: P5 (stride 16)
        x = self.conv4(p4_backbone)
        if layer_idx == 2:
            feat, a_h, a_w = self.coord_att_layers[2].forward_with_attention(x)
            p5_backbone = feat
        else:
            p5_backbone = self.coord_att_layers[2](x)

        # Stage 6: P6 (stride 32)
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

        # ===== Detection =====
        predictions = self.detect([p4, p5, p6_backbone])

        if layer_idx >= 0:
            return predictions, a_h, a_w
        return predictions, None, None
