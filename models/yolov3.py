import torch
import torch.nn as nn

from .conv import Conv, Concat
from .block import Bottleneck
from .head import Detect
from .yolo_loss import YOLOLoss

class YOLOv3(nn.Module):
    """YOLOv3 model with clear network structure"""
    def __init__(self, nc=80, anchors=None):
        """
        Args:
            nc: number of classes
            anchors: anchor boxes for 3 scales (default YOLOv3 anchors if None)
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
        
        # ===== Backbone: Darknet53 =====
        self.conv0 = Conv(3, 32, 3, 1)          # 0
        self.conv1 = Conv(32, 64, 3, 2)         # 1-P1/2
        self.bottleneck2 = Bottleneck(64, 64)   # 2
        self.conv3 = Conv(64, 128, 3, 2)        # 3-P2/4
        self.bottleneck4 = nn.Sequential(       # 4
            Bottleneck(128, 128),
            Bottleneck(128, 128)
        )
        self.conv5 = Conv(128, 256, 3, 2)       # 5-P3/8
        self.bottleneck6 = nn.Sequential(       # 6
            *[Bottleneck(256, 256) for _ in range(8)]
        )
        self.conv7 = Conv(256, 512, 3, 2)       # 7-P4/16
        self.bottleneck8 = nn.Sequential(       # 8
            *[Bottleneck(512, 512) for _ in range(8)]
        )
        self.conv9 = Conv(512, 1024, 3, 2)      # 9-P5/32
        self.bottleneck10 = nn.Sequential(      # 10
            *[Bottleneck(1024, 1024) for _ in range(4)]
        )
        
        # ===== Head: YOLOv3 Detection Head =====
        # Large objects (P5/32)
        self.bottleneck11 = Bottleneck(1024, 1024, shortcut=False)  # 11
        self.conv12 = Conv(1024, 512, 1, 1)     # 12
        self.conv13 = Conv(512, 1024, 3, 1)     # 13
        self.conv14 = Conv(1024, 512, 1, 1)     # 14
        self.conv15 = Conv(512, 1024, 3, 1)     # 15 (P5/32-large)
        
        # Medium objects (P4/16)
        self.conv16 = Conv(512, 256, 1, 1)      # 16 (from conv14)
        self.upsample17 = nn.Upsample(None, 2, mode='nearest')  # 17
        self.concat18 = Concat(dimension=1)           # 18 (concat with bottleneck8)
        self.bottleneck19 = Bottleneck(768, 512, shortcut=False)  # 19
        self.bottleneck20 = Bottleneck(512, 512, shortcut=False)  # 20
        self.conv21 = Conv(512, 256, 1, 1)      # 21
        self.conv22 = Conv(256, 512, 3, 1)      # 22 (P4/16-medium)
        
        # Small objects (P3/8)
        self.conv23 = Conv(256, 128, 1, 1)      # 23 (from conv21)
        self.upsample24 = nn.Upsample(None, 2, mode='nearest')  # 24
        self.concat25 = Concat(dimension=1)           # 25 (concat with bottleneck6)
        self.bottleneck26 = Bottleneck(384, 256, shortcut=False)  # 26
        self.bottleneck27 = nn.Sequential(      # 27 (P3/8-small)
            Bottleneck(256, 256, shortcut=False),
            Bottleneck(256, 256, shortcut=False)
        )
        
        # Detection layers
        self.detect = Detect(nc=nc, anchors=anchors, ch=(256, 512, 1024))  # 28

        # 添加loss计算器
        self.loss_fn = YOLOLoss(
            num_classes=nc,
            anchors=anchors if anchors else [
                [10, 13, 16, 30, 33, 23],
                [30, 61, 62, 45, 59, 119],
                [116, 90, 156, 198, 373, 326]
            ],
            img_size=640
        )
        
    def forward(self, x: torch.Tensor, targets=None):
        """
        Forward pass
        
        Args:
            x: input images [batch, 3, height, width]
            targets: ground truth labels [num_boxes, 6] where each row is 
                     [batch_idx, class_id, x_center, y_center, width, height]
        
        Returns:
            If targets is None: predictions
            If targets is provided: (predictions, loss) or {'predictions': ..., 'loss': ...}
        """
        # Backbone
        x = self.conv0(x)                    # 0
        x = self.conv1(x)                    # 1
        x = self.bottleneck2(x)              # 2
        x = self.conv3(x)                    # 3
        x = self.bottleneck4(x)              # 4
        x = self.conv5(x)                    # 5
        p3 = self.bottleneck6(x)             # 6 - save for P3
        x = self.conv7(p3)                   # 7
        p4 = self.bottleneck8(x)             # 8 - save for P4
        x = self.conv9(p4)                   # 9
        x = self.bottleneck10(x)             # 10
        
        # Head - Large objects (P5)
        x = self.bottleneck11(x)             # 11
        x = self.conv12(x)                   # 12
        x = self.conv13(x)                   # 13
        x_p5_route = self.conv14(x)          # 14 - route
        p5 = self.conv15(x_p5_route)         # 15 - P5 output
        
        # Head - Medium objects (P4)
        x = self.conv16(x_p5_route)          # 16
        x = self.upsample17(x)               # 17
        x = self.concat18([x, p4])           # 18 - concat with layer 8
        x = self.bottleneck19(x)             # 19
        x = self.bottleneck20(x)             # 20
        x_p4_route = self.conv21(x)          # 21 - route
        p4 = self.conv22(x_p4_route)         # 22 - P4 output
        
        # Head - Small objects (P3)
        x = self.conv23(x_p4_route)          # 23
        x = self.upsample24(x)               # 24
        x = self.concat25([x, p3])           # 25 - concat with layer 6
        x = self.bottleneck26(x)             # 26
        p3 = self.bottleneck27(x)            # 27 - P3 output
        
        # Detection
        predictions = self.detect([p3, p4, p5])     # 28 - Detect on P3, P4, P5
        
        # 如果提供了targets，计算loss
        if targets is not None:
            loss = self.loss_fn(predictions, targets)
            return {'predictions': predictions, 'loss': loss}
        
        return predictions
    
    def compute_loss(self, predictions, targets):
        """
        计算YOLO loss
        
        Args:
            predictions: 模型预测输出
            targets: [num_boxes, 6] - (batch_idx, class_id, x, y, w, h)
        
        Returns:
            总loss (标量)
        """
        # 这里需要根据你的Detect层的输出格式来实现
        # 如果你的Detect层已经实现了loss计算，可以调用它
        if hasattr(self.detect, 'compute_loss'):
            return self.detect.compute_loss(predictions, targets)
        
        # 否则这里是一个简单的示例实现
        # 你需要根据实际情况完善这个函数
        device = predictions[0].device if isinstance(predictions, (list, tuple)) else predictions.device
        
        # 简单示例：使用MSE loss（实际YOLO需要更复杂的loss）
        # 这只是占位符，你需要实现真正的YOLO loss
        loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        # TODO: 实现真正的YOLO loss计算
        # 包括: box loss, objectness loss, class loss
        
        return loss