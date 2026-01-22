import torch
import torch.nn as nn

from modules.conv import Conv, Concat
from modules.block import C3k2, SPPF, C2PSA
from modules.head import Detect
from modules.yolo_loss import YOLOLoss


class YOLOv11(nn.Module):
    """YOLOv11 model with clear network structure

    YOLOv11 uses modern building blocks:
    - C3k2: Faster CSP Bottleneck with 2 convolutions
    - SPPF: Spatial Pyramid Pooling - Fast
    - C2PSA: C2 module with Position-Sensitive Attention

    Note: While YOLOv11 is typically anchor-free, this implementation uses
    Detect layer with anchors for compatibility with the existing YOLOLoss.
    """
    def __init__(self, nc=80, anchors=None, img_size=640):
        """
        Args:
            nc: number of classes
            anchors: anchor boxes for 3 scales (default YOLOv3 anchors if None)
            img_size: input image size for loss calculation
        """
        super().__init__()
        self.nc = nc
        self.img_size = img_size

        # Default anchors (same as YOLOv3 for compatibility)
        if anchors is None:
            anchors = [
                [10, 13, 16, 30, 33, 23],      # P3/8
                [30, 61, 62, 45, 59, 119],     # P4/16
                [116, 90, 156, 198, 373, 326]  # P5/32
            ]

        # ===== Backbone =====
        # [-1, 1, Conv, [64, 3, 2]]  # 0-P1/2
        self.conv0 = Conv(3, 64, 3, 2)

        # [-1, 1, Conv, [128, 3, 2]]  # 1-P2/4
        self.conv1 = Conv(64, 128, 3, 2)

        # [-1, 2, C3k2, [256, False, 0.25]]  # 2
        self.c3k2_2 = C3k2(128, 256, 2, c3k=False, e=0.25)

        # [-1, 1, Conv, [256, 3, 2]]  # 3-P3/8
        self.conv3 = Conv(256, 256, 3, 2)

        # [-1, 2, C3k2, [512, False, 0.25]]  # 4
        self.c3k2_4 = C3k2(256, 512, 2, c3k=False, e=0.25)

        # [-1, 1, Conv, [512, 3, 2]]  # 5-P4/16
        self.conv5 = Conv(512, 512, 3, 2)

        # [-1, 2, C3k2, [512, True]]  # 6
        self.c3k2_6 = C3k2(512, 512, 2, c3k=True)

        # [-1, 1, Conv, [1024, 3, 2]]  # 7-P5/32
        self.conv7 = Conv(512, 1024, 3, 2)

        # [-1, 2, C3k2, [1024, True]]  # 8
        self.c3k2_8 = C3k2(1024, 1024, 2, c3k=True)

        # [-1, 1, SPPF, [1024, 5]]  # 9
        self.sppf9 = SPPF(1024, 1024, 5)

        # [-1, 2, C2PSA, [1024]]  # 10
        self.c2psa10 = C2PSA(1024, 1024, 2)

        # ===== Head (FPN + PAN) =====
        # Upsample path
        # [-1, 1, nn.Upsample, [None, 2, "nearest"]]  # 11
        self.upsample11 = nn.Upsample(None, 2, "nearest")

        # [[-1, 6], 1, Concat, [1]]  # 12 - cat backbone P4
        self.concat12 = Concat(1)

        # [-1, 2, C3k2, [512, False]]  # 13
        self.c3k2_13 = C3k2(1536, 512, 2, c3k=False)

        # [-1, 1, nn.Upsample, [None, 2, "nearest"]]  # 14
        self.upsample14 = nn.Upsample(None, 2, "nearest")

        # [[-1, 4], 1, Concat, [1]]  # 15 - cat backbone P3
        self.concat15 = Concat(1)

        # [-1, 2, C3k2, [256, False]]  # 16 (P3/8-small)
        self.c3k2_16 = C3k2(1024, 256, 2, c3k=False)

        # Downsample path
        # [-1, 1, Conv, [256, 3, 2]]  # 17
        self.conv17 = Conv(256, 256, 3, 2)

        # [[-1, 13], 1, Concat, [1]]  # 18 - cat head P4
        self.concat18 = Concat(1)

        # [-1, 2, C3k2, [512, False]]  # 19 (P4/16-medium)
        self.c3k2_19 = C3k2(768, 512, 2, c3k=False)

        # [-1, 1, Conv, [512, 3, 2]]  # 20
        self.conv20 = Conv(512, 512, 3, 2)

        # [[-1, 10], 1, Concat, [1]]  # 21 - cat head P5
        self.concat21 = Concat(1)

        # [-1, 2, C3k2, [1024, True]]  # 22 (P5/32-large)
        self.c3k2_22 = C3k2(1536, 1024, 2, c3k=True)

        # ===== Detection Head =====
        # [[16, 19, 22], 1, Detect, [nc]]  # Detect(P3, P4, P5)
        # Use Detect layer for compatibility with YOLOLoss
        self.detect = Detect(nc=nc, anchors=anchors, ch=(256, 512, 1024))

        # Loss function
        self.loss_fn = YOLOLoss(
            num_classes=nc,
            anchors=anchors,
            img_size=img_size,
            use_wiou=True
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
            If targets is provided: {'predictions': predictions, 'loss': loss}
        """
        # ===== Backbone =====
        x = self.conv0(x)           # 0: [B, 64, H/2, W/2]
        x = self.conv1(x)           # 1: [B, 128, H/4, W/4]
        x = self.c3k2_2(x)          # 2: [B, 256, H/4, W/4]
        x = self.conv3(x)           # 3: [B, 256, H/8, W/8]
        x = self.c3k2_4(x)          # 4: [B, 512, H/8, W/8]
        p3_backbone = x             # Save P3 (H/8) for head connection
        x = self.conv5(x)           # 5: [B, 512, H/16, W/16]
        x = self.c3k2_6(x)          # 6: [B, 512, H/16, W/16]
        p4_backbone = x             # Save P4 (H/16) for head connection
        x = self.conv7(x)           # 7: [B, 1024, H/32, W/32]
        x = self.c3k2_8(x)          # 8: [B, 1024, H/32, W/32]
        x = self.sppf9(x)           # 9: [B, 1024, H/32, W/32]
        p5 = self.c2psa10(x)        # 10: [B, 1024, H/32, W/32] - Save P5

        # ===== Head - Upsample Path (Top-down) =====
        x = self.upsample11(p5)             # 11: [B, 1024, H/16, W/16]
        x = self.concat12([x, p4_backbone])  # 12: [B, 1536, H/16, W/16] (1024 + 512)
        x = self.c3k2_13(x)                 # 13: [B, 512, H/16, W/16]
        p4_head = x                         # Save P4 head for downsample path

        x = self.upsample14(x)              # 14: [B, 512, H/8, W/8]
        x = self.concat15([x, p3_backbone])  # 15: [B, 1024, H/8, W/8] (512 + 512)
        x = self.c3k2_16(x)                 # 16: [B, 256, H/8, W/8]
        p3 = x                              # P3 output

        # ===== Head - Downsample Path (Bottom-up) =====
        x = self.conv17(p3)                 # 17: [B, 256, H/16, W/16]
        x = self.concat18([x, p4_head])     # 18: [B, 768, H/16, W/16] (256 + 512)
        x = self.c3k2_19(x)                 # 19: [B, 512, H/16, W/16]
        p4 = x                              # P4 output

        x = self.conv20(x)                  # 20: [B, 512, H/32, W/32]
        x = self.concat21([x, p5])          # 21: [B, 1536, H/32, W/32] (512 + 1024)
        p5 = self.c3k2_22(x)                # 22: [B, 1024, H/32, W/32]

        # ===== Detection Head =====
        predictions = self.detect([p3, p4, p5])

        # If targets provided, compute loss
        if targets is not None:
            loss = self.loss_fn(predictions, targets)
            return {'predictions': predictions, 'loss': loss}

        return predictions
