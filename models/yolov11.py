import torch
import torch.nn as nn

from modules.conv import Conv, Concat
from modules.block import C3k2, SPPF, C2PSA
from modules.head import Detect, DetectAnchorFree
from modules.yolo_loss import YOLOLoss, YOLOLossAnchorFree
from utils import make_divisible, compute_channels, compute_depth


# Model scaling constants
YOLOV11_SCALES = {
    'n': [0.50, 0.25, 1024],  # nano: 181 layers, 2.6M params, 6.6 GFLOPs
    's': [0.50, 0.50, 1024],  # small: 181 layers, 9.5M params, 21.7 GFLOPs
    'm': [0.50, 1.00, 512],   # medium: 231 layers, 20.1M params, 68.5 GFLOPs
    'l': [1.00, 1.00, 512],   # large: 357 layers, 25.4M params, 87.6 GFLOPs
    'x': [1.00, 1.50, 512],   # xlarge: 357 layers, 57.0M params, 196.0 GFLOPs
}


class YOLOv11(nn.Module):
    """YOLOv11 model with clear network structure and dynamic scaling

    YOLOv11 uses modern building blocks:
    - C3k2: Faster CSP Bottleneck with 2 convolutions
    - SPPF: Spatial Pyramid Pooling - Fast
    - C2PSA: C2 module with Position-Sensitive Attention

    Supports multiple model scales: n (nano), s (small), m (medium), l (large), x (xlarge)

    Note: While YOLOv11 is typically anchor-free, this implementation uses
    Detect layer with anchors for compatibility with the existing YOLOLoss.
    """
    def __init__(self, nc=80, scale='s', anchors=None, img_size=640,
                 depth_multiple=None, width_multiple=None, max_channels=None):
        """
        Args:
            nc: number of classes
            scale: model scale ('n', 's', 'm', 'l', 'x'), or custom tuple (depth, width, max_ch)
            anchors: anchor boxes for 3 scales (default YOLOv3 anchors if None)
            img_size: input image size for loss calculation
            depth_multiple: manual depth override (if None, uses scale)
            width_multiple: manual width override (if None, uses scale)
            max_channels: manual max channels override (if None, uses scale)
        """
        super().__init__()
        self.nc = nc
        self.img_size = img_size

        # Get scaling parameters
        if isinstance(scale, str):
            if scale not in YOLOV11_SCALES:
                raise ValueError(f"Invalid scale: {scale}. Must be one of {list(YOLOV11_SCALES.keys())}")
            depth_multiple, width_multiple, max_channels = YOLOV11_SCALES[scale]
        elif isinstance(scale, (list, tuple)) and len(scale) == 3:
            depth_multiple, width_multiple, max_channels = scale
        else:
            raise ValueError(f"Invalid scale format: {scale}")

        # Allow manual overrides
        if depth_multiple is None:
            depth_multiple = YOLOV11_SCALES['s'][0]
        if width_multiple is None:
            width_multiple = YOLOV11_SCALES['s'][1]
        if max_channels is None:
            max_channels = YOLOV11_SCALES['s'][2]

        self.depth_multiple = depth_multiple
        self.width_multiple = width_multiple
        self.max_channels = max_channels

        # Anchor-free: no anchors needed
        self.anchors = None  # Not used in anchor-free version

        # ===== Backbone =====
        # Channel calculations with width scaling
        c1 = compute_channels(64, width_multiple, max_channels)
        c2 = compute_channels(128, width_multiple, max_channels)
        c3 = compute_channels(256, width_multiple, max_channels)
        c4 = compute_channels(512, width_multiple, max_channels)
        c5 = compute_channels(1024, width_multiple, max_channels)

        # Depth calculations for repeating layers
        n2 = compute_depth(2, depth_multiple)

        # [-1, 1, Conv, [64, 3, 2]]  # 0-P1/2
        self.conv0 = Conv(3, c1, 3, 2)

        # [-1, 1, Conv, [128, 3, 2]]  # 1-P2/4
        self.conv1 = Conv(c1, c2, 3, 2)

        # [-1, 2, C3k2, [256, False, 0.25]]  # 2
        self.c3k2_2 = C3k2(c2, c3, n2, c3k=False, e=0.25)

        # [-1, 1, Conv, [256, 3, 2]]  # 3-P3/8
        self.conv3 = Conv(c3, c3, 3, 2)

        # [-1, 2, C3k2, [512, False, 0.25]]  # 4
        self.c3k2_4 = C3k2(c3, c4, n2, c3k=False, e=0.25)

        # [-1, 1, Conv, [512, 3, 2]]  # 5-P4/16
        self.conv5 = Conv(c4, c4, 3, 2)

        # [-1, 2, C3k2, [512, True]]  # 6
        self.c3k2_6 = C3k2(c4, c4, n2, c3k=True)

        # [-1, 1, Conv, [1024, 3, 2]]  # 7-P5/32
        self.conv7 = Conv(c4, c5, 3, 2)

        # [-1, 2, C3k2, [1024, True]]  # 8
        self.c3k2_8 = C3k2(c5, c5, n2, c3k=True)

        # [-1, 1, SPPF, [1024, 5]]  # 9
        self.sppf9 = SPPF(c5, c5, 5)

        # [-1, 2, C2PSA, [1024]]  # 10
        self.c2psa10 = C2PSA(c5, c5, n2)

        # ===== Head (FPN + PAN) =====
        # Upsample path
        # [-1, 1, nn.Upsample, [None, 2, "nearest"]]  # 11
        self.upsample11 = nn.Upsample(None, 2, "nearest")

        # [[-1, 6], 1, Concat, [1]]  # 12 - cat backbone P4
        self.concat12 = Concat(1)

        # [-1, 2, C3k2, [512, False]]  # 13
        self.c3k2_13 = C3k2(c5 + c4, c4, n2, c3k=False)

        # [-1, 1, nn.Upsample, [None, 2, "nearest"]]  # 14
        self.upsample14 = nn.Upsample(None, 2, "nearest")

        # [[-1, 4], 1, Concat, [1]]  # 15 - cat backbone P3
        self.concat15 = Concat(1)

        # [-1, 2, C3k2, [256, False]]  # 16 (P3/8-small)
        self.c3k2_16 = C3k2(c4 + c4, c3, n2, c3k=False)

        # Downsample path
        # [-1, 1, Conv, [256, 3, 2]]  # 17
        self.conv17 = Conv(c3, c3, 3, 2)

        # [[-1, 13], 1, Concat, [1]]  # 18 - cat head P4
        self.concat18 = Concat(1)

        # [-1, 2, C3k2, [512, False]]  # 19 (P4/16-medium)
        self.c3k2_19 = C3k2(c3 + c4, c4, n2, c3k=False)

        # [-1, 1, Conv, [512, 3, 2]]  # 20
        self.conv20 = Conv(c4, c4, 3, 2)

        # [[-1, 10], 1, Concat, [1]]  # 21 - cat head P5
        self.concat21 = Concat(1)

        # [-1, 2, C3k2, [1024, True]]  # 22 (P5/32-large)
        self.c3k2_22 = C3k2(c4 + c5, c5, n2, c3k=True)

        # ===== Detection Head =====
        # Anchor-free detection head (like YOLOv8/v11)
        # [[16, 19, 22], 1, DetectAnchorFree, [nc]]
        self.detect = DetectAnchorFree(nc=nc, reg_max=32, ch=(c3, c4, c5))

        # Initialize biases following ultralytics formula
        # This is CRITICAL for proper initial loss values
        self.detect.initialize_biases(img_size=img_size)

        # Anchor-free loss function with DFL
        self.loss_fn = YOLOLossAnchorFree(
            num_classes=nc,
            reg_max=32,
            img_size=img_size,
            use_dfl=True
        )

    def forward(self, x: torch.Tensor, targets=None):
        """
        Forward pass

        Args:
            x: input images [batch, 3, height, width]
            targets: ground truth labels [num_boxes, 6] where each row is
                     [batch_idx, class_id, x_center, y_center, width, height]

        Returns:
            If targets is None: predictions (inference mode)
            If targets is provided: dict with 'predictions' and 'loss'
        """
        # ===== Backbone =====
        x = self.conv0(x)           # 0
        x = self.conv1(x)           # 1
        x = self.c3k2_2(x)          # 2
        x = self.conv3(x)           # 3
        x = self.c3k2_4(x)          # 4
        p3_backbone = x             # Save P3 for head connection
        x = self.conv5(x)           # 5
        x = self.c3k2_6(x)          # 6
        p4_backbone = x             # Save P4 for head connection
        x = self.conv7(x)           # 7
        x = self.c3k2_8(x)          # 8
        x = self.sppf9(x)           # 9
        p5 = self.c2psa10(x)        # 10 - Save P5

        # ===== Head - Upsample Path (Top-down) =====
        x = self.upsample11(p5)             # 11
        x = self.concat12([x, p4_backbone])  # 12
        x = self.c3k2_13(x)                 # 13
        p4_head = x                         # Save P4 head for downsample path

        x = self.upsample14(x)              # 14
        x = self.concat15([x, p3_backbone])  # 15
        x = self.c3k2_16(x)                 # 16
        p3 = x                              # P3 output

        # ===== Head - Downsample Path (Bottom-up) =====
        x = self.conv17(p3)                 # 17
        x = self.concat18([x, p4_head])     # 18
        x = self.c3k2_19(x)                 # 19
        p4 = x                              # P4 output

        x = self.conv20(x)                  # 20
        x = self.concat21([x, p5])          # 21
        p5 = self.c3k2_22(x)                # 22

        # ===== Detection Head =====
        predictions = self.detect([p3, p4, p5])

        # If targets provided, compute loss
        if targets is not None:
            # Anchor-free: predictions is {'cls': [...], 'reg': [...]}
            # loss_fn returns (loss * batch_size, loss.detach()) following ultralytics format
            loss_for_backward, loss_items = self.loss_fn(predictions, targets)
            return loss_for_backward, loss_items, predictions

        return predictions
