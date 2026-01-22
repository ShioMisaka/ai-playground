import torch
import torch.nn as nn

from modules.conv import Conv, Concat
from modules.block import Bottleneck
from modules.head import Detect
from modules.yolo_loss import YOLOLoss


# Model scaling constants for YOLOv3
YOLOV3_SCALES = {
    'n': [0.33, 0.25, 1024],  # nano version
    's': [0.33, 0.50, 1024],  # small version
    'm': [0.67, 0.75, 768],   # medium version
    'l': [1.00, 1.00, 512],   # large version (full YOLOv3)
    'x': [1.00, 1.25, 512],   # xlarge version
}


def make_divisible(x, divisor=8):
    """Returns nearest x divisible by divisor."""
    return int(round(x / divisor) * divisor)


def compute_channels(channels, width_multiple, max_channels=None):
    """Compute scaled number of channels.

    Args:
        channels: base number of channels
        width_multiple: width scaling factor
        max_channels: maximum number of channels (optional)

    Returns:
        Scaled number of channels
    """
    channels = make_divisible(channels * width_multiple)
    if max_channels is not None:
        channels = min(channels, max_channels)
    return int(max(channels, 8))  # Ensure at least 8 channels


def compute_depth(n, depth_multiple):
    """Compute scaled number of layers/repeats.

    Args:
        n: base number of layers
        depth_multiple: depth scaling factor

    Returns:
        Scaled number of layers (at least 1)
    """
    if n <= 1:
        return n
    return max(round(n * depth_multiple), 1)


class YOLOv3(nn.Module):
    """YOLOv3 model with clear network structure and dynamic scaling

    Supports multiple model scales: n (nano), s (small), m (medium), l (large), x (xlarge)
    Uses Darknet53 backbone with FPN-style detection head.
    """
    def __init__(self, nc=80, scale='l', anchors=None, img_size=640,
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
            if scale not in YOLOV3_SCALES:
                raise ValueError(f"Invalid scale: {scale}. Must be one of {list(YOLOV3_SCALES.keys())}")
            depth_multiple, width_multiple, max_channels = YOLOV3_SCALES[scale]
        elif isinstance(scale, (list, tuple)) and len(scale) == 3:
            depth_multiple, width_multiple, max_channels = scale
        else:
            raise ValueError(f"Invalid scale format: {scale}")

        # Allow manual overrides
        if depth_multiple is None:
            depth_multiple = YOLOV3_SCALES['l'][0]
        if width_multiple is None:
            width_multiple = YOLOV3_SCALES['l'][1]
        if max_channels is None:
            max_channels = YOLOV3_SCALES['l'][2]

        self.depth_multiple = depth_multiple
        self.width_multiple = width_multiple
        self.max_channels = max_channels

        # Default YOLOv3 anchors (P3/8, P4/16, P5/32)
        if anchors is None:
            anchors = [
                [10, 13, 16, 30, 33, 23],      # P3/8
                [30, 61, 62, 45, 59, 119],     # P4/16
                [116, 90, 156, 198, 373, 326]  # P5/32
            ]

        # ===== Backbone: Darknet53 (scaled) =====
        # Channel calculations - base Darknet53 channels: 8, 16, 32, 64, 128, 256
        c1 = compute_channels(8, width_multiple, max_channels)
        c2 = compute_channels(16, width_multiple, max_channels)
        c3 = compute_channels(32, width_multiple, max_channels)
        c4 = compute_channels(64, width_multiple, max_channels)
        c5 = compute_channels(128, width_multiple, max_channels)
        c6 = compute_channels(256, width_multiple, max_channels)

        # Depth calculations - base repeats: 1, 2, 8, 8, 4
        n1 = compute_depth(1, depth_multiple)
        n2 = compute_depth(2, depth_multiple)
        n8 = compute_depth(8, depth_multiple)
        n4 = compute_depth(4, depth_multiple)

        self.conv0 = Conv(3, c1, 3, 1)           # 0
        self.conv1 = Conv(c1, c2, 3, 2)          # 1-P1/2
        self.bottleneck2 = Bottleneck(c2, c2)    # 2
        self.conv3 = Conv(c2, c3, 3, 2)          # 3-P2/4

        # Layer 4: 2 bottlenecks
        self.bottleneck4 = nn.Sequential(
            *[Bottleneck(c3, c3) for _ in range(n2)]
        )

        self.conv5 = Conv(c3, c4, 3, 2)          # 5-P3/8

        # Layer 6: 8 bottlenecks
        self.bottleneck6 = nn.Sequential(
            *[Bottleneck(c4, c4) for _ in range(n8)]
        )

        self.conv7 = Conv(c4, c5, 3, 2)          # 7-P4/16

        # Layer 8: 8 bottlenecks
        self.bottleneck8 = nn.Sequential(
            *[Bottleneck(c5, c5) for _ in range(n8)]
        )

        self.conv9 = Conv(c5, c6, 3, 2)          # 9-P5/32

        # Layer 10: 4 bottlenecks
        self.bottleneck10 = nn.Sequential(
            *[Bottleneck(c6, c6) for _ in range(n4)]
        )

        # ===== Head: YOLOv3 Detection Head =====
        # Large objects (P5/32)
        self.bottleneck11 = Bottleneck(c6, c6, shortcut=False)  # 11
        self.conv12 = Conv(c6, c5, 1, 1)         # 12
        self.conv13 = Conv(c5, c6, 3, 1)         # 13
        self.conv14 = Conv(c6, c5, 1, 1)         # 14
        self.conv15 = Conv(c5, c6, 3, 1)         # 15 (P5/32-large)

        # Medium objects (P4/16)
        self.conv16 = Conv(c5, c4, 1, 1)         # 16 (from conv14)
        self.upsample17 = nn.Upsample(None, 2, mode='nearest')  # 17
        self.concat18 = Concat(dimension=1)       # 18 (concat with bottleneck8)
        self.bottleneck19 = Bottleneck(c5 + c4, c5, shortcut=False)  # 19
        self.bottleneck20 = Bottleneck(c5, c5, shortcut=False)  # 20
        self.conv21 = Conv(c5, c4, 1, 1)         # 21
        self.conv22 = Conv(c4, c5, 3, 1)         # 22 (P4/16-medium)

        # Small objects (P3/8)
        self.conv23 = Conv(c4, c3, 1, 1)         # 23 (from conv21)
        self.upsample24 = nn.Upsample(None, 2, mode='nearest')  # 24
        self.concat25 = Concat(dimension=1)       # 25 (concat with bottleneck6)
        self.bottleneck26 = Bottleneck(c3 + c4, c4, shortcut=False)  # 26

        # Layer 27: 2 bottlenecks
        self.bottleneck27 = nn.Sequential(
            *[Bottleneck(c4, c4, shortcut=False) for _ in range(n2)]
        )

        # Detection layers
        self.detect = Detect(nc=nc, anchors=anchors, ch=(c4, c5, c6))  # 28

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

        # If targets provided, compute loss
        if targets is not None:
            loss = self.loss_fn(predictions, targets)
            return {'predictions': predictions, 'loss': loss}

        return predictions
