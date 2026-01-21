# models/__init__.py
# 从 modules 导入基础模块，保持向后兼容

from modules import (
    BiFPN_Cat,
    CoordAtt,
    CoordCrossAtt,
    BiCoordCrossAtt,
    CoordAttWithVisualization,
    CoordCrossAttWithVisualization,
    BiCoordCrossAttWithVisualization,
    Bottleneck,
    Conv,
    Concat,
    Detect,
    YOLOLoss,
)

# YOLO 检测器
from .yolo_att import (
    YOLOCoordAttDetector,
    YOLOCoordCrossAttDetector,
    YOLOBiCoordCrossAttDetector
)

# 轻量级对比网络
from .lightweight_compare import (
    LightweightCoordAttNet,
    LightweightCoordCrossAttNet
)

# YOLOv3 模型
from .yolov3 import YOLOv3

__all__ = [
    # BiFPN (from modules)
    'BiFPN_Cat',

    # 注意力 (from modules)
    'CoordAtt',
    'CoordCrossAtt',
    'BiCoordCrossAtt',
    'CoordAttWithVisualization',
    'CoordCrossAttWithVisualization',
    'BiCoordCrossAttWithVisualization',

    # YOLO 检测器
    'YOLOCoordAttDetector',
    'YOLOCoordCrossAttDetector',
    'YOLOBiCoordCrossAttDetector',

    # 轻量级对比网络
    'LightweightCoordAttNet',
    'LightweightCoordCrossAttNet',

    # 基础模块 (from modules)
    'Bottleneck',
    'Conv',
    'Concat',
    'Detect',
    'YOLOLoss',
    'YOLOv3',
]
