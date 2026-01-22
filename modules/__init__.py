# modules/__init__.py
# 基础神经网络模块

# BiFPN 特征融合
from .bifpn import BiFPN_Cat

# 注意力模块
from .att import CoordAtt, CoordCrossAtt, BiCoordCrossAtt
from .att_visualize import (
    CoordAttWithVisualization,
    CoordCrossAttWithVisualization,
    BiCoordCrossAttWithVisualization
)

# 基础卷积和块模块
from .conv import Conv, Concat
from .block import (
    Bottleneck,
    C2f,
    C3,
    C3k,
    C3k2,
    SPPF,
    Attention,
    PSABlock,
    C2PSA,
    )

# YOLO 检测头和损失
from .head import Detect
from .yolo_loss import YOLOLoss

__all__ = [
    # BiFPN
    'BiFPN_Cat',

    # 注意力
    'CoordAtt',
    'CoordCrossAtt',
    'BiCoordCrossAtt',
    'CoordAttWithVisualization',
    'CoordCrossAttWithVisualization',
    'BiCoordCrossAttWithVisualization',

    # 基础模块
    'Conv',
    'Concat',
    'Bottleneck',
    "C2f",
    "C3",
    "C3k",
    "C3k2",
    "SPPF",
    "Attention",
    "PSABlock",
    "C2PSA",

    # YOLO 组件
    'Detect',
    'YOLOLoss',
]
