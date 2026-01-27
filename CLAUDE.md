# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PyTorch-based computer vision research project focused on object detection:
- **YOLOv11** - Anchor-free detection with DFL and Task-Aligned Learning (推荐使用)
- **YOLOv3** - Anchor-based detection with Darknet-53 backbone
- **Attention Modules** - CoordAtt, CoordCrossAtt for spatial feature enhancement
- **BiFPN** - Multi-scale feature fusion with learnable weights

## Architecture

### Module Organization

```
modules/    # 基础神经网络模块
├── att.py           # CoordAtt, CoordCrossAtt
├── bifpn.py         # BiFPN_Cat
├── conv.py          # Conv (Conv2d + BN + SiLU)
├── block.py         # C3k2, SPPF, C2PSA
├── head.py          # Detect, DetectAnchorFree
└── yolo_loss.py     # YOLOLoss, YOLOLossAnchorFree

models/     # 完整模型（从 modules 导入并组合）
├── yolov11.py       # YOLOv11 (推荐)
├── yolov3.py        # YOLOv3
└── yolo_att.py      # YOLOCoordAttDetector (legacy)

engine/     # 训练引擎核心
├── train.py         # train() 主训练流程
├── training.py      # train_one_epoch() 核心训练逻辑
├── validate.py      # validate() 验证与 mAP50
└── detector.py      # 检测器专用训练逻辑

utils/      # 工具模块
├── load.py          # create_dataloaders()
├── logger.py        # TrainingLogger (CSV logging)
├── metrics.py       # compute_map50()
├── curves.py        # plot_training_curves()
└── model_summary.py # print_model_summary()
```

### Import Convention

**统一使用 `from models` 导入所有内容**（`models/__init__.py` 已重新导出 `modules` 的所有内容）：

```python
# 推荐
from models import YOLOv11, Conv, CoordAtt, BiFPN_Cat

# 不推荐（除非只需要基础模块）
from modules import Conv, CoordAtt
```

## Code Patterns

### Training YOLOv11

```python
from models import YOLOv11
from engine import train

model = YOLOv11(nc=2, scale='n')  # scale: n/s/m/l/x
train(
    model,
    config_path='datasets/MY_TEST_DATA/data.yaml',
    epochs=100,
    batch_size=16,
    img_size=640,
    lr=0.001,
    device='cuda',
    save_dir='runs/train'
)
```

### Using Conv Wrapper

```python
from models import Conv

self.cv1 = Conv(c1, c2, k=1, s=1, p=0)  # Conv+BN+SiLU
self.cv2 = Conv(c1, c2, k=3, s=1, p=1)  # 3x3 same padding
```

### Using BiFPN

```python
from models import BiFPN_Cat

bifpn = BiFPN_Cat(c1=[128, 256, 512])
out = bifpn([feat1, feat2, feat3])  # 输入通道可不同
```

### Custom Training Loop

```python
from engine import train_one_epoch, validate
from utils import TrainingLogger, plot_training_curves

with TrainingLogger('runs/training.csv', is_detection=True) as logger:
    for epoch in range(epochs):
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, epoch+1, nc=2)
        val_metrics = validate(model, val_loader, device, nc=2, img_size=640)
        logger.write_epoch(epoch+1, epoch_time, lr, train_metrics, val_metrics)

plot_training_curves('runs/training.csv', save_dir='runs')
```

## Core Training Config

### YOLOv11 Key Parameters

| Parameter | Value | 说明 |
|-----------|-------|------|
| `lr` | 0.001 | 更高值（如 0.01）会导致发散 |
| `box_loss_weight` | 7.5 | box loss 权重 |
| `cls_loss_weight` | 0.5 | cls loss 权重 |
| `dfl_loss_weight` | 1.5 | DFL loss 权重 |
| `reg_max` | 16 | DFL 分布 bin 数 |
| `iou_loss` | CIoU | Complete IoU 收敛更好 |

### Learning Rate Scheduler

**CosineAnnealingWarmRestarts** 帮助模型跳出局部最优：
- `T_0=10`: 第一次重启周期
- `T_mult=2`: 周期长度倍增
- `eta_min=1e-6`: 最小学习率

### Detection Head Mode Handling

检测头在不同模式下返回不同格式：
- **训练模式**: 返回用于 loss 计算的格式
  - `DetectAnchorFree`: `{'cls': [...], 'reg': [...]}`
- **推理模式**: 返回用于 NMS 的格式

验证时临时切换模式：
```python
model.detect.eval()  # 推理模式获取预测
# ... run validation ...
model.detect.train()  # 恢复训练模式
```

## Development Guidelines

### 添加新模型组件

1. 在 `modules/` 创建基础模块（Conv、Block、Loss 等）
2. 在 `models/` 创建完整模型
3. 在 `tests/` 创建测试文件：
   - 随机输入 tensor
   - 前向传播验证输出 shape
   - 反向传播验证可学习性
4. 在 `models/__init__.py` 导出

### 代码放置原则

| 代码类型 | 放置目录 |
|---------|---------|
| 训练/验证逻辑 | `engine/` |
| 工具函数（logging、metrics、plotting） | `utils/` |
| 神经网络组件 | `modules/` 或 `models/` |

### JSON 序列化

保存训练历史时，始终转换 PyTorch 类型：
```python
history['train_loss'].append(float(train_loss))
```

## Environment

```bash
eval "$(conda shell.bash hook)"
conda activate torch_cpu
```
