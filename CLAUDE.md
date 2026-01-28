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
├── validate.py      # validate() 验证与 mAP
└── detector.py      # 检测器专用训练逻辑

utils/      # 工具模块
├── load.py          # create_dataloaders()
├── logger.py        # TrainingLogger, LiveTableLogger
├── metrics.py       # compute_map50()
├── curves.py        # plot_training_curves()
├── path_helper.py   # get_save_dir() 自动递增目录
├── transforms.py    # MosaicTransform, MixupTransform
├── ema.py           # ModelEMA 指数移动平均
└── model_summary.py # print_model_summary()

scripts/    # 脚本
└── plot_curves.py   # 训练曲线绘制脚本
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
    save_dir='runs/train',
    use_ema=True,          # 使用 EMA（推荐）
    use_mosaic=True,       # 使用 Mosaic 增强
    close_mosaic=10,       # 最后 10 个 epoch 关闭 Mosaic
)
```

### Auto-Increment Save Directory

`save_dir` 会自动递增避免覆盖：
- `runs/train/exp` → `runs/train/exp_1` → `runs/train/exp_2` ...

```python
from utils import get_save_dir

save_dir = get_save_dir('runs/train/exp')  # 自动处理冲突
```

### Custom Training Loop

```python
from engine import train_one_epoch, validate
from utils import TrainingLogger, LiveTableLogger, plot_training_curves

# CSV 日志（YOLO 风格表头：train/loss, metrics/mAP50(B) 等）
with TrainingLogger('runs/training_log.csv', is_detection=True) as csv_logger:
    # Live 表格（动态刷新训练进度）
    live_logger = LiveTableLogger(
        columns=["total_loss", "box_loss", "cls_loss", "dfl_loss"],
        total_epochs=100,
        console_width=130,
    )

    for epoch in range(epochs):
        live_logger.start_epoch(epoch + 1, lr)

        train_metrics = train_one_epoch(model, train_loader, optimizer, device, epoch+1, nc=2)
        val_metrics = validate(model, val_loader, device, nc=2, img_size=640)

        live_logger.update_row("train", train_metrics)
        live_logger.update_row("val", val_metrics)
        live_logger.end_epoch(epoch_time)

        csv_logger.write_epoch(epoch + 1, epoch_time, lr, train_metrics, val_metrics)

    live_logger.stop()

# 绘制训练曲线（生成 4 张独立 PNG）
plot_training_curves('runs/training_log.csv', save_dir='runs')
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

**CosineAnnealingLR** 平滑下降，无中途跳变：
- `T_max=epochs`: 余弦退火周期长度
- `eta_min=1e-6`: 最小学习率

### EMA (Exponential Moving Average)

EMA 通过维护历史权重的指数移动平均，获得更稳定的模型：
- `decay=0.9999`: 衰减系数
- 动态调整公式：`decay = initial_decay * (1 - exp(-updates / 2000))`
- 验证时使用 EMA 模型可获得更平滑的 mAP 曲线

```python
from utils import ModelEMA

ema = ModelEMA(model, decay=0.9999)
for batch in dataloader:
    loss = model(imgs, targets)
    loss.backward()
    optimizer.step()
    ema.update(model)  # 更新 EMA

# 验证时使用 EMA 模型
val_metrics = validate(ema.ema, val_loader, ...)
```

### Mosaic Data Augmentation

Mosaic 将 4 张图片拼接成一张大图，是 YOLOv4/v5/v8/v11 的核心增强：
- 提升小目标检测能力
- 训练后期建议关闭（默认最后 10 个 epoch）

```python
from utils.transforms import MosaicTransform

mosaic = MosaicTransform(dataset, img_size=640, prob=1.0)
img, boxes = mosaic(img, boxes)

# 训练后期关闭
mosaic.enable = False
```

### TaskAlignedAssigner (TAL) 关键参数

**TAL 是 YOLOv8/v11 的核心正样本分配策略**，直接影响 Box Loss 收敛：

| Parameter | Value | 说明 |
|-----------|-------|------|
| `topk` | 13 | 每个 GT 选取的候选正样本数（官方: 10-13） |
| `alpha` | 0.5 | 分类分数权重 |
| `beta` | 6.0 | **IoU 指数权重（关键！）- 官方 6.0** |
| Soft Labels | Yes | 使用归一化对齐分数作为目标，而非硬标签 1.0 |

**为什么 beta=6.0 很重要？**
- `beta=2.0`（旧值）：过于宽松，容忍 IoU 低的框作为正样本
- `beta=6.0`（官方）：严格，只有 IoU 高的框才能获得高分
- 低质量匹配会导致 Box Loss 难以收敛，mAP50 停滞在 50-70%

### CSV 日志格式

CSV 表头采用 YOLO 风格的斜杠层级格式：

```
epoch,time,lr,train/loss,val/loss,train/box_loss,train/cls_loss,train/dfl_loss,val/box_loss,val/cls_loss,val/dfl_loss,metrics/precision(B),metrics/recall(B),metrics/mAP50(B),metrics/mAP50-95(B)
```

### 训练曲线输出

`plot_training_curves()` 生成 4 张独立的 PNG 图片：

| 文件名 | 内容 |
|--------|------|
| `loss_analysis.png` | Loss 曲线（2x4 布局：Train/Val × box/cls/dfl/total） |
| `map_performance.png` | mAP@0.5 和 mAP@0.5:0.95 |
| `precision_recall.png` | Precision 和 Recall |
| `training_status.png` | 训练时间和学习率 |

所有曲线都带有黄色点状虚线平滑曲线（Savitzky-Golay 滤波）。

### Detection Head Mode Handling

检测头在不同模式下返回不同格式：
- **训练模式**: 返回用于 loss 计算的格式
  - `DetectAnchorFree`: `{'cls': [...], 'reg': [...]}`
- **推理模式**: 返回用于 NMS 的格式

验证时临时切换模式：
```python
model.detect.train()  # 训练模式用于计算 loss
# ... run validation ...
model.detect.eval()   # 恢复推理模式
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
