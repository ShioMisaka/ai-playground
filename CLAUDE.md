# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PyTorch-based computer vision research project focused on object detection:
- **YOLOv11** - Anchor-free detection with DFL and Task-Aligned Learning (推荐使用)
- **YOLOv3** - Anchor-based detection with Darknet-53 backbone
- **Attention Modules** - CoordAtt, CoordCrossAtt, BiCoordCrossAtt for spatial feature enhancement
- **BiFPN** - Multi-scale feature fusion with learnable weights
- **Unified YOLO Interface** - Ultralytics-style API for training and inference

## Project Structure

```
ai-playground/
├── configs/                      # 配置文件目录
│   ├── data/                     # 数据集配置
│   │   └── coco8.yaml           # COCO8 数据集配置
│   ├── default.yaml             # 全局默认配置
│   └── models/                  # 模型配置
│       ├── yolov11n.yaml        # YOLOv11-nano 模型配置
│       └── yolov11s.yaml        # YOLOv11-small 模型配置
│
├── modules/                      # 基础神经网络模块
│   ├── __init__.py              # 模块导出
│   ├── att.py                   # 注意力机制 (CoordAtt, CoordCrossAtt, BiCoordCrossAtt)
│   ├── att_visualize.py         # 注意力可视化模块
│   ├── bifpn.py                 # 双向特征金字塔网络 (BiFPN_Cat)
│   ├── block.py                 # 基础块 (Bottleneck, C2f, C3, C3k2, SPPF, C2PSA)
│   ├── conv.py                  # 卷积模块 (Conv, Concat, autopad)
│   ├── head.py                  # 检测头 (Detect, DetectAnchorFree)
│   └── yolo_loss.py             # YOLO 损失函数 (YOLOLoss, YOLOLossAnchorFree)
│
├── models/                       # 完整模型实现
│   ├── __init__.py              # 模型导出（重新导出 modules 的所有内容）
│   ├── yolov11.py               # YOLOv11 模型（推荐）
│   ├── yolov3.py                # YOLOv3 模型
│   ├── yolo_att.py              # 带注意力机制的 YOLO（旧版）
│   ├── yolo.py                  # Ultralytics 风格的统一 YOLO 接口
│   └── lightweight_compare.py   # 轻量级对比网络
│
├── engine/                       # 训练引擎核心
│   ├── __init__.py              # 引擎导出
│   ├── train.py                 # CLI 训练脚本
│   ├── trainer.py               # DetectionTrainer 统一训练器
│   ├── training.py              # train_one_epoch 核心训练逻辑
│   ├── validate.py              # 验证与 mAP 计算 (validate, evaluate, test)
│   ├── detector.py              # 检测器专用训练逻辑
│   ├── classifier.py            # 分类器专用训练逻辑
│   ├── predictor.py             # 预测接口 (LetterBox, Results, Boxes, _post_process)
│   ├── visualize.py             # 可视化工具集合
│   ├── visualizer.py            # 可视化实现细节
│   └── comparison.py            # 模型对比工具
│
├── utils/                        # 工具模块
│   ├── __init__.py              # 工具导出
│   ├── config.py                # 配置管理 (load_yaml, save_yaml, get_config, merge_config)
│   ├── load.py                  # 数据加载 (create_dataloaders)
│   ├── logger.py                # 训练日志 (TrainingLogger, LiveTableLogger)
│   ├── metrics.py               # 评估指标 (compute_ap, compute_detection_metrics)
│   ├── curves.py                # 训练曲线可视化 (plot_training_curves)
│   ├── path_helper.py           # 路径辅助 (get_save_dir 自动递增目录)
│   ├── transforms.py            # 数据增强 (MosaicTransform, MixupTransform)
│   ├── ema.py                   # 指数移动平均 (ModelEMA)
│   ├── model_summary.py         # 模型摘要 (print_model_summary)
│   ├── scaling.py               # 模型缩放工具
│   └── table.py                 # 表格格式化工具
│
├── scripts/                      # 实用脚本
│   ├── plot_curves.py           # 训练曲线绘制
│   ├── predict_test.py          # 预测测试脚本
│   ├── test_attention.py        # 注意力机制测试
│   ├── train_test.py            # 训练测试脚本
│   └── visualization/           # 可视化脚本集合
│       ├── compare_attention_mechanisms.py
│       ├── view_improved_coordcrossatt.py
│       ├── visualize_coordatt.py
│       ├── visualize_improved_coordcrossatt.py
│       └── visualize_trained_coordatt.py
│
├── tests/                        # 单元测试
│   ├── __init__.py
│   ├── conftest.py              # Pytest 配置
│   ├── att_test.py              # 注意力模块测试
│   ├── fpn_test.py              # FPN 模块测试
│   ├── test_live_table.py       # 训练表格日志测试
│   ├── test_predict.py          # 预测功能测试
│   ├── test_trainer_fixes.py    # 训练器修复测试
│   ├── utils/                   # 工具模块测试
│   │   └── test_config.py      # 配置管理测试
│   ├── models/                  # 模型测试
│   │   └── test_yolo_train.py  # YOLO 训练测试
│   ├── engine/                  # 引擎测试
│   │   └── test_trainer.py      # 训练器测试
│   └── integration/             # 集成测试
│       ├── test_full_workflow.py
│       └── test_training_integration.py
│
├── demos/                        # 示例脚本
│   ├── mnist_demo.py            # MNIST 分类示例
│   └── yolov3_demo.py           # YOLOv3 演示
│
├── datasets/                     # 数据集存储
│   ├── MNIST/                   # MNIST 数据集
│   └── MY_TEST_DATA/            # 自定义测试数据集
│
└── outputs/                      # 输出目录（不提交）
    └── train/                   # 训练输出
```

## Import Convention

**统一使用 `from models` 导入所有内容**（`models/__init__.py` 已重新导出 `modules` 的所有内容）：

```python
# 推荐
from models import YOLOv11, YOLO, Conv, CoordAtt, BiFPN_Cat

# 不推荐（除非只需要基础模块）
from modules import Conv, CoordAtt

# 引擎和工具直接从各自模块导入
from engine import DetectionTrainer, validate, LetterBox, Results
from utils import get_config, TrainingLogger, ModelEMA, plot_training_curves
```

## Training Workflow

### Quick Start - CLI Training

```bash
# 方式1: 使用 CLI 参数（最简单）
python -m engine.train --name exp001 --epochs 100 --batch_size 16

# 方式2: 使用配置文件
python -m engine.train --config configs/experiments/my_exp.yaml

# 方式3: 结合模型配置
python -m engine.train \
  --model-config configs/models/yolov11n.yaml \
  --name exp003 \
  --epochs 200

# 嵌套参数覆盖
python -m engine.train --name exp002 optimizer.lr=0.002 scheduler.min_lr=1e-7
```

### Python API Training

#### 方式 1: 使用 DetectionTrainer（推荐）

```python
from models import YOLOv11
from engine import DetectionTrainer
from utils import get_config

# 创建配置
cfg = get_config(
    **{'train.name': 'exp001',
       'train.epochs': 100,
       'train.batch_size': 16,
       'data.name': 'coco8',
       'system.device': 'cuda'}
)

# 创建模型
model = YOLOv11(nc=2, scale='n')

# 创建训练器并训练
trainer = DetectionTrainer(model, cfg)
trainer.train()
```

#### 方式 2: 使用统一 YOLO 接口

```python
from models import YOLO

# 从配置文件创建模型
model = YOLO('configs/models/yolov11n.yaml')

# 训练
model.train(data='configs/data/coco8.yaml', epochs=100, batch=16)
```

#### 方式 3: 自定义训练循环

```python
from engine import train_one_epoch, validate
from utils import TrainingLogger, LiveTableLogger, plot_training_curves
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

# 创建模型和优化器
model = YOLOv11(nc=2, scale='n')
optimizer = Adam(model.parameters(), lr=0.001)
scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)

# CSV 日志（YOLO 风格表头：train/loss, metrics/mAP50(B) 等）
with TrainingLogger('runs/training_log.csv', is_detection=True) as csv_logger:
    # Live 表格（动态刷新训练进度）
    live_logger = LiveTableLogger(
        columns=["total_loss", "box_loss", "cls_loss", "dfl_loss"],
        total_epochs=100,
        console_width=130,
    )

    for epoch in range(100):
        live_logger.start_epoch(epoch + 1, scheduler.get_last_lr()[0])

        train_metrics = train_one_epoch(model, train_loader, optimizer, device, epoch+1, nc=2)
        val_metrics = validate(model, val_loader, device, nc=2, img_size=640)

        scheduler.step()

        live_logger.update_row("train", train_metrics)
        live_logger.update_row("val", val_metrics)
        live_logger.end_epoch(epoch_time)

        csv_logger.write_epoch(epoch + 1, epoch_time, scheduler.get_last_lr()[0], train_metrics, val_metrics)

    live_logger.stop()

# 绘制训练曲线（生成 4 张独立 PNG）
plot_training_curves('runs/training_log.csv', save_dir='runs')
```

## Configuration System

项目使用基于 YAML 的分层配置管理系统。

### 配置优先级

1. **默认配置** (`configs/default.yaml`) - 基础配置
2. **数据集配置** (`configs/data/*.yaml`) - 数据集特定配置
3. **模型配置** (`configs/models/*.yaml`) - 模型特定配置
4. **用户配置**（配置文件 OR CLI 参数）- 最高优先级

### 配置文件结构

```yaml
# configs/default.yaml
train:
  name: null              # 实验名称（必填）
  epochs: 100
  batch_size: 16
  img_size: 640
  letterbox: true         # 使用 letterbox 预处理
  mosaic: true            # 使用 Mosaic 增强
  mosaic_disable_epoch: 10 # 最后 10 个 epoch 关闭 Mosaic
  save_dir: runs/train

data:
  name: coco8             # 数据集名称
  train: images/train
  val: images/val

val:
  conf: 0.25
  iou: 0.45

predict:
  conf: 0.25
  iou: 0.45
  letterbox: true

model:
  nc: 80                  # 类别数
  scale: n                # 模型规模 (n/s/m/l/x)
  use_ema: true
  ema_decay: 0.9999

optimizer:
  type: Adam
  lr: 0.001
  betas: [0.9, 0.999]
  weight_decay: 0.0

scheduler:
  type: CosineAnnealingLR
  min_lr: 1.0e-06

system:
  device: cuda
  workers: 0
```

### CLI 参数

- `--config`: 配置文件路径
- `--model-config`: 模型配置文件路径
- `--name`: 实验名称（必填）
- `--epochs`, `--batch-size`, `--lr`, `--device`: 快捷参数
- `optimizer.lr=0.001`: 嵌套参数覆盖

## Prediction / Inference

### 统一 YOLO 接口（推荐）

```python
from models import YOLO

# 创建模型
model = YOLO('configs/models/yolov11n.yaml')
# 或从模型实例创建
model = YOLO(YOLOv11(nc=2, scale='n'))

# 推理
results = model.predict('image.jpg')
results = model.predict('image.jpg', conf=0.3, iou=0.5)

# 便捷调用
results = model('image.jpg')

# 处理结果
for r in results:
    boxes = r.boxes.xyxy  # 边界框坐标
    scores = r.boxes.conf  # 置信度
    classes = r.boxes.cls  # 类别

    # 绘制结果
    r.plot()
    r.save('output.jpg')
```

### 底层预测接口

```python
from engine.predictor import LetterBox, _post_process, Results, Boxes

# 预处理
letterbox = LetterBox(img_size=640)
img_tensor = letterbox(img)  # 保持长宽比的缩放 + 填充

# 模型推理
model.eval()
with torch.no_grad():
    predictions = model(img_tensor)

# 后处理
results = _post_process(predictions, orig_shape, conf_thresh=0.25, iou_thresh=0.45)

# 使用 Results 对象
for det in results:
    result = Results(orig_img, det['boxes'], det['scores'], det['classes'])
    result.plot()
```

### 预处理一致性

训练、验证、推理现在都使用相同的 letterbox 预处理，确保一致性：

- **训练时**: `YOLODataset` 使用 letterbox 预处理
- **验证时**: `YOLODataset` 使用 letterbox 预处理
- **推理时**: `LetterBox` 使用相同的 letterbox 预处理

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

## Training Output

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

### Auto-Increment Save Directory

`save_dir` 会自动递增避免覆盖：
- `runs/train/exp` → `runs/train/exp_1` → `runs/train/exp_2` ...

```python
from utils import get_save_dir

save_dir = get_save_dir('runs/train/exp')  # 自动处理冲突
```

## Detection Head Mode Handling

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

## Dataset Format

支持 YOLO 格式的自定义数据集：

```
datasets/MY_TEST_DATA/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
└── data.yaml          # 数据集配置
```

`data.yaml` 格式：
```yaml
path: /path/to/datasets/MY_TEST_DATA/
train: images/train
val: images/val

nc: 2  # 类别数
names:
  0: circle
  1: square
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

## Testing

```bash
# 运行所有测试
pytest

# 运行特定测试
pytest tests/att_test.py
pytest tests/engine/test_trainer.py

# 运行集成测试
pytest tests/integration/

# 带覆盖率报告
pytest --cov=modules --cov=models --cov=engine
```

## Environment

```bash
eval "$(conda shell.bash hook)"
conda activate torch_cpu
```

## Troubleshooting

### Box Loss 不下降

如果 `box_loss` 长期高于 4.0：

1. **检查学习率**：应为 0.001 或更低
2. **验证数据加载**：检查标签格式是否正确
3. **分析初始预测**：查看 IoU 分布
4. **确认 reg_max=16**：更高的值（如 32）会显著增加收敛难度
5. **验证 CIoU 已启用**：检查 `modules/yolo_loss.py` 中 `CIoU=True`
6. **确认 beta=6.0**：TAL 的 beta 参数应为 6.0

### mAP50 长期为 0%

如果 `mAP50` 超过 5 个 epoch 仍为 0.00%：

1. **验证标签格式**：类别 ID 必须从 0 开始
2. **检查边界框坐标**：应归一化到 [0, 1]（YOLO 格式）
3. **调整置信度阈值**：默认 NMS 阈值为 0.25

### 损失分量异常

训练早期各损失分量的正常范围：
- `box_loss`: 2.0-6.0（应稳定下降）
- `cls_loss`: 0.5-5.0（初始较高，逐渐下降）
- `dfl_loss`: 1.0-6.0（跟随 box_loss 趋势）
