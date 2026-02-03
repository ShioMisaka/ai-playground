# AI Playground

一个基于 PyTorch 的计算机视觉深度学习研究项目，专注于目标检测和注意力机制的实现与可视化。

## 项目简介

本项目实现了多个经典的深度学习模型，特别关注：

- **YOLOv11** - Anchor-free 目标检测，支持 DFL 和 Task-Aligned Learning（推荐）
- **YOLOv3** - 完整的目标检测系统（支持 WIoU v3 损失函数）
- **Coordinate Attention (CoordAtt)** - 坐标注意力机制
- **Coordinate Cross Attention (CoordCrossAtt)** - 坐标交叉注意力机制
- **BiFPN** - 双向特征金字塔网络，支持多尺度特征融合
- **统一 YOLO 接口** - Ultralytics 风格的训练和推理 API

## 项目特色

- **模块化设计**：清晰分离基础模块、完整模型和训练引擎
- **配置驱动**：灵活的 YAML 配置系统，支持多层配置覆盖
- **统一接口**：Ultralytics 风格的 YOLO 统一接口
- **完整测试**：从单元测试到集成测试的完整测试体系
- **丰富工具**：训练日志、曲线绘制、可视化等辅助工具
- **预处理一致性**：训练、验证、推理使用相同的 letterbox 预处理

## 环境要求

```bash
# 激活 conda 环境
eval "$(conda shell.bash hook)"
conda activate torch_cpu
```

## 快速开始

### 训练 YOLOv11（推荐）

项目使用基于 YAML 的配置管理系统，支持分层配置和灵活覆盖。

#### 方式 1: 使用 CLI 参数

```bash
# 基本训练
python -m engine.train --name exp001 --epochs 100 --batch_size 16

# 完整参数
python -m engine.train \
  --name exp001 \
  --epochs 100 \
  --batch_size 16 \
  --lr 0.001 \
  --device cuda

# 嵌套参数覆盖
python -m engine.train --name exp002 optimizer.lr=0.002 scheduler.min_lr=1e-7
```

#### 方式 2: 使用配置文件

```bash
# 使用预定义配置
python -m engine.train --config configs/experiments/my_exp.yaml

# 结合模型配置
python -m engine.train \
  --model-config configs/models/yolov11n.yaml \
  --name exp003 \
  --epochs 200
```

#### 方式 3: 在 Python 代码中训练

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
model = YOLOv11(nc=2, scale='s')

# 创建训练器并训练
trainer = DetectionTrainer(model, cfg)
trainer.train()
```

#### 方式 4: 使用统一 YOLO 接口

```python
from models import YOLO

# 从配置文件创建模型
model = YOLO('configs/models/yolov11n.yaml')

# 训练
model.train(data='configs/data/coco8.yaml', epochs=100, batch=16)

# 推理
results = model.predict('image.jpg', conf=0.3, iou=0.5)
```

### 配置系统说明

**配置优先级：**
1. 默认配置 (`configs/default.yaml`)
2. 数据集配置 (`configs/data/*.yaml`)
3. 模型配置 (`configs/models/*.yaml`)
4. 用户配置（配置文件 OR CLI 参数）

**配置文件结构：**
```
configs/
├── data/                     # 数据集配置
│   └── coco8.yaml
├── default.yaml              # 全局默认配置
└── models/                   # 模型配置
    ├── yolov11n.yaml
    └── yolov11s.yaml
```

**CLI 参数：**
- `--config`: 配置文件路径
- `--model-config`: 模型配置文件路径
- `--name`: 实验名称（必填）
- `--epochs`, `--batch-size`, `--lr`, `--device`: 快捷参数
- `optimizer.lr=0.001`: 嵌套参数覆盖

### 训练输出示例

**LiveTable 动态刷新：**
```
Epoch 1/100  lr=0.001000
           total_loss  box_loss  cls_loss  dfl_loss
Train -      10.2023    2.3450    0.9290    0.0670  100% ━━━━━━━━━━━━━━━━━━━━ 34/34 1.1s/it 37.5s<0.0s
Val   -       9.4208    2.2350    0.6850    0.1520  mAP50: 0.152  mAP50-95: 0.089
Time: 38.59s
```

**生成的文件：**
| 文件 | 说明 |
|------|------|
| `training_log.csv` | 每个 epoch 的详细指标（YOLO 风格表头） |
| `loss_analysis.png` | Loss 曲线（2x4 布局） |
| `map_performance.png` | mAP@0.5 和 mAP@0.5:0.95 |
| `precision_recall.png` | Precision 和 Recall |
| `training_status.png` | 训练时间和学习率 |
| `best.pt` | 最佳模型权重（基于 val loss） |
| `last.pt` | 最后一个 epoch 的权重 |

### 绘制训练曲线

```bash
# 使用脚本绘制
python scripts/plot_curves.py
```

生成 4 张独立的 PNG 图片，所有曲线都带有黄色点状虚线平滑曲线（Savitzky-Golay 滤波）。

### 运行测试

```bash
# 运行所有测试
pytest

# 运行特定测试
pytest tests/att_test.py
pytest tests/engine/test_trainer.py

# 带覆盖率报告
pytest --cov=modules --cov=models --cov=engine
```

## 项目结构

```
ai-playground/
├── modules/                # 基础神经网络模块
│   ├── att.py              # 注意力机制 (CoordAtt, CoordCrossAtt, BiCoordCrossAtt)
│   ├── bifpn.py            # 双向特征金字塔网络 (BiFPN_Cat)
│   ├── block.py            # 基础块 (Bottleneck, C2f, C3, C3k2, SPPF, C2PSA)
│   ├── conv.py             # 卷积模块 (Conv, Concat, autopad)
│   ├── head.py             # 检测头 (Detect, DetectAnchorFree)
│   └── yolo_loss.py        # YOLO 损失函数 (YOLOLoss, YOLOLossAnchorFree)
│
├── models/                 # 完整模型
│   ├── yolov11.py          # YOLOv11 模型（推荐）
│   ├── yolov3.py           # YOLOv3 模型
│   ├── yolo_att.py         # 带注意力机制的 YOLO（旧版）
│   └── yolo.py             # Ultralytics 风格的统一 YOLO 接口
│
├── engine/                 # 训练引擎核心
│   ├── train.py            # CLI 训练脚本
│   ├── trainer.py          # DetectionTrainer 统一训练器
│   ├── training.py         # train_one_epoch 核心训练逻辑
│   ├── validate.py         # 验证与 mAP 计算
│   ├── detector.py         # 检测器专用训练逻辑
│   ├── classifier.py       # 分类器专用训练逻辑
│   └── predictor.py        # 预测接口 (LetterBox, Results, Boxes)
│
├── utils/                  # 工具模块
│   ├── config.py           # 配置管理系统
│   ├── load.py             # 数据加载
│   ├── logger.py           # 训练日志 (TrainingLogger, LiveTableLogger)
│   ├── metrics.py          # 评估指标
│   ├── curves.py           # 训练曲线可视化
│   ├── transforms.py       # 数据增强 (MosaicTransform, MixupTransform)
│   └── ema.py              # 指数移动平均 (ModelEMA)
│
├── configs/                # 配置文件
│   ├── default.yaml        # 全局默认配置
│   ├── data/               # 数据集配置
│   └── models/             # 模型配置
│
├── scripts/                # 脚本
│   ├── plot_curves.py      # 训练曲线绘制脚本
│   └── visualization/      # 可视化脚本
│
├── tests/                  # 单元测试
│   ├── utils/              # 工具模块测试
│   ├── models/             # 模型测试
│   ├── engine/             # 引擎测试
│   └── integration/        # 集成测试
│
├── demos/                  # 快速演示脚本
├── datasets/               # 数据集存储
└── outputs/                # 输出结果
```

## 核心模型

### YOLOv11（推荐）

```python
from models import YOLOv11

# 支持多种模型规模
model = YOLOv11(
    nc=2,           # 类别数
    scale='n',      # n/s/m/l/x (nano/small/medium/large/xlarge)
    img_size=640
)

# 模型规模对应：
# n - nano:   ~2.6M params,  ~6.6 GFLOPs
# s - small:  ~9.5M params, ~21.7 GFLOPs
# m - medium: ~20.1M params, ~68.5 GFLOPs
# l - large:  ~25.4M params, ~87.6 GFLOPs
# x - xlarge: ~57.0M params, ~196.0 GFLOPs
```

**特点：**
- Anchor-free 架构，无需预设 anchor boxes
- DFL (Distribution Focal Loss) 边界框回归
- Task-Aligned Learning 样本分配策略
- 支持 CIoU、GIoU 等多种 IoU 损失
- EMA 指数移动平均
- Mosaic 数据增强

### BiFPN 多尺度特征融合

```python
from models import BiFPN_Cat

# 融合不同通道的特征图
feat1 = torch.randn(1, 128, 40, 40)
feat2 = torch.randn(1, 256, 40, 40)
feat3 = torch.randn(1, 512, 40, 40)

bifpn = BiFPN_Cat(c1=[128, 256, 512])
out = bifpn([feat1, feat2, feat3])  # 输出: (1, 512, 40, 40)
```

## 训练特性

### EMA (Exponential Moving Average)

EMA 通过维护历史权重的指数移动平均，获得更稳定的模型：

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

**优势：**
- 衰减系数 0.9999，动态调整
- 验证时使用 EMA 模型可获得更平滑的 mAP 曲线
- 显著减少训练震荡

### Mosaic 数据增强

Mosaic 将 4 张图片拼接成一张大图，是 YOLOv4/v5/v8/v11 的核心增强：

```python
from utils.transforms import MosaicTransform

mosaic = MosaicTransform(dataset, img_size=640, prob=1.0)
img, boxes = mosaic(img, boxes)

# 训练后期关闭
mosaic.enable = False
```

**优势：**
- 提升小目标检测能力
- 增加样本多样性
- 训练后期建议关闭（默认最后 10 个 epoch）

### LiveTable 动态日志

训练时使用动态刷新的表格显示进度：

```python
from utils import LiveTableLogger

live_logger = LiveTableLogger(
    columns=["total_loss", "box_loss", "cls_loss", "dfl_loss"],
    total_epochs=100,
    console_width=130,
)

live_logger.start_epoch(1, lr=0.001)
live_logger.update_row("train", train_metrics)
live_logger.update_row("val", val_metrics)
live_logger.end_epoch(epoch_time)
```

### 自动递增保存目录

保存目录会自动递增避免覆盖：

```python
from utils import get_save_dir

save_dir = get_save_dir('runs/train/exp')
# runs/train/exp -> runs/train/exp_1 -> runs/train/exp_2 ...
```

## 数据集格式

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

## 核心训练配置

### YOLOv11 关键参数

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `lr` | 0.001 | 更高值（如 0.01）会导致发散 |
| `box_loss_weight` | 7.5 | box loss 权重 |
| `cls_loss_weight` | 0.5 | cls loss 权重 |
| `dfl_loss_weight` | 1.5 | DFL loss 权重 |
| `reg_max` | 16 | DFL 分布 bin 数 |
| `iou_loss` | CIoU | Complete IoU 收敛更好 |

### 学习率调度器

**CosineAnnealingLR** 平滑下降，无中途跳变：
- `T_max=epochs`: 余弦退火周期长度
- `eta_min=1e-6`: 最小学习率

### TaskAlignedAssigner (TAL) 参数

**TAL 是 YOLOv8/v11 的核心正样本分配策略**，直接影响 Box Loss 收敛：

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `topk` | 13 | 每个 GT 选取的候选正样本数 |
| `alpha` | 0.5 | 分类分数权重 |
| `beta` | 6.0 | **IoU 指数权重（关键！）** |
| Soft Labels | Yes | 使用归一化对齐分数作为目标 |

**为什么 beta=6.0 很重要？**
- `beta=2.0`（旧值）：过于宽松，容忍 IoU 低的框作为正样本
- `beta=6.0`（官方）：严格，只有 IoU 高的框才能获得高分
- 低质量匹配会导致 Box Loss 难以收敛，mAP50 停滞在 50-70%

## 常见训练问题排查

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

## 开发状态

| 模块 | 状态 |
|------|------|
| YOLOv3 | ✅ 完成 |
| YOLOv11 | ✅ 完成 |
| BiFPN | ✅ 完成 |
| CoordAtt | ✅ 完成 |
| CoordCrossAtt | ✅ 完成 |
| DFL Loss | ✅ 完成 |
| Task-Aligned Learning | ✅ 完成 |
| mAP50/mAP50-95 | ✅ 完成 |
| Precision/Recall | ✅ 完成 |
| EMA | ✅ 完成 |
| Mosaic 增强 | ✅ 完成 |
| LiveTable 日志 | ✅ 完成 |
| 训练曲线 | ✅ 完成 |
| 自动递增目录 | ✅ 完成 |
| 统一 YOLO 接口 | ✅ 完成 |
| 配置管理系统 | ✅ 完成 |

## 许可证

MIT License
