# AI Playground

一个基于 PyTorch 的计算机视觉深度学习研究项目，专注于目标检测和注意力机制的实现与可视化。

## 项目简介

本项目实现了多个经典的深度学习模型，特别关注：

- **YOLOv11** - Anchor-free 目标检测，支持 DFL 和 Task-Aligned Learning（推荐）
- **YOLOv3** - 完整的目标检测系统（支持 WIoU v3 损失函数）
- **Coordinate Attention (CoordAtt)** - 坐标注意力机制
- **Coordinate Cross Attention (CoordCrossAtt)** - 坐标交叉注意力机制
- **BiFPN** - 双向特征金字塔网络，支持多尺度特征融合

## 环境要求

```bash
# 激活 conda 环境
eval "$(conda shell.bash hook)"
conda activate torch_cpu
```

## 快速开始

### 训练 YOLOv11（推荐）

```python
from models import YOLOv11
from engine import train

# 创建模型（支持动态缩放）
model = YOLOv11(nc=2, scale='s')

# 训练（包含 EMA、Mosaic 增强、损失分量跟踪、mAP 验证、曲线绘制）
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

**训练输出（LiveTable 动态刷新）：**
```
Epoch 1/100  lr=0.001000
           total_loss  box_loss  cls_loss  dfl_loss
Train -      10.2023    2.3450    0.9290    0.0670  100% ━━━━━━━━━━━━━━━━━━━━ 34/34 1.1s/it 37.5s<0.0s
Val   -       9.4208    2.2350    0.6850    0.1520  mAP50: 0.152  mAP50-95: 0.089
Time: 38.59s
```

### 绘制训练曲线

```bash
# 使用脚本绘制（推荐）
python scripts/plot_curves.py
```

生成 4 张独立的 PNG 图片：
- `loss_analysis.png` - Loss 曲线（2x4 布局）
- `map_performance.png` - mAP@0.5 和 mAP@0.5:0.95
- `precision_recall.png` - Precision 和 Recall
- `training_status.png` - 训练时间和学习率

所有曲线都带有黄色点状虚线平滑曲线（Savitzky-Golay 滤波）。

### 运行测试

```bash
# 测试 BiFPN 模块
python tests/fpn_test.py

# 测试 Coordinate Attention 模块
python tests/att_test.py
```

### 诊断工具

```bash
# 诊断训练问题（数据加载、标签、模型预测）
python scripts/diagnose_training.py

# 调试 TaskAlignedAssigner 和损失计算
python scripts/debug_assigner.py
```

这些工具会生成：
- 可视化图片（保存到 `outputs/diagnosis/`）
- 详细的训练信息输出
- IoU 分析和损失分量统计

### 训练和可视化（传统方法）

#### 1. 训练 YOLOv3

```bash
python demos/yolov3_demo.py
```

#### 2. 训练 YOLO + CoordAtt 检测器

```bash
python visualization/visualize_trained_coordatt.py
```

训练完成后会在 `outputs/yolo_coordatt_<timestamp>/` 生成：
- `best_model.pth` - 最佳模型权重
- `training_history.json` - 训练历史记录
- `detection_attention.png` - 检测注意力可视化
- `attention_comparison.png` - 训练前后注意力对比

#### 3. 对比 CoordAtt vs CoordCrossAtt

```bash
python visualization/compare_attention_mechanisms.py
```

对比训练会在 `outputs/attention_comparison/run_<timestamp>/` 生成：
- `coordatt/` 和 `crossatt/` - 两个模型的训练结果
- `attention_comparison.png` - 注意力对比图
- `cross_attention_matrix.png` - Cross-Attention 相关性矩阵
- `training_progress.png` - 训练进度对比

## 项目结构

```
ai-playground/
├── modules/                # 基础神经网络模块
│   ├── att.py              # CoordAtt, CoordCrossAtt
│   ├── bifpn.py            # BiFPN_Cat
│   ├── conv.py             # Conv (Conv2d + BN + SiLU)
│   ├── block.py            # C3k2, SPPF, C2PSA
│   ├── head.py             # Detect, DetectAnchorFree
│   └── yolo_loss.py        # YOLOLoss, YOLOLossAnchorFree
│
├── models/                 # 完整模型
│   ├── yolov11.py          # YOLOv11 (推荐)
│   ├── yolov3.py           # YOLOv3
│   └── yolo_att.py         # YOLOCoordAttDetector (legacy)
│
├── engine/                 # 训练引擎核心
│   ├── train.py            # train() 主训练流程
│   ├── training.py         # train_one_epoch() 核心训练逻辑
│   ├── validate.py         # validate() 验证与 mAP
│   ├── detector.py         # 检测器专用训练逻辑
│   └── ema.py              # ModelEMA 指数移动平均
│
├── utils/                  # 工具模块
│   ├── load.py             # create_dataloaders()
│   ├── logger.py           # TrainingLogger, LiveTableLogger
│   ├── metrics.py          # compute_map50()
│   ├── curves.py           # plot_training_curves()
│   ├── path_helper.py      # get_save_dir() 自动递增目录
│   ├── transforms.py       # MosaicTransform, MixupTransform
│   └── model_summary.py    # print_model_summary()
│
├── scripts/                # 脚本
│   └── plot_curves.py      # 训练曲线绘制脚本
│
├── tests/                  # 单元测试
├── visualization/          # 可视化脚本
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
from engine.ema import ModelEMA

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

### 自动递增保存目录

保存目录会自动递增避免覆盖：

```python
from utils import get_save_dir

save_dir = get_save_dir('runs/train/exp')
# runs/train/exp -> runs/train/exp_1 -> runs/train/exp_2 ...
```

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
test: images/test

nc: 2  # 类别数
names:
  0: circle
  1: square
```

## 训练输出

训练过程会自动记录和保存：

| 文件 | 说明 |
|------|------|
| `training_log.csv` | 每个 epoch 的详细指标（YOLO 风格表头） |
| `loss_analysis.png` | Loss 曲线（2x4 布局） |
| `map_performance.png` | mAP@0.5 和 mAP@0.5:0.95 |
| `precision_recall.png` | Precision 和 Recall |
| `training_status.png` | 训练时间和学习率 |
| `best.pt` | 最佳模型权重（基于 val loss） |
| `last.pt` | 最后一个 epoch 的权重 |

**CSV 日志格式（YOLO 风格斜杠层级）：**
```csv
epoch,time,lr,train/loss,val/loss,train/box_loss,train/cls_loss,train/dfl_loss,val/box_loss,val/cls_loss,val/dfl_loss,metrics/precision(B),metrics/recall(B),metrics/mAP50(B),metrics/mAP50-95(B)
1,38.59,0.001000,10.2023,9.4208,2.345,0.929,0.067,2.235,0.685,0.152,0.1234,0.0987,0.1523,0.0891
2,41.29,0.000987,9.6480,8.9609,2.127,0.793,0.053,2.056,0.623,0.131,0.1456,0.1123,0.1823,0.1056
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
2. **验证数据加载**：运行 `python scripts/diagnose_training.py` 检查标签
3. **分析初始预测**：运行 `python scripts/debug_assigner.py` 查看 IoU
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

## 可视化效果

训练后，模型学会关注形状的关键特征区域：

- **检测注意力** - 显示模型在检测时关注的位置
- **注意力对比** - 对比 CoordAtt 和 CoordCrossAtt 的注意力分布
- **相关性矩阵** - 展示水平和垂直位置之间的关联
- **训练曲线** - Loss、mAP、Precision、Recall、学习率变化趋势
- **损失分量** - Box Loss、Cls Loss、DFL Loss 分别跟踪

## 许可证

MIT License
