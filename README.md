# AI Playground

一个基于 PyTorch 的计算机视觉深度学习研究项目，专注于目标检测和注意力机制的实现与可视化。

## 项目简介

本项目实现了多个经典的深度学习模型，特别关注：

- **YOLOv3** - 完整的目标检测系统（支持 WIoU v3 损失函数）
- **YOLOv11** - Anchor-free 目标检测，支持 DFL 和 Task-Aligned Learning
- **Coordinate Attention (CoordAtt)** - 坐标注意力机制
- **Coordinate Cross Attention (CoordCrossAtt)** - 坐标交叉注意力机制
- **BiFPN** - 双向特征金字塔网络，支持多尺度特征融合

## 环境要求

```bash
# 激活 conda 环境
conda activate torch_cpu
```

## 快速开始

### 训练 YOLOv11（推荐）

```python
from models import YOLOv11
from engine import train

# 创建模型（支持动态缩放）
model = YOLOv11(nc=2, scale='s')

# 训练（包含损失分量跟踪、mAP50 验证、曲线绘制）
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

**训练输出：**
```
Epoch 1/100
--------------------------------------------------
Learning Rate: 0.001000
Epoch 1 [0/34] Loss: 10.2023 (box: 2.345, cls: 0.929, dfl: 0.067)
...
Train - Loss: 8.1521 (box: 2.277, cls: 0.717, dfl: 0.158) | mAP: N/A
Val   - Loss: 7.0718 (box: 2.235, cls: 0.685, dfl: 0.152) | mAP50: 15.23%
```

### 运行测试

```bash
# 测试 BiFPN 模块
python tests/fpn_test.py

# 测试 Coordinate Attention 模块
python tests/att_test.py
```

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

#### 4. 测试已训练模型（不进行训练）

```bash
# 测试完整数据集
python scripts/test_attention.py \
    --model outputs/yolo_coordatt_xxx/best_model.pth \
    --config datasets/MY_TEST_DATA/data.yaml \
    --output outputs/test_results

# 测试单张图像
python scripts/test_attention.py \
    --model outputs/yolo_coordatt_xxx/best_model.pth \
    --image datasets/MY_TEST_DATA/images/test/circle_0001.jpg \
    --output outputs/test_results
```

#### 5. 可视化未训练的注意力

```bash
# 查看 CoordAtt 在随机初始化时的注意力分布
python visualization/visualize_coordatt.py
```

## 项目结构

```
ai-playground/
├── models/                 # 神经网络组件
│   ├── att.py              # 基础注意力模块 (CoordAtt, CoordCrossAtt)
│   ├── att_visualize.py    # 带可视化功能的注意力模块
│   ├── yolo_att.py         # YOLO + Attention 检测器
│   ├── yolov3.py           # YOLOv3 目标检测
│   ├── yolov11.py          # YOLOv11 anchor-free 目标检测
│   ├── yolo_loss.py        # YOLO loss function (WIoU v3, Anchor-Free)
│   ├── bifpn.py            # BiFPN 特征融合
│   ├── conv.py             # 自定义卷积层 (Conv+BN+SiLU)
│   ├── head.py             # Detect, DetectAnchorFree 检测头
│   └── block.py            # C3k2, SPPF, C2PSA 构建块
├── engine/                 # 训练引擎核心
│   ├── train.py            # 主训练流程
│   ├── training.py         # 核心 epoch 训练逻辑
│   ├── validate.py         # 验证函数（含 mAP50）
│   ├── detector.py         # YOLO 检测训练
│   ├── classifier.py       # 分类任务训练
│   ├── visualize.py        # 注意力可视化函数
│   └── simple.py           # 简单/遗留训练函数
├── utils/                  # 通用工具模块
│   ├── load.py             # 数据加载
│   ├── logger.py           # CSV 训练日志（含损失分量）
│   ├── curves.py           # 训练曲线绘制
│   ├── metrics.py          # 评估指标计算（含 mAP50）
│   └── model_summary.py    # 模型信息展示
├── tests/                  # 单元测试
├── visualization/          # 训练入口脚本
├── scripts/                # 测试脚本
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

### YOLO + CoordAtt 检测器

```python
from models import YOLOCoordAttDetector
from engine import train

# 创建模型并训练
model = YOLOCoordAttDetector(nc=2)  # 2 个类别

# 训练（包含 CSV 日志、曲线绘制、模型信息输出）
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

### YOLO + CoordCrossAtt 检测器

```python
from models import YOLOCoordCrossAttDetector

# 创建模型 (num_heads 控制多头注意力的头数)
model = YOLOCoordCrossAttDetector(nc=2, num_heads=1)
```

### BiFPN 多尺度特征融合

```python
from modules import BiFPN_Cat

# 融合不同通道的特征图
feat1 = torch.randn(1, 128, 40, 40)
feat2 = torch.randn(1, 256, 40, 40)
feat3 = torch.randn(1, 512, 40, 40)

bifpn = BiFPN_Cat(c1=[128, 256, 512])
out = bifpn([feat1, feat2, feat3])  # 输出: (1, 512, 40, 40)
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

- **CSV 日志** (`training_log.csv`) - 每个 epoch 的 loss、loss 分量、mAP50、学习率、时间
- **训练曲线** (`training_curves.png`) - Loss、指标、学习率、时间曲线
- **模型权重** (`best.pt`, `last.pt`) - 最佳和最后一个 epoch 的权重
- **模型摘要** - 训练前自动输出层数、参数量、FLOPs

**CSV 日志格式（检测任务）：**
```csv
epoch,time,lr,train_loss,val_loss,train_box_loss,train_cls_loss,train_dfl_loss,val_box_loss,val_cls_loss,val_dfl_loss,val_map50
1,38.59,0.006667,10.2023,9.4208,2.345,0.929,0.067,2.235,0.685,0.152,0.0000
2,41.29,0.010000,9.6480,8.9609,2.127,0.793,0.053,2.056,0.623,0.131,15.23
```

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
| mAP50 计算 | ✅ 完成 |
| 训练引擎 | ✅ 完成 |
| CSV 日志 | ✅ 完成 |
| 曲线绘制 | ✅ 完成 |
| 可视化工具 | ✅ 完成 |
| 模型对比 | ✅ 完成 |

## 可视化效果

训练后，模型学会关注形状的关键特征区域：

- **检测注意力** - 显示模型在检测时关注的位置
- **注意力对比** - 对比 CoordAtt 和 CoordCrossAtt 的注意力分布
- **相关性矩阵** - 展示水平和垂直位置之间的关联
- **训练曲线** - Loss、准确率/mAP、学习率变化趋势
- **损失分量** - Box Loss、Cls Loss、DFL Loss 分别跟踪

## 关键特性

### YOLOv11 训练优化

- **Per-Scale Bbox Clamping** - 每个尺度使用正确的 grid size 进行 clamp
  - P3 (stride=8): max=80
  - P4 (stride=16): max=40
  - P5 (stride=32): max=20
- **优化的损失权重** - box: 7.5, cls: 0.5, dfl: 1.5 (与 ultralytics 一致)
- **CIoU Loss** - 使用 Complete IoU，考虑重叠面积、中心距和长宽比
- **改进的 DFL 初始化** - regression bias=-3.5，使初始预测更接近目标范围 [0-15]
- **损失分量跟踪** - 实时显示 box_loss、cls_loss、dfl_loss
- **Task-Aligned Assignment** - 自动调整 topk=64, beta=2.0 获得更多正样本
- **梯度裁剪** - max_norm=10.0 防止梯度爆炸
- **学习率调度器** - CosineAnnealingWarmRestarts 帮助模型跳出局部最优
  - T_0=10, T_mult=2, eta_min=1e-6
  - 每 10/20/40... 个 epoch 重启学习率

### 训练参数建议

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| 学习率 (lr) | 0.001 | 过高（如 0.01）会导致训练发散 |
| Warmup epochs | 3 | 线性 warmup，从 0.00033 到 0.001 |
| Batch size | 8-16 | 根据 GPU 内存调整 |
| Image size | 640 | 标准输入尺寸 |
| Epochs | 100-300 | 根据数据集大小调整 |

### 常见训练问题排查

#### Box Loss 不下降

如果 `box_loss` 长期高于 4.0：

1. **检查学习率**：应为 0.001 或更低
2. **验证数据加载**：运行 `python scripts/diagnose_training.py` 检查标签
3. **分析初始预测**：运行 `python scripts/debug_assigner.py` 查看 IoU
4. **确认 reg_max=16**：更高的值（如 32）会显著增加收敛难度
5. **验证 CIoU 已启用**：检查 `modules/yolo_loss.py:583` 中 `CIoU=True`

#### mAP50 长期为 0%

如果 `mAP50` 超过 5 个 epoch 仍为 0.00%：

1. **验证标签格式**：类别 ID 必须从 0 开始
2. **检查边界框坐标**：应归一化到 [0, 1]（YOLO 格式）
3. **调整置信度阈值**：默认 NMS 阈值为 0.25

#### 损失分量异常

训练早期各损失分量的正常范围：
- `box_loss`: 2.0-6.0（应稳定下降）
- `cls_loss`: 0.5-5.0（初始较高，逐渐下降）
- `dfl_loss`: 1.0-6.0（跟随 box_loss 趋势）

### 模型动态缩放

YOLOv11 支持通过 `scale` 参数动态调整模型大小：
- 深度乘数 (`depth_multiple`)
- 宽度乘数 (`width_multiple`)
- 最大通道数 (`max_channels`)

## 许可证

MIT License
