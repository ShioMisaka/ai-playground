# AI Playground

一个基于 PyTorch 的计算机视觉深度学习研究项目，专注于目标检测和注意力机制的实现与可视化。

## 项目简介

本项目实现了多个经典的深度学习模型，特别关注：

- **YOLOv3** - 完整的目标检测系统（支持 WIoU v3 损失函数）
- **Coordinate Attention (CoordAtt)** - 坐标注意力机制
- **Coordinate Cross Attention (CoordCrossAtt)** - 坐标交叉注意力机制
- **BiFPN** - 双向特征金字塔网络，支持多尺度特征融合

## 环境要求

```bash
# 激活 conda 环境
conda activate torch_cpu
```

## 快速开始

### 运行测试

```bash
# 测试 BiFPN 模块
python tests/fpn_test.py

# 测试 Coordinate Attention 模块
python tests/att_test.py
```

### 训练和可视化

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
│   ├── yolo_loss.py        # YOLO loss function
│   ├── bifpn.py            # BiFPN 特征融合
│   ├── conv.py             # 自定义卷积层 (Conv+BN+SiLU)
│   └── ...
├── engine/                 # 训练引擎核心
│   ├── train.py            # 主训练流程
│   ├── training.py         # 核心 epoch 训练逻辑
│   ├── validate.py         # 验证函数
│   ├── detector.py         # YOLO 检测训练
│   ├── classifier.py       # 分类任务训练
│   ├── visualize.py        # 注意力可视化函数
│   ├── comparison.py       # 模型对比训练
│   └── simple.py           # 简单/遗留训练函数
├── utils/                  # 通用工具模块
│   ├── load.py             # 数据加载
│   ├── logger.py           # CSV 训练日志
│   ├── curves.py           # 训练曲线绘制
│   ├── metrics.py          # 评估指标计算
│   └── model_summary.py    # 模型信息展示
├── tests/                  # 单元测试
├── visualization/          # 训练入口脚本
├── scripts/                # 测试脚本
├── demos/                  # 快速演示脚本
├── datasets/               # 数据集存储
└── outputs/                # 输出结果
```

## 核心模型

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

- **CSV 日志** (`training_log.csv`) - 每个 epoch 的 loss、accuracy/mAP、学习率、时间
- **训练曲线** (`training_curves.png`) - Loss、指标、学习率、时间曲线
- **模型权重** (`best.pt`, `last.pt`) - 最佳和最后一个 epoch 的权重
- **模型摘要** - 训练前自动输出层数、参数量、FLOPs

## 开发状态

| 模块 | 状态 |
|------|------|
| YOLOv3 | ✅ 完成 |
| BiFPN | ✅ 完成 |
| CoordAtt | ✅ 完成 |
| CoordCrossAtt | ✅ 完成 |
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

## 许可证

MIT License
