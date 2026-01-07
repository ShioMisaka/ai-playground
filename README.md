# AI Playground

一个基于 PyTorch 的计算机视觉深度学习研究项目，专注于目标检测和注意力机制的实现与可视化。

## 项目简介

本项目实现了多个经典的深度学习模型，特别关注：

- **YOLOv3** - 完整的目标检测系统
- **BiFPN** - 双向特征金字塔网络，支持多尺度特征融合
- **Coordinate Attention** - 坐标注意力机制，带完整可视化

## 特性

- 模块化的神经网络组件设计
- 完整的训练和验证流程
- YOLO 格式数据集支持
- 注意力机制可视化工具
- 支持自定义目标检测任务

## 项目结构

```
ai-playground/
├── models/              # 神经网络组件
│   ├── yolov3.py       # YOLOv3 目标检测
│   ├── bifpn.py        # BiFPN 特征融合
│   ├── att.py          # Coordinate Attention
│   ├── conv.py         # 自定义卷积层 (Conv+BN+SiLU)
│   └── block.py        # 基础模块
├── engine/             # 训练和验证逻辑
├── utils/              # 数据加载工具
├── tests/              # 单元测试
├── demos/              # 演示脚本
├── visualization/      # 可视化工具
├── datasets/           # 数据集存储
└── outputs/            # 输出结果
```

## 快速开始

### 环境要求

```bash
# 激活 conda 环境
conda activate torch_cpu
```

### 运行测试

```bash
# 测试 BiFPN 模块
python tests/fpn_test.py

# 测试 Coordinate Attention 模块
python tests/att_test.py
```

### 运行演示

```bash
# MNIST 分类演示
python demos/mnist_demo.py

# YOLOv3 目标检测训练
python demos/yolov3_demo.py
```

### 注意力可视化

```bash
# 训练模型并可视化注意力效果
python visualization/visualize_trained_coordatt.py
```

## 核心模型

### YOLOv3

完整的目标检测实现，包含：
- Darknet-53 骨干网络
- 多尺度检测 (P3/P4/P5)
- 自定义锚框生成
- YOLO 损失函数

### BiFPN

支持多通道特征融合的特征金字塔网络：

```python
from models import BiFPN_Cat

# 融合不同通道的特征图
feat1 = torch.randn(1, 128, 40, 40)
feat2 = torch.randn(1, 256, 40, 40)
feat3 = torch.randn(1, 512, 40, 40)

bifpn = BiFPN_Cat(c1=[128, 256, 512])
out = bifpn([feat1, feat2, feat3])  # 输出: (1, 512, 40, 40)
```

### Coordinate Attention

坐标注意力机制，带完整可视化工具：

```python
from models import CoordAtt

# 创建注意力模块
att = CoordAtt(inp=256, oup=256, reduction=32)
out = att(x)  # 输出尺寸与输入相同，应用注意力权重
```

## 可视化效果

使用 4 层堆叠的 CoordAtt，展示训练前后注意力的变化：

![Attention Visualization](outputs/attention_after_training_comparison.png)

训练后，模型学会关注形状的关键特征区域。

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

## 开发状态

| 模块 | 状态 |
|------|------|
| YOLOv3 | ✅ 完成 |
| BiFPN | ✅ 完成 |
| Coordinate Attention | ✅ 完成 |
| 训练引擎 | ✅ 完成 |
| 数据加载 | ✅ 完成 |
| 可视化工具 | ✅ 完成 |

## 许可证

MIT License
