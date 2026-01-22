# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AI Playground - a PyTorch-based computer vision research project focused on:
- **YOLOv3** object detection implementation
- **YOLOv11** anchor-free object detection with DFL and Task-Aligned Learning
- **Coordinate Attention (CoordAtt)** - Captures long-range spatial dependencies along horizontal and vertical directions
- **Coordinate Cross Attention (CoordCrossAtt)** - Cross-attention mechanism for H-W feature interaction
- **BiFPN** with learnable fusion weights
- Custom YOLO-format dataset training

## Environment

Activate conda environment before running:
```bash
eval "$(conda shell.bash hook)"
conda activate torch_cpu
```

## Running Scripts

### Training YOLOv11 (Recommended)

```bash
# Train YOLOv11 with anchor-free detection
python -c "
from models import YOLOv11
from engine import train

model = YOLOv11(nc=2, scale='n')
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
"
```

### Training + Visualization (Legacy)

```bash
# Train YOLO + CoordAtt detector
python visualization/visualize_trained_coordatt.py

# Compare CoordAtt vs CoordCrossAtt
python visualization/compare_attention_mechanisms.py

# Train YOLOv3
python demos/yolov3_demo.py
```

### Testing (No Training)

```bash
# Test trained model on dataset
python scripts/test_attention.py --model outputs/best_model.pth --config datasets/MY_TEST_DATA/data.yaml

# Test on single image
python scripts/test_attention.py --model outputs/best_model.pth --image test.jpg
```

### Unit Tests

```bash
python tests/fpn_test.py    # Test BiFPN module
python tests/att_test.py    # Test Attention modules
```

### Visualization Only (Untrained)

```bash
python visualization/visualize_coordatt.py
```

## Architecture

### models/ - Neural Network Components

**File organization:**
- `att.py` - Base attention modules: `CoordAtt`, `CoordCrossAtt`
- `att_visualize.py` - Attention with visualization hooks
- `yolo_att.py` - YOLO detectors: `YOLOCoordAttDetector`, `YOLOCoordCrossAttDetector`
- `yolov3.py` - YOLOv3 with Darknet-53 backbone
- `yolov11.py` - YOLOv11 anchor-free architecture with DFL
- `yolo_loss.py` - YOLO loss functions (supports WIoU v3, Anchor-Free with DFL)
- `bifpn.py` - `BiFPN_Cat` for multi-scale feature fusion
- `conv.py` - `Conv` wrapper (Conv2d + BatchNorm + SiLU)
- `head.py` - `Detect`, `DetectAnchorFree` detection heads
- `block.py` - C3k2, SPPF, C2PSA building blocks

### engine/ - Training Engine Core

**Core training modules:**
- `train.py` - Main training flow (`train()`)
- `training.py` - Core epoch training logic (`train_one_epoch()`, `print_metrics()`)
- `validate.py` - Validation logic with mAP50 (`validate()`, `evaluate()`, `test()`)
- `detector.py` - Detection-specific training
- `classifier.py` - Classification-specific training
- `visualize.py` - All visualization functions

### utils/ - Utility Modules

**Utility modules:**
- `load.py` - Data loading utilities (`create_dataloaders()`)
- `logger.py` - CSV training logger with loss components (`TrainingLogger`)
- `curves.py` - Training curve plotting (`plot_training_curves()`)
- `metrics.py` - Evaluation metrics including mAP50 (`compute_map50()`, `compute_*_metrics()`)
- `model_summary.py` - Model/training info display (`print_training_info()`, `print_model_summary()`)

### Directory Structure

```
models/         # Neural network components
engine/         # Training engine core
utils/          # Utility modules (logging, plotting, metrics)
tests/          # Unit tests for individual modules
visualization/  # Training entry scripts
scripts/        # Testing scripts (no training)
demos/          # Quick demo scripts
datasets/       # YOLO-format datasets
outputs/        # Generated outputs (models, images, logs)
```

## Code Patterns

### YOLOv11 Model (Recommended)

```python
from models import YOLOv11
from engine import train

# Create model with dynamic scaling
model = YOLOv11(
    nc=2,           # number of classes
    scale='n',      # n/s/m/l/x for nano/small/medium/large/xlarge
    reg_max=32,     # DFL distribution bins
    img_size=640
)

# Train (includes CSV logging with loss components, curve plotting, mAP50)
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

### Custom Conv Wrapper

Use `modules.conv.Conv` instead of raw `nn.Conv2d`:
```python
from modules import Conv
self.cv1 = Conv(c1, c2, k=1, s=1, p=0)  # Conv+BN+SiLU
self.cv2 = Conv(c1, c2, k=3, s=1, p=1)  # 3x3 with same padding
```

### BiFPN Usage

Accepts list of tensors with different channels, outputs fused tensor:
```python
from modules import BiFPN_Cat
feat1, feat2, feat3 = torch.randn(1, 128, 40, 40), torch.randn(1, 256, 40, 40), torch.randn(1, 512, 40, 40)
bifpn = BiFPN_Cat(c1=[128, 256, 512])
out = bifpn([feat1, feat2, feat3])  # Shape: (1, 512, 40, 40)
```

### Training a Model

```python
from models import YOLOv11
from engine import train
from utils import create_dataloaders

# Create model
model = YOLOv11(nc=2, scale='s')

# Train (includes CSV logging, curve plotting, model summary)
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

### Using Training Components Directly

```python
from engine import train_one_epoch, print_metrics, validate
from utils import TrainingLogger, plot_training_curves, print_model_summary

# Print model info before training
print_model_summary(model, img_size=640, nc=2)

# Custom training loop with CSV logging
with TrainingLogger('runs/training.csv', is_detection=True) as logger:
    for epoch in range(epochs):
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, epoch+1, nc=2)
        val_metrics = validate(model, val_loader, device, nc=2, img_size=640)

        print_metrics(train_metrics, val_metrics, is_detection=True)
        logger.write_epoch(epoch+1, epoch_time, lr, train_metrics, val_metrics)

# Plot training curves after training
plot_training_curves('runs/training.csv', save_dir='runs')
```

## Important Implementation Notes

### YOLOv11 Loss Format

YOLOv11 returns loss in ultralytics-style format:
- Returns `(loss * batch_size, loss_items, predictions)` where:
  - `loss * batch_size`: scalar for backward pass
  - `loss_items`: tensor `[box_loss, cls_loss, dfl_loss]` (not multiplied by batch_size)
  - `predictions`: dict with `{'cls': [...], 'reg': [...]}`

### Detection Training Mode Handling

When using YOLO's detection head, different modes:
- **YOLOv3 Detect** (anchor-based):
  - Training mode: Returns list for loss computation
  - Inference mode: Returns tuple for NMS
- **DetectAnchorFree** (anchor-free):
  - Training mode: Returns `{'cls': [...], 'reg': [...]}`
  - Inference mode: Returns `(concatenated, dict)`

The `engine/validate.py` temporarily switches `model.detect.eval()` for inference predictions during validation.

### JSON Serialization

When saving training history to JSON, always convert PyTorch types to Python native types:
```python
history['train_loss'].append(float(train_loss))
history['lr'].append(float(optimizer.param_groups[0]['lr']))
```

## Testing New Components

When adding a new model, create a corresponding test file in `tests/`:
1. Create random input tensor(s)
2. Instantiate module
3. Forward pass and validate output shape
4. Backward pass with optimizer to verify learnability
5. Test multiple input sizes

Export the model class in `models/__init__.py`.

## Module Organization Principles

- **engine/** - Training engine core logic only (training loops, validation)
- **utils/** - General utilities (logging, plotting, metrics, data loading)
- **models/** - Neural network components

When adding new code:
- If it's training/validation logic → `engine/`
- If it's a utility (logging, metrics, plotting) → `utils/`
- If it's a network component → `models/`

## Git Commit Conventions

所有提交使用中文，遵循 **Conventional Commits** 格式：

### 格式
```
<类型>: <描述>

<正文>
```

### 类型
- `feat:` - 新功能
- `fix:` - 修复 bug
- `docs:` - 文档变更
- `refactor:` - 代码重构（无功能变化）
- `test:` - 添加或更新测试
- `chore:` - 维护任务、依赖、配置

### 规则
1. **标题行**：不超过 50 字符
2. **正文**：多文件变更时必须用列表列出
3. **无 AI 标识**：不包含 "生成于 AI"、"AI 协作" 等信息

### 示例

**简单变更**：
```
feat: 添加坐标注意力可视化
```

**多文件变更**（需要正文）：
```
refactor: 重构训练模块代码结构

- 将 train.py 拆分为主训练流程和核心训练逻辑
- 新建 training.py 存放核心 epoch 训练函数
- 将日志、绘图、指标模块移至 utils/
- 更新所有相关文件的导入路径
```
