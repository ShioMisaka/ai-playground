# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AI Playground - a PyTorch-based computer vision research project focused on:
- **YOLOv3** object detection implementation
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

### Training + Visualization

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

**File organization (refactored):**
- `att.py` - Base attention modules: `CoordAtt`, `CoordCrossAtt`
- `att_visualize.py` - Attention with visualization hooks: `CoordAttWithVisualization`, `CoordCrossAttWithVisualization`
- `yolo_att.py` - YOLO detectors: `YOLOCoordAttDetector`, `YOLOCoordCrossAttDetector`
- `yolov3.py` - YOLOv3 with Darknet-53 backbone
- `yolo_loss.py` - YOLO loss function (supports WIoU v3)
- `bifpn.py` - `BiFPN_Cat` for multi-scale feature fusion
- `conv.py` - `Conv` wrapper (Conv2d + BatchNorm + SiLU)
- `head.py` - `Detect` detection head for YOLO

**Export in `models/__init__.py`:**
```python
from .att import CoordAtt, CoordCrossAtt
from .att_visualize import CoordAttWithVisualization, CoordCrossAttWithVisualization
from .yolo_att import YOLOCoordAttDetector, YOLOCoordCrossAttDetector
from .yolov3 import YOLOv3
```

### engine/ - Training and Visualization Engine

**Core modules:**
- `detector.py` - YOLO detection training (`train_detector()`, `train_one_epoch()`, `validate()`)
- `visualize.py` - All visualization functions
- `comparison.py` - Model comparison training (`train_and_compare_models()`, `print_comparison_results()`)

**Visualization functions (all in `engine/visualize.py`):**
```python
# Basic tools
load_image()                    # Load and preprocess image
enhance_contrast()               # Enhance attention map contrast
get_coordatt_attention()        # Get CoordAtt attention map
get_crossatt_attention()        # Get CoordCrossAtt attention map

# Detection task
visualize_detection_attention()       # Detection attention with boxes
visualize_attention_comparison()       # Before/after training comparison

# Single/Multiple images
visualize_single_image_attention()     # Single image attention
visualize_multiple_images_attention()  # Multiple images attention

# Model comparison
visualize_model_comparison()           # Compare CoordAtt vs CoordCrossAtt
visualize_cross_attention_matrix()     # Cross-Attention correlation matrix
visualize_training_progress()           # Training progress comparison
```

### Directory Structure

```
models/         # Neural network components
engine/         # Training and visualization engine
utils/          # Data loading utilities
tests/          # Unit tests for individual modules
visualization/  # Training entry scripts
scripts/        # Testing scripts (no training)
datasets/       # YOLO-format datasets
outputs/        # Generated outputs (models, images, logs)
```

## Code Patterns

### Custom Conv Wrapper

Use `models.conv.Conv` instead of raw `nn.Conv2d`:
```python
from models import Conv
self.cv1 = Conv(c1, c2, k=1, s=1, p=0)  # Conv+BN+SiLU
self.cv2 = Conv(c1, c2, k=3, s=1, p=1)  # 3x3 with same padding
```

### BiFPN Usage

Accepts list of tensors with different channels, outputs fused tensor:
```python
from models import BiFPN_Cat
feat1, feat2, feat3 = torch.randn(1, 128, 40, 40), torch.randn(1, 256, 40, 40), torch.randn(1, 512, 40, 40)
bifpn = BiFPN_Cat(c1=[128, 256, 512])
out = bifpn([feat1, feat2, feat3])  # Shape: (1, 512, 40, 40)
```

### Coordinate Attention (Basic)

```python
from models import CoordAtt
att = CoordAtt(inp=256, oup=256, reduction=32)
out = att(x)  # Same shape as input, attention-weighted
```

### YOLO + CoordAtt Detector (For Object Detection)

```python
from models import YOLOCoordAttDetector
from engine import train_detector, visualize_detection_attention
from utils import create_dataloaders

# Create model with CoordAtt-enhanced backbone
model = YOLOCoordAttDetector(nc=2)  # nc = number of classes

# Train the detector
train_loader, val_loader, _ = create_dataloaders(config_path, batch_size=4, img_size=640)
train_detector(model, train_loader, val_loader, epochs=100, lr=0.001, device='cuda')

# Visualize attention after training
visualize_detection_attention(model, val_loader, device, save_path='attention.png')
```

### Model Comparison

```python
from models import YOLOCoordAttDetector, YOLOCoordCrossAttDetector
from engine import train_and_compare_models, print_comparison_results

# Define models to compare
model_dict = {
    'CoordAtt': (YOLOCoordAttDetector, {'nc': 2}),
    'CoordCrossAtt': (YOLOCoordCrossAttDetector, {'nc': 2, 'num_heads': 1}),
}

# Train and compare
results = train_and_compare_models(
    model_dict, train_loader, val_loader,
    epochs=50, lr=0.001, device='cuda',
    save_dir='outputs/comparison'
)

# Print results
print_comparison_results(results)
```

## Important Implementation Notes

### Detection Training Mode Handling

When using YOLO's `Detect` head, it returns different formats based on training mode:
- **Training mode** (`model.training=True`): Returns list of predictions for loss computation
- **Inference mode** (`model.eval()`): Returns tuple `(concatenated, list)` for NMS/post-processing

The `engine/detector.py` `validate()` function temporarily sets `model.detect.train()` during validation.

### Stride Alignment

The `YOLOCoordAttDetector` backbone produces feature maps at strides 4/8/16/32, but the Detect head expects inputs at strides 8/16/32. Therefore:
- Use `p4` (stride 8), `p5` (stride 16), `p6_backbone` (stride 32) for detection
- The FPN neck fuses features but maintains correct stride alignment

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
refactor: 重构注意力模块代码结构

- 将 att.py 拆分为基础模块和可视化模块
- 新建 att_visualize.py 存放带可视化功能的注意力模块
- 新建 yolo_att.py 存放 YOLO 检测器
- 更新 models/__init__.py 导入路径
```
