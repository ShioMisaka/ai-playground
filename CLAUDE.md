# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AI Playground - a PyTorch-based computer vision research project focused on:
- **YOLOv3** object detection implementation
- **BiFPN** (Bidirectional Feature Pyramid Network) with learnable fusion weights
- **Coordinate Attention** mechanisms
- Custom YOLO-format dataset training

## Environment

Activate conda environment before running:
```bash
eval "$(conda shell.bash hook)"
conda activate torch_cpu
```

## Running Tests

**Unit tests** (in `tests/`):
```bash
python tests/fpn_test.py    # Test BiFPN module
python tests/att_test.py    # Test Coordinate Attention module
```

**Demos** (in `demos/`):
```bash
python demos/mnist_demo.py     # Train CNN on MNIST
python demos/yolov3_demo.py    # Train YOLOv3 on custom dataset
```

**Visualization** (in `visualization/`):
```bash
python visualization/visualize_coordatt.py          # Visualize attention (untrained)
python visualization/visualize_trained_coordatt.py  # Train model, then visualize attention
```

**Output files** are saved in `outputs/` directory.

## Architecture

### models/ - Neural Network Components

**Core modules:**
- `yolov3.py` - Complete YOLOv3 with Darknet-53 backbone
- `bifpn.py` - `BiFPN_Cat` class for multi-scale feature fusion with learnable weights
- `att.py` - `CoordAtt` coordinate attention module
- `conv.py` - `Conv` wrapper (Conv2d + BatchNorm + SiLU activation)
- `block.py` - `Bottleneck` building block
- `head.py` - Detection head for YOLO

**When adding new models:** Export in `models/__init__.py` following the pattern:
```python
from .your_module import YourClass
```

### engine/ - Training Logic

- `train.py` - `train_one_epoch()` function
- `validate.py` - `validate()` and `evaluate()` functions

### utils/ - Data Loading

- `load.py` - `get_yolo_dataset()` for YOLO-format datasets, supports MNIST

### Dataset Format

Custom YOLO dataset in `datasets/MY_TEST_DATA/`:
- `images/` - train/val/test splits
- `labels/` - YOLO format .txt annotations (class x_center y_center width height, normalized)
- `data.yaml` - dataset configuration

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

### Coordinate Attention

```python
from models import CoordAtt
att = CoordAtt(inp=256, oup=256, reduction=32)
out = att(x)  # Same shape as input, attention-weighted
```

## Testing New Components

When adding a new model (like `att.py`), create a corresponding test file in `tests/` following the existing pattern:
1. Create random input tensor(s)
2. Instantiate module
3. Forward pass and validate output shape
4. Backward pass with optimizer to verify learnability
5. Test multiple input sizes

Then export the model class in `models/__init__.py`.

**Directory structure:**
```
tests/          # Unit tests for individual modules
demos/          # Training/usage demonstration scripts
visualization/  # Visualization and analysis scripts
outputs/        # Generated outputs (images, models, logs)
```
