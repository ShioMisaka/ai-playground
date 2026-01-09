# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AI Playground - a PyTorch-based computer vision research project focused on:
- **YOLOv3** object detection implementation
- **Coordinate Attention for Object Detection** - Integrating CoordAtt into YOLO for better bounding box regression
- **BiFPN** (Bidirectional Feature Pyramid Network) with learnable fusion weights
- Custom YOLO-format dataset training

**Key Design**: The Coordinate Attention mechanism helps the detection network better regress to target objects by capturing long-range spatial dependencies along horizontal and vertical directions.

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
- `att.py` - `CoordAtt` coordinate attention module, `CoordAttWithVisualization`, and `YOLOCoordAttDetector`
  - `YOLOCoordAttDetector`: YOLO detector with 4 CoordAtt layers in backbone at strides 4/8/16/32
  - Uses FPN neck for multi-scale feature fusion
  - Detect head outputs at 3 scales (stride 8/16/32) for object detection
- `bifpn.py` - `BiFPN_Cat` class for multi-scale feature fusion with learnable weights
- `conv.py` - `Conv` wrapper (Conv2d + BatchNorm + SiLU activation)
- `block.py` - `Bottleneck` building block
- `head.py` - `Detect` detection head for YOLO (used by YOLOCoordAttDetector)
- `yolo_loss.py` - YOLO loss function (box, objectness, class losses)

**When adding new models:** Export in `models/__init__.py` following the pattern:
```python
from .your_module import YourClass
```

### engine/ - Training Logic

- `train.py` - Generic `train()` function for classification
- `validate.py` - `validate()` and `evaluate()` functions for classification
- `detector.py` - **YOLO detection training** (`train_one_epoch()`, `validate()`, `train_detector()`)
  - Handles training/validation mode switching for Detect head
  - Supports warmup + cosine annealing learning rate schedule
  - Early stopping with patience
  - Saves best model and training history to JSON
- `visualize.py` - **Attention visualization for detection**
  - `visualize_detection_attention()` - Visualize attention maps with detection boxes
  - `visualize_attention_comparison()` - Compare attention across multiple CoordAtt layers
  - `enhance_contrast()` - Image enhancement utilities

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

**Basic CoordAtt module:**
```python
from models import CoordAtt
att = CoordAtt(inp=256, oup=256, reduction=32)
out = att(x)  # Same shape as input, attention-weighted
```

**YOLO + CoordAtt Detector (for object detection):**
```python
from models import YOLOCoordAttDetector
from engine import train_detector, visualize_detection_attention
from utils import create_dataloaders

# Create model with CoordAtt-enhanced backbone
model = YOLOCoordAttDetector(nc=1)  # nc = number of classes

# Train the detector
train_loader, val_loader, _ = create_dataloaders(config_path, batch_size=4, img_size=640)
train_detector(model, train_loader, val_loader, epochs=100, lr=0.001, device='cuda')

# Visualize attention after training
visualize_detection_attention(model, val_loader, device, save_path='attention.png')
```

## Important Implementation Notes

### Detection Training Mode Handling

When using YOLO's `Detect` head, it returns different formats based on training mode:
- **Training mode** (`model.training=True`): Returns list of predictions for loss computation
- **Inference mode** (`model.eval()`): Returns tuple `(concatenated, list)` for NMS/post-processing

**Important**: The `engine/detector.py` `validate()` function temporarily sets `model.detect.train()` during validation to ensure loss computation works correctly, while keeping other layers (BatchNorm, Dropout) in eval mode.

### Stride Alignment

The `YOLOCoordAttDetector` backbone produces feature maps at strides 4/8/16/32, but the Detect head expects inputs at strides 8/16/32. Therefore:
- Use `p4` (stride 8), `p5` (stride 16), `p6_backbone` (stride 32) for detection
- The FPN neck fuses features but maintains correct stride alignment

### JSON Serialization

When saving training history to JSON, always convert PyTorch types to Python native types:
```python
history['train_loss'].append(float(train_loss))  # Not train_loss.item()
history['lr'].append(float(optimizer.param_groups[0]['lr']))
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

## Git Commit Conventions

Follow **Conventional Commits** format for all commits:

### Format
```
<type>: <subject>

<body>
```

### Types
- `feat:` - New feature or functionality
- `fix:` - Bug fix
- `docs:` - Documentation changes
- `refactor:` - Code refactoring (no functional change)
- `test:` - Adding or updating tests
- `chore:` - Maintenance tasks, dependencies, configuration

### Rules
1. **Subject line**: Max 50 characters, lowercase first letter
2. **Body**: Required for multi-file changes, use bullet points
3. **No AI attribution**: Do not include "generated by AI", "co-authored-by AI", etc.

### Examples

**Simple change**:
```
feat: add coordinate attention visualization
```

**Multi-file change** (body required):
```
refactor: reorganize coordinate attention implementation

- Split CoordAtt into separate module
- Add visualization hooks to attention layers
- Update detector to use new CoordAtt module
- Add unit tests for attention computation
```

**Bug fix**:
```
fix: correct stride alignment in detect head
```
