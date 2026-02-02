# YOLO Training Guide

This guide covers how to train YOLO models using the unified training API.

## Quick Start

### Python API

```python
from models import YOLO

# Create model from config
model = YOLO('configs/models/yolov11n.yaml')

# Train
results = model.train(
    data='data/coco8/data.yaml',
    epochs=100,
    batch=16,
    imgsz=640,
)

# Immediately predict with trained model (best weights auto-loaded)
predictions = model.predict('image.jpg')

print(f"Best mAP50: {results['best_map']:.4f}")
print(f"Save directory: {results['save_dir']}")
```

### Command Line

```bash
python -m engine.train \
    --name my_experiment \
    --data data/coco8/data.yaml \
    --epochs 100 \
    --batch-size 16 \
    --img-size 640 \
    --device cpu
```

## Configuration System

### Priority Order

Configuration is merged from multiple sources (in order of priority, lowest to highest):

1. **Default config** (`configs/default.yaml`) - Base configuration
2. **Model config** (`configs/models/*.yaml`) - Model-specific settings
3. **User config file** (`--config my_exp.yaml`) - Experiment configuration
4. **Function parameters / CLI arguments** - Runtime overrides

Higher priority sources override lower priority sources.

### Configuration Files

**Model Config** (`configs/models/yolov11n.yaml`):
```yaml
model:
  nc: 2        # Number of classes
  scale: n     # Model scale (n/s/m/l/x)
  use_ema: true
```

**Data Config** (`data/coco8/data.yaml`):
```yaml
path: /path/to/data
train: images/train
val: images/val
nc: 2
names: ['cat', 'dog']
```

**Experiment Config** (optional):
```yaml
train:
  epochs: 100
  batch_size: 16
  img_size: 640
  mosaic: true

optimizer:
  type: Adam
  lr: 0.001

scheduler:
  type: CosineAnnealingLR
  min_lr: 1e-6
```

## Training Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | str | None | Dataset config YAML path (required) |
| `epochs` | int | 100 | Number of training epochs |
| `batch` | int | 16 | Batch size |
| `imgsz` | int | 640 | Training image size |
| `lr` | float | 0.001 | Learning rate |
| `device` | str | auto | Device (cpu/cuda/cuda:0) |
| `save_dir` | str | runs/train | Base save directory |
| `config` | dict | None | Full config dict (overrides other params) |
| `**kwargs` | - | - | Nested parameter overrides |

### Nested Overrides

You can override any nested config parameter using dot notation:

```python
model.train(
    data='data.yaml',
    epochs=100,
    optimizer__lr=0.0001,      # Nested: optimizer.lr
    train__mosaic_prob=0.9,    # Nested: train.mosaic_prob
    model__ema_decay=0.9999,   # Nested: model.ema_decay
)
```

## Output Directory Structure

Training creates the following structure:

```
runs/train/exp/              # Auto-increments to exp_1, exp_2, ...
├── weights/
│   ├── best.pt              # Best model (by mAP50)
│   └── last.pt              # Last epoch checkpoint
├── results.csv              # Training metrics log
├── loss_analysis.png        # Loss curves (box/cls/dfl/total)
├── map_performance.png      # mAP@0.5 and mAP@0.5:0.95
├── precision_recall.png     # Precision and Recall curves
└── training_status.png      # Training time and learning rate
```

### CSV Log Format

The CSV log uses YOLO-style slash notation:

```csv
epoch,time,lr,train/loss,val/loss,train/box_loss,train/cls_loss,...
1,12.34,0.001,2.45,2.67,1.23,0.45,...
```

## State Synchronization

After training completes, the YOLO model automatically synchronizes its state:

1. **Best weights loaded**: `model.model` loads `weights/best.pt`
2. **Class info updated**: `model.nc` and `model.names` from dataset
3. **Ready for inference**: You can immediately call `model.predict()`

```python
model = YOLO('configs/models/yolov11n.yaml')
results = model.train(data='data.yaml', epochs=100)

# Model now has trained weights
predictions = model.predict('test.jpg')  # Uses best.pt weights
```

## Advanced Usage

### Custom Configuration Dictionary

For full control, pass a complete configuration dictionary:

```python
config = {
    'train': {
        'name': 'custom_exp',
        'epochs': 200,
        'batch_size': 32,
        'img_size': 640,
        'mosaic': True,
        'mosaic_prob': 0.9,
        'mosaic_disable_epoch': 10,
        'letterbox': True,
    },
    'data': {
        'train': 'data/coco8/data.yaml',
        'nc': 2,
    },
    'optimizer': {
        'type': 'AdamW',
        'lr': 0.0001,
        'weight_decay': 0.0005,
    },
    'scheduler': {
        'type': 'CosineAnnealingLR',
        'min_lr': 1e-6,
    },
    'model': {
        'use_ema': True,
        'ema_decay': 0.9999,
    },
    'device': 'cuda:0',
}

results = model.train(config=config)
```

### Resume Training

To resume from a checkpoint:

```python
# Load last checkpoint
model = YOLO('runs/train/exp/weights/last.pt')

# Continue training (weights are preserved)
results = model.train(
    data='data.yaml',
    epochs=150,  # Total epochs (not additional)
)
```

### Custom Save Directory

```python
results = model.train(
    data='data.yaml',
    epochs=100,
    save_dir='/path/to/custom/dir',
)
```

The save directory will auto-increment: `/path/to/custom/dir/exp`, `/path/to/custom/dir/exp_1`, etc.

## Training Features

### EMA (Exponential Moving Average)

EMA is enabled by default and provides more stable validation metrics:

```python
config = {
    'model': {
        'use_ema': True,
        'ema_decay': 0.9999,
    }
}
```

Validation uses EMA model weights automatically.

### Mosaic Data Augmentation

Mosaic is enabled by default and disabled in the last 10 epochs:

```python
config = {
    'train': {
        'mosaic': True,
        'mosaic_prob': 1.0,
        'mosaic_disable_epoch': 10,  # Disable in final 10 epochs
    }
}
```

### Learning Rate Scheduling

Default is Cosine Annealing:

```python
config = {
    'scheduler': {
        'type': 'CosineAnnealingLR',
        'min_lr': 1e-6,
    }
}
```

### Optimizer Configuration

Default is Adam with parameter groups:

```python
config = {
    'optimizer': {
        'type': 'Adam',
        'lr': 0.001,
        'weight_decay': 0.0,
    }
}
```

Parameter groups:
- BatchNorm weights: no decay
- Other weights: with decay
- Biases: no decay

## Return Values

The `train()` method returns a dictionary with:

```python
{
    'best_map': 0.85,              # Best mAP50 achieved
    'final_epoch': 100,            # Final epoch number
    'save_dir': '/path/to/exp',    # Save directory path
}
```

## Error Handling

### Missing Data Parameter

```python
# This will raise ValueError
model.train(epochs=100)

# Error: Data parameter is required. Provide either 'data' argument
# or include 'data' in config.
```

### Invalid Data Path

```python
# This will raise FileNotFoundError
model.train(data='nonexistent.yaml', epochs=100)

# Error: YAML config file not found: nonexistent.yaml
```

## Best Practices

1. **Start with defaults**: Use default config and override only what you need
2. **Use experiment configs**: Save your experiment settings as YAML files
3. **Monitor training**: Check `results.csv` and generated plots
4. **Save often**: Checkpoints are saved automatically (best and last)
5. **Use EMA**: Keep EMA enabled for more stable metrics
6. **Disable Mosaic at end**: Default behavior improves final accuracy

## Examples

### Minimal Training

```python
from models import YOLO

model = YOLO('configs/models/yolov11n.yaml')
model.train(data='data/coco8/data.yaml', epochs=100)
```

### Custom Dataset

```python
# Create data.yaml
# path: /datasets/my_dataset
# train: images/train
# val: images/val
# nc: 10
# names: ['class1', 'class2', ...]

model.train(data='datasets/my_dataset/data.yaml', epochs=200)
```

### Quick Test Run

```python
# Test with minimal epochs and small batch
model.train(
    data='data.yaml',
    epochs=1,
    batch=2,
    imgsz=64,
    device='cpu',
)
```

### Production Training

```python
model.train(
    data='data.yaml',
    epochs=300,
    batch=32,
    imgsz=640,
    lr=0.001,
    device='cuda:0',
    optimizer__type='AdamW',
    optimizer__weight_decay=0.0005,
)
```
