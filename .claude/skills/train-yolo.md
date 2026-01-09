# Train YOLO Detector

## Description
Train the YOLOCoordAttDetector on a custom dataset with the standard training pipeline.

## Parameters
- `--config`: Path to data.yaml config file (default: datasets/MY_TEST_DATA/data.yaml)
- `--epochs`: Number of training epochs (default: 100)
- `--batch`: Batch size (default: 4)
- `--img`: Image size (default: 640)
- `--lr`: Learning rate (default: 0.001)
- `--device`: Device to use (default: cuda)

## Implementation
Create a training script that:
1. Activates the conda environment
2. Creates dataloaders from config
3. Instantiates YOLOCoordAttDetector
4. Runs train_detector() with early stopping
5. Saves best model to outputs/
6. Visualizes attention after training

Example usage:
```bash
python -c "
from models import YOLOCoordAttDetector
from engine import train_detector, visualize_detection_attention
from utils import create_dataloaders

train_loader, val_loader, _ = create_dataloaders('datasets/MY_TEST_DATA/data.yaml', batch_size=4, img_size=640)
model = YOLOCoordAttDetector(nc=1)
train_detector(model, train_loader, val_loader, epochs=100, lr=0.001, device='cuda')
visualize_detection_attention(model, val_loader, 'cuda', save_path='outputs/attention.png')
"
```
