import tempfile
import torch
from models import YOLOv11
from engine.trainer import DetectionTrainer


def test_training_execution():
    """Test that training loop executes without errors."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model = YOLOv11(nc=2, scale='n')

        # Minimal config for quick test
        config = {
            'train': {
                'name': 'test_train',
                'epochs': 1,
                'batch_size': 2,
                'img_size': 64,  # Small for speed
                'save_dir': tmpdir,
                'letterbox': True,
            },
            'data': {
                'train': 'data/coco8/data.yaml',  # Use YAML config path
                'val': 'data/coco8/data.yaml',
                'nc': 2,
                'names': ['cat', 'dog']
            },
            'optimizer': {'type': 'Adam', 'lr': 0.001},
            'scheduler': {'type': 'CosineAnnealingLR', 'min_lr': 1e-6},
            'model': {'use_ema': True, 'ema_decay': 0.9999},
            'device': 'cpu',
        }

        trainer = DetectionTrainer(model, config)

        # Should complete without errors
        results = trainer.train()

        assert 'best_map' in results
        assert 'final_epoch' in results
        assert trainer.best_map >= 0
