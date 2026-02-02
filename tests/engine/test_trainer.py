"""Tests for engine.trainer.DetectionTrainer"""
import pytest
import torch
from models import YOLOv11
from engine.trainer import DetectionTrainer


def test_trainer_initialization():
    """Test that DetectionTrainer can be initialized with minimal config."""
    model = YOLOv11(nc=2, scale='n')

    config = {
        'train': {
            'name': 'test_exp',
            'epochs': 2,
            'batch_size': 4,
            'img_size': 640,
            'letterbox': True,
        },
        'data': {
            'train': 'dummy_train.txt',
            'val': 'dummy_val.txt',
            'nc': 2,
            'names': ['cat', 'dog']
        },
        'device': 'cpu',
    }

    # Should not raise
    trainer = DetectionTrainer(model, config)

    assert trainer.model is model
    assert trainer.config == config
    assert trainer.device == torch.device('cpu')


def test_optimizer_creation():
    """Test optimizer creation with different configs."""
    model = YOLOv11(nc=2, scale='n')

    config = {
        'train': {'name': 'test', 'epochs': 1, 'batch_size': 4},
        'data': {'train': 'dummy.txt', 'nc': 2},
        'optimizer': {'type': 'Adam', 'lr': 0.001, 'weight_decay': 0.0005},
        'device': 'cpu',
    }
    trainer = DetectionTrainer(model, config)
    trainer._setup_model()
    trainer._setup_optimizer()

    assert trainer.optimizer is not None
    assert trainer.optimizer.param_groups[0]['lr'] == 0.001

    # Verify parameter groups are correctly configured
    assert len(trainer.optimizer.param_groups) == 3, "Optimizer should have 3 parameter groups"

    # pg0 (BatchNorm weights) should have weight_decay=0.0
    assert trainer.optimizer.param_groups[0]['weight_decay'] == 0.0, "BatchNorm weights should have no weight decay"

    # pg1 (Other weights) should have the configured weight_decay
    assert trainer.optimizer.param_groups[1]['weight_decay'] == 0.0005, "Other weights should have configured weight decay"

    # pg2 (Biases) should have weight_decay=0.0
    assert trainer.optimizer.param_groups[2]['weight_decay'] == 0.0, "Biases should have no weight decay"


def test_scheduler_creation():
    """Test scheduler creation."""
    model = YOLOv11(nc=2, scale='n')

    config = {
        'train': {'name': 'test', 'epochs': 10, 'batch_size': 4},
        'data': {'train': 'dummy.txt', 'nc': 2},
        'optimizer': {'type': 'Adam', 'lr': 0.001},
        'scheduler': {'type': 'CosineAnnealingLR', 'min_lr': 1e-6},
        'device': 'cpu',
    }
    trainer = DetectionTrainer(model, config)
    trainer._setup_model()
    trainer._setup_optimizer()
    trainer._setup_scheduler()

    assert trainer.scheduler is not None
