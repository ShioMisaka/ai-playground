"""Tests for engine.trainer.DetectionTrainer"""
import pytest
import tempfile
from pathlib import Path
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


def test_ema_setup():
    """Test EMA initialization."""
    model = YOLOv11(nc=2, scale='n')

    config = {
        'train': {'name': 'test', 'epochs': 1, 'batch_size': 4},
        'data': {'train': 'dummy.txt', 'nc': 2},
        'model': {'use_ema': True, 'ema_decay': 0.9999},
        'device': 'cpu',
    }
    trainer = DetectionTrainer(model, config)
    trainer._setup_model()
    trainer._setup_ema()

    assert trainer.ema is not None
    assert trainer.ema.decay == 0.9999


def test_ema_disabled():
    """Test EMA can be disabled."""
    model = YOLOv11(nc=2, scale='n')

    config = {
        'train': {'name': 'test', 'epochs': 1, 'batch_size': 4},
        'data': {'train': 'dummy.txt', 'nc': 2},
        'model': {'use_ema': False},
        'device': 'cpu',
    }
    trainer = DetectionTrainer(model, config)
    trainer._setup_ema()

    assert trainer.ema is None


def test_mosaic_setup():
    """Test Mosaic initialization."""
    model = YOLOv11(nc=2, scale='n')

    config = {
        'train': {
            'name': 'test',
            'epochs': 1,
            'batch_size': 4,
            'img_size': 640,
            'mosaic': True,
            'mosaic_prob': 0.9,
        },
        'data': {'train': 'dummy.txt', 'nc': 2},
        'device': 'cpu',
    }
    trainer = DetectionTrainer(model, config)
    trainer._setup_mosaic()

    # Mosaic is set to None initially, will be configured after data loaders are created
    assert trainer.mosaic_enabled is True
    assert trainer.mosaic_prob == 0.9
    assert trainer.mosaic is None  # Will be set after data loaders are created


def test_logging_setup():
    """Test CSV and LiveTable logger initialization."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model = YOLOv11(nc=2, scale='n')

        config = {
            'train': {
                'name': 'test',
                'epochs': 2,
                'batch_size': 4,
                'save_dir': tmpdir,
            },
            'data': {'train': 'dummy.txt', 'nc': 2},
            'device': 'cpu',
        }
        trainer = DetectionTrainer(model, config)
        trainer._setup_save_dir()
        trainer._setup_logging()

        assert trainer.csv_logger is not None
        assert trainer.live_logger is not None

        # Verify CSV logger is opened (writer should be initialized)
        assert trainer.csv_logger._writer is not None, "CSV logger should be opened"
        assert trainer.csv_logger._file is not None, "CSV file should be open"

        # Verify total_loss column is included in LiveTableLogger
        assert "total_loss" in trainer.live_logger.columns, "total_loss should be in columns"

        # Close the CSV logger to release the file
        trainer.csv_logger.close()
