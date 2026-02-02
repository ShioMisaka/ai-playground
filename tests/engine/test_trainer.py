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
