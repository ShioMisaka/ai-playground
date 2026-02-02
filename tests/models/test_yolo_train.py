"""Tests for YOLO.train() method"""
import tempfile
from pathlib import Path
import pytest
from unittest.mock import Mock, patch


def test_yolo_train_api():
    """Test that YOLO.train() works with config."""
    # Skip if coco8 data doesn't exist
    data_path = Path('/datasets/coco8')
    if not data_path.exists():
        pytest.skip(f"Dataset not found: {data_path}")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create model from config
        from models import YOLO
        model = YOLO('configs/models/yolov11n.yaml')

        # Train with overrides
        results = model.train(
            data='configs/data/coco8.yaml',
            epochs=1,
            batch=2,
            imgsz=64,
            save_dir=tmpdir,
            device='cpu',
        )

        # Check results
        assert 'best_map' in results
        assert 'save_dir' in results

        # Verify model state was updated
        assert model.nc == 2  # coco8 has 2 classes


def test_yolo_train_with_dict_config():
    """Test YOLO.train() with dict config."""
    # Skip if coco8 data doesn't exist
    data_path = Path('/datasets/coco8')
    if not data_path.exists():
        pytest.skip(f"Dataset not found: {data_path}")

    with tempfile.TemporaryDirectory() as tmpdir:
        from models import YOLO
        model = YOLO('configs/models/yolov11n.yaml')

        config = {
            'train': {
                'name': 'dict_test',
                'epochs': 1,
                'batch_size': 2,
                'img_size': 64,
            },
            'data': {
                'train': 'configs/data/coco8.yaml',
                'nc': 2,
            },
        }

        results = model.train(config=config, save_dir=tmpdir)
        assert 'best_map' in results


def test_yolo_create_from_config():
    """Test that YOLO can be created from model config file."""
    from models import YOLO

    # Create model from config
    model = YOLO('configs/models/yolov11n.yaml')

    # Verify basic properties
    assert hasattr(model, 'model')
    assert hasattr(model, 'train')
    assert hasattr(model, 'predict')
    assert model.nc == 2  # From config
    assert model.device is not None


def test_yolo_train_signature():
    """Test that YOLO.train() has correct signature."""
    from models import YOLO
    import inspect

    model = YOLO('configs/models/yolov11n.yaml')

    # Check method exists and has correct signature
    assert hasattr(model, 'train')

    sig = inspect.signature(model.train)
    params = list(sig.parameters.keys())

    # Verify key parameters exist
    assert 'data' in params
    assert 'epochs' in params
    assert 'batch' in params
    assert 'imgsz' in params
    assert 'lr' in params
    assert 'device' in params
    assert 'config' in params
    assert 'save_dir' in params
    assert 'kwargs' in params


def test_yolo_load_from_weights_still_works():
    """Test backward compatibility - loading from weights still works."""
    from models import YOLO
    import torch
    from models.yolov11 import YOLOv11

    # Create a dummy weights file
    with tempfile.TemporaryDirectory() as tmpdir:
        weights_path = Path(tmpdir) / 'dummy.pt'

        # Create and save a model
        dummy_model = YOLOv11(nc=2, scale='n')
        torch.save(dummy_model.state_dict(), weights_path)

        # Load using YOLO class
        model = YOLO(str(weights_path))

        # Verify it loaded correctly
        assert hasattr(model, 'model')
        assert hasattr(model, 'predict')
        assert model.nc == 2


def test_yolo_train_creates_correct_config():
    """Test that train() method creates correct config structure."""
    from models import YOLO

    model = YOLO('configs/models/yolov11n.yaml')

    # Mock both the trainer and load_yaml to avoid actual file operations
    with patch('models.yolo.DetectionTrainer') as mock_trainer_class, \
         patch('models.yolo.load_yaml') as mock_load_yaml:
        mock_load_yaml.return_value = {'nc': 2, 'names': ['cat', 'dog']}

        mock_trainer = Mock()
        mock_trainer.train.return_value = {
            'best_map': 0.85,
            'final_epoch': 1,
            'save_dir': '/tmp/test'
        }
        mock_trainer_class.return_value = mock_trainer

        # Call train with parameters (data is now required)
        results = model.train(
            data='dummy_data.yaml',  # data parameter is now required
            epochs=10,
            batch=8,
            imgsz=320,
            lr=0.001,
            device='cpu',
        )

        # Verify trainer was called
        assert mock_trainer_class.called

        # Get the config that was passed to trainer
        call_args = mock_trainer_class.call_args
        config = call_args[1]['config']

        # Verify config structure
        assert 'train' in config
        assert config['train']['epochs'] == 10
        assert config['train']['batch_size'] == 8
        assert config['train']['img_size'] == 320
        assert config['device'] == 'cpu'
        assert 'optimizer' in config
        assert config['optimizer']['lr'] == 0.001

        # Verify results
        assert results['best_map'] == 0.85
