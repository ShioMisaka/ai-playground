"""
Unit tests for utils/config.py module.

Tests cover:
- Default config loading
- File path inputs for model/user configs
- Dict inputs for model/user configs
- Priority order verification
- Flat key conversion in overrides
- Error cases (missing files, invalid types)
"""
import tempfile
from pathlib import Path
import pytest
import yaml

from utils.config import (
    load_yaml,
    merge_configs,
    merge_training_config,
    _flatten_to_nested,
)


class TestLoadYaml:
    """Tests for load_yaml function."""

    def test_load_valid_yaml(self):
        """Test loading a valid YAML file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump({'train': {'epochs': 100, 'batch_size': 16}}, f)
            f.flush()

            result = load_yaml(f.name)
            assert result == {'train': {'epochs': 100, 'batch_size': 16}}

            Path(f.name).unlink()

    def test_load_empty_yaml(self):
        """Test loading an empty YAML file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write('')
            f.flush()

            result = load_yaml(f.name)
            assert result == {}

            Path(f.name).unlink()

    def test_load_nonexistent_file(self):
        """Test loading a non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="配置文件不存在"):
            load_yaml('nonexistent_file.yaml')

    def test_load_invalid_yaml(self):
        """Test loading invalid YAML raises yaml.YAMLError."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write('invalid: yaml: content: [')
            f.flush()

            with pytest.raises(yaml.YAMLError):
                load_yaml(f.name)

            Path(f.name).unlink()


class TestMergeConfigs:
    """Tests for merge_configs function."""

    def test_simple_merge(self):
        """Test simple dictionary merging."""
        base = {'a': 1, 'b': 2}
        override = {'b': 3, 'c': 4}
        result = merge_configs(base, override)

        assert result == {'a': 1, 'b': 3, 'c': 4}

    def test_nested_merge(self):
        """Test nested dictionary merging."""
        base = {'train': {'epochs': 100, 'batch_size': 16}}
        override = {'train': {'batch_size': 32, 'lr': 0.001}}
        result = merge_configs(base, override)

        assert result == {
            'train': {'epochs': 100, 'batch_size': 32, 'lr': 0.001}
        }

    def test_deeply_nested_merge(self):
        """Test deeply nested dictionary merging."""
        base = {'optimizer': {'params': {'lr': 0.001, 'momentum': 0.9}}}
        override = {'optimizer': {'params': {'lr': 0.0001}, 'type': 'Adam'}}
        result = merge_configs(base, override)

        assert result == {
            'optimizer': {
                'params': {'lr': 0.0001, 'momentum': 0.9},
                'type': 'Adam'
            }
        }

    def test_override_does_not_modify_base(self):
        """Test that merge doesn't modify the original base dict."""
        base = {'a': 1, 'b': 2}
        override = {'b': 3}
        result = merge_configs(base, override)

        assert base == {'a': 1, 'b': 2}
        assert result == {'a': 1, 'b': 3}


class TestFlattenToNested:
    """Tests for _flatten_to_nested function."""

    def test_single_level_key(self):
        """Test converting single-level flat key."""
        flat = {'epochs': 100}
        result = _flatten_to_nested(flat)

        assert result == {'epochs': 100}

    def test_two_level_key(self):
        """Test converting two-level flat key."""
        flat = {'train.epochs': 100}
        result = _flatten_to_nested(flat)

        assert result == {'train': {'epochs': 100}}

    def test_three_level_key(self):
        """Test converting three-level flat key."""
        flat = {'optimizer.params.lr': 0.001}
        result = _flatten_to_nested(flat)

        assert result == {'optimizer': {'params': {'lr': 0.001}}}

    def test_multiple_keys(self):
        """Test converting multiple flat keys."""
        flat = {
            'train.epochs': 100,
            'train.batch_size': 16,
            'optimizer.lr': 0.001
        }
        result = _flatten_to_nested(flat)

        assert result == {
            'train': {'epochs': 100, 'batch_size': 16},
            'optimizer': {'lr': 0.001}
        }

    def test_mixed_levels(self):
        """Test converting mixed level keys."""
        flat = {
            'epochs': 100,
            'train.batch_size': 16,
            'optimizer.params.momentum': 0.9
        }
        result = _flatten_to_nested(flat)

        assert result == {
            'epochs': 100,
            'train': {'batch_size': 16},
            'optimizer': {'params': {'momentum': 0.9}}
        }


class TestMergeTrainingConfig:
    """Tests for merge_training_config function."""

    def test_default_config_only(self):
        """Test loading only default config."""
        config = merge_training_config()

        assert 'train' in config
        assert 'val' in config
        assert 'predict' in config
        assert config['train']['epochs'] == 100

    def test_with_model_config_file(self):
        """Test merging model config from file."""
        # Create a temporary model config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            model_config = {
                'model': {
                    'type': 'YOLOv11',
                    'scale': 'n',
                    'nc': 2
                }
            }
            yaml.dump(model_config, f)
            f.flush()

            config = merge_training_config(model_config=f.name)

            assert config['model']['type'] == 'YOLOv11'
            assert config['model']['nc'] == 2
            # Default values should still be present
            assert 'train' in config

            Path(f.name).unlink()

    def test_with_model_config_dict(self):
        """Test merging model config from dict."""
        model_config = {
            'model': {
                'type': 'YOLOv11',
                'scale': 's',
                'nc': 3
            }
        }

        config = merge_training_config(model_config=model_config)

        assert config['model']['type'] == 'YOLOv11'
        assert config['model']['scale'] == 's'
        assert config['model']['nc'] == 3

    def test_with_user_config_file(self):
        """Test merging user config from file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            user_config = {
                'train': {
                    'name': 'my_experiment',
                    'epochs': 200,
                    'batch_size': 32
                }
            }
            yaml.dump(user_config, f)
            f.flush()

            config = merge_training_config(user_config=f.name)

            assert config['train']['name'] == 'my_experiment'
            assert config['train']['epochs'] == 200
            assert config['train']['batch_size'] == 32

            Path(f.name).unlink()

    def test_with_user_config_dict(self):
        """Test merging user config from dict."""
        user_config = {
            'train': {
                'name': 'test_exp',
                'lr': 0.0001
            }
        }

        config = merge_training_config(user_config=user_config)

        assert config['train']['name'] == 'test_exp'
        assert config['train']['lr'] == 0.0001

    def test_priority_order(self):
        """Test that configs are merged in correct priority order."""
        model_config = {
            'train': {
                'epochs': 50,
                'batch_size': 8,
                'lr': 0.01
            }
        }

        user_config = {
            'train': {
                'epochs': 100,
                'batch_size': 16
            }
        }

        overrides = {
            'train.epochs': 200
        }

        config = merge_training_config(
            model_config=model_config,
            user_config=user_config,
            overrides=overrides
        )

        # Default < model < user < overrides
        # Default has lr=0.001, model has lr=0.01 -> model wins
        assert config['train']['lr'] == 0.01

        # Model has batch_size=8, user has 16 -> user wins
        assert config['train']['batch_size'] == 16

        # Model has epochs=50, user has 100, overrides has 200 -> overrides wins
        assert config['train']['epochs'] == 200

    def test_with_overrides_flat_keys(self):
        """Test that overrides with flat keys are properly converted."""
        overrides = {
            'train.epochs': 150,
            'train.batch_size': 32,
            'optimizer.lr': 0.0001
        }

        config = merge_training_config(overrides=overrides)

        assert config['train']['epochs'] == 150
        assert config['train']['batch_size'] == 32
        assert config['optimizer']['lr'] == 0.0001

    def test_overrides_with_nested_keys(self):
        """Test that overrides with nested structure work."""
        overrides = {
            'train': {'epochs': 150},
            'optimizer': {'lr': 0.0001}
        }

        config = merge_training_config(overrides=overrides)

        assert config['train']['epochs'] == 150
        assert config['optimizer']['lr'] == 0.0001

    def test_all_sources_combined(self):
        """Test combining all config sources."""
        model_config = {
            'model': {'type': 'YOLOv11', 'nc': 2}
        }

        user_config = {
            'train': {'name': 'combined_test', 'epochs': 50}
        }

        overrides = {
            'train.batch_size': 32
        }

        config = merge_training_config(
            model_config=model_config,
            user_config=user_config,
            overrides=overrides
        )

        # All configs should be present
        assert config['model']['type'] == 'YOLOv11'
        assert config['model']['nc'] == 2
        assert config['train']['name'] == 'combined_test'
        assert config['train']['epochs'] == 50
        assert config['train']['batch_size'] == 32
        # Defaults should be present for unspecified values
        assert 'val' in config

    def test_missing_model_config_file(self):
        """Test that missing model config file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            merge_training_config(model_config='nonexistent_model.yaml')

    def test_missing_user_config_file(self):
        """Test that missing user config file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            merge_training_config(user_config='nonexistent_user.yaml')

    def test_none_parameters(self):
        """Test that None parameters are handled correctly."""
        config = merge_training_config(
            model_config=None,
            user_config=None,
            overrides=None
        )

        # Should just return default config
        assert 'train' in config
        assert 'val' in config


class TestMergeTrainingConfigIntegration:
    """Integration tests for merge_training_config."""

    def test_realistic_training_config(self):
        """Test a realistic training configuration scenario."""
        # Model config specifies architecture
        model_config = {
            'model': {
                'type': 'YOLOv11',
                'scale': 'n',
                'nc': 80,
                'use_ema': True
            }
        }

        # User config specifies training parameters
        user_config = {
            'train': {
                'name': 'yolov11_coco',
                'epochs': 300,
                'batch_size': 64,
                'img_size': 640
            },
            'data': {
                'train': 'datasets/coco/train.txt',
                'val': 'datasets/coco/val.txt',
                'nc': 80,
                'names': ['person', 'bicycle', 'car']  # abbreviated
            }
        }

        # CLI overrides for quick experimentation
        overrides = {
            'train.epochs': 10,  # Quick test run
            'train.lr': 0.001,
            'device': 'cpu'
        }

        config = merge_training_config(
            model_config=model_config,
            user_config=user_config,
            overrides=overrides
        )

        # Verify all sources are merged
        assert config['model']['type'] == 'YOLOv11'
        assert config['model']['nc'] == 80
        assert config['train']['name'] == 'yolov11_coco'
        assert config['train']['batch_size'] == 64
        assert config['train']['epochs'] == 10  # Override wins
        assert config['train']['lr'] == 0.001
        assert config['device'] == 'cpu'
        assert 'val' in config  # Defaults present

    def test_deep_override_preserves_siblings(self):
        """Test that deep override preserves sibling values."""
        model_config = {
            'optimizer': {
                'type': 'SGD',
                'params': {
                    'lr': 0.01,
                    'momentum': 0.9,
                    'weight_decay': 0.0005
                }
            }
        }

        overrides = {
            'optimizer.params.lr': 0.001
        }

        config = merge_training_config(
            model_config=model_config,
            overrides=overrides
        )

        # lr should be overridden
        assert config['optimizer']['params']['lr'] == 0.001
        # siblings should be preserved
        assert config['optimizer']['params']['momentum'] == 0.9
        assert config['optimizer']['params']['weight_decay'] == 0.0005
        assert config['optimizer']['type'] == 'SGD'
