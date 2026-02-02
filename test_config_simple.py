#!/usr/bin/env python
"""
Simple test runner for config tests that works without pytest.
"""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.config import (
    load_yaml,
    merge_configs,
    merge_training_config,
    _flatten_to_nested,
)
import tempfile
from pathlib import Path
import yaml


def test_load_valid_yaml():
    """Test loading a valid YAML file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump({'train': {'epochs': 100, 'batch_size': 16}}, f)
        f.flush()

        result = load_yaml(f.name)
        assert result == {'train': {'epochs': 100, 'batch_size': 16}}

        Path(f.name).unlink()
    print("✓ test_load_valid_yaml passed")


def test_load_empty_yaml():
    """Test loading an empty YAML file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write('')
        f.flush()

        result = load_yaml(f.name)
        assert result == {}

        Path(f.name).unlink()
    print("✓ test_load_empty_yaml passed")


def test_simple_merge():
    """Test simple dictionary merging."""
    base = {'a': 1, 'b': 2}
    override = {'b': 3, 'c': 4}
    result = merge_configs(base, override)

    assert result == {'a': 1, 'b': 3, 'c': 4}
    print("✓ test_simple_merge passed")


def test_nested_merge():
    """Test nested dictionary merging."""
    base = {'train': {'epochs': 100, 'batch_size': 16}}
    override = {'train': {'batch_size': 32, 'lr': 0.001}}
    result = merge_configs(base, override)

    assert result == {
        'train': {'epochs': 100, 'batch_size': 32, 'lr': 0.001}
    }
    print("✓ test_nested_merge passed")


def test_single_level_key():
    """Test converting single-level flat key."""
    flat = {'epochs': 100}
    result = _flatten_to_nested(flat)

    assert result == {'epochs': 100}
    print("✓ test_single_level_key passed")


def test_two_level_key():
    """Test converting two-level flat key."""
    flat = {'train.epochs': 100}
    result = _flatten_to_nested(flat)

    assert result == {'train': {'epochs': 100}}
    print("✓ test_two_level_key passed")


def test_multiple_keys():
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
    print("✓ test_multiple_keys passed")


def test_default_config_only():
    """Test loading only default config."""
    config = merge_training_config()

    assert 'train' in config
    assert 'val' in config
    assert 'predict' in config
    assert config['train']['epochs'] == 100
    print("✓ test_default_config_only passed")


def test_with_model_config_dict():
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
    print("✓ test_with_model_config_dict passed")


def test_with_user_config_dict():
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
    print("✓ test_with_user_config_dict passed")


def test_priority_order():
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
    assert config['train']['lr'] == 0.01  # model wins
    assert config['train']['batch_size'] == 16  # user wins
    assert config['train']['epochs'] == 200  # overrides wins
    print("✓ test_priority_order passed")


def test_with_overrides_flat_keys():
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
    print("✓ test_with_overrides_flat_keys passed")


def test_all_sources_combined():
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
    print("✓ test_all_sources_combined passed")


def test_missing_model_config_file():
    """Test that missing model config file raises FileNotFoundError."""
    try:
        merge_training_config(model_config='nonexistent_model.yaml')
        assert False, "Should have raised FileNotFoundError"
    except FileNotFoundError:
        pass
    print("✓ test_missing_model_config_file passed")


def test_missing_user_config_file():
    """Test that missing user config file raises FileNotFoundError."""
    try:
        merge_training_config(user_config='nonexistent_user.yaml')
        assert False, "Should have raised FileNotFoundError"
    except FileNotFoundError:
        pass
    print("✓ test_missing_user_config_file passed")


def test_none_parameters():
    """Test that None parameters are handled correctly."""
    config = merge_training_config(
        model_config=None,
        user_config=None,
        overrides=None
    )

    # Should just return default config
    assert 'train' in config
    assert 'val' in config
    print("✓ test_none_parameters passed")


def main():
    """Run all tests."""
    tests = [
        test_load_valid_yaml,
        test_load_empty_yaml,
        test_simple_merge,
        test_nested_merge,
        test_single_level_key,
        test_two_level_key,
        test_multiple_keys,
        test_default_config_only,
        test_with_model_config_dict,
        test_with_user_config_dict,
        test_priority_order,
        test_with_overrides_flat_keys,
        test_all_sources_combined,
        test_missing_model_config_file,
        test_missing_user_config_file,
        test_none_parameters,
    ]

    print("Running config tests...")
    print("=" * 60)

    failed = []
    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"✗ {test.__name__} failed: {e}")
            failed.append(test.__name__)

    print("=" * 60)
    if failed:
        print(f"\n{len(failed)} test(s) failed:")
        for name in failed:
            print(f"  - {name}")
        return 1
    else:
        print(f"\nAll {len(tests)} tests passed!")
        return 0


if __name__ == '__main__':
    sys.exit(main())
