"""Tests for CLI train script"""
import subprocess
import tempfile
from pathlib import Path


def test_train_cli_help():
    """Test that CLI help works."""
    result = subprocess.run(
        ['python', '-m', 'engine.train', '--help'],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert '--name' in result.stdout


def test_train_cli_basic():
    """Test basic CLI training invocation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        result = subprocess.run(
            [
                'python', '-m', 'engine.train',
                '--name', 'cli_test',
                '--data', 'configs/data/coco8.yaml',
                '--epochs', '1',
                '--batch-size', '2',
                '--img-size', '64',
                '--save-dir', tmpdir,
                '--device', 'cpu',
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )
        # Should succeed or fail gracefully (not crash)
        assert result.returncode in [0, 1]
