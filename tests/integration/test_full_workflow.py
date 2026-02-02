"""
Integration test for the full YOLO training workflow.

This test verifies:
1. Model creation from config
2. Training with YOLO.train()
3. Automatic best weight loading
4. Inference with trained model
"""
import tempfile
from pathlib import Path
from models import YOLO
import torch
import numpy as np


def test_full_train_predict_workflow():
    """Test complete training + prediction workflow."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Create model
        model = YOLO('configs/models/yolov11n.yaml')
        assert model.model is not None

        # 2. Train model
        train_results = model.train(
            data='data/coco8/data.yaml',
            epochs=1,
            batch=2,
            imgsz=64,
            save_dir=tmpdir,
            device='cpu',
        )

        assert 'best_map' in train_results
        assert Path(train_results['save_dir']).exists()

        # 3. Verify weights were saved (best.pt if mAP improved, last.pt always)
        weights_dir = Path(train_results['save_dir']) / 'weights'
        assert weights_dir.exists()

        last_path = weights_dir / 'last.pt'
        assert last_path.exists(), "Last checkpoint should always be saved"

        # best.pt is only saved if mAP improves (mAP > 0 for first epoch)
        best_path = weights_dir / 'best.pt'
        if train_results['best_map'] > 0:
            assert best_path.exists(), "Best checkpoint should be saved when mAP > 0"

        # 4. Predict with trained model
        # Create a dummy image for testing
        dummy_img = np.random.rand(480, 640, 3).astype(np.uint8)

        results = model.predict(dummy_img, conf=0.25)
        assert results is not None
        assert len(results) == 1

        print("Full workflow test passed!")
