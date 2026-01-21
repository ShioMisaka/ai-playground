import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from models import YOLOv3

def test01():
    model = YOLOv3(nc=80)
    model.eval()

    # 测试前向传播
    x = torch.randn(1, 3, 640, 640)
    output = model(x)

    if isinstance(output, tuple):
        predictions, raw_outputs = output
        print(f"Predictions shape: {predictions.shape}")  # (1, num_predictions, 85)
        print(f"Raw outputs shapes: {[o.shape for o in raw_outputs]}")
    else:
        print(f"Training outputs: {[o.shape for o in output]}")

from engine import train


def test02():
    # 1. 导入你的YOLO模型
    model = YOLOv3(nc=2)
    
    # 2. 开始训练
    trained_model = train(
        model=model,
        config_path='datasets/MY_TEST_DATA/data.yaml',
        epochs=2,
        batch_size=8,
        img_size=640,
        lr=0.01,
        device='cuda',
        save_dir='runs/train/exp1'
    )



if __name__ == "__main__":
    test02()