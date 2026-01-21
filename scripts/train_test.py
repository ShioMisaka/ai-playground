import os
import sys
# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import YOLOv3
from engine import train

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
    save_dir='outputs/train/exp1'
)
