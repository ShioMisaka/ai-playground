import os
import sys
# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import YOLO

# 使用新的 YOLO API 进行训练
# 1. 创建模型（从配置文件）
model = YOLO('configs/models/yolov11n.yaml')

# 2. 开始训练
results = model.train(
    data='datasets/MY_TEST_DATA/data.yaml',
    epochs=30,
    batch=8,
    imgsz=640,
    lr=0.001,
    device='cpu',
    name='test_exp',
    save_dir='runs/train',
    mosaic=False,  # 关闭 Mosaic 数据增强
)

print(f"训练完成！Best mAP50: {results['best_map']:.4f}")
print(f"模型保存位置: {results['save_dir']}")
