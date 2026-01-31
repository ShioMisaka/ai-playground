import os
import sys
# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import YOLOv11
from engine import train
from utils.config import get_config

# 1. 创建配置（使用嵌套键名）
cfg = get_config(
    **{
        'train.name': 'test_exp',
        'train.epochs': 30,
        'train.batch_size': 8,
        'train.letterbox': False,  # 训练时使用简单 resize（不使用 letterbox）
        'train.save_dir': 'runs/train',
        'model.nc': 2,  # 类别数
        'model.scale': 'n',  # 模型规模
        'model.img_size': 640,
        'model.use_ema': True,
        'augment.use_mosaic': False,
        'augment.close_mosaic': 10,  # 最后10个epoch关闭mosaic
        'optimizer.type': 'Adam',
        'optimizer.lr': 0.001,
        'optimizer.weight_decay': 0.0005,
        'optimizer.betas': [0.9, 0.999],  # Adam betas
        'optimizer.eps': 1e-8,
        'scheduler.type': 'CosineAnnealingLR',
        'scheduler.min_lr': 1e-6,
        'system.device': 'cpu',
        'system.workers': 0,
        'data': 'datasets/MY_TEST_DATA/data.yaml',
    }
)

# 2. 创建模型
model = YOLOv11(nc=2, scale='n')

# 3. 开始训练
print("开始训练...")
trained_model = train(model, cfg)
print("训练完成！")
