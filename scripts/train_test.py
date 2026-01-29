import os
import sys
# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import YOLOv11
from engine import train
from utils.config import get_config

# 1. 创建配置（使用嵌套键名）
cfg = get_config(
    **{'train.name': 'test_exp',
       'train.epochs': 30,
       'train.batch_size': 8,
       'optimizer.lr': 0.001,
       'system.device': 'cpu'}
)

# 2. 创建模型
model = YOLOv11(nc=2, scale='n')

# 3. 开始训练
trained_model = train(model, cfg)
