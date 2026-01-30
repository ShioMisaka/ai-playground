import os
import sys
# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import YOLOv11
from engine import train
from utils.config import get_config

# 1. 创建配置（使用嵌套键名）
# 新版：添加 train.letterbox=true 确保训练/推理一致性
cfg = get_config(
    **{
        'train.name': 'test_exp',
        'train.epochs': 30,
        'train.batch_size': 8,
        'train.letterbox': True,  # 使用 letterbox 预处理（确保一致性）
        'train.save_dir': 'runs/train',
        'model.nc': 2,  # 类别数
        'model.scale': 'n',  # 模型规模
        'model.img_size': 640,
        'model.use_ema': True,
        'augment.use_mosaic': True,
        'augment.close_mosaic': 10,  # 最后10个epoch关闭mosaic（30个epoch训练时，在第21个epoch关闭）
        'optimizer.type': 'Adam',
        'optimizer.lr': 0.001,
        'optimizer.weight_decay': 0.0005,
        'optimizer.momentum': 0.9,
        'optimizer.betas': [0.9, 0.999],  # Adam betas
        'optimizer.eps': 1e-8,
        'scheduler.type': 'CosineAnnealingLR',  # 学习率调度器
        'scheduler.min_lr': 1e-6,
        'system.device': 'cpu',
        'system.workers': 0,
        'data': 'datasets/MY_TEST_DATA/data.yaml',  # 测试数据路径（需要存在）
    }
)

# 2. 创建模型
# 新版 YOLOv11 返回格式：
# - 推理模式: (bs, n_anchors, 4+nc) 张量
# - 训练模式: {'loss': ..., 'loss_items': [...], 'predictions': ...}
model = YOLOv11(nc=2, scale='n')

# 3. 开始训练
# train() 函数会自动处理新版输出格式
trained_model = train(model, cfg)
