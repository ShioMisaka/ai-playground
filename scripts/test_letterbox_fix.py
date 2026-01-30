#!/usr/bin/env python3
"""
测试完整的 Letterbox 支持

验证修复后的 letterbox 预处理能够正常训练。
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import YOLOv11
from engine import train
from utils.config import get_config

print("="*70)
print("测试完整的 Letterbox 支持")
print("="*70)
print()
print("目标: 验证 letterbox=True 时，损失能正常下降")
print("配置: 5 个 epoch，启用 letterbox")
print()

# 创建配置 - 启用 letterbox
cfg = get_config(
    **{
        'train.name': 'test_letterbox_fix',
        'train.epochs': 5,
        'train.batch_size': 4,
        'train.letterbox': True,  # 关键测试：启用 letterbox
        'train.save_dir': 'runs/train',
        'model.nc': 2,
        'model.scale': 'n',
        'model.img_size': 640,
        'model.use_ema': False,
        'augment.use_mosaic': False,  # 暂时禁用 Mosaic 排查问题
        'augment.close_mosaic': 0,
        'system.device': 'cpu',
        'system.workers': 0,
        'optimizer.type': 'Adam',
        'optimizer.lr': 0.001,
        'optimizer.betas': [0.9, 0.999],
        'optimizer.eps': 1e-8,
        'optimizer.weight_decay': 0.0005,
        'scheduler.type': 'CosineAnnealingLR',
        'scheduler.min_lr': 1e-6,
        'data': 'datasets/MY_TEST_DATA/data.yaml',
    }
)

print("训练配置:")
print(f"  Epochs: {cfg['train']['epochs']}")
print(f"  Batch Size: {cfg['train']['batch_size']}")
print(f"  Letterbox: {cfg['train']['letterbox']} ← 关键测试")
print(f"  Learning Rate: {cfg['train']['lr']}")
print(f"  Device: {cfg['train']['device']}")
print()

# 创建模型
print("创建 YOLOv11 模型...")
model = YOLOv11(nc=2, scale='n')
print(f"模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
print()

# 开始训练
print("开始训练...")
print("-"*70)
try:
    trained_model = train(model, cfg)
    print("-"*70)
    print()
    print("✓ 训练完成！")
    print()
    print("请检查训练日志，确认损失是否正常下降：")
    print(f"  日志文件: runs/train/test_letterbox_fix/training_log.csv")
    print()
    print("预期结果:")
    print("  - 损失应该稳定下降（类似禁用 letterbox 的效果）")
    print("  - 如果损失仍然很高或波动，说明修复不完整")
    print()
except Exception as e:
    print()
    print("✗ 训练失败！")
    print(f"错误: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("="*70)
print("测试完成")
print("="*70)
