"""
Coordinate Attention YOLO 训练与可视化脚本 (目标检测版本)

使用 engine 模块进行 YOLO 检测训练和注意力可视化
"""
import os
import sys
import torch
from pathlib import Path
from datetime import datetime

# 添加父目录到路径以导入 models 和 engine
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import YOLOCoordAttDetector
from engine import train_detector, visualize_detection_attention, visualize_attention_comparison
from utils import create_dataloaders
from utils.load import load_yaml_config

# 创建带时间戳的输出目录
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = os.path.join('outputs', f'yolo_coordatt_{TIMESTAMP}')
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"输出目录: {OUTPUT_DIR}")


if __name__ == '__main__':
    # 配置
    config_path = 'datasets/MY_TEST_DATA/data.yaml'
    img_size = 640
    batch_size = 1
    epochs = 2  # 测试用，少量 epochs
    lr = 0.001
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"使用设备: {device}")

    # 加载数据集配置
    config = load_yaml_config(config_path)
    print(f"类别数: {config['nc']}")
    print(f"类别名称: {config['names']}")

    # 创建数据加载器
    print("\n加载数据集...")
    train_loader, val_loader, _ = create_dataloaders(
        config_path=config_path,
        batch_size=batch_size,
        img_size=img_size,
        workers=0
    )

    # 创建 YOLO + CoordAtt 检测器
    print(f"\n创建 YOLO + CoordAtt 检测器 (nc={config['nc']})...")
    model = YOLOCoordAttDetector(nc=config['nc']).to(device)

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")

    # 训练模型 (使用 engine.detector.train_detector)
    train_detector(model, train_loader, val_loader,
                   epochs=epochs, lr=lr, device=device,
                   save_dir=OUTPUT_DIR, patience=15)

    # 训练后可视化
    print("\n" + "=" * 50)
    print("训练后注意力可视化 (已学习):")
    print("=" * 50)

    visualize_detection_attention(
        model, val_loader, device,
        save_path=os.path.join(OUTPUT_DIR, 'detection_attention.png'),
        img_size=img_size
    )

    # 创建对比图
    visualize_attention_comparison(
        model, val_loader, device,
        save_path=os.path.join(OUTPUT_DIR, 'attention_comparison.png'),
        img_size=img_size
    )

    print("\n完成!")
