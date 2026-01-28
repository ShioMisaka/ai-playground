"""
YOLO + CoordAtt 训练脚本

训练带有 Coordinate Attention 的 YOLO 检测器并可视化注意力效果
"""
import os
import sys
import torch
from pathlib import Path
from datetime import datetime

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import YOLOCoordAttDetector
from engine import train_detector, visualize_detection_attention, visualize_attention_comparison
from utils import create_dataloaders
from utils.load import load_yaml_config


def main():
    # 配置
    config_path = 'datasets/MY_TEST_DATA/data.yaml'
    img_size = 640
    batch_size = 8
    epochs = 25
    lr = 0.001
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path('outputs') / f'yolo_coordatt_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"输出目录: {output_dir}")

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

    # 训练模型
    train_detector(model, train_loader, val_loader,
                   epochs=epochs, lr=lr, device=device,
                   save_dir=str(output_dir), patience=15)

    # 训练后可视化
    print("\n" + "=" * 50)
    print("训练后注意力可视化:")
    print("=" * 50)

    visualize_detection_attention(
        model, val_loader, device,
        save_path=str(output_dir / 'detection_attention.png'),
        img_size=img_size
    )

    # 创建对比图
    visualize_attention_comparison(
        model, val_loader, device,
        save_path=str(output_dir / 'attention_comparison.png'),
        img_size=img_size
    )

    print(f"\n所有输出已保存到: {output_dir}")
    print("完成!")


if __name__ == '__main__':
    main()
