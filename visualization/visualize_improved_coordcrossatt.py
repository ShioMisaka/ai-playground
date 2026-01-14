"""
ImprovedCoordCrossAtt 训练 + 双向注意力可视化脚本

训练带 ImprovedCoordCrossAtt 的 YOLO 检测器，并可视化其双向注意力机制。
"""
import os
import sys
import torch
from pathlib import Path
from datetime import datetime

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import YOLOBiCoordCrossAttDetector
from engine import (train_detector, visualize_improved_cross_attention,
                    visualize_improved_cross_attention_overlay)
from utils import create_dataloaders
from utils.load import load_yaml_config


def main():
    # 配置
    config_path = 'datasets/MY_TEST_DATA/data.yaml'
    img_size = 256
    batch_size = 8
    epochs = 30
    lr = 0.001
    num_heads = 4  # ImprovedCoordCrossAtt 建议使用更多头
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path('outputs/improved_coordcrossatt') / f'run_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("ImprovedCoordCrossAtt 双向注意力机制训练与可视化")
    print("=" * 60)
    print(f"\n配置:")
    print(f"  - 设备: {device.upper()}")
    print(f"  - 图像尺寸: {img_size}x{img_size}")
    print(f"  - 批大小: {batch_size}")
    print(f"  - 训练轮数: {epochs}")
    print(f"  - 学习率: {lr}")
    print(f"  - 注意力头数: {num_heads}")
    print(f"  - 输出目录: {output_dir}")

    # 加载数据集配置
    config = load_yaml_config(config_path)
    print(f"\n数据集配置:")
    print(f"  - 类别数: {config['nc']}")
    print(f"  - 类别名称: {config['names']}")

    # 加载数据
    print("\n加载数据集...")
    train_loader, val_loader, _ = create_dataloaders(
        config_path=config_path,
        batch_size=batch_size,
        img_size=img_size,
        workers=0
    )
    print(f"  - 训练集: {len(train_loader)} 批次")
    print(f"  - 验证集: {len(val_loader)} 批次")

    # 创建模型
    print("\n创建 ImprovedCoordCrossAtt 检测器...")
    model = YOLOBiCoordCrossAttDetector(nc=config['nc'], num_heads=num_heads)
    model = model.to(device)

    print("\n模型架构:")
    print("  - Backbone: Darknet-53 with ImprovedCoordCrossAtt")
    print("  - 注意力机制: 对称双向 Cross-Attention")
    print("    * Branch H: 利用宽度信息增强高度特征")
    print("    * Branch W: 利用高度信息增强宽度特征")
    print("  - Neck: FPN (Feature Pyramid Network)")
    print("  - Head: YOLO Detection Head")

    # 训练模型
    print("\n" + "=" * 60)
    print("开始训练...")
    print("=" * 60)

    history = train_detector(
        model, train_loader, val_loader,
        epochs=epochs, lr=lr, device=device,
        save_dir=str(output_dir), patience=10
    )

    model_save_path = output_dir / 'best_model.pth'
    print(f"\n训练完成！最佳模型已保存到: {model_save_path}")

    # 加载最佳模型进行可视化
    print("\n" + "=" * 60)
    print("生成双向注意力可视化...")
    print("=" * 60)

    best_model = YOLOBiCoordCrossAttDetector(nc=config['nc'], num_heads=num_heads)
    checkpoint = torch.load(model_save_path, weights_only=False)
    best_model.load_state_dict(checkpoint['model_state_dict'])
    best_model = best_model.to(device)
    best_model.eval()

    # 1. 双向注意力矩阵可视化
    print("\n--- 生成双向注意力矩阵可视化 ---")
    visualize_improved_cross_attention(
        best_model, val_loader, device,
        save_path=str(output_dir / 'bidirectional_attention_matrix.png'),
        img_size=img_size
    )

    # 2. 注意力叠加可视化
    print("\n--- 生成注意力叠加可视化 ---")
    visualize_improved_cross_attention_overlay(
        best_model, val_loader, device,
        save_path=str(output_dir / 'attention_overlay.png'),
        img_size=img_size
    )

    print(f"\n{'=' * 60}")
    print(f"所有输出已保存到: {output_dir}")
    print(f"{'=' * 60}")
    print("\n生成的可视化文件:")
    print("  - bidirectional_attention_matrix.png  双向注意力矩阵 (H→W & W→H)")
    print("  - attention_overlay.png                注意力叠加效果图")
    print("\nImprovedCoordCrossAtt 特点:")
    print("  - 对称结构: 同时计算 H→W 和 W→H 的注意力")
    print("  - 效率优化: 直接对池化特征进行投影")
    print("  - 双向增强: 两个方向都被平等地增强")
    print("\n完成!")


if __name__ == '__main__':
    main()
