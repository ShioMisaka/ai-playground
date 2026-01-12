"""
注意力机制对比训练脚本 - 对比 CoordAtt 和 CoordCrossAtt

训练两个模型并对比它们在 YOLO 检测任务上的表现
"""
import os
import sys
import torch
from pathlib import Path
from datetime import datetime

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import YOLOCoordAttDetector, YOLOCoordCrossAttDetector
from engine import (train_and_compare_models, print_comparison_results,
                    visualize_single_model_attention, visualize_model_comparison,
                    visualize_cross_attention_matrix, visualize_training_progress)
from utils import create_dataloaders
from utils.load import load_yaml_config


def main():
    # 配置
    config_path = 'datasets/MY_TEST_DATA/data.yaml'
    img_size = 256
    batch_size = 8
    epochs = 15
    lr = 0.001
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path('outputs/attention_comparison') / f'run_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("CoordAtt vs CoordCrossAtt 注意力机制对比 (YOLO 检测)")
    print("=" * 60)
    print(f"\n配置:")
    print(f"  - 设备: {device.upper()}")
    print(f"  - 图像尺寸: {img_size}x{img_size}")
    print(f"  - 批大小: {batch_size}")
    print(f"  - 训练轮数: {epochs}")
    print(f"  - 学习率: {lr}")
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

    # 定义模型字典
    model_dict = {
        'CoordAtt': (YOLOCoordAttDetector, {'nc': config['nc']}),
        'CoordCrossAtt': (YOLOCoordCrossAttDetector, {'nc': config['nc'], 'num_heads': 1}),
    }

    # 训练并对比模型
    results = train_and_compare_models(
        model_dict, train_loader, val_loader,
        epochs=epochs, lr=lr, device=device,
        save_dir=str(output_dir), patience=10
    )

    # 打印对比结果
    print_comparison_results(results)

    # 可视化对比
    print("\n生成可视化对比...")

    coordatt_model = results['CoordAtt']['model']
    crossatt_model = results['CoordCrossAtt']['model']

    # ========== 单个模型详细可视化 ==========
    print("\n--- 生成 CoordAtt 独立可视化 ---")
    visualize_single_model_attention(
        coordatt_model, 'CoordAtt (Coordinate Attention)',
        val_loader, device,
        save_path=str(output_dir / 'coordatt_detail.png'),
        img_size=img_size
    )

    print("\n--- 生成 CoordCrossAtt 独立可视化 ---")
    visualize_single_model_attention(
        crossatt_model, 'CoordCrossAtt (Coordinate Cross Attention)',
        val_loader, device,
        save_path=str(output_dir / 'crossatt_detail.png'),
        img_size=img_size
    )

    # ========== 模型对比可视化 ==========
    visualize_model_comparison(
        coordatt_model, crossatt_model, val_loader, device,
        save_path=str(output_dir / 'attention_comparison.png'),
        img_size=img_size
    )

    # Cross-Attention 相关性矩阵
    visualize_cross_attention_matrix(
        crossatt_model, val_loader, device,
        save_path=str(output_dir / 'cross_attention_matrix.png'),
        img_size=img_size
    )

    # 训练进度对比
    history_dict = {
        'CoordAtt': results['CoordAtt']['history'],
        'CoordCrossAtt': results['CoordCrossAtt']['history'],
    }
    visualize_training_progress(
        history_dict,
        save_path=str(output_dir / 'training_progress.png')
    )

    print(f"\n{'=' * 60}")
    print(f"所有输出已保存到: {output_dir}")
    print(f"{'=' * 60}")
    print("\n生成的可视化文件:")
    print("  - coordatt_detail_sample*.png      CoordAtt 各样本注意力详情")
    print("  - crossatt_detail_sample*.png      CoordCrossAtt 各样本注意力详情")
    print("  - attention_comparison.png         两种注意力机制对比")
    print("  - cross_attention_matrix.png       Cross-Attention 相关性矩阵")
    print("  - training_progress.png            训练进度对比")
    print("\n完成!")


if __name__ == '__main__':
    main()
