"""
ImprovedCoordCrossAtt 模型可视化脚本（无需训练）

直接加载已训练的 pth 文件，生成双向注意力可视化。
"""
import os
import sys
import torch
from pathlib import Path

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import YOLOBiCoordCrossAttDetector
from engine import (visualize_improved_cross_attention,
                    visualize_improved_cross_attention_overlay)
from utils import create_dataloaders
from utils.load import load_yaml_config


def main():
    # 配置
    config_path = 'datasets/MY_TEST_DATA/data.yaml'
    img_size = 256

    # 模型路径（可以修改为你的模型路径）
    default_model_path = 'outputs/improved_coordcrossatt/run_20260114_105748/best_model.pth'

    # 查找最新的模型文件
    model_dir = Path('outputs/improved_coordcrossatt')
    if model_dir.exists():
        run_dirs = sorted(model_dir.glob('run_*'))
        if run_dirs:
            latest_run = run_dirs[-1]
            model_path = latest_run / 'best_model.pth'
            if model_path.exists():
                default_model_path = str(model_path)

    print("=" * 60)
    print("ImprovedCoordCrossAtt 双向注意力可视化")
    print("=" * 60)

    # 让用户输入模型路径
    user_input = input(f"\n请输入模型路径 (直接回车使用默认路径):\n{default_model_path}\n> ").strip()

    if user_input:
        model_path = user_input
    else:
        model_path = default_model_path

    model_path = Path(model_path)

    if not model_path.exists():
        print(f"\n错误: 模型文件不存在: {model_path}")
        print("\n可用的模型文件:")
        if model_dir.exists():
            for run_dir in sorted(model_dir.glob('run_*')):
                pth_file = run_dir / 'best_model.pth'
                if pth_file.exists():
                    print(f"  - {pth_file}")
        return

    print(f"\n使用模型: {model_path}")

    # 设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"设备: {device.upper()}")

    # 加载数据集配置
    config = load_yaml_config(config_path)
    print(f"\n数据集: {config['names']}")
    print(f"类别数: {config['nc']}")

    # 加载数据
    print("\n加载数据集...")
    _, val_loader, _ = create_dataloaders(
        config_path=config_path,
        batch_size=4,
        img_size=img_size,
        workers=0
    )

    # 加载模型
    print("\n加载模型...")
    checkpoint = torch.load(model_path, weights_only=False, map_location=device)

    # 获取 num_heads（从模型配置或使用默认值）
    num_heads = checkpoint.get('num_heads', 4)
    print(f"注意力头数: {num_heads}")

    # 创建模型并加载权重
    model = YOLOBiCoordCrossAttDetector(nc=config['nc'], num_heads=num_heads)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"最佳验证 Loss: {checkpoint.get('best_loss', 'N/A')}")
    print(f"训练轮数: {checkpoint.get('epoch', 'N/A')}")

    # 输出目录
    output_dir = model_path.parent
    print(f"\n输出目录: {output_dir}")

    # 生成可视化
    print("\n" + "=" * 60)
    print("生成可视化...")
    print("=" * 60)

    # 1. 双向注意力矩阵可视化
    print("\n--- 生成双向注意力矩阵可视化 ---")
    visualize_improved_cross_attention(
        model, val_loader, device,
        save_path=str(output_dir / 'bidirectional_attention_matrix.png'),
        img_size=img_size
    )

    # 2. 注意力叠加可视化
    print("\n--- 生成注意力叠加可视化 ---")
    visualize_improved_cross_attention_overlay(
        model, val_loader, device,
        save_path=str(output_dir / 'attention_overlay.png'),
        img_size=img_size
    )

    print(f"\n{'=' * 60}")
    print(f"可视化完成！文件已保存到: {output_dir}")
    print(f"{'=' * 60}")
    print("\n生成的文件:")
    print(f"  - {output_dir / 'bidirectional_attention_matrix.png'}")
    print(f"  - {output_dir / 'attention_overlay.png'}")


if __name__ == '__main__':
    main()
