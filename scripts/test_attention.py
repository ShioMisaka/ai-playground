"""
已训练模型的注意力可视化脚本

加载已训练的模型并可视化注意力效果（不进行训练）
"""
import os
import sys
import argparse
from pathlib import Path

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import YOLOCoordAttDetector, YOLOCoordCrossAttDetector
from engine import (load_best_model, visualize_detection_attention,
                    visualize_attention_comparison, load_image,
                    visualize_single_image_attention)
from utils import create_dataloaders
from utils.load import load_yaml_config


# 模型类型映射
MODEL_TYPES = {
    'coordatt': YOLOCoordAttDetector,
    'crossatt': YOLOCoordCrossAttDetector,
}


def test_yolo_attention(model_path, config_path, output_dir='outputs/test',
                        device='cpu', model_type='coordatt'):
    """
    测试 YOLO + Attention 模型的注意力可视化

    Args:
        model_path: 模型检查点路径
        config_path: 数据集配置路径
        output_dir: 输出目录
        device: 设备
        model_type: 模型类型 ('coordatt' 或 'crossatt')
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载数据集配置
    config = load_yaml_config(config_path)
    print(f"类别数: {config['nc']}")
    print(f"类别名称: {config['names']}")

    # 加载数据
    print("\n加载数据集...")
    _, val_loader, _ = create_dataloaders(
        config_path=config_path,
        batch_size=4, img_size=640, workers=0
    )

    # 获取模型类
    model_class = MODEL_TYPES.get(model_type.lower(), YOLOCoordAttDetector)
    print(f"使用模型类型: {model_class.__name__}")

    # 加载最佳模型
    print(f"\n加载模型: {model_path}")
    model = load_best_model(model_class, model_path, device=device, nc=config['nc'])

    # 可视化注意力
    print("\n生成注意力可视化...")

    visualize_detection_attention(
        model, val_loader, device,
        save_path=str(output_dir / 'detection_attention.png'),
        img_size=640
    )

    visualize_attention_comparison(
        model, val_loader, device,
        save_path=str(output_dir / 'attention_comparison.png'),
        img_size=640
    )

    print(f"\n所有输出已保存到: {output_dir}")


def test_single_attention(model_path, image_path, output_dir='outputs/test',
                          device='cpu', model_type='coordatt'):
    """
    测试单张图像的注意力可视化

    Args:
        model_path: 模型检查点路径
        image_path: 测试图像路径
        output_dir: 输出目录
        device: 设备
        model_type: 模型类型 ('coordatt' 或 'crossatt')
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 获取模型类
    model_class = MODEL_TYPES.get(model_type.lower(), YOLOCoordAttDetector)
    print(f"使用模型类型: {model_class.__name__}")

    # 加载模型
    model = load_best_model(model_class, model_path, device=device, nc=2)

    # 加载图像
    img_tensor, img_display = load_image(image_path, img_size=640)

    # 可视化
    visualize_single_image_attention(
        model, img_tensor, img_display,
        save_path=str(output_dir / 'single_attention.png')
    )

    print(f"单图可视化已保存到: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='测试已训练模型的注意力可视化')
    parser.add_argument('--model', type=str, required=True,
                        help='模型检查点路径')
    parser.add_argument('--config', type=str, default='datasets/MY_TEST_DATA/data.yaml',
                        help='数据集配置路径')
    parser.add_argument('--output', type=str, default='outputs/test',
                        help='输出目录')
    parser.add_argument('--device', type=str,
                        default='cuda' if __import__('torch').cuda.is_available() else 'cpu',
                        help='设备 (cuda/cpu)')
    parser.add_argument('--model-type', type=str, default='coordatt',
                        choices=['coordatt', 'crossatt'],
                        help='模型类型 (coordatt/crossatt，默认: coordatt)')
    parser.add_argument('--image', type=str, default=None,
                        help='单张图像测试（指定此选项则只测试单张图像）')

    args = parser.parse_args()

    if args.image:
        test_single_attention(args.model, args.image, args.output, args.device, args.model_type)
    else:
        test_yolo_attention(args.model, args.config, args.output, args.device, args.model_type)


if __name__ == '__main__':
    main()
