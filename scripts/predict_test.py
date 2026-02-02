#!/usr/bin/env python
"""
YOLOv11 预测测试脚本（极简版）

使用训练好的模型对测试图像进行目标检测，并生成 3x3 拼图展示。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models import YOLO


def main():
    # 配置
    weights_path = 'runs/train/test_exp/best.pt'
    data_config = 'datasets/MY_TEST_DATA/data.yaml'
    test_images_dir = 'datasets/MY_TEST_DATA/images/test'
    output_dir = 'runs/train/test_exp/test'
    grid_save_path = 'runs/train/test_exp/test/grid.jpg'

    print("=" * 60)
    print("YOLOv11 批量预测测试")
    print("=" * 60)
    print(f"权重: {weights_path}")
    print(f"数据: {data_config}")
    print(f"测试集: {test_images_dir}")
    print("=" * 60)

    # 加载模型
    model = YOLO(
        weights_path=weights_path,
        data_config=data_config
    )
    print(f"✓ 模型加载成功 | 类别: {model.nc} | 设备: {model.device}")

    # 批量预测
    print(f"\n开始预测...")
    results = model.predict(
        source=test_images_dir,
        conf=0.25,
        iou=0.45,
        save=True,
        save_dir=output_dir
    )

    # 统计
    total_detections = sum(len(r.boxes) for r in results)
    print(f"✓ 预测完成 | 图片: {len(results)} | 检测: {total_detections}")

    # 生成 3x3 拼图
    print(f"\n生成 3x3 拼图...")
    model.visualize_grid(
        save_path=grid_save_path,
        num_samples=9,
        source_dir=test_images_dir
    )

    print(f"\n完成！结果保存在: {output_dir}/")
    return 0


if __name__ == "__main__":
    exit(main())
