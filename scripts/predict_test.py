#!/usr/bin/env python
"""
YOLOv11 预测测试脚本

使用训练好的模型对图像进行目标检测。
使用新的统一 YOLO 接口。
"""
import os
import sys
import cv2
import numpy as np
from pathlib import Path

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import YOLO


def main():
    # 模型权重路径
    weights_path = 'runs/train/test_exp_9/last.pt'

    # 测试图片路径
    image_path = 'datasets/MY_TEST_DATA/images/val/circle_0023.jpg'

    # 验证文件存在
    if not Path(weights_path).exists():
        print(f"错误: 权重文件不存在: {weights_path}")
        print("请先运行训练脚本生成权重文件")
        return 1

    if not Path(image_path).exists():
        print(f"错误: 图片不存在: {image_path}")
        return 1

    print("=" * 60)
    print("YOLOv11 预测测试（新版统一接口）")
    print("=" * 60)
    print(f"权重文件: {weights_path}")
    print(f"测试图片: {image_path}")
    print("=" * 60)

    # 加载模型（使用新的统一 YOLO 接口）
    print("\n加载模型...")
    model = YOLO(weights_path)
    print(f"✓ 模型加载成功")

    # 读取图片
    img = cv2.imread(image_path)
    if img is None:
        print(f"错误: 无法读取图片: {image_path}")
        return 1

    print(f"图片尺寸: {img.shape}")

    # 执行预测
    print("\n开始预测...")
    results = model.predict(
        img,
        conf=0.25,
        iou=0.45
    )

    # 打印结果
    print("\n" + "=" * 60)
    print("预测结果")
    print("=" * 60)

    if len(results) > 0:
        r = results[0]
        boxes = r.boxes
        print(f"检测到 {len(boxes)} 个目标")

        if len(boxes) > 0:
            print("\n详细信息:")
            for i in range(len(boxes)):
                box = boxes.xyxy[i]  # xyxy 格式
                conf = boxes.conf[i]
                cls = int(boxes.cls[i])
                cls_name = r.names.get(cls, f"cls{cls}")

                print(f"  [{i}] {cls_name}:")
                print(f"      置信度: {conf:.3f}")
                print(f"      边界框: x1={box[0]:.1f}, y1={box[1]:.1f}, x2={box[2]:.1f}, y2={box[3]:.1f}")

            # 可视化结果
            print("\n生成可视化结果...")
            vis_img = img.copy()

            for i in range(len(boxes)):
                box = boxes.xyxy[i].astype(int)
                conf = boxes.conf[i]
                cls = int(boxes.cls[i])
                cls_name = r.names.get(cls, f"cls{cls}")

                # 绘制边界框
                cv2.rectangle(vis_img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

                # 绘制标签
                label = f"{cls_name}: {conf:.2f}"
                (label_width, label_height), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                cv2.rectangle(
                    vis_img,
                    (box[0], box[1] - label_height - 10),
                    (box[0] + label_width, box[1]),
                    (0, 255, 0),
                    -1
                )
                cv2.putText(
                    vis_img,
                    label,
                    (box[0], box[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    1
                )

            # 保存结果
            save_dir = Path('runs/predict/test')
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / 'circle_0023_result.jpg'
            cv2.imwrite(str(save_path), vis_img)
            print(f"✓ 可视化结果已保存到: {save_path}")
        else:
            print("未检测到任何目标")
    else:
        print("未返回任何结果")

    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    exit(main())
