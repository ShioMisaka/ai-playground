#!/usr/bin/env python
"""
YOLOv11 预测脚本

使用训练好的模型对图像或视频进行目标检测，并保存标注结果。

用法:
    # 预测单张图片
    python scripts/predict.py --weights runs/train/exp/best.pt --source image.jpg

    # 预测目录中的所有图片
    python scripts/predict.py --weights runs/train/exp/best.pt --source images/

    # 预测视频
    python scripts/predict.py --weights runs/train/exp/best.pt --source video.mp4

    # 自定义置信度阈值和保存路径
    python scripts/predict.py --weights runs/train/exp/best.pt --source image.jpg \\
        --conf 0.3 --save-dir runs/predict/custom
"""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="YOLOv11 目标检测预测脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # 必填参数
    parser.add_argument(
        "--weights", "-w",
        type=str,
        required=True,
        help="模型权重文件路径 (.pt)"
    )

    parser.add_argument(
        "--source", "-s",
        type=str,
        required=True,
        help="输入源：图片路径、图片目录或视频文件"
    )

    # 可选参数
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="置信度阈值 (默认: 0.25)"
    )

    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="NMS IoU 阈值 (默认: 0.45)"
    )

    parser.add_argument(
        "--img-size",
        type=int,
        default=640,
        help="目标图像尺寸，None 为动态模式 (默认: 640)"
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="设备 (cuda:0, cpu，None 自动选择)"
    )

    parser.add_argument(
        "--save-dir",
        type=str,
        default="runs/predict",
        help="保存目录 (默认: runs/predict)"
    )

    parser.add_argument(
        "--no-save",
        action="store_true",
        help="不保存结果，仅打印检测信息"
    )

    parser.add_argument(
        "--auto",
        action="store_true",
        help="动态 letterbox 模式（根据输入图像尺寸自动调整）"
    )

    parser.add_argument(
        "--simple-resize",
        action="store_true",
        help="使用直接 resize（匹配训练时的处理，不保持长宽比）"
    )

    args = parser.parse_args()

    # 导入 YOLO（添加项目根目录到路径）
    import sys
    from pathlib import Path as SysPath
    sys.path.insert(0, str(SysPath(__file__).parent.parent))
    from engine.predict import YOLO

    # 验证权重文件
    weights_path = Path(args.weights)
    if not weights_path.exists():
        print(f"错误: 权重文件不存在: {weights_path}")
        return 1

    # 验证输入源
    source_path = Path(args.source)
    if not source_path.exists():
        print(f"错误: 输入源不存在: {source_path}")
        return 1

    # 打印配置
    print("=" * 60)
    print("YOLOv11 预测")
    print("=" * 60)
    print(f"权重文件: {weights_path}")
    print(f"输入源: {source_path}")
    print(f"置信度阈值: {args.conf}")
    print(f"NMS IoU 阈值: {args.iou}")
    print(f"目标尺寸: {args.img_size if not args.auto else '动态'}")
    print(f"设备: {args.device or '自动'}")
    print(f"保存目录: {args.save_dir}")
    print("=" * 60)

    # 加载模型
    print("\n加载模型...")
    model = YOLO(
        str(weights_path),
        device=args.device,
        conf=args.conf,
        iou=args.iou,
        img_size=None if args.auto else args.img_size,
        simple_resize=args.simple_resize
    )
    print(f"✓ 模型加载成功 (类别数: {model.nc})")
    if args.simple_resize:
        print(f"  模式: 直接 resize (匹配训练时的处理)")
    else:
        print(f"  模式: Letterbox (保持长宽比)")

    # 执行预测
    print(f"\n开始预测...")
    results = model.predict(
        str(source_path),
        conf=args.conf,
        iou=args.iou,
        img_size=None if args.auto else args.img_size,
        save=not args.no_save,
        save_dir=args.save_dir
    )

    # 打印结果统计
    print("\n" + "=" * 60)
    print("预测完成")
    print("=" * 60)
    print(f"处理数量: {len(results)}")

    total_detections = sum(len(r.boxes) for r in results)
    print(f"总检测框数: {total_detections}")

    if len(results) > 0:
        # 打印每个结果的检测数
        for i, r in enumerate(results):
            if len(r.boxes) > 0:
                print(f"  结果 {i}: {len(r.boxes)} 个检测, 形状={r.orig_shape}")

                # 打印前 3 个检测
                for j in range(min(3, len(r.boxes))):
                    box = r.boxes.xyxy[j]
                    conf = r.boxes.conf[j]
                    cls = int(r.boxes.cls[j])
                    cls_name = r.names.get(cls, f"cls{cls}")
                    print(f"    [{j}] {cls_name}: conf={conf:.3f}, box={box.astype(int)}")

                if len(r.boxes) > 3:
                    print(f"    ... 还有 {len(r.boxes) - 3} 个检测")

    if not args.no_save:
        print(f"\n结果已保存到: {args.save_dir}")

    return 0


if __name__ == "__main__":
    exit(main())
