"""
YOLOv11 预测模块测试

测试 letterbox 预处理、坐标映射、模型推理等功能。
"""
import sys
import torch
import numpy as np
import cv2
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.predict import LetterBox, _scale_coords, _post_process, YOLO


def test_letterbox():
    """测试 LetterBox 预处理"""
    print("\n=== 测试 LetterBox ===")

    # 创建测试图像 (720x1280, 16:9)
    img = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)

    letterbox = LetterBox(auto=False)
    transformed, params = letterbox(img, target_size=640)

    ratio, (pad_x, pad_y) = params

    print(f"原始图像尺寸: {img.shape[:2]}")
    print(f"变换后尺寸: {transformed.shape[:2]}")
    print(f"缩放比例: {ratio:.4f}")
    print(f"Padding: ({pad_x:.2f}, {pad_y:.2f})")

    # 验证变换后是正方形
    assert transformed.shape[0] == 640, f"高度应为 640, 实际为 {transformed.shape[0]}"
    assert transformed.shape[1] == 640, f"宽度应为 640, 实际为 {transformed.shape[1]}"

    # 验证缩放比例 (640/1280 = 0.5, 640/720 = 0.888..., min is 0.5)
    expected_ratio = min(640 / 720, 640 / 1280)
    assert abs(ratio - expected_ratio) < 0.01, f"缩放比例应为 {expected_ratio}, 实际为 {ratio}"

    print("✓ LetterBox 测试通过")


def test_scale_coords():
    """测试坐标映射"""
    print("\n=== 测试坐标映射 ===")

    # 模拟场景：1280x720 图像，letterbox 到 640x640
    orig_h, orig_w = 720, 1280
    target_size = 640

    # 计算 letterbox 参数
    ratio = min(target_size / orig_h, target_size / orig_w)  # 0.5
    scaled_h, scaled_w = int(orig_h * ratio), int(orig_w * ratio)  # 360, 640
    pad_x = (target_size - scaled_w) / 2  # 0
    pad_y = (target_size - scaled_h) / 2  # 140

    print(f"原始尺寸: ({orig_w}, {orig_h})")
    print(f"缩放后尺寸: ({scaled_w}, {scaled_h})")
    print(f"Padding: ({pad_x}, {pad_y})")

    # 创建测试边界框 (在 640x640 空间)
    # 假设在 640x640 空间中检测到一个框: cx=320, cy=320, w=100, h=100
    coords_letterbox = np.array([[320, 320, 100, 100]], dtype=np.float32)

    print(f"\nLetterbox 空间的框: {coords_letterbox[0]}")

    # 映射回原图空间
    coords_original = _scale_coords(
        coords_letterbox,
        (orig_h, orig_w),
        ratio,
        (pad_x, pad_y)
    )

    print(f"原图空间的框: {coords_original[0]}")

    # 验证映射
    # cx: (320 - 0) / 0.5 = 640
    # cy: (320 - 140) / 0.5 = 360
    # w: 100 / 0.5 = 200
    # h: 100 / 0.5 = 200
    expected_cx = (320 - pad_x) / ratio
    expected_cy = (320 - pad_y) / ratio
    expected_w = 100 / ratio
    expected_h = 100 / ratio

    np.testing.assert_almost_equal(coords_original[0, 0], expected_cx, decimal=1)
    np.testing.assert_almost_equal(coords_original[0, 1], expected_cy, decimal=1)
    np.testing.assert_almost_equal(coords_original[0, 2], expected_w, decimal=1)
    np.testing.assert_almost_equal(coords_original[0, 3], expected_h, decimal=1)

    print("✓ 坐标映射测试通过")


def test_model_loading():
    """测试模型加载"""
    print("\n=== 测试模型加载 ===")

    # 查找可用的权重文件
    weights_paths = [
        "runs/train/test_exp_2/best.pt",
        "runs/outputs/train/exp_2/best.pt",
        "runs/outputs/train/exp1v11/best.pt",
    ]

    weights_path = None
    for path in weights_paths:
        if Path(path).exists():
            weights_path = path
            break

    if weights_path is None:
        print("⚠ 跳过模型加载测试：未找到权重文件")
        return

    print(f"使用权重文件: {weights_path}")

    try:
        model = YOLO(weights_path, device="cpu")
        print(f"✓ 模型加载成功")
        print(f"  - 类别数: {model.nc}")
        print(f"  - 设备: {model.device}")
        print(f"  - 模型: {model}")

    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
        raise


def test_prediction():
    """测试完整预测流程"""
    print("\n=== 测试预测流程 ===")

    # 查找可用的权重文件和测试图像
    weights_paths = [
        "runs/train/test_exp_2/best.pt",
        "runs/outputs/train/exp_2/best.pt",
        "runs/outputs/train/exp1v11/best.pt",
    ]

    test_images = [
        "datasets/MY_TEST_DATA/images/test/circle_0014.jpg",
        "datasets/MY_TEST_DATA/images/test/square_0060.jpg",
    ]

    weights_path = None
    for path in weights_paths:
        if Path(path).exists():
            weights_path = path
            break

    test_image = None
    for path in test_images:
        if Path(path).exists():
            test_image = path
            break

    if weights_path is None or test_image is None:
        print("⚠ 跳过预测测试：未找到权重文件或测试图像")
        return

    print(f"使用权重文件: {weights_path}")
    print(f"使用测试图像: {test_image}")

    try:
        # 加载模型
        model = YOLO(weights_path, device="cpu", conf=0.25)

        # 执行预测
        results = model.predict(test_image, conf=0.25, save=False)

        print(f"✓ 预测成功")
        print(f"  - 结果数量: {len(results)}")
        print(f"  - 检测框数量: {len(results[0].boxes)}")
        print(f"  - 原始图像形状: {results[0].orig_shape}")

        # 如果有检测结果，打印前 3 个
        if len(results[0].boxes) > 0:
            print(f"\n  前 {min(3, len(results[0].boxes))} 个检测结果:")
            for i in range(min(3, len(results[0].boxes))):
                box = results[0].boxes.xyxy[i]
                conf = results[0].boxes.conf[i]
                cls = results[0].boxes.cls[i]
                print(f"    [{i}] box={box}, conf={conf:.3f}, cls={cls}")

    except Exception as e:
        print(f"✗ 预测失败: {e}")
        import traceback
        traceback.print_exc()
        raise


def main():
    """运行所有测试"""
    print("=" * 60)
    print("YOLOv11 预测模块测试")
    print("=" * 60)

    test_letterbox()
    test_scale_coords()
    test_model_loading()
    test_prediction()

    print("\n" + "=" * 60)
    print("所有测试通过!")
    print("=" * 60)


if __name__ == "__main__":
    main()
