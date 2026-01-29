"""测试训练和推理的一致性"""
import torch
import numpy as np
from models.yolov11 import YOLOv11
from engine.preprocessor import Preprocessor
from engine.postprocessor import Postprocessor


def test_preprocessing_consistency():
    """测试预处理一致性"""
    print("测试预处理一致性...")

    # 创建测试图像
    img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # 训练时使用的预处理器
    train_preprocessor = Preprocessor(img_size=640, letterbox=True)

    # 推理时使用的预处理器（通过 YOLO 类）
    # 应该完全相同
    infer_preprocessor = Preprocessor(img_size=640, letterbox=True)

    train_tensor, train_params = train_preprocessor(img)
    infer_tensor, infer_params = infer_preprocessor(img)

    # 验证输出相同
    assert torch.allclose(train_tensor, infer_tensor), "预处理输出不一致"
    assert train_params == infer_params, "预处理参数不一致"

    print("  ✓ 预处理一致性测试通过")


def test_model_output_format():
    """测试模型输出格式"""
    print("测试模型输出格式...")

    model = YOLOv11(nc=2, scale='n')
    model.eval()

    # 创建测试输入
    imgs = torch.randn(2, 3, 640, 640)

    # 推理模式
    with torch.no_grad():
        predictions = model(imgs)

    # 验证输出格式
    assert isinstance(predictions, torch.Tensor), "推理输出应该是张量"
    assert predictions.shape[0] == 2, "batch size 应该是 2"
    assert predictions.shape[2] == 6, "应该是 4 + 2 类"

    print("  ✓ 模型输出格式测试通过")


def test_postprocessing_consistency():
    """测试后处理一致性"""
    print("测试后处理一致性...")

    # 创建模拟预测
    predictions = torch.rand(1, 100, 6)
    predictions[0, :5, 4:] = 0.9  # 高置信度

    preprocess_params = {
        'letterbox': True,
        'ratio': 640 / 480,
        'pad': (0.0, 0.0)
    }

    postprocessor = Postprocessor(conf_threshold=0.25, iou_threshold=0.45)

    result = postprocessor(predictions, (480, 640), preprocess_params)

    # 验证返回格式
    assert 'boxes' in result
    assert 'scores' in result
    assert 'labels' in result

    print("  ✓ 后处理一致性测试通过")


def test_yolo_interface():
    """测试 YOLO 接口"""
    print("测试 YOLO 接口...")

    # 创建模型（从权重或配置）
    # 这里用模型实例
    from models.yolo import YOLO
    model = YOLO(YOLOv11(nc=2, scale='n'))

    # 创建测试图像
    img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # 推理
    results = model.predict(img, conf=0.25)

    # 验证返回
    assert len(results) == 1
    assert hasattr(results[0], 'boxes')

    print("  ✓ YOLO 接口测试通过")


if __name__ == '__main__':
    print("=" * 50)
    print("运行一致性测试...")
    print("=" * 50)

    test_preprocessing_consistency()
    test_model_output_format()
    test_postprocessing_consistency()
    test_yolo_interface()

    print("=" * 50)
    print("所有测试通过！")
    print("=" * 50)
