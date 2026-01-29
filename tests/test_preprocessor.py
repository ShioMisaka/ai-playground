"""测试 Preprocessor"""
import pytest
import numpy as np
import torch
from engine.preprocessor import Preprocessor


def test_preprocessor_letterbox():
    """测试 letterbox 预处理"""
    preprocessor = Preprocessor(img_size=640, letterbox=True)

    # 创建测试图像 (非正方形)
    img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    img_tensor, params = preprocessor(img)

    # 验证输出形状
    assert img_tensor.shape == (1, 3, 640, 640)

    # 验证归一化
    assert img_tensor.min() >= 0.0
    assert img_tensor.max() <= 1.0

    # 验证参数
    assert params['letterbox'] == True
    assert 'ratio' in params
    assert 'pad' in params
    assert params['orig_shape'] == (480, 640)


def test_preprocessor_simple_resize():
    """测试简单 resize 预处理"""
    preprocessor = Preprocessor(img_size=640, letterbox=False)

    img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    img_tensor, params = preprocessor(img)

    # 验证输出形状
    assert img_tensor.shape == (1, 3, 640, 640)

    # 验证参数
    assert params['letterbox'] == False
    assert 'scale_x' in params
    assert 'scale_y' in params


def test_preprocessor_auto_mode():
    """测试动态模式"""
    preprocessor = Preprocessor(img_size=640, letterbox=True, auto=True)

    img = np.random.randint(0, 255, (800, 600, 3), dtype=np.uint8)

    img_tensor, params = preprocessor(img)

    # 动态模式：目标尺寸为最长边
    assert img_tensor.shape == (1, 3, 800, 800)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
