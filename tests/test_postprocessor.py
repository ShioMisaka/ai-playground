"""测试 Postprocessor"""
import pytest
import torch
import numpy as np
from engine.postprocessor import Postprocessor


def test_postprocessor_basic():
    """测试基本后处理流程"""
    postprocessor = Postprocessor(conf_threshold=0.25, iou_threshold=0.45)

    # 创建模拟预测输出
    # (1, 100, 7) = (bs, n_anchors, 4+nc) 假设 3 类
    predictions = torch.rand(1, 100, 7)
    # 设置高置信度
    predictions[0, :5, 4:] = 0.9  # 前 5 个高置信度

    orig_shape = (480, 640)
    preprocess_params = {
        'letterbox': False,
        'scale_x': 640 / 640,
        'scale_y': 640 / 480,
        'pad': (0.0, 0.0)
    }

    result = postprocessor(predictions, orig_shape, preprocess_params)

    # 验证返回格式
    assert 'boxes' in result
    assert 'scores' in result
    assert 'labels' in result
    assert result['boxes'].shape[0] == result['scores'].shape[0]


def test_postprocessor_empty():
    """测试空检测结果"""
    postprocessor = Postprocessor(conf_threshold=0.9, iou_threshold=0.45)

    # 低置信度预测
    predictions = torch.rand(1, 100, 7) * 0.5

    result = postprocessor(predictions, (480, 640), {
        'letterbox': False,
        'scale_x': 1.0,
        'scale_y': 1.0,
        'pad': (0.0, 0.0)
    })

    # 应该返回空结果
    assert len(result['boxes']) == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
