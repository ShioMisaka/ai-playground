#!/usr/bin/env python
"""
调试预测脚本 - 检查模型状态和输出
"""
import sys
import torch
import numpy as np
import cv2
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.predict import LetterBox
from models import YOLOv11


def debug_prediction():
    """调试预测流程"""

    # 加载图像
    img_path = "datasets/MY_TEST_DATA/images/train/circle_0007.jpg"
    img = cv2.imread(img_path)
    orig_h, orig_w = img.shape[:2]
    print(f"原始图像: {img.shape} (H={orig_h}, W={orig_w})")

    # Letterbox 预处理
    letterbox = LetterBox(auto=False)
    transformed, (ratio, (pad_x, pad_y)) = letterbox(img, target_size=640)
    print(f"变换后: {transformed.shape}")
    print(f"参数: ratio={ratio}, pad=({pad_x}, {pad_y})")

    # 转换为 Tensor
    img_tensor = torch.from_numpy(transformed).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0)
    print(f"Tensor shape: {img_tensor.shape}")

    # 加载模型
    print("\n=== 加载模型 ===")
    ckpt = torch.load("runs/train/test_exp_2/best.pt", map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)

    # 推断参数
    nc = 2
    model = YOLOv11(nc=nc, scale="n")

    # 加载权重
    model.load_state_dict(state_dict, strict=False)

    # 检查训练模式
    print(f"模型 training 模式: {model.training}")
    print(f"Detect 层 training 模式: {model.detect.training}")

    # 设置为 eval 模式
    model.eval()
    print(f"设置 eval 后:")
    print(f"  模型 training 模式: {model.training}")
    print(f"  Detect 层 training 模式: {model.detect.training}")

    # 推理
    print("\n=== 推理 ===")
    with torch.no_grad():
        predictions = model(img_tensor)

    print(f"输出类型: {type(predictions)}")

    if isinstance(predictions, tuple):
        pred_output, detail = predictions
        print(f"预测输出 shape: {pred_output.shape}")
        print(f"Detail keys: {detail.keys() if isinstance(detail, dict) else type(detail)}")
    else:
        pred_output = predictions
        print(f"预测输出 shape: {pred_output.shape}")

    # 检查预测值
    pred_single = pred_output[0]  # (n_anchors, 4+nc)
    print(f"\n单张图像预测 shape: {pred_single.shape}")

    # 提取类别分数
    cls_scores = pred_single[:, 4:]  # (n_anchors, nc)
    scores, labels = torch.max(cls_scores, dim=1)

    print(f"\n=== 置信度分析 ===")
    print(f"最大置信度: {scores.max().item():.4f}")
    print(f"置信度 > 0.5 的数量: {(scores > 0.5).sum().item()}")
    print(f"置信度 > 0.25 的数量: {(scores > 0.25).sum().item()}")
    print(f"置信度 > 0.1 的数量: {(scores > 0.1).sum().item()}")

    # 打印 top 10 预测
    top_values, top_indices = torch.topk(scores, min(10, len(scores)))
    print(f"\nTop 10 预测:")
    for i, (score, idx) in enumerate(zip(top_values, top_indices)):
        box = pred_single[idx, :4]
        print(f"  [{i}] score={score:.4f}, cls={labels[idx].item()}, box={box.tolist()}")


if __name__ == "__main__":
    debug_prediction()
