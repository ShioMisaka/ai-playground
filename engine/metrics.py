"""
评估指标计算模块

提供分类和检测任务的评估指标计算功能。
"""
import torch
import numpy as np
from typing import Optional, Dict, List, Tuple


def compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """计算平均精度 (Average Precision)

    使用 11 点插值法计算 AP

    Args:
        recall: 召回率数组
        precision: 精确率数组

    Returns:
        AP 值
    """
    # 11 点插值
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        p = np.max(precision[recall >= t]) if np.any(recall >= t) else 0
        ap += p / 11
    return ap


def compute_detection_metrics(predictions: List, targets: torch.Tensor, nc: int,
                             iou_threshold: float = 0.5) -> Dict[str, float]:
    """计算检测任务的评估指标

    Args:
        predictions: 预测结果列表，每个元素为 (batch, anchors, grid_h, grid_w, output_dim)
        targets: 目标张量 [num_boxes, 6] (batch_idx, class_id, x, y, w, h)
        nc: 类别数量
        iou_threshold: IoU 阈值

    Returns:
        包含 mAP, precision, recall 等指标的字典
    """
    # 这是一个简化的实现，实际应用中可能需要更复杂的 NMS 和匹配逻辑
    device = targets.device

    # 收集所有预测和目标
    all_preds = []
    all_gts = []

    batch_size = len(predictions)

    for batch_idx in range(batch_size):
        # 获取当前 batch 的目标
        batch_targets = targets[targets[:, 0] == batch_idx]
        if len(batch_targets) == 0:
            continue

        # 解析预测结果
        pred = predictions[batch_idx]  # (anchors, grid_h, grid_w, output_dim)

        # 简化处理：这里假设预测已经经过适当的处理
        # 实际应用中需要更复杂的解析逻辑

        # 将预测转换为 [x, y, w, h, conf, class]
        # 这里仅作为示例框架

        # 暂时返回占位符
        return {
            'mAP@0.5': 0.0,
            'mAP@0.5:0.95': 0.0,
            'precision': 0.0,
            'recall': 0.0
        }

    # TODO: 实现完整的检测指标计算
    # 1. 解析预测框
    # 2. NMS 去重
    # 3. IoU 匹配
    # 4. 计算 precision/recall
    # 5. 计算 mAP

    return {
        'mAP@0.5': 0.0,
        'mAP@0.5:0.95': 0.0,
        'precision': 0.0,
        'recall': 0.0
    }


def compute_classification_metrics(predictions: torch.Tensor, targets: torch.Tensor,
                                   nc: int) -> Dict[str, float]:
    """计算分类任务的评估指标

    Args:
        predictions: 预测结果 [N, num_classes] 或 [N]
        targets: 目标标签 [N]
        nc: 类别数量

    Returns:
        包含 accuracy, per_class_accuracy 等指标的字典
    """
    # 获取预测类别
    if predictions.dim() > 1 and predictions.size(1) > 1:
        predicted = torch.argmax(predictions, dim=1)
    else:
        predicted = predictions

    # 总体准确率
    correct = (predicted == targets).sum().item()
    total = targets.size(0)
    accuracy = correct / total if total > 0 else 0.0

    # 每个类别的准确率
    per_class_acc = {}
    for c in range(nc):
        mask = targets == c
        if mask.sum() > 0:
            class_acc = (predicted[mask] == targets[mask]).float().mean().item()
            per_class_acc[f'class_{c}'] = class_acc

    return {
        'accuracy': accuracy,
        **per_class_acc
    }


def format_metrics(metrics: Dict[str, float]) -> str:
    """格式化指标输出

    Args:
        metrics: 指标字典

    Returns:
        格式化后的字符串
    """
    parts = []
    for key, value in metrics.items():
        if isinstance(value, float):
            if key in ['mAP@0.5', 'mAP@0.5:0.95', 'precision', 'recall', 'accuracy']:
                parts.append(f"{key}: {value*100:.2f}%")
            else:
                parts.append(f"{key}: {value:.4f}")
        else:
            parts.append(f"{key}: {value}")
    return " | ".join(parts)
