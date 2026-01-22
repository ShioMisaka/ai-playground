"""
评估指标计算模块

提供分类和检测任务的评估指标计算功能。
"""
import torch
import numpy as np
from typing import Optional, Dict, List, Tuple
import torchvision.ops as ops


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


def bbox_iou(box1: np.ndarray, box2: np.ndarray) -> np.ndarray:
    """计算 IoU (numpy 版本)

    Args:
        box1: [N, 4] (x1, y1, x2, y2)
        box2: [M, 4] (x1, y1, x2, y2)

    Returns:
        [N, M] IoU 矩阵
    """
    # 计算交集区域
    x1 = np.maximum(box1[:, None, 0], box2[None, :, 0])  # [N, M]
    y1 = np.maximum(box1[:, None, 1], box2[None, :, 1])
    x2 = np.minimum(box1[:, None, 2], box2[None, :, 2])
    y2 = np.minimum(box1[:, None, 3], box2[None, :, 3])

    inter_area = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

    # 计算并集区域
    box1_area = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    box2_area = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    union_area = box1_area[:, None] + box2_area[None, :] - inter_area

    return inter_area / (union_area + 1e-7)


def compute_map50(predictions: List[Dict[str, np.ndarray]],
                  targets: torch.Tensor,
                  nc: int,
                  iou_threshold: float = 0.5,
                  conf_threshold: float = 0.25) -> Dict[str, float]:
    """计算 mAP@0.5

    Args:
        predictions: 预测结果列表，每个元素是 dict，包含:
            - boxes: [N, 4] (x1, y1, x2, y2) in pixels
            - scores: [N] 置信度分数
            - labels: [N] 类别标签
        targets: 目标张量 [num_boxes, 6] (batch_idx, class_id, x, y, w, h) normalized
        nc: 类别数量
        iou_threshold: IoU 阈值 (default 0.5 for mAP50)
        conf_threshold: 置信度阈值

    Returns:
        包含 mAP50, precision, recall 等指标的字典
    """
    device = targets.device
    targets_np = targets.cpu().numpy()

    # 按类别组织预测和目标
    pred_by_class = {c: [] for c in range(nc)}
    gt_by_class = {c: [] for c in range(nc)}
    gt_matched = {c: [] for c in range(nc)}  # 跟踪每个 GT 是否被匹配

    # 收集所有 batch 的预测和目标
    for batch_idx, batch_preds in enumerate(predictions):
        # 获取当前 batch 的目标
        batch_targets = targets_np[targets_np[:, 0] == batch_idx]
        if len(batch_targets) == 0 and len(batch_preds.get('boxes', [])) == 0:
            continue

        # 处理目标 - 转换从 xywh 到 xyxy，并反归一化到像素坐标
        for gt in batch_targets:
            class_id = int(gt[1])
            if 0 <= class_id < nc:
                x, y, w, h = gt[2:6]
                x1, y1 = x - w/2, y - h/2
                x2, y2 = x + w/2, y + h/2
                gt_by_class[class_id].append([x1, y1, x2, y2])
                gt_matched[class_id].append(False)

        # 处理预测
        if 'boxes' in batch_preds and len(batch_preds['boxes']) > 0:
            boxes = batch_preds['boxes']  # [N, 4]
            scores = batch_preds['scores']  # [N]
            labels = batch_preds['labels']  # [N]

            for box, score, label in zip(boxes, scores, labels):
                if 0 <= label < nc and score >= conf_threshold:
                    pred_by_class[int(label)].append({
                        'box': box,
                        'score': score
                    })

    # 计算每个类别的 AP
    aps = []
    precisions = []
    recalls = []

    for class_id in range(nc):
        n_gt = len(gt_by_class[class_id])
        if n_gt == 0:
            continue  # 没有该类别的目标，跳过

        # 获取该类别的预测
        class_preds = pred_by_class[class_id]
        if len(class_preds) == 0:
            # 没有预测，AP = 0
            aps.append(0.0)
            precisions.append(0.0)
            recalls.append(0.0)
            continue

        # 按分数排序
        class_preds = sorted(class_preds, key=lambda x: x['score'], reverse=True)

        # 初始化匹配标记
        tp = np.zeros(len(class_preds))
        fp = np.zeros(len(class_preds))
        gt_matched_class = gt_matched[class_id].copy()

        # 对每个预测进行匹配
        for i, pred in enumerate(class_preds):
            pred_box = np.array([pred['box']])

            # 找到最佳匹配的 GT
            best_iou = 0
            best_gt_idx = -1

            for gt_idx, gt_box in enumerate(gt_by_class[class_id]):
                if gt_matched_class[gt_idx]:
                    continue  # 已经被匹配过

                iou = bbox_iou(pred_box, np.array([gt_box]))[0, 0]
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            # 判断是否为有效匹配
            if best_iou >= iou_threshold:
                tp[i] = 1
                gt_matched_class[best_gt_idx] = True
            else:
                fp[i] = 1

        # 计算 precision 和 recall
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)

        recalls_arr = tp_cumsum / n_gt
        precisions_arr = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-7)

        # 计算 AP
        ap = compute_ap(recalls_arr, precisions_arr)
        aps.append(ap)
        precisions.append(precisions_arr[-1] if len(precisions_arr) > 0 else 0)
        recalls.append(recalls_arr[-1] if len(recalls_arr) > 0 else 0)

    # 计算 mAP
    if len(aps) > 0:
        mAP50 = np.mean(aps)
        avg_precision = np.mean(precisions) if len(precisions) > 0 else 0
        avg_recall = np.mean(recalls) if len(recalls) > 0 else 0
    else:
        mAP50 = 0.0
        avg_precision = 0.0
        avg_recall = 0.0

    return {
        'mAP50': mAP50,
        'precision': avg_precision,
        'recall': avg_recall
    }


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
