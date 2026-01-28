import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Dict, List
import torchvision.ops as ops

from utils.metrics import compute_classification_metrics, format_metrics, compute_map50


def evaluate(test_data, model):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad(): # 测试时不需要计算梯度
        for data, target in test_data:
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1) # 取概率最大的索引作为预测值
            total += target.size(0)
            correct += (predicted == target).sum().item()
    return correct / total


def test(model: nn.Module, test_data, test_size = 2):
    for (n, (x, _)) in enumerate(test_data):
        if n > test_size: break

        # 1. 准备输入数据
        # x[0] 是 [1, 28, 28]，unsqueeze(0) 后变成 [1, 1, 28, 28] 以符合 CNN 的 4D 要求
        single_img = x[0].unsqueeze(0)

        # 2. 预测
        model.eval()
        with torch.no_grad():
            output = model(single_img) # 直接传入 4D 张量
            predict = torch.argmax(output, dim=1)

        # 3. 可视化
        plt.figure(n)
        # imshow 需要 [28, 28]，所以用 squeeze 掉多余维度
        plt.imshow(x[0].squeeze())
        plt.title(f"prediction: {predict.item()}")

    plt.show()


def validate(model, dataloader, device, nc=None, img_size=640):
    """验证模型，返回详细指标

    Args:
        model: 模型
        dataloader: 数据加载器
        device: 设备
        nc: 类别数量，用于计算准确率
        img_size: 输入图像尺寸（用于检测任务）

    Returns:
        dict: 包含 loss, box_loss, cls_loss, dfl_loss, mAP50 等指标的字典
    """
    # 保存 Detect 层的原始训练状态
    detect_training_state = None
    if hasattr(model, 'detect'):
        detect_training_state = model.detect.training
        model.detect.train()  # 确保 Detect 层在训练模式，以便正确计算 loss

    model.eval()

    # model.eval() 会将所有子模块设为 eval 模式，需要重新设置 detect 为 train 模式
    if hasattr(model, 'detect'):
        model.detect.train()

    total_loss = 0
    # 损失分量累计（用于检测任务）
    total_box_loss = 0.0
    total_cls_loss = 0.0
    total_dfl_loss = 0.0

    all_predictions = []
    all_targets = []

    # 检测任务：收集所有预测（用于 mAP50 计算）
    all_detections = []
    all_gt_boxes = []

    with torch.no_grad():
        for batch_idx, (imgs, targets, paths) in enumerate(dataloader):
            imgs = imgs.to(device)
            targets = targets.to(device)

            # 尝试不同的调用方式
            loss_items = None  # Track loss_items for printing
            try:
                outputs = model(imgs, targets)
                # Check if outputs is tuple/list with 3 elements (new ultralytics-style format)
                if isinstance(outputs, (tuple, list)) and len(outputs) >= 2:
                    loss_for_backward = outputs[0]
                    loss_items = outputs[1]
                    predictions = outputs[2] if len(outputs) > 2 else None
                    loss = loss_for_backward
                else:
                    raise TypeError("Unexpected output format")
            except TypeError:
                outputs = model(imgs)
                if hasattr(model, 'compute_loss'):
                    outputs = {'predictions': outputs, 'loss': model.compute_loss(outputs, targets)}
                elif hasattr(model, 'detect') and hasattr(model.detect, 'compute_loss'):
                    outputs = {'predictions': outputs, 'loss': model.detect.compute_loss(outputs, targets)}
                else:
                    loss = torch.tensor(1.0, device=device)
                    outputs = {'loss': loss}
                loss = outputs.get('loss', loss)
                predictions = outputs.get('predictions', outputs)

            # 确保loss是标量
            if hasattr(loss, 'dim') and loss.dim() > 0:
                loss = loss.sum()

            # Use loss_items for logging (not multiplied by batch_size)
            if loss_items is not None:
                # loss_items: [box_loss, cls_loss, dfl_loss]
                box_loss = loss_items[0].item()
                cls_loss = loss_items[1].item()
                dfl_loss = loss_items[2].item()
                current_loss = box_loss + cls_loss + dfl_loss
                # 累计损失分量
                total_box_loss += box_loss
                total_cls_loss += cls_loss
                total_dfl_loss += dfl_loss
            else:
                current_loss = loss.item()

            total_loss += current_loss

            # 收集预测和目标用于指标计算
            is_detection = hasattr(model, 'detect')

            if is_detection:
                # 检测任务：获取推理模式下的预测用于 mAP50 计算
                # 先切换到推理模式获取预测
                model.detect.eval()
                with torch.no_grad():
                    inference_preds = model(imgs)
                    # inference_preds 是 (concatenated, dict) 格式
                    if isinstance(inference_preds, tuple) and len(inference_preds) >= 1:
                        pred_output = inference_preds[0]  # (bs, total_anchors, 4+nc)
                        # 后处理：提取框、分数、标签
                        batch_detections = _post_process_predictions(pred_output, img_size)
                        all_detections.extend(batch_detections)
                        all_gt_boxes.append(targets.cpu())
                # 恢复训练模式
                model.detect.train()
            else:
                # 分类任务：收集类别预测
                if isinstance(predictions, torch.Tensor):
                    if predictions.dim() > 1 and predictions.size(1) > 1:
                        # [N, num_classes] -> 收集所有预测
                        all_predictions.append(predictions)
                    else:
                        all_predictions.append(predictions)
                    all_targets.append(targets)

    # 恢复 Detect 层的原始状态
    if hasattr(model, 'detect') and detect_training_state is not None:
        if not detect_training_state:
            model.detect.eval()

    num_batches = len(dataloader)
    metrics = {
        'loss': total_loss / num_batches,
    }

    # 计算额外的指标
    if hasattr(model, 'detect'):
        # 检测任务：添加损失分量
        metrics['box_loss'] = total_box_loss / num_batches
        metrics['cls_loss'] = total_cls_loss / num_batches
        metrics['dfl_loss'] = total_dfl_loss / num_batches

        # 计算 mAP50、mAP50-95、precision、recall
        if nc is not None and len(all_detections) > 0:
            # 合并所有 batch 的 GT
            all_gt = torch.cat(all_gt_boxes, dim=0)
            map_results = compute_map50(all_detections, all_gt, nc, img_size=img_size)
            metrics['mAP50'] = map_results['mAP50']
            metrics['mAP50-95'] = map_results['mAP50-95']
            metrics['precision'] = map_results['precision']
            metrics['recall'] = map_results['recall']
        else:
            metrics['mAP50'] = 0.0
            metrics['mAP50-95'] = 0.0
            metrics['precision'] = 0.0
            metrics['recall'] = 0.0
    else:
        # 分类任务：计算准确率
        if nc is not None and all_predictions:
            all_predictions = torch.cat(all_predictions, dim=0)
            all_targets = torch.cat(all_targets, dim=0)
            cls_metrics = compute_classification_metrics(all_predictions, all_targets, nc)
            metrics.update(cls_metrics)

    return metrics


def _post_process_predictions(pred_output: torch.Tensor, img_size: int,
                              conf_threshold: float = 0.25,
                              iou_threshold: float = 0.45) -> List[Dict]:
    """后处理预测结果，提取框、分数、标签

    Args:
        pred_output: (bs, n_anchors, 4+nc) 预测输出，格式为 [cx, cy, w, h, cls1, cls2, ...]
        img_size: 图像尺寸
        conf_threshold: 置信度阈值
        iou_threshold: NMS IoU 阈值

    Returns:
        每个图像的检测结果列表，每个元素是 dict: {'boxes': [N,4], 'scores': [N], 'labels': [N]}
    """
    bs = pred_output.shape[0]
    nc = pred_output.shape[2] - 4  # 类别数

    results = []

    for b in range(bs):
        pred = pred_output[b]  # (n_anchors, 4+nc)

        # 提取 bbox (cx, cy, w, h)
        boxes = pred[:, :4]  # (n_anchors, 4)

        # 提取类别分数并取最大值
        cls_scores = pred[:, 4:]  # (n_anchors, nc)
        scores, labels = torch.max(cls_scores, dim=1)  # (n_anchors,), (n_anchors,)

        # 置信度过滤
        mask = scores > conf_threshold
        if mask.sum() == 0:
            # 没有满足阈值的检测
            results.append({'boxes': np.empty((0, 4)), 'scores': np.empty(0), 'labels': np.empty(0)})
            continue

        boxes = boxes[mask]
        scores = scores[mask]
        labels = labels[mask]

        # 转换从 cxcywh 到 x1y1x2y2
        x1 = boxes[:, 0] - boxes[:, 2] / 2
        y1 = boxes[:, 1] - boxes[:, 3] / 2
        x2 = boxes[:, 0] + boxes[:, 2] / 2
        y2 = boxes[:, 1] + boxes[:, 3] / 2
        boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=1)  # (n, 4)

        # NMS
        keep = ops.nms(boxes_xyxy, scores, iou_threshold)

        # 保留 NMS 后的结果
        final_boxes = boxes_xyxy[keep].cpu().numpy()
        final_scores = scores[keep].cpu().numpy()
        final_labels = labels[keep].cpu().numpy()

        results.append({
            'boxes': final_boxes,
            'scores': final_scores,
            'labels': final_labels
        })

    return results
