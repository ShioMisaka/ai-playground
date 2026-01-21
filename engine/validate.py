import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from typing import Optional, Dict, List

from .metrics import compute_classification_metrics, format_metrics

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


def validate(model, dataloader, device, nc=None):
    """验证模型，返回详细指标

    Args:
        model: 模型
        dataloader: 数据加载器
        device: 设备
        nc: 类别数量，用于计算准确率

    Returns:
        dict: 包含 loss, accuracy 等指标的字典
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
    all_predictions = []
    all_targets = []

    # 检测任务：收集所有预测
    all_detections = []
    all_gt_boxes = []

    with torch.no_grad():
        for imgs, targets, paths in dataloader:
            imgs = imgs.to(device)
            targets = targets.to(device)

            # 尝试不同的调用方式
            try:
                outputs = model(imgs, targets)
            except TypeError:
                outputs = model(imgs)
                if hasattr(model, 'compute_loss'):
                    outputs = {'predictions': outputs, 'loss': model.compute_loss(outputs, targets)}
                elif hasattr(model, 'detect') and hasattr(model.detect, 'compute_loss'):
                    outputs = {'predictions': outputs, 'loss': model.detect.compute_loss(outputs, targets)}
                else:
                    loss = torch.tensor(1.0, device=device)
                    outputs = {'loss': loss}

            # 获取loss
            if isinstance(outputs, dict):
                loss = outputs.get('loss', None)
                predictions = outputs.get('predictions', outputs)
                if loss is None:
                    raise ValueError("无法获取loss")
            elif isinstance(outputs, (tuple, list)):
                loss = outputs[-1]
                predictions = outputs[0]
            else:
                loss = outputs
                predictions = outputs

            # 确保loss是标量
            if hasattr(loss, 'dim') and loss.dim() > 0:
                loss = loss.mean()

            total_loss += loss.item()

            # 收集预测和目标用于指标计算
            is_detection = hasattr(model, 'detect')

            if is_detection:
                # 检测任务：收集检测框
                if isinstance(predictions, (tuple, list)):
                    # (concatenated, list) 格式
                    if len(predictions) >= 2:
                        all_detections.append(predictions[0])
                        all_gt_boxes.append(targets)
                else:
                    all_detections.append(predictions)
                    all_gt_boxes.append(targets)
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

    metrics = {
        'loss': total_loss / len(dataloader),
    }

    # 计算额外的指标
    if hasattr(model, 'detect'):
        # 检测任务
        # TODO: 实现 mAP 计算
        metrics['mAP'] = 0.0  # 占位符
    else:
        # 分类任务：计算准确率
        if nc is not None and all_predictions:
            all_predictions = torch.cat(all_predictions, dim=0)
            all_targets = torch.cat(all_targets, dim=0)
            cls_metrics = compute_classification_metrics(all_predictions, all_targets, nc)
            metrics.update(cls_metrics)

    return metrics