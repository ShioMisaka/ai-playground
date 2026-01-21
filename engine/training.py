"""
核心训练循环模块

提供通用的单 epoch 训练逻辑。
"""
import torch
from typing import Optional, Dict


def train_one_epoch(model, dataloader, optimizer, device, epoch, nc: Optional[int] = None):
    """训练一个 epoch

    Args:
        model: 模型
        dataloader: 数据加载器
        optimizer: 优化器
        device: 设备
        epoch: 当前 epoch
        nc: 类别数量（用于计算准确率）

    Returns:
        dict: 包含 loss, accuracy/mAP 等指标的字典
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (imgs, targets, paths) in enumerate(dataloader):
        imgs = imgs.to(device)
        targets = targets.to(device)

        # 前向传播
        optimizer.zero_grad()

        # 尝试不同的调用方式
        try:
            # 方式1: 模型接受 targets 参数
            outputs = model(imgs, targets)
        except TypeError:
            # 方式2: 模型不接受 targets 参数
            outputs = model(imgs)
            # 如果模型有 compute_loss 方法
            if hasattr(model, 'compute_loss'):
                outputs = {'predictions': outputs, 'loss': model.compute_loss(outputs, targets)}
            elif hasattr(model, 'detect') and hasattr(model.detect, 'compute_loss'):
                outputs = {'predictions': outputs, 'loss': model.detect.compute_loss(outputs, targets)}
            else:
                print("警告: 模型没有 compute_loss 方法，使用占位 loss")
                loss = torch.tensor(1.0, device=device, requires_grad=True)
                outputs = {'loss': loss}

        # 获取 loss
        if isinstance(outputs, dict):
            loss = outputs.get('loss', None)
            predictions = outputs.get('predictions', outputs)
            if loss is None:
                raise ValueError("无法获取 loss，请检查模型实现")
        elif isinstance(outputs, (tuple, list)):
            loss = outputs[-1]
            predictions = outputs[0]
        else:
            loss = outputs
            predictions = outputs

        # 确保 loss 是标量
        if hasattr(loss, 'dim') and loss.dim() > 0:
            loss = loss.mean()

        # 反向传播
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # 收集预测用于指标计算
        is_detection = hasattr(model, 'detect')

        if is_detection:
            # 检测任务：暂不计算训练时 mAP（计算成本高）
            pass
        else:
            # 分类任务：收集类别预测
            if isinstance(predictions, torch.Tensor):
                if predictions.dim() > 1 and predictions.size(1) > 1:
                    predicted = torch.argmax(predictions, dim=1)
                else:
                    predicted = predictions
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        if batch_idx % 10 == 0:
            print(f"Epoch {epoch} [{batch_idx}/{len(dataloader)}] Loss: {loss.item():.4f}")

    metrics = {'loss': total_loss / len(dataloader)}

    # 添加额外指标
    if hasattr(model, 'detect'):
        # 检测任务：训练时不计算 mAP（计算成本高），返回占位符
        metrics['mAP'] = -1.0  # -1 表示未计算
    else:
        # 分类任务：计算准确率
        if nc is not None:
            metrics['accuracy'] = correct / total if total > 0 else 0.0

    return metrics


def print_metrics(train_metrics: Dict[str, float], val_metrics: Dict[str, float], is_detection: bool):
    """打印训练和验证指标

    Args:
        train_metrics: 训练集指标
        val_metrics: 验证集指标
        is_detection: 是否为检测任务
    """
    print(f"Train Loss: {train_metrics['loss']:.4f}", end='')

    # 打印训练集额外指标
    if is_detection:
        # 检测任务
        if 'mAP' in train_metrics and train_metrics['mAP'] >= 0:
            print(f" | mAP: {train_metrics['mAP']*100:.2f}%", end='')
        else:
            print(f" | mAP: N/A", end='')
    else:
        # 分类任务
        if 'accuracy' in train_metrics:
            print(f" | Acc: {train_metrics['accuracy']*100:.2f}%", end='')
    print()

    print(f"Val Loss: {val_metrics['loss']:.4f}", end='')

    # 打印验证集额外指标
    if is_detection:
        # 检测任务
        if 'mAP' in val_metrics and val_metrics['mAP'] >= 0:
            print(f" | mAP: {val_metrics['mAP']*100:.2f}%", end='')
        else:
            print(f" | mAP: N/A", end='')
    else:
        # 分类任务
        if 'accuracy' in val_metrics:
            print(f" | Acc: {val_metrics['accuracy']*100:.2f}%", end='')
    print()
