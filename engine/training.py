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
        dict: 包含 loss, box_loss, cls_loss, dfl_loss, accuracy/mAP 等指标的字典
    """
    model.train()
    total_loss = 0
    # 损失分量累计（用于检测任务）
    total_box_loss = 0.0
    total_cls_loss = 0.0
    total_dfl_loss = 0.0
    # 分类任务统计
    correct = 0
    total = 0

    for batch_idx, (imgs, targets, paths) in enumerate(dataloader):
        imgs = imgs.to(device)
        targets = targets.to(device)

        # 前向传播
        optimizer.zero_grad()

        # 尝试不同的调用方式
        loss_items = None  # Track loss_items for printing
        try:
            # 方式1: 模型接受 targets 参数，返回 (loss, loss_items, predictions)
            outputs = model(imgs, targets)
            # Check if outputs is tuple/list with 3 elements (new ultralytics-style format)
            if isinstance(outputs, (tuple, list)) and len(outputs) >= 2:
                loss_for_backward = outputs[0]
                loss_items = outputs[1]
                predictions = outputs[2] if len(outputs) > 2 else None
                # loss_items contains [box_loss, cls_loss, dfl_loss] (not multiplied by batch_size)
                loss = loss_for_backward
            else:
                raise TypeError("Unexpected output format")
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
            loss = outputs.get('loss', loss)
            predictions = outputs.get('predictions', outputs)

        # 确保 loss 是标量
        if hasattr(loss, 'dim') and loss.dim() > 0:
            loss = loss.sum()

        # 反向传播
        loss.backward()

        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)

        optimizer.step()

        # Use loss_items for printing (not multiplied by batch_size)
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
            if loss_items is not None:
                # 打印各损失分量
                print(f"Epoch {epoch} [{batch_idx}/{len(dataloader)}] "
                      f"Loss: {current_loss:.4f} (box: {box_loss:.4f}, cls: {cls_loss:.4f}, dfl: {dfl_loss:.4f})")
            else:
                print(f"Epoch {epoch} [{batch_idx}/{len(dataloader)}] Loss: {loss.item():.4f}")

    num_batches = len(dataloader)
    metrics = {
        'loss': total_loss / num_batches,
    }

    # 添加额外指标
    if hasattr(model, 'detect'):
        # 检测任务：添加损失分量
        metrics['box_loss'] = total_box_loss / num_batches
        metrics['cls_loss'] = total_cls_loss / num_batches
        metrics['dfl_loss'] = total_dfl_loss / num_batches
        metrics['mAP'] = -1.0  # -1 表示未计算（训练时不计算 mAP）
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
    if is_detection:
        # 检测任务：打印各损失分量
        print(f"Train - Loss: {train_metrics['loss']:.4f}", end='')
        if 'box_loss' in train_metrics:
            print(f" (box: {train_metrics['box_loss']:.4f}, "
                  f"cls: {train_metrics['cls_loss']:.4f}, "
                  f"dfl: {train_metrics['dfl_loss']:.4f})", end='')
        if 'mAP50' in train_metrics and train_metrics['mAP50'] >= 0:
            print(f" | mAP50: {train_metrics['mAP50']*100:.2f}%", end='')
        elif 'mAP' in train_metrics and train_metrics['mAP'] >= 0:
            print(f" | mAP: {train_metrics['mAP']*100:.2f}%", end='')
        else:
            print(f" | mAP: N/A", end='')
        print()

        print(f"Val   - Loss: {val_metrics['loss']:.4f}", end='')
        if 'box_loss' in val_metrics:
            print(f" (box: {val_metrics['box_loss']:.4f}, "
                  f"cls: {val_metrics['cls_loss']:.4f}, "
                  f"dfl: {val_metrics['dfl_loss']:.4f})", end='')
        if 'mAP50' in val_metrics and val_metrics['mAP50'] >= 0:
            print(f" | mAP50: {val_metrics['mAP50']*100:.2f}%", end='')
        elif 'mAP' in val_metrics and val_metrics['mAP'] >= 0:
            print(f" | mAP: {val_metrics['mAP']*100:.2f}%", end='')
        else:
            print(f" | mAP: N/A", end='')
        print()
    else:
        # 分类任务
        print(f"Train Loss: {train_metrics['loss']:.4f}", end='')
        if 'accuracy' in train_metrics:
            print(f" | Acc: {train_metrics['accuracy']*100:.2f}%", end='')
        print()

        print(f"Val Loss: {val_metrics['loss']:.4f}", end='')
        if 'accuracy' in val_metrics:
            print(f" | Acc: {val_metrics['accuracy']*100:.2f}%", end='')
        print()
