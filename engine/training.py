"""
核心训练循环模块

提供通用的单 epoch 训练逻辑。
"""
import time
import torch
from typing import Optional, Dict, TYPE_CHECKING, Any

if TYPE_CHECKING:
    from utils import LiveTableLogger

from utils import format_detection_train_line, format_detection_val_line


def _format_progress_bar(current: int, total: int, elapsed: float) -> str:
    """格式化进度条

    Args:
        current: 当前批次索引（从0开始）
        total: 总批次数
        elapsed: 已用时间（秒）

    Returns:
        格式化的进度条字符串
    """
    progress = (current + 1) / total
    percent = int(progress * 100)

    # 进度条宽度（字符数）
    bar_width = 20
    filled = int(progress * bar_width)
    bar = '━' * filled + '─' * (bar_width - filled)

    # 时间信息
    it_time = elapsed / (current + 1) if current > 0 else 0
    eta = it_time * (total - current - 1)

    return f"{percent}% ━{bar} {current + 1}/{total} {it_time:.1f}s/it {elapsed:.1f}s<{eta:.1f}s"


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    device,
    epoch,
    total_epochs,
    nc: Optional[int] = None,
    live_logger: Optional["LiveTableLogger"] = None,
    ema: Optional[Any] = None,
):
    """训练一个 epoch

    Args:
        model: 模型
        dataloader: 数据加载器
        optimizer: 优化器
        device: 设备
        epoch: 当前 epoch（从1开始）
        total_epochs: 总 epoch 数
        nc: 类别数量（用于计算准确率）
        live_logger: LiveTableLogger 实例（可选），用于动态表格显示
        ema: ModelEMA 实例（可选），用于更新 EMA 权重

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

    epoch_start_time = time.time()

    for batch_idx, batch_data in enumerate(dataloader):
        # 兼容新旧数据格式
        if len(batch_data) == 4:
            # 新格式：(imgs, targets, paths, letterbox_params_list)
            imgs, targets, paths, letterbox_params_list = batch_data
        else:
            # 旧格式：(imgs, targets, paths)
            imgs, targets, paths = batch_data
            letterbox_params_list = None

        imgs = imgs.to(device)
        targets = targets.to(device)

        # 前向传播
        optimizer.zero_grad()

        # 尝试不同的调用方式
        loss_items = None  # Track loss_items for printing
        try:
            # 新版 YOLOv11: 调用模型时传入 targets，返回字典格式
            # {'loss': loss, 'loss_items': [box_loss, cls_loss, dfl_loss], 'predictions': predictions}
            outputs = model(imgs, targets)

            # 新格式：返回字典
            if isinstance(outputs, dict):
                loss = outputs['loss']
                loss_items = outputs.get('loss_items')
                predictions = outputs.get('predictions')
            # 旧格式兼容：返回 tuple/list
            elif isinstance(outputs, (tuple, list)) and len(outputs) >= 2:
                loss_for_backward = outputs[0]
                loss_items = outputs[1]
                predictions = outputs[2] if len(outputs) > 2 else None
                loss = loss_for_backward
            else:
                raise TypeError("Unexpected output format")
        except Exception as e:
            # 方式2: 模型不接受 targets 参数（旧版本兼容）
            outputs = model(imgs)
            # 如果模型有 compute_loss 方法
            if hasattr(model, 'compute_loss'):
                outputs = {'predictions': outputs, 'loss': model.compute_loss(outputs, targets)}
            elif hasattr(model, 'detect') and hasattr(model.detect, 'compute_loss'):
                outputs = {'predictions': outputs, 'loss': model.detect.compute_loss(outputs, targets)}
            else:
                print(f"警告: 模型前向传播失败 - {e}")
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

        # 更新 EMA 权重
        if ema is not None:
            ema.update(model)

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

        # 每个 iter 都更新打印
        elapsed = time.time() - epoch_start_time

        if loss_items is not None:
            # 检测任务：使用 LiveTableLogger 或传统打印
            if live_logger is not None:
                # 使用 LiveTableLogger 更新
                live_logger.update_row(
                    "train",
                    {
                        "total_loss": current_loss,
                        "box_loss": box_loss,
                        "cls_loss": cls_loss,
                        "dfl_loss": dfl_loss,
                    },
                    progress={
                        "current": batch_idx,
                        "total": len(dataloader),
                        "elapsed": elapsed,
                    },
                )
            else:
                # 传统打印方式（向后兼容）
                progress_bar = _format_progress_bar(batch_idx, len(dataloader), elapsed)
                line = format_detection_train_line(
                    current_loss, box_loss, cls_loss, dfl_loss, progress_bar
                )
                print(f"\r{line}", end="", flush=True)
        else:
            # 分类任务
            if live_logger is not None:
                live_logger.update_row(
                    "train",
                    {"total_loss": loss.item()},
                    progress={
                        "current": batch_idx,
                        "total": len(dataloader),
                        "elapsed": elapsed,
                    },
                )
            else:
                progress_bar = _format_progress_bar(batch_idx, len(dataloader), elapsed)
                print(
                    f"\rEpoch [{epoch}/{total_epochs}]    Loss: {loss.item():>7.4f}    {progress_bar}",
                    end="",
                    flush=True,
                )

    # epoch 结束时换行（仅传统打印方式）
    if live_logger is None:
        print()

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


def print_metrics(
    train_metrics: Dict[str, float],
    val_metrics: Dict[str, float],
    is_detection: bool,
    live_logger: Optional["LiveTableLogger"] = None,
):
    """打印训练和验证指标

    Args:
        train_metrics: 训练集指标
        val_metrics: 验证集指标
        is_detection: 是否为检测任务
        live_logger: LiveTableLogger 实例（可选），用于动态表格显示
    """
    if is_detection:
        # 检测任务：更新 Val 行到 LiveTableLogger 或传统打印
        if live_logger is not None:
            live_logger.update_row(
                "val",
                {
                    "total_loss": val_metrics["loss"],
                    "box_loss": val_metrics.get("box_loss", 0),
                    "cls_loss": val_metrics.get("cls_loss", 0),
                    "dfl_loss": val_metrics.get("dfl_loss", 0),
                    "mAP50": val_metrics.get("mAP50"),
                    "mAP50-95": val_metrics.get("mAP50-95"),
                },
            )
        else:
            # 传统打印方式（Train 行已在训练过程中显示，只打印 Val 行）
            map50 = val_metrics.get("mAP50", None)
            val_line = format_detection_val_line(
                val_metrics["loss"],
                val_metrics["box_loss"],
                val_metrics["cls_loss"],
                val_metrics["dfl_loss"],
                map50,
            )
            print(val_line)
    else:
        # 分类任务
        if live_logger is not None:
            live_logger.update_row(
                "val",
                {
                    "total_loss": val_metrics["loss"],
                    "accuracy": val_metrics.get("accuracy"),
                },
            )
        else:
            print(f"Train Loss: {train_metrics['loss']:>7.4f}", end="")
            if "accuracy" in train_metrics:
                print(f"    Acc: {train_metrics['accuracy']*100:>6.2f}%", end="")
            print()

            print(f"Val Loss: {val_metrics['loss']:>7.4f}", end="")
            if "accuracy" in val_metrics:
                print(f"    Acc: {val_metrics['accuracy']*100:>6.2f}%", end="")
            print()
