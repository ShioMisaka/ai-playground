"""
主训练流程模块

提供完整的训练流程，包括数据准备、模型训练、验证和保存。
"""
import time
from pathlib import Path
from typing import Optional

import torch
from rich.console import Console

from .training import train_one_epoch, print_metrics
from .validate import validate
from utils import (create_dataloaders, TrainingLogger, LiveTableLogger, plot_training_curves,
                   print_training_info, print_model_summary, print_detection_header, get_save_dir,
                   ModelEMA, print_training_completion, print_mosaic_disabled, print_plotting_status)
from utils.transforms import MosaicTransform


def _create_optimizer(model, cfg: dict):
    """创建优化器

    Args:
        model: 模型
        cfg: 完整配置字典

    Returns:
        优化器实例
    """
    optim_cfg = cfg['optimizer']
    optimizer_type = optim_cfg.get('type', 'Adam')

    if optimizer_type == 'Adam':
        return torch.optim.Adam(
            model.parameters(),
            lr=optim_cfg['lr'],
            betas=tuple(optim_cfg['betas']),
            eps=optim_cfg['eps'],
            weight_decay=optim_cfg.get('weight_decay', 0.0)
        )

    raise ValueError(f"Unsupported optimizer: {optimizer_type}")


def _create_scheduler(optimizer, cfg: dict, epochs: int):
    """创建学习率调度器

    Args:
        optimizer: 优化器
        cfg: 完整配置字典
        epochs: 训练总轮数

    Returns:
        学习率调度器实例
    """
    sched_cfg = cfg['scheduler']
    scheduler_type = sched_cfg.get('type', 'CosineAnnealingLR')

    if scheduler_type == 'CosineAnnealingLR':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=sched_cfg.get('min_lr', 1e-6)
        )

    raise ValueError(f"Unsupported scheduler: {scheduler_type}")


def _save_checkpoint(model, optimizer, epoch, loss, save_path: Path):
    """保存模型检查点"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, save_path)


def train(model, config_path, epochs=100, batch_size=16, img_size=640,
          lr=0.001, device='cuda', save_dir='runs/train',
          use_ema=True, use_mosaic=True, close_mosaic=10):
    """完整训练流程

    Args:
        model: 模型
        config_path: 数据集配置文件路径
        epochs: 训练轮数
        batch_size: 批大小
        img_size: 图像尺寸
        lr: 学习率
        device: 设备
        save_dir: 保存目录
        use_ema: 是否使用 EMA（指数移动平均）
        use_mosaic: 是否使用 Mosaic 数据增强
        close_mosaic: 最后 N 个 epoch 关闭 Mosaic（默认 10）

    Returns:
        训练后的模型
    """
    # 创建保存目录（自动递增避免冲突）
    save_dir = get_save_dir(save_dir)

    # 创建数据加载器
    train_loader, val_loader, config = create_dataloaders(
        config_path=config_path,
        batch_size=batch_size,
        img_size=img_size,
        workers=0
    )

    # 添加 Mosaic 增强到训练数据集
    if use_mosaic and epochs > close_mosaic:
        mosaic_transform = MosaicTransform(
            dataset=train_loader.dataset,
            img_size=img_size,
            prob=1.0,  # 训练时始终应用
            enable=True
        )
        train_loader.dataset.transform = mosaic_transform
    else:
        mosaic_transform = None

    nc = config.get('nc')  # 类别数量

    # 打印训练配置信息（包含 Mosaic 和 EMA）
    print_training_info(
        config_path,
        epochs,
        batch_size,
        img_size,
        lr,
        device,
        save_dir,
        num_train_samples=len(train_loader.dataset),
        num_val_samples=len(val_loader.dataset),
        nc=nc,
        use_mosaic=use_mosaic and epochs > close_mosaic,
        use_ema=use_ema,
        close_mosaic=close_mosaic if use_mosaic and epochs > close_mosaic else None,
    )

    # 设置设备
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # 打印模型摘要
    print_model_summary(model, img_size, nc=nc)

    # 创建优化器和调度器
    optimizer = _create_optimizer(model, lr)
    scheduler = _create_scheduler(optimizer, epochs)

    # 创建 EMA
    ema = None
    if use_ema:
        ema = ModelEMA(model, decay=0.9999)

    # 初始化日志记录器
    is_detection = hasattr(model, 'detect')
    csv_path = save_dir / 'training_log.csv'

    best_loss = float('inf')

    # 创建 LiveTableLogger
    if is_detection:
        live_logger = LiveTableLogger(
            columns=["total_loss", "box_loss", "cls_loss", "dfl_loss"],
            total_epochs=epochs,
            console_width=130,  # 建议使用 130+ 的宽度以完整显示进度条
        )
    else:
        live_logger = LiveTableLogger(
            columns=["total_loss", "accuracy"],
            total_epochs=epochs,
            console_width=130,
        )

    # 训练循环（使用 try...except...finally 确保异常时也能正确关闭 LiveTableLogger）
    try:
        with TrainingLogger(csv_path, is_detection) as csv_logger:
            for epoch in range(epochs):
                # 开始新的 epoch
                live_logger.start_epoch(epoch + 1, optimizer.param_groups[0]["lr"])

                epoch_start_time = time.time()

                # 关闭 Mosaic（最后 N 个 epoch）
                if mosaic_transform is not None and epoch == epochs - close_mosaic:
                    mosaic_transform.enable = False
                    print_mosaic_disabled(epoch + 1)

                # 训练一个 epoch
                train_metrics = train_one_epoch(
                    model, train_loader, optimizer, device, epoch + 1, epochs, nc=nc, live_logger=live_logger, ema=ema
                )

                # 定期清理内存
                if device.type == 'cuda':
                    torch.cuda.empty_cache()

                # 验证（使用 EMA 模型）
                val_model = ema.ema if ema is not None else model
                val_metrics = validate(val_model, val_loader, device, nc=nc, img_size=img_size)

                # 清理内存
                if device.type == 'cuda':
                    torch.cuda.empty_cache()

                # 更新学习率
                scheduler.step()

                epoch_time = time.time() - epoch_start_time

                # 打印结果到 LiveTableLogger
                print_metrics(train_metrics, val_metrics, is_detection, live_logger)

                # 结束当前 epoch
                live_logger.end_epoch(epoch_time)

                # 写入 CSV 日志
                csv_logger.write_epoch(
                    epoch + 1, epoch_time, optimizer.param_groups[0]["lr"], train_metrics, val_metrics
                )

                # 保存最佳模型
                if val_metrics["loss"] < best_loss:
                    best_loss = val_metrics["loss"]
                    save_model = ema.ema if ema is not None else model
                    _save_checkpoint(save_model, optimizer, epoch, best_loss, save_dir / "best.pt")

                # 保存最后一个 epoch（使用 EMA 模型）
                save_model = ema.ema if ema is not None else model
                _save_checkpoint(save_model, optimizer, epoch, val_metrics["loss"], save_dir / "last.pt")

    except KeyboardInterrupt:
        # 捕获 Ctrl+C，优雅地中断训练
        live_logger._console.print("\n\n[yellow]训练被用户中断 (KeyboardInterrupt)[/yellow]")

    except Exception as e:
        # 捕获其他未知错误
        live_logger._console.print(f"\n\n[red]训练发生错误: {e}[/red]")
        raise e  # 抛出异常以便调试

    finally:
        # 无论成功、中断还是报错，都会执行这里
        # 强制停止 LiveTableLogger 并恢复光标
        live_logger.stop()

    # 打印训练完成信息
    print_training_completion(save_dir, csv_path, best_loss)

    # 绘制训练曲线
    print_plotting_status(csv_path, save_dir)
    plot_training_curves(csv_path, save_dir)

    return model
