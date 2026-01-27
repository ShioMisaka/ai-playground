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
                   print_training_info, print_model_summary, print_detection_header)


def _create_optimizer(model, lr: float):
    """创建优化器"""
    return torch.optim.Adam(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.999),
        eps=1e-8
    )


def _create_scheduler(optimizer, epochs: int, warmup_epochs: int = 3):
    """创建学习率调度器

    使用 warmup + CosineAnnealingWarmRestarts 策略
    周期性重启学习率，帮助模型跳出局部最优解

    Args:
        optimizer: 优化器
        epochs: 总训练轮数
        warmup_epochs: warmup 轮数

    Returns:
        学习率调度器
    """
    # T_0: 第一次重启的周期长度（epoch数）
    # T_mult: 每次重启后周期长度的倍增因子
    # eta_min: 最小学习率
    T_0 = 10
    T_mult = 2
    eta_min = 1e-6

    return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=T_0,
        T_mult=T_mult,
        eta_min=eta_min
    )


def _save_checkpoint(model, optimizer, epoch, loss, save_path: Path):
    """保存模型检查点"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, save_path)


def train(model, config_path, epochs=100, batch_size=16, img_size=640,
          lr=0.001, device='cuda', save_dir='runs/train'):
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

    Returns:
        训练后的模型
    """
    # 打印训练配置信息
    print_training_info(config_path, epochs, batch_size, img_size, lr, device, save_dir)

    # 创建保存目录
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 创建数据加载器
    train_loader, val_loader, config = create_dataloaders(
        config_path=config_path,
        batch_size=batch_size,
        img_size=img_size,
        workers=0
    )

    nc = config.get('nc')  # 类别数量

    print(f"类别数: {nc}")
    print(f"类别名称: {config.get('names', [])}")
    print(f"训练集: {len(train_loader.dataset)} 张图片")
    print(f"验证集: {len(val_loader.dataset)} 张图片")

    # 设置设备
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    model = model.to(device)

    # 打印模型摘要
    print_model_summary(model, img_size, nc=nc)

    # 创建优化器和调度器
    optimizer = _create_optimizer(model, lr)
    scheduler = _create_scheduler(optimizer, epochs)

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

                # 训练一个 epoch
                train_metrics = train_one_epoch(
                    model, train_loader, optimizer, device, epoch + 1, epochs, nc=nc, live_logger=live_logger
                )

                # 定期清理内存
                if device.type == 'cuda':
                    torch.cuda.empty_cache()

                # 验证
                val_metrics = validate(model, val_loader, device, nc=nc, img_size=img_size)

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
                    _save_checkpoint(model, optimizer, epoch, best_loss, save_dir / "best.pt")

                # 保存最后一个 epoch
                _save_checkpoint(model, optimizer, epoch, val_metrics["loss"], save_dir / "last.pt")

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

    print("\n训练完成!")
    print(f"训练日志已保存到: {csv_path}")

    # 绘制训练曲线
    print("\n正在绘制训练曲线...")
    plot_training_curves(csv_path, save_dir)

    return model
