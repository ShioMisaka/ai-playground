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
from .ema import ModelEMA
from utils import (create_dataloaders, TrainingLogger, LiveTableLogger, plot_training_curves,
                   print_training_info, print_model_summary, print_detection_header, get_save_dir)
from utils.transforms import MosaicTransform


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

    使用 CosineAnnealingLR 策略
    学习率平滑下降，贯穿整个训练过程，无中途跳变

    Args:
        optimizer: 优化器
        epochs: 总训练轮数
        warmup_epochs: warmup 轮数（预留参数，当前未启用）

    Returns:
        学习率调度器
    """
    # T_max: 余弦退火周期长度（设置为总 epoch 数）
    # eta_min: 最小学习率
    T_max = epochs
    eta_min = 1e-6

    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=T_max,
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
    # 打印训练配置信息
    print_training_info(config_path, epochs, batch_size, img_size, lr, device, save_dir)

    # 创建保存目录（自动递增避免冲突）
    save_dir = get_save_dir(save_dir)
    print(f"保存目录: {save_dir}")

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
        print(f"Mosaic 增强: 启用 (最后 {close_mosaic} 个 epoch 关闭)")
    else:
        mosaic_transform = None
        print("Mosaic 增强: 禁用")

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

    # 创建 EMA
    ema = None
    if use_ema:
        ema = ModelEMA(model, decay=0.9999)
        print("EMA: 启用 (decay=0.9999)")
    else:
        print("EMA: 关闭")

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
                    print(f"\n[Epoch {epoch + 1}] 关闭 Mosaic 增强，使用原始数据精调")

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

    print("\n训练完成!")
    print(f"训练日志已保存到: {csv_path}")

    # 绘制训练曲线
    print("\n正在绘制训练曲线...")
    plot_training_curves(csv_path, save_dir)

    return model
