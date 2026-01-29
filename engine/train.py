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

# 直接从子模块导入，避免循环导入
from utils.load import create_dataloaders
from utils.logger import TrainingLogger, LiveTableLogger
from utils.curves import plot_training_curves
from utils.model_summary import print_training_start_2x2, print_training_completion, print_mosaic_disabled
from utils.table import print_detection_header
from utils.path_helper import get_save_dir
from utils.ema import ModelEMA
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


def train(model, cfg: dict, data_config=None):
    """完整训练流程

    Args:
        model: 模型实例
        cfg: 训练配置字典（由 get_config() 生成）
        data_config: 数据集配置字典（可选，从 data.yaml 加载）

    Returns:
        训练后的模型
    """
    # 解析配置
    train_cfg = cfg['train']
    model_cfg = cfg['model']
    augment_cfg = cfg['augment']
    sys_cfg = cfg['system']

    # 获取参数
    epochs = train_cfg['epochs']
    batch_size = train_cfg['batch_size']
    name = train_cfg['name']
    save_dir_base = train_cfg['save_dir']
    img_size = model_cfg['img_size']
    use_ema = model_cfg['use_ema']
    ema_decay = model_cfg.get('ema_decay', 0.9999)
    use_mosaic = augment_cfg['use_mosaic']
    close_mosaic = augment_cfg['close_mosaic']
    workers = sys_cfg['workers']

    # 获取数据集路径
    data_path = cfg.get('data')

    # 自动递增保存目录
    save_dir = get_save_dir(save_dir_base, name)

    # 创建数据加载器
    train_loader, val_loader, config = create_dataloaders(
        config_path=data_path,
        batch_size=batch_size,
        img_size=img_size,
        workers=workers
    )

    # 添加 Mosaic 增强到训练数据集
    # close_mosaic 表示最后 N 个 epoch 关闭 mosaic
    # 只有当 total_epochs > close_mosaic 时才启用 mosaic（确保有足够的时间使用）
    if use_mosaic and close_mosaic > 0 and epochs > close_mosaic:
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

    # 设置设备
    device_str = sys_cfg['device']
    device = torch.device(device_str if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # 打印训练信息（2x2 布局：Environment + Dataset, Hyperparameters + Model Summary）
    print_training_start_2x2(
        data_path,
        epochs,
        batch_size,
        img_size,
        cfg['optimizer']['lr'],
        device_str,
        save_dir,
        model,
        num_train_samples=len(train_loader.dataset),
        num_val_samples=len(val_loader.dataset),
        nc=nc,
        use_mosaic=use_mosaic and epochs > close_mosaic,
        use_ema=use_ema,
        close_mosaic=close_mosaic if use_mosaic and epochs > close_mosaic else None,
    )

    # 创建优化器和调度器
    optimizer = _create_optimizer(model, cfg)
    scheduler = _create_scheduler(optimizer, cfg, epochs)

    # 创建 EMA
    ema = None
    if use_ema:
        ema = ModelEMA(model, decay=ema_decay)

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
    plot_training_curves(csv_path, save_dir)

    return model


if __name__ == '__main__':
    from utils.config import parse_args, get_config, print_config, _flatten_to_nested, _parse_value
    from models import YOLOv11

    # 解析 CLI 参数
    args = parse_args()

    # 收集覆盖参数
    overrides = {}
    for item in args.overrides:
        if '=' in item:
            key, value = item.split('=', 1)
            overrides[key] = _parse_value(value)

    # 构建 CLI 参数字典
    kwargs = {
        'name': args.name,
        'data': args.data,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'device': args.device,
        **overrides
    }
    # 过滤 None 值
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    # 获取配置
    if args.config:
        cfg = get_config(config_file=args.config, model_config=args.model_config)
    else:
        cfg = get_config(model_config=args.model_config, **kwargs)

    # 打印配置
    print_config(cfg)

    # 创建模型
    model_cfg = cfg.get('model', {})
    nc = model_cfg.get('nc', 80)
    scale = model_cfg.get('scale', 'n')
    model = YOLOv11(nc=nc, scale=scale)

    # 开始训练
    train(model, cfg)
