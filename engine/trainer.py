"""
DetectionTrainer: Unified training abstraction for YOLO models.

This class encapsulates all training logic including:
- Configuration management
- Data loader creation
- Optimizer/scheduler setup
- Training loop execution
- Validation and checkpointing
- Logging (CSV + LiveTable)
"""
import time
from pathlib import Path
from typing import Optional, Union, Dict, Any, TYPE_CHECKING
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from utils import (
    create_dataloaders,
    get_save_dir,
    TrainingLogger,
    LiveTableLogger,
    ModelEMA,
    print_mosaic_disabled,
    format_detection_train_line,
    format_detection_val_line,
)
from utils.transforms import MosaicTransform
from engine.validate import validate

if TYPE_CHECKING:
    from utils import LiveTableLogger


class DetectionTrainer:
    """
    Unified trainer for YOLO detection models.

    Args:
        model: YOLO model instance (YOLOv11, etc.)
        config: Training configuration dict
        save_dir: Base directory for saving outputs
    """

    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        save_dir: Optional[Union[str, Path]] = None,
    ):
        self.model = model
        self.config = config
        self.train_cfg = config.get('train', {})
        self.data_cfg = config.get('data', {})
        self.optimizer_cfg = config.get('optimizer', {})
        self.scheduler_cfg = config.get('scheduler', {})

        # Setup device
        device_str = config.get('device', 'cpu')
        self.device = torch.device(device_str)

        # Placeholders
        self.train_loader: Optional[DataLoader] = None
        self.val_loader: Optional[DataLoader] = None
        self.optimizer: Optional[Optimizer] = None
        self.scheduler: Optional[_LRScheduler] = None
        self.ema: Optional[ModelEMA] = None
        self.mosaic: Optional[MosaicTransform] = None
        self.save_dir: Optional[Path] = None
        self.csv_logger: Optional[TrainingLogger] = None
        self.live_logger: Optional[LiveTableLogger] = None
        self.best_map: float = 0.0

    def setup(self):
        """Setup all training components."""
        try:
            self._setup_save_dir()
            self._setup_data_loaders()
            self._setup_model()
            self._setup_optimizer()
            self._setup_scheduler()
            self._setup_ema()
            self._setup_mosaic()
            self._setup_logging()
        except Exception as e:
            raise RuntimeError(f"Failed to setup trainer: {e}") from e

    def _setup_save_dir(self):
        """Setup save directory with auto-increment."""
        name = self.train_cfg.get('name', 'exp')
        base_dir = self.train_cfg.get('save_dir', 'runs/train')
        self.save_dir = get_save_dir(Path(base_dir) / name)

    def _setup_data_loaders(self):
        """Setup training and validation data loaders."""
        # Extract data config path (YAML file)
        # 优先使用 _yaml_path（原始 YAML 文件路径），否则使用 train 键
        data_path = self.data_cfg.get('_yaml_path', self.data_cfg.get('train'))

        self.train_loader, self.val_loader, _ = create_dataloaders(
            config_path=data_path,
            batch_size=self.train_cfg.get('batch_size', 16),
            img_size=self.train_cfg.get('img_size', 640),
            workers=0,
            letterbox=self.train_cfg.get('letterbox', True),
        )

    def _setup_model(self):
        """Move model to device."""
        self.model.to(self.device)

    def _setup_optimizer(self):
        """Setup optimizer."""
        optim_type = self.optimizer_cfg.get('type', 'Adam')
        lr = self.optimizer_cfg.get('lr', 0.001)
        weight_decay = self.optimizer_cfg.get('weight_decay', 0.0)

        # Get trainable parameters
        pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
        for _, v in self.model.named_modules():
            if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
                pg2.append(v.bias)  # biases
            if isinstance(v, nn.BatchNorm2d):
                pg0.append(v.weight)  # no decay
            elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
                pg1.append(v.weight)  # apply decay

        optimizer_cls = getattr(torch.optim, optim_type, torch.optim.Adam)
        self.optimizer = optimizer_cls([
            {'params': pg0, 'weight_decay': 0.0},  # BatchNorm weights - no decay
            {'params': pg1, 'weight_decay': weight_decay},  # Other weights - with decay
            {'params': pg2, 'weight_decay': 0.0},  # Biases - no decay
        ], lr=lr)

    def _setup_scheduler(self):
        """Setup learning rate scheduler."""
        scheduler_type = self.scheduler_cfg.get('type', 'CosineAnnealingLR')
        epochs = self.train_cfg.get('epochs', 100)
        min_lr = self.scheduler_cfg.get('min_lr', 1e-6)

        if scheduler_type == 'CosineAnnealingLR':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=epochs,
                eta_min=min_lr,
            )
        else:
            # Default to CosineAnnealingLR
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=epochs,
                eta_min=min_lr,
            )

    def _setup_ema(self):
        """Setup EMA."""
        model_cfg = self.config.get('model', {})
        use_ema = model_cfg.get('use_ema', True)
        if not use_ema:
            self.ema = None
            return

        decay = model_cfg.get('ema_decay', 0.9999)
        self.ema = ModelEMA(self.model, decay=decay)

    def _setup_mosaic(self):
        """Setup Mosaic data augmentation."""
        enable_mosaic = self.train_cfg.get('mosaic', True)
        if not enable_mosaic:
            self.mosaic = None
            return

        # 获取配置参数
        img_size = self.train_cfg.get('img_size', 640)
        mosaic_prob = self.train_cfg.get('mosaic_prob', 1.0)

        # 创建 MosaicTransform 并设置到训练集 dataset
        # 注意：train_loader 必须已经在 _setup_data_loaders 中创建
        self.mosaic = MosaicTransform(
            dataset=self.train_loader.dataset,
            img_size=img_size,
            prob=mosaic_prob,
            enable=True,
        )
        self.train_loader.dataset.transform = self.mosaic

    def _setup_logging(self):
        """Setup logging systems."""
        # CSV logger
        csv_path = self.save_dir / 'results.csv'
        self.csv_logger = TrainingLogger(
            csv_path,
            is_detection=True,
        )
        self.csv_logger.open()

        # Live table logger - 只包含训练时的 loss 列（验证指标通过 print_metrics 打印）
        self.live_logger = LiveTableLogger(
            columns=["total_loss", "box_loss", "cls_loss", "dfl_loss"],
            total_epochs=self.train_cfg.get('epochs', 100),
            console_width=self.train_cfg.get('console_width', 130),
        )

    def train(self) -> Dict[str, Any]:
        """
        Execute the full training loop.

        Returns:
            Dict containing training results (best_map, final_loss, etc.)
        """
        import time

        self.setup()

        epochs = self.train_cfg.get('epochs', 100)
        nc = self.data_cfg.get('nc', 80)
        img_size = self.train_cfg.get('img_size', 640)
        mosaic_disable_epoch = self.train_cfg.get('mosaic_disable_epoch', None)

        try:
            epoch = 0
            for epoch in range(epochs):
                epoch_start = time.time()
                lr = self.optimizer.param_groups[0]['lr']

                self.live_logger.start_epoch(epoch + 1, lr)

                # Disable Mosaic in final epochs
                if mosaic_disable_epoch and epoch >= epochs - mosaic_disable_epoch:
                    if self.mosaic is not None:
                        self.mosaic.enable = False
                    if epoch == epochs - mosaic_disable_epoch:
                        print_mosaic_disabled(epoch)

                # Training
                train_metrics = train_one_epoch(
                    model=self.model,
                    dataloader=self.train_loader,
                    optimizer=self.optimizer,
                    device=self.device,
                    epoch=epoch + 1,
                    total_epochs=epochs,
                    nc=nc,
                    live_logger=self.live_logger,
                )

                # Update EMA
                if self.ema is not None:
                    self.ema.update(self.model)

                # Validation
                val_model = self.ema.ema if self.ema else self.model
                # Switch detect head to training mode for loss computation
                if hasattr(val_model, 'detect'):
                    val_model.detect.train()

                val_metrics = validate(
                    val_model,
                    self.val_loader,
                    self.device,
                    nc=nc,
                    img_size=img_size,
                )

                # Restore detect head mode
                if hasattr(val_model, 'detect'):
                    val_model.detect.eval()

                epoch_time = time.time() - epoch_start

                # 打印训练和验证指标（使用与 train.py 相同的方式）
                is_detection = hasattr(self.model, 'detect')
                print_metrics(train_metrics, val_metrics, is_detection, self.live_logger)

                self.live_logger.end_epoch(epoch_time)

                self.csv_logger.write_epoch(
                    epoch + 1, epoch_time, lr, train_metrics, val_metrics
                )

                # Step scheduler
                self.scheduler.step()

                # Save checkpoint
                current_map = val_metrics.get('mAP50', 0.0)
                is_best = current_map > self.best_map
                if is_best:
                    self.best_map = current_map

                self._save_checkpoint(epoch + 1, is_best=is_best)

        except KeyboardInterrupt:
            self.live_logger._console.print("\n\n[yellow]训练被用户中断 (KeyboardInterrupt)[/yellow]")
            self._save_checkpoint(epoch, is_best=False)

        finally:
            self.live_logger.stop()
            self.csv_logger.close()

        # 注意：训练完成信息和曲线绘制由外部接口 (models/yolo.py) 处理
        # 这里只返回训练结果

        return {
            'best_map': self.best_map,
            'final_epoch': epoch + 1,
            'save_dir': str(self.save_dir),
        }

    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        # Save last checkpoint
        last_path = self.save_dir / 'weights' / 'last.pt'
        last_path.parent.mkdir(parents=True, exist_ok=True)

        ckpt = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_map': self.best_map,
        }

        if self.ema is not None:
            ckpt['ema_state_dict'] = self.ema.ema.state_dict()

        torch.save(ckpt, last_path)

        # Save best checkpoint
        if is_best:
            best_path = self.save_dir / 'weights' / 'best.pt'
            torch.save(ckpt, best_path) 

# ============================================================================
# Functions migrated from engine.training.py
# ============================================================================

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
