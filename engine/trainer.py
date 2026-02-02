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
from pathlib import Path
from typing import Optional, Union, Dict, Any
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
)
from utils.transforms import MosaicTransform
from engine.training import train_one_epoch
from engine.validate import validate


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
        data_path = self.data_cfg.get('train')

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
            self.mosaic_enabled = False
            self.mosaic_prob = 0.0
            return

        # Mosaic requires dataset access - will be set during data loading
        # For now, create a placeholder that will be configured later
        self.mosaic_enabled = enable_mosaic
        self.mosaic_prob = self.train_cfg.get('mosaic_prob', 1.0)
        self.mosaic = None  # Will be set after data loaders are created

    def _setup_logging(self):
        """Setup logging systems."""
        # CSV logger
        csv_path = self.save_dir / 'results.csv'
        self.csv_logger = TrainingLogger(
            csv_path,
            is_detection=True,
        )
        self.csv_logger.open()

        # Live table logger
        self.live_logger = LiveTableLogger(
            columns=[
                "total_loss", "box_loss", "cls_loss", "dfl_loss",
                "precision", "recall", "mAP50", "mAP50-95"
            ],
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
        from utils.curves import plot_training_curves

        self.setup()

        epochs = self.train_cfg.get('epochs', 100)
        nc = self.data_cfg.get('nc', 80)
        img_size = self.train_cfg.get('img_size', 640)
        mosaic_disable_epoch = self.train_cfg.get('mosaic_disable_epoch', None)

        print(f"Starting training for {epochs} epochs...")
        print(f"Save directory: {self.save_dir}")

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

                # Update loggers
                self.live_logger.update_row("train", train_metrics)
                self.live_logger.update_row("val", val_metrics)

                epoch_time = time.time() - epoch_start
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
            print("\nTraining interrupted by user. Saving checkpoint...")
            self._save_checkpoint(epoch, is_best=False)

        finally:
            self.live_logger.stop()
            self.csv_logger.close()

        # Plot training curves
        csv_path = self.save_dir / 'results.csv'
        plot_training_curves(str(csv_path), save_dir=str(self.save_dir))

        print(f"\nTraining complete. Best mAP50: {self.best_map:.4f}")

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
            print(f"New best model saved with mAP50: {self.best_map:.4f}")
