"""
训练日志记录模块

提供 CSV 格式的训练日志记录功能。
"""
import csv
from pathlib import Path
from typing import Dict, Any, Optional


class TrainingLogger:
    """训练日志记录器

    将训练过程中的指标记录到 CSV 文件。
    """

    def __init__(self, csv_path: Path, is_detection: bool = False):
        """初始化日志记录器

        Args:
            csv_path: CSV 文件路径
            is_detection: 是否为检测任务（决定 CSV 列名）
        """
        self.csv_path = csv_path
        self.is_detection = is_detection
        self._file: Optional[Any] = None
        self._writer: Optional[csv.DictWriter] = None
        self._fieldnames: Optional[list] = None

    def _get_fieldnames(self) -> list:
        """获取 CSV 字段名"""
        base_fields = ['epoch', 'time', 'lr', 'train_loss', 'val_loss']
        if self.is_detection:
            base_fields.extend(['train_map', 'val_map'])
        else:
            base_fields.extend(['train_accuracy', 'val_accuracy'])
        return base_fields

    def open(self):
        """打开 CSV 文件并写入表头"""
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.csv_path, 'w', newline='')
        self._fieldnames = self._get_fieldnames()
        self._writer = csv.DictWriter(self._file, fieldnames=self._fieldnames)
        self._writer.writeheader()
        self._file.flush()

    def write_epoch(self, epoch: int, epoch_time: float, lr: float,
                    train_metrics: Dict[str, float], val_metrics: Dict[str, float]):
        """写入一个 epoch 的数据

        Args:
            epoch: 当前 epoch
            epoch_time: epoch 耗时（秒）
            lr: 学习率
            train_metrics: 训练集指标 {'loss': ..., 'accuracy': ... 或 'mAP': ...}
            val_metrics: 验证集指标
        """
        if self._writer is None:
            raise RuntimeError("Logger 未打开，请先调用 open()")

        row = {
            'epoch': epoch,
            'time': f'{epoch_time:.2f}',
            'lr': f'{lr:.6f}',
            'train_loss': f'{train_metrics["loss"]:.4f}',
            'val_loss': f'{val_metrics["loss"]:.4f}'
        }

        if self.is_detection:
            # 检测任务：mAP（-1 表示未计算，写入为空字符串）
            train_map = train_metrics.get('mAP', -1)
            val_map = val_metrics.get('mAP', -1)
            row['train_map'] = f'{train_map:.4f}' if train_map >= 0 else ''
            row['val_map'] = f'{val_map:.4f}' if val_map >= 0 else ''
        else:
            # 分类任务：accuracy
            row['train_accuracy'] = f'{train_metrics.get("accuracy", 0):.4f}'
            row['val_accuracy'] = f'{val_metrics.get("accuracy", 0):.4f}'

        self._writer.writerow(row)
        self._file.flush()

    def close(self):
        """关闭 CSV 文件"""
        if self._file is not None:
            self._file.close()
            self._file = None

    def __enter__(self):
        """支持 with 语句"""
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """支持 with 语句"""
        self.close()
