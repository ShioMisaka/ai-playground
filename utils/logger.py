"""
训练日志记录模块

提供 CSV 格式的训练日志记录功能和 LiveTable 动态表格显示功能。
"""
import csv
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable
from rich.console import Group, Console
from rich.table import Table
from rich.live import Live
from rich.text import Text
from rich import box


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
            # 检测任务：添加损失分量和 mAP50
            base_fields.extend([
                'train_box_loss', 'train_cls_loss', 'train_dfl_loss',
                'val_box_loss', 'val_cls_loss', 'val_dfl_loss',
                'val_map50'
            ])
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
            train_metrics: 训练集指标 {'loss': ..., 'box_loss': ..., 'cls_loss': ..., 'dfl_loss': ..., 'mAP50': ...}
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
            # 检测任务：损失分量和 mAP50
            row['train_box_loss'] = f'{train_metrics.get("box_loss", 0):.4f}'
            row['train_cls_loss'] = f'{train_metrics.get("cls_loss", 0):.4f}'
            row['train_dfl_loss'] = f'{train_metrics.get("dfl_loss", 0):.4f}'

            row['val_box_loss'] = f'{val_metrics.get("box_loss", 0):.4f}'
            row['val_cls_loss'] = f'{val_metrics.get("cls_loss", 0):.4f}'
            row['val_dfl_loss'] = f'{val_metrics.get("dfl_loss", 0):.4f}'

            # mAP50（如果存在）
            map50 = val_metrics.get('mAP50', 0)
            row['val_map50'] = f'{map50:.4f}' if map50 >= 0 else ''
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


class LiveTableLogger:
    """使用 rich.table 的动态表格 Logger

    每个 epoch 显示一个独立表格，支持动态刷新当前 epoch 的数据。
    """

    def __init__(
        self,
        columns: List[str],
        total_epochs: int,
        column_formatters: Optional[Dict[str, Callable]] = None,
        console_width: Optional[int] = None,
    ):
        """初始化 LiveTableLogger

        Args:
            columns: 列名列表，如 ["total_loss", "box_loss", "cls_loss", "dfl_loss"]
            total_epochs: 总 epoch 数
            column_formatters: 每列的格式化函数，如 {"total_loss": lambda x: f"{x:.4f}"}
            console_width: 控制台宽度（默认自动检测）
        """
        self.columns = columns
        self.total_epochs = total_epochs
        self.column_formatters = column_formatters or {}
        self._console_width = console_width

        # 状态管理
        self._console = Console(width=console_width) if console_width else Console()
        self._live: Optional[Live] = None
        self._completed_tables: List[Table] = []
        self._current_table: Optional[Table] = None
        self._current_epoch: Optional[int] = None
        self._current_lr: Optional[float] = None
        self._train_data: Optional[Dict[str, Any]] = None
        self._val_data: Optional[Dict[str, Any]] = None
        self._train_progress: Optional[str] = None

    def _format_value(self, column: str, value: Any) -> str:
        """格式化列值

        Args:
            column: 列名
            value: 原始值

        Returns:
            格式化后的字符串
        """
        if column in self.column_formatters:
            return self.column_formatters[column](value)
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value)

    def _create_table(self, epoch: int, lr: float) -> Table:
        """创建一个新的 epoch 表格

        Args:
            epoch: 当前 epoch
            lr: 学习率

        Returns:
            rich.table.Table
        """
        table = Table(
            title=None,  # 不使用 title，手动添加
            box=None,  # 无边框
            show_header=True,
            header_style="bold magenta",
            padding=(0, 1),
            show_edge=False,
            show_lines=False,
        )

        # 添加列
        table.add_column("", style="cyan", width=9)  # Train/Val 标签列
        for col in self.columns:
            table.add_column(col, justify="right", width=10)
        table.add_column("", width=65, overflow="fold")  # 进度条/mAP50 列

        return table

    def _create_epoch_header(self, epoch: int, lr: float) -> str:
        """创建 epoch 标题行

        Args:
            epoch: 当前 epoch
            lr: 学习率

        Returns:
            标题字符串
        """
        return f"[bold cyan]Epoch {epoch}/{self.total_epochs}[/bold cyan]  [dim]Learning Rate: {lr:.6f}[/dim]"

    def _render_progress_bar(
        self,
        current: int,
        total: int,
        elapsed: float,
        bar_width: int = 20,
    ) -> str:
        """渲染进度条

        Args:
            current: 当前批次索引（从0开始）
            total: 总批次数
            elapsed: 已用时间（秒）
            bar_width: 进度条宽度（字符数）

        Returns:
            进度条字符串，如 "100% ━━━━━━━━━━━━━━━━━━━━ 34/34 1.1s/it 36.8s<0.0s"
        """
        progress = (current + 1) / total
        percent = int(progress * 100)
        filled = int(progress * bar_width)
        bar = "━" * filled + "─" * (bar_width - filled)

        # 时间信息
        it_time = elapsed / (current + 1) if current > 0 else 0
        eta = it_time * (total - current - 1)

        return f"{percent}% {bar} {current + 1}/{total} {it_time:.1f}s/it {elapsed:.1f}s<{eta:.1f}s"

    def start(self):
        """启动 LiveTableLogger，初始化 rich.live"""
        self._live = Live(
            Group(),
            console=self._console,
            refresh_per_second=10,
        )
        self._live.start()

    def start_epoch(self, epoch: int, lr: float):
        """开始一个新的 epoch

        Args:
            epoch: 当前 epoch 编号
            lr: 学习率
        """
        self._current_epoch = epoch
        self._current_lr = lr
        self._current_table = self._create_table(epoch, lr)
        self._train_data = None
        self._val_data = None
        self._train_progress = None
        self._refresh()

    def update_row(
        self,
        phase: str,
        data: Dict[str, Any],
        progress: Optional[Dict[str, Any]] = None,
    ):
        """更新当前 epoch 的某一行

        Args:
            phase: "train" 或 "val"
            data: 指标数据字典，键名对应 columns
            progress: 进度信息（仅 train 需要），包含：
                - current: 当前批次索引（从0开始）
                - total: 总批次数
                - elapsed: 已用时间（秒）
                - bar_width: 进度条宽度（可选，默认 20）
        """
        if phase == "train":
            self._train_data = data
            if progress:
                bar_width = progress.get("bar_width", 20)
                self._train_progress = self._render_progress_bar(
                    progress["current"],
                    progress["total"],
                    progress["elapsed"],
                    bar_width,
                )
        elif phase == "val":
            self._val_data = data

        self._refresh()

    def end_epoch(self, epoch_time: float):
        """结束当前 epoch，锁定表格并添加到已完成列表

        Args:
            epoch_time: epoch 耗时（秒）
        """
        if self._current_epoch is None:
            return

        # 创建最终的表格
        final_table = self._create_table(self._current_epoch, self._current_lr)

        # 添加 Train 行
        if self._train_data is not None:
            train_cells = ["Train -"]
            for col in self.columns:
                value = self._train_data.get(col, "")
                train_cells.append(self._format_value(col, value))
            # 进度条列（100%完成）
            progress = self._train_progress or "100%"
            train_cells.append(progress)
            final_table.add_row(*train_cells)

        # 添加 Val 行
        if self._val_data is not None:
            val_cells = ["Val   -"]
            for col in self.columns:
                value = self._val_data.get(col, "")
                val_cells.append(self._format_value(col, value))
            # mAP50 列
            map50 = self._val_data.get("mAP50")
            map50_str = f"mAP50: {map50*100:>6.2f}%" if map50 is not None else ""
            val_cells.append(map50_str)
            final_table.add_row(*val_cells)

        # 将标题和表格作为元组保存
        header = self._create_epoch_header(self._current_epoch, self._current_lr)
        self._completed_tables.append((header, final_table))

        # 清空当前状态
        self._current_table = None
        self._current_epoch = None
        self._current_lr = None
        self._train_data = None
        self._val_data = None
        self._train_progress = None

        self._refresh()

    def _refresh(self):
        """刷新显示"""
        if self._live is None:
            return

        # 构建显示内容：已完成的表格 + 当前正在训练的表格
        content = []

        # 添加已完成的表格
        for item in self._completed_tables:
            header, table = item  # 解包元组
            content.append(header)  # 添加标题
            content.append(table)
            content.append("")  # 空行分隔

        # 添加当前正在训练的表格（每次重新创建）
        if self._current_epoch is not None:
            # 添加标题
            header = self._create_epoch_header(self._current_epoch, self._current_lr)
            content.append(header)

            # 重新创建当前表格
            current_display_table = self._create_table(
                self._current_epoch, self._current_lr
            )

            # 添加 Train 行
            if self._train_data is not None:
                train_cells = ["Train -"]
                for col in self.columns:
                    value = self._train_data.get(col, "")
                    train_cells.append(self._format_value(col, value))
                # 进度条列
                progress = self._train_progress or ""
                train_cells.append(progress)
                current_display_table.add_row(*train_cells)

            # 添加 Val 行
            if self._val_data is not None:
                val_cells = ["Val   -"]
                for col in self.columns:
                    value = self._val_data.get(col, "")
                    val_cells.append(self._format_value(col, value))
                # mAP50 列
                map50 = self._val_data.get("mAP50")
                map50_str = f"mAP50: {map50*100:>6.2f}%" if map50 is not None else ""
                val_cells.append(map50_str)
                current_display_table.add_row(*val_cells)

            content.append(current_display_table)

        # 更新 Live 显示
        self._live.update(Group(*content))

    def stop(self):
        """停止 LiveTableLogger"""
        if self._live is not None:
            self._live.stop()
            self._live = None
