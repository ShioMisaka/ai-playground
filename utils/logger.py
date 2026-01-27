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
            # 检测任务：添加损失分量和 mAP50、mAP50-95
            base_fields.extend([
                'train_box_loss', 'train_cls_loss', 'train_dfl_loss',
                'val_box_loss', 'val_cls_loss', 'val_dfl_loss',
                'val_map50', 'val_map50_95'
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
            # 检测任务：损失分量和 mAP50、mAP50-95
            row['train_box_loss'] = f'{train_metrics.get("box_loss", 0):.4f}'
            row['train_cls_loss'] = f'{train_metrics.get("cls_loss", 0):.4f}'
            row['train_dfl_loss'] = f'{train_metrics.get("dfl_loss", 0):.4f}'

            row['val_box_loss'] = f'{val_metrics.get("box_loss", 0):.4f}'
            row['val_cls_loss'] = f'{val_metrics.get("cls_loss", 0):.4f}'
            row['val_dfl_loss'] = f'{val_metrics.get("dfl_loss", 0):.4f}'

            # mAP50 和 mAP50-95（如果存在）
            map50 = val_metrics.get('mAP50', 0)
            row['val_map50'] = f'{map50:.4f}' if map50 >= 0 else ''
            map50_95 = val_metrics.get('mAP50-95', 0)
            row['val_map50_95'] = f'{map50_95:.4f}' if map50_95 >= 0 else ''
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

    设计原则：
    - 已完成的 epoch：直接打印到终端，保留历史但不刷新
    - 当前正在训练的 epoch：使用 Live 动态刷新
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

        # 状态管理
        self._console = Console(width=console_width) if console_width else Console()
        self._live: Optional[Live] = None
        # 当前正在训练的 epoch 数据
        self._current_epoch: Optional[int] = None
        self._current_lr: Optional[float] = None
        self._train_data: Optional[Dict[str, Any]] = None
        self._val_data: Optional[Dict[str, Any]] = None
        self._train_progress: Optional[str] = None

    def _format_value(self, column: str, value: Any) -> str:
        """格式化列值"""
        if column in self.column_formatters:
            return self.column_formatters[column](value)
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value)

    def _create_table(self) -> Table:
        """创建数据表格（无边框）"""
        table = Table(
            show_header=True,
            header_style="bold magenta",
            padding=(0, 1),
            show_edge=False,
            box=None,  # 无边框
        )
        table.add_column("", style="cyan", width=9)
        for col in self.columns:
            table.add_column(col, justify="right", width=10)
        table.add_column("", width=65, overflow="fold")
        return table

    def _create_epoch_header(self, epoch: int, lr: float) -> str:
        """创建 epoch 标题行"""
        return f"[bold cyan]Epoch {epoch}/{self.total_epochs}[/bold cyan]  [dim]lr={lr:.6f}[/dim]"

    def _add_row_to_table(self, table: Table, phase: str, data: Dict[str, Any], progress: Optional[str] = None):
        """向表格添加一行"""
        if phase == "train":
            cells = ["Train -"]
            for col in self.columns:
                cells.append(self._format_value(col, data.get(col, "")))
            cells.append(progress or "")
        else:
            cells = ["Val   -"]
            for col in self.columns:
                cells.append(self._format_value(col, data.get(col, "")))
            # 显示 mAP50 和 mAP50-95，使用小数格式，精确到小数点后三位
            map50 = data.get("mAP50")
            map50_95 = data.get("mAP50-95")
            if map50 is not None and map50_95 is not None:
                map_str = f"mAP50: {map50:.3f}  mAP50-95: {map50_95:.3f}"
            elif map50 is not None:
                map_str = f"mAP50: {map50:.3f}"
            else:
                map_str = ""
            cells.append(map_str)
        table.add_row(*cells)

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
        """启动 LiveTableLogger"""
        self._live = Live(
            Group(),
            console=self._console,
            refresh_per_second=4,
            transient=False,  # 保留最后输出的内容
        )
        self._live.start()

    def start_epoch(self, epoch: int, lr: float):
        """开始一个新的 epoch，启动 Live 显示

        Args:
            epoch: 当前 epoch 编号
            lr: 学习率
        """
        self._current_epoch = epoch
        self._current_lr = lr
        self._train_data = None
        self._val_data = None
        self._train_progress = None

        # 确保 Live 已启动
        if self._live is None:
            self.start()
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
            data: 指标数据字典
            progress: 进度信息（仅 train 需要）
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
        """结束当前 epoch，停止 Live 并打印 Time 信息

        Args:
            epoch_time: epoch 耗时（秒）
        """
        if self._current_epoch is None:
            return

        # 停止 Live（最后刷新的内容会保留在屏幕上）
        if self._live is not None:
            self._live.stop()
            self._live = None

        # 只打印 Time 信息（Live 的内容已保留）
        self._console.print(f"[dim]Time: {epoch_time:.2f}s[/dim]")
        self._console.print("")  # 空行分隔

        # 确保光标显示（修复 VSCode 终端光标问题）
        self._console.show_cursor(True)

        # 清空当前状态
        self._current_epoch = None
        self._current_lr = None
        self._train_data = None
        self._val_data = None
        self._train_progress = None

    def _refresh(self):
        """刷新 Live 显示（只显示当前正在训练的 epoch）"""
        if self._live is None:
            return

        content = []

        if self._current_epoch is not None:
            # 标题
            content.append(self._create_epoch_header(self._current_epoch, self._current_lr))

            # 表格
            table = self._create_table()
            if self._train_data is not None:
                self._add_row_to_table(table, "train", self._train_data, self._train_progress)
            if self._val_data is not None:
                self._add_row_to_table(table, "val", self._val_data)
            content.append(table)

        self._live.update(Group(*content))

    def stop(self):
        """停止 LiveTableLogger 并恢复终端状态

        使用 try-except 确保即使发生异常也能恢复光标显示。
        此方法可被多次调用，不会产生副作用。
        """
        # 1. 停止 Live 刷新
        if self._live is not None:
            try:
                self._live.stop()
            except Exception:
                pass  # 忽略停止时的错误
            self._live = None

        # 2. 强制显示光标（这是关键修复）
        # 即使 self._live 已经是 None 了，也要确保光标被恢复
        try:
            self._console.show_cursor(True)
        except Exception:
            pass
