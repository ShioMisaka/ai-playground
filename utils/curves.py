"""
训练曲线可视化模块

提供训练过程中的曲线绘制功能，使用 pandas 读取 CSV 并生成独立图表。

重构后采用样式配置 + 通用绘图函数的设计，消除重复代码。
"""
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.signal import savgol_filter


# 设置全局样式
sns.set_style("whitegrid")


@dataclass
class StyleConfig:
    """绘图样式配置

    集中管理所有视觉样式，方便统一修改。
    """
    # 颜色方案
    colors: list[str]
    smooth_color: str = '#FFC107'

    # 线条样式
    linewidth: float = 3.0
    smooth_linewidth: float = 3.0
    smooth_linestyle: str = ':'
    smooth_alpha: float = 0.8
    marker: str = 'o'
    markersize: float = 6.0

    # 字体大小
    font_size: int = 14
    axes_labelsize: int = 15
    axes_titlesize: int = 16
    legend_fontsize: int = 13
    suptitle_fontsize: int = 18

    # 其他
    dpi: int = 100
    save_dpi: int = 150
    grid_alpha: float = 0.3

    # 平滑参数
    smooth_window_length: int = 19
    smooth_polyorder: int = 2


# 创建全局样式实例
# 使用 seaborn husl 调色板，替换第 4 个颜色（黄色）为深青色
_default_colors = list(sns.color_palette("husl", 8))
_default_colors[3] = '#008B8B'  # Dark Cyan

STYLE = StyleConfig(colors=_default_colors)

# 应用全局字体设置
plt.rcParams['font.size'] = STYLE.font_size
plt.rcParams['axes.labelsize'] = STYLE.axes_labelsize
plt.rcParams['axes.titlesize'] = STYLE.axes_titlesize
plt.rcParams['legend.fontsize'] = STYLE.legend_fontsize
plt.rcParams['figure.dpi'] = STYLE.dpi


def _smooth_data(data: np.ndarray, window_length: int = 5, polyorder: int = 2) -> np.ndarray:
    """使用 Savitzky-Golay 滤波器平滑数据

    Args:
        data: 输入数据数组
        window_length: 窗口大小（必须为奇数且大于 polyorder）
        polyorder: 多项式阶数

    Returns:
        平滑后的数据
    """
    data = np.asarray(data)
    n = len(data)

    # 确保窗口大小不超过数据长度且为奇数
    if window_length >= n:
        window_length = n - 1 if (n - 1) % 2 == 1 else n - 2
    if window_length <= polyorder:
        window_length = polyorder + 1 if (polyorder + 1) % 2 == 1 else polyorder + 2
    if window_length < 3:
        window_length = 3

    # 如果窗口大小调整后仍不满足条件，直接返回原数据
    if window_length >= n or window_length <= polyorder:
        return data

    try:
        return savgol_filter(data, window_length, polyorder)
    except Exception:
        return data


def _load_and_clean_csv(csv_path: Path) -> pd.DataFrame:
    """读取并清洗 CSV 文件

    Args:
        csv_path: CSV 文件路径

    Returns:
        清洗后的 DataFrame
    """
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    return df


def _get_column(df: pd.DataFrame, possible_names: list[str]) -> str | None:
    """尝试从多个可能的列名中获取存在的列

    Args:
        df: DataFrame
        possible_names: 可能的列名列表

    Returns:
        找到的列名，未找到则返回 None
    """
    for name in possible_names:
        if name in df.columns:
            return name
    return None


def _draw_fallback_text(
    ax: plt.Axes,
    text: str,
    title: str | None = None,
    fontsize: int = 12
) -> None:
    """在 axes 上绘制"数据不可用"的提示文字

    Args:
        ax: matplotlib axes 对象
        text: 显示的文本内容
        title: 可选的子图标题
        fontsize: 字体大小
    """
    ax.text(0.5, 0.5, text,
            ha='center', va='center', transform=ax.transAxes,
            fontsize=fontsize, color='gray')
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title)


def draw_metric_curve(
    ax: plt.Axes,
    epochs: np.ndarray,
    values: np.ndarray,
    label: str,
    title: str,
    color: str | None = None,
    ylabel: str | None = None,
    xlabel: str = 'Epoch',
    style: StyleConfig = STYLE,
    *,
    smooth: bool = True,
    yscale: str | None = None,
    ylim: tuple[float, float] | None = None,
    marker: str | None = None,
) -> None:
    """在指定 axes 上绘制单条指标曲线

    这是通用的绘图函数，统一处理：
    - 主曲线绘制
    - 平滑虚线绘制（可选）
    - 标题、坐标轴、网格、legend 设置

    Args:
        ax: matplotlib axes 对象
        epochs: epoch 数组
        values: 指标值数组
        label: 图例标签
        title: 子图标题
        color: 曲线颜色（None 则从 style.colors 选择）
        ylabel: Y 轴标签
        xlabel: X 轴标签
        style: 样式配置
        smooth: 是否绘制平滑虚线
        yscale: Y 轴缩放（如 'log'）
        ylim: Y 轴范围
        marker: marker 样式（None 则使用 style.marker）
    """
    # 确定颜色
    if color is None:
        color = style.colors[0]

    # 确定 marker
    if marker is None:
        marker = style.marker

    # 绘制主曲线
    ax.plot(epochs, values, color=color, linewidth=style.linewidth,
            marker=marker, markersize=style.markersize, label=label)

    # 绘制平滑虚线
    if smooth:
        smoothed = _smooth_data(values, style.smooth_window_length, style.smooth_polyorder)
        ax.plot(epochs, smoothed, color=style.smooth_color,
                linestyle=style.smooth_linestyle, linewidth=style.smooth_linewidth,
                alpha=style.smooth_alpha)

    # 设置标题和标签
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    # 设置网格和图例
    ax.grid(True, alpha=style.grid_alpha)
    ax.legend(loc='best')

    # 可选的 Y 轴设置
    if yscale is not None:
        ax.set_yscale(yscale)
    if ylim is not None:
        ax.set_ylim(ylim)


def plot_loss_analysis(csv_path: Path, save_dir: Path) -> None:
    """绘制 Loss 曲线图 (2x4 布局)

    第一行：Train 的 box_loss, cls_loss, dfl_loss, total_loss
    第二行：Val 的 box_loss, cls_loss, dfl_loss, total_loss

    Args:
        csv_path: CSV 文件路径
        save_dir: 保存目录
    """
    df = _load_and_clean_csv(csv_path)

    # 检测任务特有的损失列（支持新旧格式）
    train_cols = {
        'box': _get_column(df, ['train/box_loss', 'train_box_loss', 'train_box']),
        'cls': _get_column(df, ['train/cls_loss', 'train_cls_loss', 'train_cls']),
        'dfl': _get_column(df, ['train/dfl_loss', 'train_dfl_loss', 'train_dfl']),
    }
    val_cols = {
        'box': _get_column(df, ['val/box_loss', 'val_box_loss', 'val_box']),
        'cls': _get_column(df, ['val/cls_loss', 'val_cls_loss', 'val_cls']),
        'dfl': _get_column(df, ['val/dfl_loss', 'val_dfl_loss', 'val_dfl']),
    }

    # 计算 total_loss（如果不存在，支持新旧格式）
    train_total = _get_column(df, ['train/loss', 'train_loss', 'train_total_loss'])
    if train_total is None and all(v is not None for v in train_cols.values()):
        df['train_total_loss'] = df[train_cols['box']] + df[train_cols['cls']] + df[train_cols['dfl']]
        train_total = 'train_total_loss'

    val_total = _get_column(df, ['val/loss', 'val_loss', 'val_total_loss'])
    if val_total is None and all(v is not None for v in val_cols.values()):
        df['val_total_loss'] = df[val_cols['box']] + df[val_cols['cls']] + df[val_cols['dfl']]
        val_total = 'val_total_loss'

    # 配置每个子图：(title, column_name, color_index)
    train_plots = [
        ('Box Loss', train_cols['box'], 0),
        ('Cls Loss', train_cols['cls'], 1),
        ('DFL Loss', train_cols['dfl'], 2),
        ('Total Loss', train_total, 3),
    ]
    val_plots = [
        ('Box Loss', val_cols['box'], 0),
        ('Cls Loss', val_cols['cls'], 1),
        ('DFL Loss', val_cols['dfl'], 2),
        ('Total Loss', val_total, 3),
    ]

    # 创建子图
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))

    # 绘制第一行：Train
    for title, col, color_idx in train_plots:
        ax = axes[0, color_idx]
        if col is not None:
            draw_metric_curve(
                ax, df['epoch'].values, df[col].values,
                label='Train',
                title=f'Train {title}',
                ylabel='Loss',
                color=STYLE.colors[color_idx],
                style=STYLE,
            )
        else:
            _draw_fallback_text(ax, f'{title}\nNot Available', title=f'Train {title}')

    # 绘制第二行：Val（使用方形 marker）
    for title, col, color_idx in val_plots:
        ax = axes[1, color_idx]
        if col is not None:
            draw_metric_curve(
                ax, df['epoch'].values, df[col].values,
                label='Val',
                title=f'Val {title}',
                ylabel='Loss',
                color=STYLE.colors[color_idx],
                marker='s',
                style=STYLE,
            )
        else:
            _draw_fallback_text(ax, f'{title}\nNot Available', title=f'Val {title}')

    plt.suptitle('Loss Analysis', fontsize=STYLE.suptitle_fontsize, fontweight='bold', y=0.995)
    plt.tight_layout()

    save_path = save_dir / 'loss_analysis.png'
    plt.savefig(save_path, dpi=STYLE.save_dpi, bbox_inches='tight')
    plt.close()


def plot_map_performance(csv_path: Path, save_dir: Path) -> None:
    """绘制 mAP 性能图 (1x2 布局)

    左图：mAP@0.5
    右图：mAP@0.5:0.95

    Args:
        csv_path: CSV 文件路径
        save_dir: 保存目录
    """
    df = _load_and_clean_csv(csv_path)

    # 查找 mAP 列（支持新旧格式）
    map50_col = _get_column(df, ['metrics/mAP50(B)', 'val_map50', 'map50', 'mAP50'])
    map50_95_col = _get_column(df, ['metrics/mAP50-95(B)', 'val_map50_95', 'map50_95', 'mAP50-95'])

    # 检查是否有 mAP 数据
    if map50_col is None and map50_95_col is None:
        warnings.warn("No mAP columns found in CSV. Skipping mAP performance plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左图：mAP@0.5
    if map50_col is not None:
        draw_metric_curve(
            axes[0], df['epoch'].values, df[map50_col].values * 100,
            label='mAP@0.5',
            title='Validation mAP@0.5',
            ylabel='mAP@0.5 (%)',
            color=STYLE.colors[0],
            ylim=(0, 105),
            style=STYLE,
        )
    else:
        _draw_fallback_text(axes[0], 'mAP@0.5\nNot Available', title='Validation mAP@0.5')

    # 右图：mAP@0.5:0.95
    if map50_95_col is not None:
        draw_metric_curve(
            axes[1], df['epoch'].values, df[map50_95_col].values * 100,
            label='mAP@0.5:0.95',
            title='Validation mAP@0.5:0.95',
            ylabel='mAP@0.5:0.95 (%)',
            color=STYLE.colors[1],
            ylim=(0, 105),
            marker='s',
            style=STYLE,
        )
    else:
        _draw_fallback_text(axes[1], 'mAP@0.5:0.95\nNot Available', title='Validation mAP@0.5:0.95')

    plt.suptitle('mAP Performance', fontsize=STYLE.suptitle_fontsize, fontweight='bold')
    plt.tight_layout()

    save_path = save_dir / 'map_performance.png'
    plt.savefig(save_path, dpi=STYLE.save_dpi, bbox_inches='tight')
    plt.close()


def plot_precision_recall(csv_path: Path, save_dir: Path) -> None:
    """绘制 Precision 和 Recall 曲线图 (1x2 布局)

    左图：Precision
    右图：Recall

    Args:
        csv_path: CSV 文件路径
        save_dir: 保存目录
    """
    df = _load_and_clean_csv(csv_path)

    # 查找 precision 和 recall 列（支持新旧格式）
    precision_col = _get_column(df, ['metrics/precision(B)', 'val_precision', 'precision'])
    recall_col = _get_column(df, ['metrics/recall(B)', 'val_recall', 'recall'])

    # 检查是否有数据
    if precision_col is None and recall_col is None:
        warnings.warn("No precision/recall columns found in CSV. Skipping precision-recall plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左图：Precision
    if precision_col is not None:
        draw_metric_curve(
            axes[0], df['epoch'].values, df[precision_col].values * 100,
            label='Precision',
            title='Validation Precision',
            ylabel='Precision (%)',
            color=STYLE.colors[2],
            ylim=(0, 105),
            style=STYLE,
        )
    else:
        _draw_fallback_text(axes[0], 'Precision\nNot Available', title='Validation Precision')

    # 右图：Recall
    if recall_col is not None:
        draw_metric_curve(
            axes[1], df['epoch'].values, df[recall_col].values * 100,
            label='Recall',
            title='Validation Recall',
            ylabel='Recall (%)',
            color=STYLE.colors[3],
            ylim=(0, 105),
            marker='s',
            style=STYLE,
        )
    else:
        _draw_fallback_text(axes[1], 'Recall\nNot Available', title='Validation Recall')

    plt.suptitle('Precision & Recall', fontsize=STYLE.suptitle_fontsize, fontweight='bold')
    plt.tight_layout()

    save_path = save_dir / 'precision_recall.png'
    plt.savefig(save_path, dpi=STYLE.save_dpi, bbox_inches='tight')
    plt.close()


def plot_training_status(csv_path: Path, save_dir: Path) -> None:
    """绘制训练状态图 (1x2 布局)

    左图：训练时间
    右图：学习率

    Args:
        csv_path: CSV 文件路径
        save_dir: 保存目录
    """
    df = _load_and_clean_csv(csv_path)

    # 查找列
    time_col = _get_column(df, ['time', 'epoch_time', 'training_time'])
    lr_col = _get_column(df, ['lr', 'learning_rate', 'lr/pg0'])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左图：训练时间
    if time_col is not None:
        draw_metric_curve(
            axes[0], df['epoch'].values, df[time_col].values,
            label='Epoch Time',
            title='Epoch Training Time',
            ylabel='Time (seconds)',
            color=STYLE.colors[4],
            style=STYLE,
        )
        # 添加平均时间线
        avg_time = df[time_col].mean()
        axes[0].axhline(y=avg_time, color=STYLE.colors[5], linestyle='--',
                       linewidth=1.5, alpha=0.7, label=f'Avg: {avg_time:.1f}s')
        axes[0].legend(loc='best')
    else:
        _draw_fallback_text(axes[0], 'Time Data\nNot Available', title='Epoch Training Time')

    # 右图：学习率
    if lr_col is not None:
        draw_metric_curve(
            axes[1], df['epoch'].values, df[lr_col].values,
            label='Learning Rate',
            title='Learning Rate Schedule',
            ylabel='Learning Rate',
            color=STYLE.colors[6],
            yscale='log',
            marker='s',
            style=STYLE,
        )
    else:
        _draw_fallback_text(axes[1], 'Learning Rate\nNot Available', title='Learning Rate Schedule')

    plt.suptitle('Training Status', fontsize=STYLE.suptitle_fontsize, fontweight='bold')
    plt.tight_layout()

    save_path = save_dir / 'training_status.png'
    plt.savefig(save_path, dpi=STYLE.save_dpi, bbox_inches='tight')
    plt.close()


def plot_training_curves(csv_path: Path, save_dir: Path) -> None:
    """绘制所有训练曲线（主入口函数）

    生成四张独立的 PNG 图片：
    - loss_analysis.png: Loss 曲线分析
    - map_performance.png: mAP 性能
    - precision_recall.png: Precision 和 Recall
    - training_status.png: 训练状态

    Args:
        csv_path: CSV 文件路径
        save_dir: 保存图片的目录
    """
    csv_path = Path(csv_path)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating training curves from: {csv_path}")

    # 绘制四张图
    plot_loss_analysis(csv_path, save_dir)
    plot_map_performance(csv_path, save_dir)
    plot_precision_recall(csv_path, save_dir)
    plot_training_status(csv_path, save_dir)

    print(f"\nAll curves saved to: {save_dir}")
