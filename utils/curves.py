"""
训练曲线可视化模块

提供训练过程中的曲线绘制功能，使用 pandas 读取 CSV 并生成独立图表。
"""
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# 设置样式
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.dpi'] = 100

# 颜色方案（从 seaborn 调色板）
COLORS = sns.color_palette("husl", 8)


def _load_and_clean_csv(csv_path: Path) -> pd.DataFrame:
    """读取并清洗 CSV 文件

    Args:
        csv_path: CSV 文件路径

    Returns:
        清洗后的 DataFrame
    """
    df = pd.read_csv(csv_path)

    # 去除列名的首尾空格
    df.columns = df.columns.str.strip()

    return df


def _get_column(df: pd.DataFrame, possible_names: list) -> str | None:
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


def plot_loss_analysis(csv_path: Path, save_dir: Path):
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

    # 创建 2x4 子图
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))

    # 标题映射
    titles = {
        'box': 'Box Loss',
        'cls': 'Cls Loss',
        'dfl': 'DFL Loss',
        'total': 'Total Loss'
    }

    # 第一行：Train losses
    for idx, (key, title) in enumerate(titles.items()):
        ax = axes[0, idx]
        if key == 'total':
            col = train_total
        else:
            col = train_cols[key]

        if col is not None:
            ax.plot(df['epoch'], df[col], color=COLORS[idx], linewidth=2, marker='o', markersize=3, label='Loss')
            ax.set_title(f'Train {title}')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best')
        else:
            ax.text(0.5, 0.5, f'{title}\nNot Available',
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=10, color='gray')
            ax.set_xticks([])
            ax.set_yticks([])

    # 第二行：Val losses
    for idx, (key, title) in enumerate(titles.items()):
        ax = axes[1, idx]
        if key == 'total':
            col = val_total
        else:
            col = val_cols[key]

        if col is not None:
            ax.plot(df['epoch'], df[col], color=COLORS[idx + 2], linewidth=2, marker='s', markersize=3, label='Loss')
            ax.set_title(f'Val {title}')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best')
        else:
            ax.text(0.5, 0.5, f'{title}\nNot Available',
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=10, color='gray')
            ax.set_xticks([])
            ax.set_yticks([])

    plt.suptitle('Loss Analysis', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()

    save_path = save_dir / 'loss_analysis.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Loss analysis saved to: {save_path}")
    plt.close()


def plot_map_performance(csv_path: Path, save_dir: Path):
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
        axes[0].plot(df['epoch'], df[map50_col] * 100,
                    color=COLORS[0], linewidth=2, marker='o', markersize=4, label='mAP@0.5')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('mAP@0.5 (%)')
        axes[0].set_title('Validation mAP@0.5')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim([0, 105])
        axes[0].legend(loc='best')
    else:
        axes[0].text(0.5, 0.5, 'mAP@0.5\nNot Available',
                    ha='center', va='center', transform=axes[0].transAxes,
                    fontsize=12, color='gray')
        axes[0].set_xticks([])
        axes[0].set_yticks([])
        axes[0].set_title('Validation mAP@0.5')

    # 右图：mAP@0.5:0.95
    if map50_95_col is not None:
        axes[1].plot(df['epoch'], df[map50_95_col] * 100,
                    color=COLORS[1], linewidth=2, marker='s', markersize=4, label='mAP@0.5:0.95')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('mAP@0.5:0.95 (%)')
        axes[1].set_title('Validation mAP@0.5:0.95')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim([0, 105])
        axes[1].legend(loc='best')
    else:
        axes[1].text(0.5, 0.5, 'mAP@0.5:0.95\nNot Available',
                    ha='center', va='center', transform=axes[1].transAxes,
                    fontsize=12, color='gray')
        axes[1].set_xticks([])
        axes[1].set_yticks([])
        axes[1].set_title('Validation mAP@0.5:0.95')

    plt.suptitle('mAP Performance', fontsize=16, fontweight='bold')
    plt.tight_layout()

    save_path = save_dir / 'map_performance.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"mAP performance saved to: {save_path}")
    plt.close()


def plot_training_status(csv_path: Path, save_dir: Path):
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
        axes[0].plot(df['epoch'], df[time_col],
                    color=COLORS[4], linewidth=2, marker='o', markersize=4, label='Epoch Time')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Time (seconds)')
        axes[0].set_title('Epoch Training Time')
        axes[0].grid(True, alpha=0.3)

        # 添加平均时间线
        avg_time = df[time_col].mean()
        axes[0].axhline(y=avg_time, color=COLORS[5], linestyle='--',
                       linewidth=1.5, alpha=0.7, label=f'Avg: {avg_time:.1f}s')
        axes[0].legend(loc='best')
    else:
        axes[0].text(0.5, 0.5, 'Time Data\nNot Available',
                    ha='center', va='center', transform=axes[0].transAxes,
                    fontsize=12, color='gray')
        axes[0].set_xticks([])
        axes[0].set_yticks([])
        axes[0].set_title('Epoch Training Time')

    # 右图：学习率
    if lr_col is not None:
        axes[1].plot(df['epoch'], df[lr_col],
                    color=COLORS[6], linewidth=2, marker='s', markersize=4, label='Learning Rate')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Learning Rate')
        axes[1].set_title('Learning Rate Schedule')
        axes[1].set_yscale('log')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(loc='best')
    else:
        axes[1].text(0.5, 0.5, 'Learning Rate\nNot Available',
                    ha='center', va='center', transform=axes[1].transAxes,
                    fontsize=12, color='gray')
        axes[1].set_xticks([])
        axes[1].set_yticks([])
        axes[1].set_title('Learning Rate Schedule')

    plt.suptitle('Training Status', fontsize=16, fontweight='bold')
    plt.tight_layout()

    save_path = save_dir / 'training_status.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Training status saved to: {save_path}")
    plt.close()


def plot_training_curves(csv_path: Path, save_dir: Path):
    """绘制所有训练曲线（主入口函数）

    生成三张独立的 PNG 图片：
    - loss_analysis.png: Loss 曲线分析
    - map_performance.png: mAP 性能
    - training_status.png: 训练状态

    Args:
        csv_path: CSV 文件路径
        save_dir: 保存图片的目录
    """
    csv_path = Path(csv_path)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating training curves from: {csv_path}")

    # 绘制三张图
    plot_loss_analysis(csv_path, save_dir)
    plot_map_performance(csv_path, save_dir)
    plot_training_status(csv_path, save_dir)

    print(f"\nAll curves saved to: {save_dir}")
