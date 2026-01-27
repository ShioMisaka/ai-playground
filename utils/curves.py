"""
训练曲线可视化模块

提供训练过程中的曲线绘制功能。
"""
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_training_curves(csv_path: Path, save_dir: Path):
    """绘制训练曲线

    Args:
        csv_path: CSV 文件路径
        save_dir: 保存图片的目录
    """
    # 读取 CSV 数据
    epochs = []
    train_loss = []
    val_loss = []
    train_metric = []  # 可以是 accuracy 或 mAP
    val_metric = []
    val_map50 = []  # mAP50 指标
    val_map50_95 = []  # mAP50-95 指标
    lr = []
    epoch_time = []
    metric_name = ''  # 'accuracy' 或 'detection'

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row['epoch']))
            train_loss.append(float(row['train_loss']))
            val_loss.append(float(row['val_loss']))
            lr.append(float(row['lr']))
            epoch_time.append(float(row['time']))

            # 检查是哪种指标
            if not metric_name:
                if 'train_accuracy' in row:
                    metric_name = 'accuracy'
                elif 'val_map50' in row:
                    metric_name = 'detection'

            # 读取指标值
            if metric_name == 'accuracy':
                if row.get('train_accuracy'):
                    train_metric.append(float(row['train_accuracy']))
                if row.get('val_accuracy'):
                    val_metric.append(float(row['val_accuracy']))
            elif metric_name == 'detection':
                # 检测任务：读取 mAP50 和 mAP50-95
                if row.get('val_map50'):
                    val = row['val_map50']
                    if val:
                        val_map50.append(float(val))
                if row.get('val_map50_95'):
                    val = row['val_map50_95']
                    if val:
                        val_map50_95.append(float(val))

    epochs = np.array(epochs)

    # 根据任务类型决定图表布局
    if metric_name == 'detection':
        # 检测任务：使用 2x3 布局，包含 mAP50 和 mAP50-95
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    else:
        # 分类任务或通用任务：使用 2x2 布局
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Loss 曲线
    ax1 = axes[0, 0]
    ax1.plot(epochs, train_loss, 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, val_loss, 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. 指标曲线
    if metric_name == 'accuracy':
        ax2 = axes[0, 1]
        if train_metric or val_metric:
            if train_metric:
                ax2.plot(epochs[:len(train_metric)], np.array(train_metric) * 100,
                        'b-', label='Train Acc', linewidth=2)
            if val_metric:
                ax2.plot(epochs[:len(val_metric)], np.array(val_metric) * 100,
                        'r-', label='Val Acc', linewidth=2)
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy (%)')
            ax2.set_title('Training and Validation Accuracy')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'No accuracy data available\n(Metric not computed during training)',
                    ha='center', va='center', transform=ax2.transAxes,
                    fontsize=12, color='gray')
            ax2.set_title('Training and Validation Accuracy')
            ax2.set_xticks([])
            ax2.set_yticks([])
    elif metric_name == 'detection':
        # 检测任务：分别绘制 mAP50 和 mAP50-95
        ax2 = axes[0, 1]
        if val_map50:
            ax2.plot(epochs[:len(val_map50)], np.array(val_map50) * 100,
                    'r-', label='Val mAP50', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('mAP50 (%)')
        ax2.set_title('Validation mAP@0.5')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        ax5 = axes[0, 2]
        if val_map50_95:
            ax5.plot(epochs[:len(val_map50_95)], np.array(val_map50_95) * 100,
                     'r-', label='Val mAP50-95', linewidth=2)
        ax5.set_xlabel('Epoch')
        ax5.set_ylabel('mAP50-95 (%)')
        ax5.set_title('Validation mAP@0.5:0.95')
        ax5.legend()
        ax5.grid(True, alpha=0.3)

        # 同时显示 mAP50 和 mAP50-95 的对比图
        ax6 = axes[1, 2]
        if val_map50:
            ax6.plot(epochs[:len(val_map50)], np.array(val_map50) * 100,
                    'b-', label='mAP@0.5', linewidth=2)
        if val_map50_95:
            ax6.plot(epochs[:len(val_map50_95)], np.array(val_map50_95) * 100,
                    'r-', label='mAP@0.5:0.95', linewidth=2)
        ax6.set_xlabel('Epoch')
        ax6.set_ylabel('mAP (%)')
        ax6.set_title('mAP@0.5 vs mAP@0.5:0.95')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
    else:
        # 通用任务
        ax2 = axes[0, 1]
        ax2.text(0.5, 0.5, 'No metric data available',
                ha='center', va='center', transform=ax2.transAxes,
                fontsize=12, color='gray')
        ax2.set_title('Training and Validation Metric')
        ax2.set_xticks([])
        ax2.set_yticks([])

    # 3. Learning Rate 曲线
    ax3 = axes[1, 0]
    ax3.plot(epochs, lr, 'g-', linewidth=2)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Learning Rate')
    ax3.set_title('Learning Rate Schedule')
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3)

    # 4. Epoch Time 曲线
    ax4 = axes[1, 1]
    ax4.plot(epochs, epoch_time, 'm-', linewidth=2)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Time (seconds)')
    ax4.set_title('Epoch Training Time')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / 'training_curves.png', dpi=150, bbox_inches='tight')
    print(f"训练曲线已保存到: {save_dir / 'training_curves.png'}")
    plt.close()
