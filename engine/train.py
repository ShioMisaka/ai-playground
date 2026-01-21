import torch
import torch.nn as nn
import torch.optim as optim
import time
import csv
from pathlib import Path
from typing import Optional, Dict, Any

import matplotlib.pyplot as plt
import numpy as np

from .validate import evaluate
from .validate import validate
from .model_info import print_training_info, print_model_summary
from utils import create_dataloaders


def train_fc(model: nn.Module, train_data, val_data, epochs=3,
             optimizer=optim.Adam, criterion=nn.CrossEntropyLoss()):
    op = optimizer(model.parameters(), lr=0.001)
    print("Training begins...")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_data):
            op.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            op.step()

            running_loss += loss.item()
            if (batch_idx + 1) % 500 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{batch_idx+1}/{len(train_data)}], Loss: {loss.item():.4f}")

        print(f"Epoch [{epoch+1}/{epochs}], Accuracy: {100 * evaluate(val_data, model):.2f}%")

    print("Training completed...")


def train_one_epoch(model, dataloader, optimizer, device, epoch, nc: Optional[int] = None):
    """训练一个epoch

    Args:
        model: 模型
        dataloader: 数据加载器
        optimizer: 优化器
        device: 设备
        epoch: 当前 epoch
        nc: 类别数量（用于计算准确率）

    Returns:
        dict: 包含 loss, accuracy/mAP 等指标的字典
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    # 收集所有预测用于指标计算
    all_predictions = []
    all_targets = []

    for batch_idx, (imgs, targets, paths) in enumerate(dataloader):
        imgs = imgs.to(device)
        targets = targets.to(device)

        # 前向传播
        optimizer.zero_grad()

        # 尝试不同的调用方式
        try:
            # 方式1: 模型接受targets参数
            outputs = model(imgs, targets)
        except TypeError:
            # 方式2: 模型不接受targets参数
            outputs = model(imgs)
            # 如果模型有compute_loss方法
            if hasattr(model, 'compute_loss'):
                outputs = {'predictions': outputs, 'loss': model.compute_loss(outputs, targets)}
            elif hasattr(model, 'detect') and hasattr(model.detect, 'compute_loss'):
                outputs = {'predictions': outputs, 'loss': model.detect.compute_loss(outputs, targets)}
            else:
                # 简单的占位loss（需要你自己实现真正的loss函数）
                print("警告: 模型没有compute_loss方法，使用占位loss")
                loss = torch.tensor(1.0, device=device, requires_grad=True)
                outputs = {'loss': loss}

        # 获取loss
        if isinstance(outputs, dict):
            loss = outputs.get('loss', None)
            predictions = outputs.get('predictions', outputs)
            if loss is None:
                raise ValueError("无法获取loss，请检查模型实现")
        elif isinstance(outputs, (tuple, list)):
            loss = outputs[-1]
            predictions = outputs[0]
        else:
            loss = outputs
            predictions = outputs

        # 确保loss是标量
        if hasattr(loss, 'dim') and loss.dim() > 0:
            loss = loss.mean()

        # 反向传播
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

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

        if batch_idx % 10 == 0:
            print(f"Epoch {epoch} [{batch_idx}/{len(dataloader)}] Loss: {loss.item():.4f}")

    metrics = {'loss': total_loss / len(dataloader)}

    # 添加额外指标
    if hasattr(model, 'detect'):
        # 检测任务：训练时不计算 mAP（计算成本高），返回占位符
        metrics['mAP'] = -1.0  # -1 表示未计算
    else:
        # 分类任务：计算准确率
        if nc is not None:
            metrics['accuracy'] = correct / total if total > 0 else 0.0

    return metrics


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
    lr = []
    epoch_time = []
    metric_name = ''  # 'accuracy' 或 'mAP'

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
                elif 'train_map' in row:
                    metric_name = 'mAP'

            # 读取指标值
            if metric_name == 'accuracy':
                if row.get('train_accuracy'):
                    train_metric.append(float(row['train_accuracy']))
                if row.get('val_accuracy'):
                    val_metric.append(float(row['val_accuracy']))
            elif metric_name == 'mAP':
                if row.get('train_map'):
                    val = row['train_map']
                    if val:
                        train_metric.append(float(val))
                if row.get('val_map'):
                    val = row['val_map']
                    if val:
                        val_metric.append(float(val))

    epochs = np.array(epochs)

    # 创建图表
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

    # 2. 指标曲线（accuracy 或 mAP）
    ax2 = axes[0, 1]
    if train_metric or val_metric:
        label = 'Acc' if metric_name == 'accuracy' else 'mAP'
        ylabel = 'Accuracy (%)' if metric_name == 'accuracy' else 'mAP (%)'
        if train_metric:
            ax2.plot(epochs[:len(train_metric)], np.array(train_metric) * 100,
                    'b-', label=f'Train {label}', linewidth=2)
        if val_metric:
            ax2.plot(epochs[:len(val_metric)], np.array(val_metric) * 100,
                    'r-', label=f'Val {label}', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel(ylabel)
        ax2.set_title(f'Training and Validation {label}')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    else:
        # 没有指标数据，显示提示信息
        ax2.text(0.5, 0.5, f'No {metric_name} data available\n(Metric not computed during training)',
                ha='center', va='center', transform=ax2.transAxes,
                fontsize=12, color='gray')
        ax2.set_title(f'Training and Validation {metric_name.capitalize() if metric_name else "Metric"}')
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


def train(model, config_path, epochs=100, batch_size=16, img_size=640,
          lr=0.001, device='cuda', save_dir='runs/train'):
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
    """
    # 打印训练配置信息
    print_training_info(config_path, epochs, batch_size, img_size, lr, device, save_dir)

    # 创建保存目录
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 创建数据加载器
    train_loader, val_loader, config = create_dataloaders(
        config_path=config_path,
        batch_size=batch_size,
        img_size=img_size,
        workers=0
    )

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

    # 优化器
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.999),
        eps=1e-8
    )

    # 学习率调度器 - 使用warmup
    def warmup_lambda(epoch):
        warmup_epochs = 3
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            # Cosine annealing
            progress = (epoch - warmup_epochs) / (epochs - warmup_epochs)
            return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lambda)

    best_loss = float('inf')

    # 创建 CSV 文件保存训练数据
    csv_path = save_dir / 'training_log.csv'
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)

    # 写入 CSV 表头
    headers = ['epoch', 'time', 'lr', 'train_loss', 'val_loss']
    # 根据任务类型添加额外的指标列
    if hasattr(model, 'detect'):
        # 检测任务
        headers.extend(['train_map', 'val_map'])
    else:
        # 分类任务
        headers.extend(['train_accuracy', 'val_accuracy'])
    csv_writer.writerow(headers)
    csv_file.flush()

    # 训练循环
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        print("-" * 50)
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

        epoch_start_time = time.time()

        # 训练
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, epoch+1, nc=nc)

        # 定期清理内存
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        # 验证
        val_metrics = validate(model, val_loader, device, nc=nc)

        # 清理内存
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        # 更新学习率
        scheduler.step()

        epoch_time = time.time() - epoch_start_time

        # 打印结果
        print(f"Train Loss: {train_metrics['loss']:.4f}", end='')

        # 打印训练集额外指标
        if hasattr(model, 'detect'):
            # 检测任务
            if 'mAP' in train_metrics and train_metrics['mAP'] >= 0:
                print(f" | mAP: {train_metrics['mAP']*100:.2f}%", end='')
            else:
                print(f" | mAP: N/A", end='')
        else:
            # 分类任务
            if 'accuracy' in train_metrics:
                print(f" | Acc: {train_metrics['accuracy']*100:.2f}%", end='')
        print()

        print(f"Val Loss: {val_metrics['loss']:.4f}", end='')

        # 打印验证集额外指标
        if hasattr(model, 'detect'):
            # 检测任务
            if 'mAP' in val_metrics and val_metrics['mAP'] >= 0:
                print(f" | mAP: {val_metrics['mAP']*100:.2f}%", end='')
            else:
                print(f" | mAP: N/A", end='')
        else:
            # 分类任务
            if 'accuracy' in val_metrics:
                print(f" | Acc: {val_metrics['accuracy']*100:.2f}%", end='')
        print()

        print(f"Epoch Time: {epoch_time:.2f}s")

        # 写入 CSV
        row = [
            epoch + 1,
            f"{epoch_time:.2f}",
            f"{optimizer.param_groups[0]['lr']:.6f}",
            f"{train_metrics['loss']:.4f}",
            f"{val_metrics['loss']:.4f}"
        ]

        # 添加额外指标到 CSV
        if hasattr(model, 'detect'):
            # 检测任务：mAP（-1 表示未计算，写入为空字符串）
            train_map = train_metrics.get('mAP', -1)
            val_map = val_metrics.get('mAP', -1)
            row.append(f"{train_map:.4f}" if train_map >= 0 else "")
            row.append(f"{val_map:.4f}" if val_map >= 0 else "")
        else:
            # 分类任务：accuracy
            train_acc = train_metrics.get('accuracy', 0)
            val_acc = val_metrics.get('accuracy', 0)
            row.append(f"{train_acc:.4f}")
            row.append(f"{val_acc:.4f}")

        csv_writer.writerow(row)
        csv_file.flush()

        # 保存最佳模型
        if val_metrics['loss'] < best_loss:
            best_loss = val_metrics['loss']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, save_dir / 'best.pt')
            print(f"保存最佳模型: {save_dir / 'best.pt'}")

        # 保存最后一个epoch
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': val_metrics['loss'],
        }, save_dir / 'last.pt')

    # 关闭 CSV 文件
    csv_file.close()

    print("\n训练完成!")
    print(f"训练日志已保存到: {csv_path}")

    # 绘制训练曲线
    print("\n正在绘制训练曲线...")
    plot_training_curves(csv_path, save_dir)

    return model
