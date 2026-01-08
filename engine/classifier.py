"""
分类器训练模块 - 用于训练 CoordAtt 等分类模型
"""
import os
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """训练一个 epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for imgs, labels, _ in dataloader:
        imgs = imgs.to(device)
        labels = labels.to(device)

        # 前向传播
        outputs = model(imgs)
        loss = criterion(outputs, labels)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 统计
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    avg_loss = total_loss / len(dataloader)
    acc = 100. * correct / total
    return avg_loss, acc


def validate(model, dataloader, criterion, device):
    """验证"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, labels, _ in dataloader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            outputs = model(imgs)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    avg_loss = total_loss / len(dataloader)
    acc = 100. * correct / total
    return avg_loss, acc


def train_classifier(model, train_loader, val_loader, epochs=50, lr=0.001,
                     device='cpu', save_dir='outputs', patience=10):
    """
    训练分类器 - 稳定版训练策略

    Args:
        model: 要训练的模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        epochs: 最大训练轮数
        lr: 初始学习率
        device: 训练设备
        save_dir: 保存目录
        patience: 早停耐心值

    Returns:
        history: 训练历史记录字典
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    # 使用 ReduceLROnPlateau：当验证损失不再下降时降低学习率
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
    )

    # 创建保存目录
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 训练数据记录
    history = {
        'epochs': [],
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'lr': [],
        'time_sec': []
    }

    print("=" * 50)
    print("开始训练...")
    print("=" * 50)

    best_acc = 0
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    total_start_time = time.time()

    for epoch in range(epochs):
        epoch_start_time = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # 根据验证损失调整学习率
        scheduler.step(val_loss)

        epoch_time = time.time() - epoch_start_time
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch [{epoch+1}/{epochs}] "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% | "
              f"LR: {current_lr:.6f} | "
              f"Time: {epoch_time:.2f}s")

        # 记录数据
        history['epochs'].append(epoch + 1)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)
        history['time_sec'].append(epoch_time)

        # 保存最佳模型（基于验证准确率）
        if val_acc > best_acc:
            best_acc = val_acc
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            print(f"  → 保存最佳模型 (Val Acc: {val_acc:.2f}%)")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"早停: 验证准确率 {patience} 轮未提升")
            break

    total_time = time.time() - total_start_time

    # 恢复最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print("\n已恢复最佳模型权重")

    print("=" * 50)
    print(f"训练完成! 最佳验证准确率: {best_acc:.2f}%")
    print(f"总训练时间: {total_time:.2f}s ({total_time/60:.2f}min)")
    print("=" * 50)

    # 保存训练数据
    history['total_time_sec'] = total_time
    history['best_val_acc'] = best_acc
    history['total_epochs'] = len(history['epochs'])

    history_path = save_dir / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"训练数据已保存到: {history_path}")

    # 保存最佳模型
    model_save_path = save_dir / 'best_model.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'best_acc': best_acc,
        'best_val_loss': best_val_loss,
        'epoch': len(history['epochs'])
    }, model_save_path)
    print(f"最佳模型已保存到: {model_save_path}")

    return history
