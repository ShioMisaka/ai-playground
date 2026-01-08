"""
YOLO 检测器训练模块 - 用于训练 YOLOv3 和 CoordAtt 检测模型
"""
import os
import json
import time
import torch
import torch.optim as optim
from pathlib import Path


def train_one_epoch(model, dataloader, optimizer, device, epoch):
    """训练一个 epoch"""
    model.train()
    total_loss = 0

    for batch_idx, (imgs, targets, paths) in enumerate(dataloader):
        imgs = imgs.to(device)
        targets = targets.to(device)

        # 前向传播
        optimizer.zero_grad()
        outputs = model(imgs, targets)

        # 获取 loss
        if isinstance(outputs, dict):
            loss = outputs.get('loss', None)
            if loss is None:
                raise ValueError("模型输出未包含 loss")
        elif isinstance(outputs, (tuple, list)):
            loss = outputs[-1]
        else:
            loss = outputs

        # 确保loss是标量
        if hasattr(loss, 'dim') and loss.dim() > 0:
            loss = loss.mean()

        # 反向传播
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if batch_idx % 10 == 0:
            print(f"Epoch {epoch} [{batch_idx}/{len(dataloader)}] Loss: {loss.item():.4f}")

    return total_loss / len(dataloader)


def validate(model, dataloader, device):
    """验证模型"""
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for imgs, targets, paths in dataloader:
            imgs = imgs.to(device)
            targets = targets.to(device)

            # 注意：需要将 detect head 设为训练模式以获取正确格式的输出
            # 但其他层保持 eval 模式（如 BatchNorm, Dropout）
            if hasattr(model, 'detect'):
                model.detect.train()
            if hasattr(model, 'training'):
                was_training = model.training
            model.training = True  # 临时设置为训练模式

            # 前向传播
            outputs = model(imgs, targets)

            # 恢复 eval 模式
            model.eval()
            if hasattr(model, 'detect'):
                model.detect.eval()

            # 获取 loss
            if isinstance(outputs, dict):
                loss = outputs.get('loss', None)
                if loss is None:
                    raise ValueError("模型输出未包含 loss")
            elif isinstance(outputs, (tuple, list)):
                loss = outputs[-1]
            else:
                loss = outputs

            # 确保loss是标量
            if hasattr(loss, 'dim') and loss.dim() > 0:
                loss = loss.mean()

            total_loss += loss.item()

    return total_loss / len(dataloader)


def train_detector(model, train_loader, val_loader, epochs=100, lr=0.001,
                   device='cpu', save_dir='outputs', patience=15):
    """
    训练 YOLO 检测器

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
    # 确保 device 是 torch.device 类型
    if isinstance(device, str):
        device = torch.device(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)

    # 学习率调度器 - 使用 warmup + cosine annealing
    def warmup_lambda(epoch):
        warmup_epochs = 3
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            # 避免除零错误
            if epochs <= warmup_epochs:
                return 1.0
            # Cosine annealing
            progress = (epoch - warmup_epochs) / (epochs - warmup_epochs)
            return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lambda)

    # 创建保存目录
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 训练数据记录
    history = {
        'epochs': [],
        'train_loss': [],
        'val_loss': [],
        'lr': [],
        'time_sec': []
    }

    print("=" * 50)
    print("开始训练...")
    print("=" * 50)

    best_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    total_start_time = time.time()

    for epoch in range(epochs):
        epoch_start_time = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch+1)

        # 定期清理内存
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        val_loss = validate(model, val_loader, device)

        # 清理内存
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        # 更新学习率
        scheduler.step()

        epoch_time = time.time() - epoch_start_time
        current_lr = float(optimizer.param_groups[0]['lr'])  # 确保是 Python float

        print(f"\nEpoch [{epoch+1}/{epochs}] "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"LR: {current_lr:.6f} | "
              f"Time: {epoch_time:.2f}s")

        # 记录数据（确保都是 Python 原生类型）
        history['epochs'].append(epoch + 1)
        history['train_loss'].append(float(train_loss))
        history['val_loss'].append(float(val_loss))
        history['lr'].append(current_lr)
        history['time_sec'].append(float(epoch_time))

        # 保存最佳模型（基于验证 loss）
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            print(f"  → 保存最佳模型 (Val Loss: {val_loss:.4f})")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"早停: 验证 loss {patience} 轮未提升")
            break

    total_time = time.time() - total_start_time

    # 恢复最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print("\n已恢复最佳模型权重")

    print("=" * 50)
    print(f"训练完成! 最佳验证 loss: {best_loss:.4f}")
    print(f"总训练时间: {total_time:.2f}s ({total_time/60:.2f}min)")
    print("=" * 50)

    # 保存训练数据
    history['total_time_sec'] = total_time
    history['best_val_loss'] = best_loss
    history['total_epochs'] = len(history['epochs'])

    history_path = save_dir / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"训练数据已保存到: {history_path}")

    # 保存最佳模型
    model_save_path = save_dir / 'best_model.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'best_loss': best_loss,
        'epoch': len(history['epochs'])
    }, model_save_path)
    print(f"最佳模型已保存到: {model_save_path}")

    return history
