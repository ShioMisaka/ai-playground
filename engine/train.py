import torch
import torch.nn as nn
import torch.optim as optim

from pathlib import Path

from .validate import evaluate
from .validate import validate
from utils import create_dataloaders

def train_fc(model: nn.Module, train_data, val_data, epochs = 3, optimizer = optim.Adam, criterion = nn.CrossEntropyLoss()):
    op = optimizer(model.parameters(), lr=0.001)
    print("Training begins...")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_data):
            op.zero_grad()                   # 1. 梯度清零
            output = model(data)             # 2. 前向传播
            loss = criterion(output, target) # 3. 计算损失
            loss.backward()                  # 4. 反向传播
            op.step()                        # 5. 更新参数

            running_loss += loss.item()
            if (batch_idx + 1) % 500 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{batch_idx+1}/{len(train_data)}], Loss: {loss.item():.4f}")

        print(f"Epoch [{epoch+1}/{epochs}], Accuracy: {100 * evaluate(val_data, model):.2f}%")
    
    print("Training completed...")

def train_one_epoch(model, dataloader, optimizer, device, epoch):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    
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
            if loss is None:
                raise ValueError("无法获取loss，请检查模型实现")
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


def train(model, config_path, epochs=100, batch_size=16, img_size=640, 
          lr=0.001, device='cuda', save_dir='runs/train'):
    """完整训练流程"""
    
    # 创建保存目录
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建数据加载器
    train_loader, val_loader, config = create_dataloaders(
        config_path=config_path,
        batch_size=batch_size,
        img_size=img_size,
        workers=0  # 使用0避免多进程内存问题
    )
    
    print(f"\n类别数: {config['nc']}")
    print(f"类别名称: {config['names']}")
    print(f"训练集: {len(train_loader.dataset)} 张图片")
    print(f"验证集: {len(val_loader.dataset)} 张图片\n")
    
    # 设置设备
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    model = model.to(device)
    
    # 优化器 - 使用Adam，更稳定
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
    
    # 训练循环
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        print("-" * 50)
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # 训练
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch+1)
        
        # 定期清理内存
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        
        # 验证
        val_loss = validate(model, val_loader, device)
        
        # 清理内存
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        
        # 更新学习率
        scheduler.step()
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # 保存最佳模型
        if val_loss < best_loss:
            best_loss = val_loss
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
            'loss': val_loss,
        }, save_dir / 'last.pt')
    
    print("\n训练完成!")
    return model