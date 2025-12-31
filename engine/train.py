import torch
import torch.nn as nn
import torch.optim as optim

from .validate import evaluate

def train(model: nn.Module, train_data, val_data, epochs = 3, optimizer = optim.Adam, criterion = nn.CrossEntropyLoss()):
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

def train(model, config_path, epochs=100, batch_size=16, img_size=640, 
          lr=0.01, device='cuda', save_dir='runs/train'):
    """完整训练流程"""
    
    # 创建保存目录
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建数据加载器
    train_loader, val_loader, config = create_dataloaders(
        config_path=config_path,
        batch_size=batch_size,
        img_size=img_size
    )
    
    print(f"\n类别数: {config['nc']}")
    print(f"类别名称: {config['names']}")
    print(f"训练集: {len(train_loader.dataset)} 张图片")
    print(f"验证集: {len(val_loader.dataset)} 张图片\n")
    
    # 设置设备
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # 优化器
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=0.937,
        weight_decay=0.0005
    )
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs
    )
    
    best_loss = float('inf')
    
    # 训练循环
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        print("-" * 50)
        
        # 训练
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch+1)
        
        # 验证
        val_loss = validate(model, val_loader, device)
        
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