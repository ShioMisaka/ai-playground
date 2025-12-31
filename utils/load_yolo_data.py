import os
import yaml
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from pathlib import Path


class YOLODataset(Dataset):
    """YOLO格式数据集加载器"""
    
    def __init__(self, img_dir, label_dir, img_size=640, augment=False):
        """
        Args:
            img_dir: 图片目录路径
            label_dir: 标签目录路径
            img_size: 输入图像尺寸
            augment: 是否进行数据增强
        """
        self.img_dir = Path(img_dir)
        self.label_dir = Path(label_dir)
        self.img_size = img_size
        self.augment = augment
        
        # 获取所有图片文件
        self.img_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            self.img_files.extend(list(self.img_dir.glob(ext)))
            self.img_files.extend(list(self.img_dir.glob(ext.upper())))
        
        self.img_files = sorted(self.img_files)
        
        # 对应的标签文件
        self.label_files = [
            self.label_dir / f"{img_file.stem}.txt" 
            for img_file in self.img_files
        ]
        
        print(f"找到 {len(self.img_files)} 张图片")
    
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        # 加载图片
        img_path = self.img_files[idx]
        img = Image.open(img_path).convert('RGB')
        orig_w, orig_h = img.size
        
        # 调整图片大小
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        img = np.array(img).astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)  # HWC -> CHW
        
        # 加载标签
        label_path = self.label_files[idx]
        boxes = []
        
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    line = line.strip()
                    if line:
                        # YOLO格式: class_id x_center y_center width height (归一化坐标)
                        values = line.split()
                        class_id = int(values[0])
                        x_center = float(values[1])
                        y_center = float(values[2])
                        width = float(values[3])
                        height = float(values[4])
                        boxes.append([class_id, x_center, y_center, width, height])
        
        if len(boxes) == 0:
            # 没有标注框，创建空张量
            boxes = torch.zeros((0, 5), dtype=torch.float32)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
        
        return img, boxes, str(img_path)


def collate_fn(batch):
    """自定义批处理函数，处理不同数量的边界框"""
    imgs, boxes, paths = zip(*batch)
    
    # 堆叠图片
    imgs = torch.stack(imgs, 0)
    
    # 为每个标签添加batch索引
    boxes_with_idx = []
    for i, box in enumerate(boxes):
        if box.shape[0] > 0:
            # 在第一列添加batch索引
            batch_idx = torch.full((box.shape[0], 1), i, dtype=torch.float32)
            box_with_idx = torch.cat([batch_idx, box], dim=1)
            boxes_with_idx.append(box_with_idx)
    
    # 合并所有boxes
    if len(boxes_with_idx) > 0:
        targets = torch.cat(boxes_with_idx, 0)
    else:
        targets = torch.zeros((0, 6), dtype=torch.float32)
    
    return imgs, targets, paths


def load_yaml_config(yaml_path):
    """加载YAML配置文件"""
    with open(yaml_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def create_dataloaders(config_path, batch_size=16, img_size=640, workers=0):
    """创建训练和验证数据加载器"""
    
    # 加载配置
    config = load_yaml_config(config_path)
    
    # 获取数据集根路径
    data_yaml_path = Path(config_path)
    if data_yaml_path.parent.name == config['path']:
        # data.yaml在数据集根目录下
        root_path = data_yaml_path.parent
    else:
        # 使用配置中的path
        root_path = Path(config['path'])
    
    print(f"数据集根路径: {root_path}")
    
    # 检测是否有GPU
    use_cuda = torch.cuda.is_available()
    
    # 创建训练集
    # config['train'] 是 'images/train'
    train_img_dir = root_path / config['train']
    # 将 'images/train' 替换为 'labels/train'
    train_label_path = config['train'].replace('images', 'labels')
    train_label_dir = root_path / train_label_path
    
    print(f"训练集图片路径: {train_img_dir}")
    print(f"训练集标签路径: {train_label_dir}")
    
    train_dataset = YOLODataset(
        img_dir=train_img_dir,
        label_dir=train_label_dir,
        img_size=img_size,
        augment=True
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        collate_fn=collate_fn,
        pin_memory=False,
        persistent_workers=False
    )
    
    # 创建验证集
    val_img_dir = root_path / config['val']
    val_label_path = config['val'].replace('images', 'labels')
    val_label_dir = root_path / val_label_path
    
    print(f"验证集图片路径: {val_img_dir}")
    print(f"验证集标签路径: {val_label_dir}")
    
    val_dataset = YOLODataset(
        img_dir=val_img_dir,
        label_dir=val_label_dir,
        img_size=img_size,
        augment=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        collate_fn=collate_fn,
        pin_memory=False,
        persistent_workers=False
    )
    
    return train_loader, val_loader, config


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


def validate(model, dataloader, device):
    """验证模型"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for imgs, targets, paths in dataloader:
            imgs = imgs.to(device)
            targets = targets.to(device)
            
            # 尝试不同的调用方式
            try:
                outputs = model(imgs, targets)
            except TypeError:
                outputs = model(imgs)
                if hasattr(model, 'compute_loss'):
                    outputs = {'predictions': outputs, 'loss': model.compute_loss(outputs, targets)}
                elif hasattr(model, 'detect') and hasattr(model.detect, 'compute_loss'):
                    outputs = {'predictions': outputs, 'loss': model.detect.compute_loss(outputs, targets)}
                else:
                    loss = torch.tensor(1.0, device=device)
                    outputs = {'loss': loss}
            
            # 获取loss
            if isinstance(outputs, dict):
                loss = outputs.get('loss', None)
                if loss is None:
                    raise ValueError("无法获取loss")
            elif isinstance(outputs, (tuple, list)):
                loss = outputs[-1]
            else:
                loss = outputs
            
            # 确保loss是标量
            if hasattr(loss, 'dim') and loss.dim() > 0:
                loss = loss.mean()
            
            total_loss += loss.item()
    
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


# ============ 使用示例 ============

if __name__ == "__main__":
    # 1. 导入你的YOLO模型
    # from your_model import YOLOModel
    # model = YOLOModel(num_classes=2)
    
    # 2. 开始训练
    # trained_model = train(
    #     model=model,
    #     config_path='MY_TEST_DATA/data.yaml',
    #     epochs=100,
    #     batch_size=16,
    #     img_size=640,
    #     lr=0.01,
    #     device='cuda',
    #     save_dir='runs/train/exp1'
    # )
    
    # 3. 或者单独使用数据加载器
    train_loader, val_loader, config = create_dataloaders(
        config_path='MY_TEST_DATA/data.yaml',
        batch_size=16,
        img_size=640
    )
    
    # 测试数据加载
    print("测试数据加载...")
    for imgs, targets, paths in train_loader:
        print(f"图片批次形状: {imgs.shape}")
        print(f"目标形状: {targets.shape}")
        print(f"目标内容 (batch_idx, class_id, x, y, w, h):")
        print(targets[:5])  # 打印前5个目标
        break