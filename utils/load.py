import yaml
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Any

def get_data_loader(path : str, is_train: bool):
    to_tensor = transforms.Compose([transforms.ToTensor()])
    data_set = datasets.MNIST(path, is_train, transform=to_tensor, download=True)
    return DataLoader(data_set, batch_size=15, shuffle=True)

class YOLODataset(Dataset):
    """YOLO格式数据集加载器"""

    def __init__(self, img_dir, label_dir, img_size=640, augment=False, transform=None):
        """
        Args:
            img_dir: 图片目录路径
            label_dir: 标签目录路径
            img_size: 输入图像尺寸
            augment: 是否进行数据增强（保留用于兼容）
            transform: 自定义数据增强变换（Mosaic, Mixup 等）
        """
        self.img_dir = Path(img_dir)
        self.label_dir = Path(label_dir)
        self.img_size = img_size
        self.augment = augment
        self.transform = transform
        
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

    def _load_raw_item(self, idx: int):
        """加载原始图片和标签（不经过 transform，避免 Mosaic 递归）

        Args:
            idx: 样本索引

        Returns:
            (img, boxes): PIL Image 和归一化坐标的 boxes 张量
        """
        img_path = self.img_files[idx]
        img = Image.open(img_path).convert('RGB')

        label_path = self.label_files[idx]
        boxes = []

        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    line = line.strip()
                    if line:
                        values = line.split()
                        class_id = int(values[0])
                        x_center = float(values[1])
                        y_center = float(values[2])
                        width = float(values[3])
                        height = float(values[4])
                        boxes.append([class_id, x_center, y_center, width, height])

        if len(boxes) == 0:
            boxes = torch.zeros((0, 5), dtype=torch.float32)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)

        return img, boxes

    def __getitem__(self, idx):
        # 加载原始图片和标签
        img, boxes = self._load_raw_item(idx)

        # 应用数据增强（Mosaic, Mixup 等）
        if self.transform is not None:
            img, boxes = self.transform(img, boxes)

        # 调整图片大小
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        img = np.array(img).astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)  # HWC -> CHW

        return img, boxes, str(self.img_files[idx])

def load_yaml_config(yaml_path):
    """加载YAML配置文件"""
    with open(yaml_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


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