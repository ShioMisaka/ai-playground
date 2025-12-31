import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import xml.etree.ElementTree as ET
from PIL import Image
import os
import urllib.request
import tarfile
from pathlib import Path

def get_data_loader(path : str, is_train: bool):
    to_tensor = transforms.Compose([transforms.ToTensor()])
    data_set = datasets.MNIST(path, is_train, transform=to_tensor, download=True)
    return DataLoader(data_set, batch_size=15, shuffle=True)

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import xml.etree.ElementTree as ET
from PIL import Image
import os
import urllib.request
import tarfile
from pathlib import Path


def voc_collate_fn(batch):
    """
    自定义collate函数，因为每张图片的目标数量不同
    """
    images = []
    targets = []
    
    for image, target in batch:
        images.append(image)
        targets.append(target)
    
    images = torch.stack(images, 0)
    return images, targets


def manual_download_voc(root, year='2007'):
    """
    手动下载VOC数据集（备用方案）
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    
    # 镜像下载链接
    urls = {
        '2007': [
            'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar',
            'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar',
        ],
        '2012': [
            'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar',
        ]
    }
    
    print(f"开始下载VOC{year}数据集...")
    for url in urls.get(year, []):
        filename = url.split('/')[-1]
        filepath = root / filename
        
        if not filepath.exists():
            print(f"下载 {filename}...")
            try:
                urllib.request.urlretrieve(url, filepath)
                print(f"下载完成: {filename}")
            except Exception as e:
                print(f"下载失败: {e}")
                print(f"\n请手动下载:")
                print(f"  链接: {url}")
                print(f"  保存到: {filepath}")
                raise
        
        # 解压
        extract_path = root / 'VOCdevkit'
        if not extract_path.exists():
            print(f"解压 {filename}...")
            with tarfile.open(filepath, 'r') as tar:
                tar.extractall(root)
            print(f"解压完成")


class VOCDataset(torch.utils.data.Dataset):
    """
    PASCAL VOC数据集封装
    """
    def __init__(self, root, year='2007', image_set='train', transform=None, download=True):
        """
        Args:
            root: 数据集根目录
            year: '2007' 或 '2012'
            image_set: 'train', 'val', 'trainval', 或 'test'
            transform: 图像变换
            download: 是否自动下载
        """
        self.root = root
        self.year = year
        self.image_set = image_set
        self.transform = transform
        
        # 尝试使用torchvision下载，失败则手动下载
        try:
            self.voc = datasets.VOCDetection(
                root=root,
                year=year,
                image_set=image_set,
                download=download
            )
        except RuntimeError as e:
            if download and "not found or corrupted" in str(e):
                print("torchvision下载失败，尝试手动下载...")
                manual_download_voc(root, year)
                # 重新尝试加载
                self.voc = datasets.VOCDetection(
                    root=root,
                    year=year,
                    image_set=image_set,
                    download=False
                )
            else:
                raise
        
        # VOC类别
        self.classes = [
            'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
            'bus', 'car', 'cat', 'chair', 'cow',
            'diningtable', 'dog', 'horse', 'motorbike', 'person',
            'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
        ]
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
    def __len__(self):
        return len(self.voc)
    
    def __getitem__(self, idx):
        """
        返回:
            image: tensor (C, H, W)
            target: dict包含boxes, labels等信息
        """
        image, annotation = self.voc[idx]
        
        # 解析XML标注
        boxes = []
        labels = []
        
        for obj in annotation['annotation']['object']:
            # 获取类别
            class_name = obj['name']
            if class_name not in self.class_to_idx:
                continue
            label = self.class_to_idx[class_name]
            
            # 获取bbox坐标 (xmin, ymin, xmax, ymax)
            bbox = obj['bndbox']
            xmin = float(bbox['xmin'])
            ymin = float(bbox['ymin'])
            xmax = float(bbox['xmax'])
            ymax = float(bbox['ymax'])
            
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(label)
        
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)
        
        # 应用transform
        if self.transform:
            image = self.transform(image)
        
        # 构建target字典
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([idx])
        }
        
        return image, target


def get_data_loader(path: str, is_train: bool, batch_size=8, year='2007'):
    """
    获取VOC数据加载器
    
    Args:
        path: 数据集根目录
        is_train: True为训练集，False为验证集
        batch_size: batch大小
        year: VOC年份 '2007' 或 '2012'
    
    Returns:
        DataLoader
    """
    # 数据预处理
    if is_train:
        transform = transforms.Compose([
            transforms.Resize((416, 416)),  # YOLOv3常用输入尺寸
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        image_set = 'train'
        shuffle = True
    else:
        transform = transforms.Compose([
            transforms.Resize((416, 416)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        image_set = 'val'
        shuffle = False
    
    # 创建数据集
    dataset = VOCDataset(
        root=path,
        year=year,
        image_set=image_set,
        transform=transform,
        download=True
    )
    
    # 创建DataLoader
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=voc_collate_fn,
        num_workers=4,
        pin_memory=True
    )


def get_simple_data_loader(path: str, is_train: bool, batch_size=8):
    """
    简化版数据加载器（如果你只需要简单的tensor）
    """
    to_tensor = transforms.Compose([
        transforms.Resize((416, 416)),
        transforms.ToTensor()
    ])
    
    image_set = 'train' if is_train else 'val'
    
    dataset = VOCDataset(
        root=path,
        year='2007',
        image_set=image_set,
        transform=to_tensor,
        download=True
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        collate_fn=voc_collate_fn
    )