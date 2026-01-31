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

    def __init__(self, img_dir, label_dir, img_size=640, augment=False, transform=None, letterbox=True):
        """
        Args:
            img_dir: 图片目录路径
            label_dir: 标签目录路径
            img_size: 输入图像尺寸
            augment: 是否进行数据增强（保留用于兼容）
            transform: 自定义数据增强变换（Mosaic, Mixup 等）
            letterbox: 是否使用 letterbox 预处理
        """
        self.img_dir = Path(img_dir)
        self.label_dir = Path(label_dir)
        self.img_size = img_size
        self.augment = augment
        self.transform = transform
        self.letterbox = letterbox  # 新增
        
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

        # 转换为 numpy 数组
        img = np.array(img).astype(np.float32)

        # 预处理：letterbox 或简单 resize
        letterbox_params = None  # 存储letterbox参数 (r, pad_w, pad_h)

        import cv2
        img_h, img_w = img.shape[:2]

        # 检测是否是 Mosaic 输出的图像（2*img_size × 2*img_size）
        is_mosaic = (img_h == self.img_size * 2) and (img_w == self.img_size * 2)

        if self.letterbox:
            if is_mosaic:
                # Mosaic 图像：从 2*img_size × 2*img_size resize 到 img_size × img_size
                img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)

                # Boxes 坐标需要缩放 0.5（因为图像尺寸缩小了一半）
                if len(boxes) > 0:
                    boxes[:, 1:] *= 0.5  # 所有坐标缩放 0.5

                # Mosaic 不需要 letterbox 参数（没有填充偏移）
                letterbox_params = None
            else:
                # 非 Mosaic 图像：使用标准 letterbox
                r = min(self.img_size / img_h, self.img_size / img_w)
                scaled_h, scaled_w = int(round(img_h * r)), int(round(img_w * r))

                # 缩放
                img = cv2.resize(img, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)

                # 填充 - 确保最终尺寸精确为 img_size
                pad_h = self.img_size - scaled_h
                pad_w = self.img_size - scaled_w
                # 上下左右均分填充
                top, bottom = pad_h // 2, pad_h - pad_h // 2
                left, right = pad_w // 2, pad_w - pad_w // 2
                img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))

                # 存储 letterbox 参数
                letterbox_params = (r, left, top)

                # 调整边界框坐标以匹配 letterbox 变换
                if len(boxes) > 0:
                    # Boxes 格式: [class_id, x_center, y_center, width, height] (归一化坐标)
                    # 1. 应用缩放因子 r
                    boxes[:, 1] *= r  # x_center
                    boxes[:, 2] *= r  # y_center
                    boxes[:, 3] *= r  # width
                    boxes[:, 4] *= r  # height

                    # 2. 调整中心点坐标（加上填充偏移）
                    # 归一化的偏移量 = pad / img_size
                    boxes[:, 1] += left / self.img_size  # x_center
                    boxes[:, 2] += top / self.img_size  # y_center
        else:
            # 简单 resize - 不需要letterbox参数
            img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)

            # 归一化坐标需要随图像尺寸缩放
            # 坐标缩放因子 = old_size / new_size
            if len(boxes) > 0:
                scale_x = img_w / self.img_size  # x 方向缩放
                scale_y = img_h / self.img_size  # y 方向缩放
                boxes[:, 1] *= scale_x  # x_center
                boxes[:, 2] *= scale_y  # y_center
                boxes[:, 3] *= scale_x  # width
                boxes[:, 4] *= scale_y  # height

        # 归一化
        img = img.astype(np.float32) / 255.0

        # HWC -> CHW
        img = torch.from_numpy(img).permute(2, 0, 1)

        return img, boxes, str(self.img_files[idx]), letterbox_params

def load_yaml_config(yaml_path):
    """加载YAML配置文件"""
    with open(yaml_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def collate_fn(batch):
    """自定义批处理函数，处理不同数量的边界框和 letterbox 参数

    Args:
        batch: (img, boxes, path, letterbox_params) 元组列表

    Returns:
        imgs: 堆叠的图像张量
        targets: 合并的目标张量 (N, 9) - [batch_idx, class_id, x, y, w, h, r, pad_w, pad_h]
        paths: 图像路径列表
        letterbox_params_list: 每个 sample 的 letterbox 参数列表
    """
    imgs, boxes, paths, letterbox_params_list = zip(*batch)

    # 堆叠图片
    imgs = torch.stack(imgs, 0)

    # 为每个标签添加batch索引和letterbox参数
    boxes_with_idx = []
    for i, (box, lb_params) in enumerate(zip(boxes, letterbox_params_list)):
        if box.shape[0] > 0:
            # 在第一列添加batch索引
            batch_idx = torch.full((box.shape[0], 1), i, dtype=torch.float32)
            box_with_idx = torch.cat([batch_idx, box], dim=1)

            # 添加 letterbox 参数 (r, left, top) 到每个box
            # 这样每行变成: [batch_idx, class_id, x, y, w, h, r, left, top]
            if lb_params is not None:
                r, left, top = lb_params
                # 使用 expand 创建相同值的 tensor
                lb_values = torch.tensor([[r, left, top]], dtype=torch.float32)
                lb_tensor = lb_values.expand(box.shape[0], 3)
                box_with_idx = torch.cat([box_with_idx, lb_tensor], dim=1)
            else:
                # 没有letterbox (使用简单resize)，用0填充
                lb_tensor = torch.zeros((box.shape[0], 3), dtype=torch.float32)
                box_with_idx = torch.cat([box_with_idx, lb_tensor], dim=1)

            boxes_with_idx.append(box_with_idx)

    # 合并所有boxes
    if len(boxes_with_idx) > 0:
        targets = torch.cat(boxes_with_idx, 0)
    else:
        # 空目标，格式：[batch_idx, class_id, x, y, w, h, r, pad_w, pad_h]
        targets = torch.zeros((0, 9), dtype=torch.float32)

    return imgs, targets, paths, list(letterbox_params_list)

def create_dataloaders(config_path, batch_size=16, img_size=640, workers=0, letterbox=True):
    """创建训练和验证数据加载器

    Args:
        config_path: 数据配置文件路径
        batch_size: 批大小
        img_size: 图像尺寸
        workers: 数据加载线程数
        letterbox: 是否使用 letterbox 预处理
    """
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

    # 创建训练集
    # config['train'] 是 'images/train'
    train_img_dir = root_path / config['train']
    # 将 'images/train' 替换为 'labels/train'
    train_label_path = config['train'].replace('images', 'labels')
    train_label_dir = root_path / train_label_path

    train_dataset = YOLODataset(
        img_dir=train_img_dir,
        label_dir=train_label_dir,
        img_size=img_size,
        augment=True,
        letterbox=letterbox  # 新增
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

    val_dataset = YOLODataset(
        img_dir=val_img_dir,
        label_dir=val_label_dir,
        img_size=img_size,
        augment=False,
        letterbox=letterbox  # 新增
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