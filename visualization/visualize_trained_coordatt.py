"""
Coordinate Attention 训练与可视化脚本 (优化版)

使用多个 CoordAtt 层堆叠，增强注意力效果的可视化
"""
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from datetime import datetime

# 添加父目录到路径以导入 models
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import CoordAtt, Conv

# 创建带时间戳的输出目录
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = os.path.join('outputs', f'attention_vis_{TIMESTAMP}')
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"输出目录: {OUTPUT_DIR}")


class CoordAttWithVisualization(CoordAtt):
    """带可视化功能的 Coordinate Attention，返回注意力权重"""

    def forward_with_attention(self, x):
        """返回输出和注意力权重"""
        n, c, h, w = x.size()

        # 信息嵌入 (H, W 维度聚合)
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        # 拼接与融合
        y = self.cv1(torch.cat([x_h, x_w], dim=2))

        # 拆分
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        # 加权输出
        a_h = self.cv_h(x_h).sigmoid()
        a_w = self.cv_w(x_w).sigmoid()

        output = x * a_h * a_w

        return output, a_h, a_w


class SimpleShapeDataset(Dataset):
    """简单的形状分类数据集 (圆形 vs 方形)"""

    def __init__(self, img_dir, img_size=224, augment=False):
        self.img_dir = Path(img_dir)
        self.img_size = img_size
        self.augment = augment
        self.img_files = list(self.img_dir.glob('*.jpg')) + list(self.img_dir.glob('*.png'))

        # 根据文件名确定标签
        self.labels = []
        for f in self.img_files:
            if 'circle' in f.name.lower():
                self.labels.append(0)
            elif 'square' in f.name.lower():
                self.labels.append(1)
            else:
                self.labels.append(0)

        print(f"加载了 {len(self.img_files)} 张图片")

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        label = self.labels[idx]

        # 加载图片
        img = Image.open(img_path).convert('RGB')
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        img = np.array(img).astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)  # HWC -> CHW

        return img, label, str(img_path)


class DeepCoordAttClassifier(nn.Module):
    """使用多个 CoordAtt 层的深度分类器"""

    def __init__(self, num_classes=2):
        super().__init__()

        # 使用更小的 reduction 值 (8 而不是 16) 来获得更强的注意力表达能力
        # 堆叠多个 CoordAtt 层

        # 第一阶段: 低层特征
        self.stage1 = nn.Sequential(
            Conv(3, 32, k=7, s=2, p=3),  # 大卷积核捕获更多上下文
            CoordAttWithVisualization(inp=32, oup=32, reduction=4),  # 第一层注意力
        )

        # 第二阶段: 中层特征
        self.stage2 = nn.Sequential(
            Conv(32, 64, k=3, s=2, p=1),
            CoordAttWithVisualization(inp=64, oup=64, reduction=4),  # 第二层注意力
        )

        # 第三阶段: 高层特征
        self.stage3 = nn.Sequential(
            Conv(64, 128, k=3, s=2, p=1),
            CoordAttWithVisualization(inp=128, oup=128, reduction=8),  # 第三层注意力
            Conv(128, 256, k=3, s=2, p=1),
            CoordAttWithVisualization(inp=256, oup=256, reduction=8),  # 第四层注意力
        )

        # 分类头
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

        # 存储所有的 CoordAtt 层用于可视化
        self.coord_att_layers = nn.ModuleList([
            self.stage1[1],
            self.stage2[1],
            self.stage3[1],
            self.stage3[3]
        ])

    def forward(self, x, return_attention=False, layer_idx=-1):
        """
        Args:
            x: 输入图像
            return_attention: 是否返回注意力权重
            layer_idx: 要返回哪一层的注意力 (-1 表示最后一层, 0-3 表示特定层)
        """
        # Stage 1
        x = self.stage1(x)

        # Stage 2
        x = self.stage2(x)

        # Stage 3
        x = self.stage3[:2](x)  # Conv + CoordAtt
        x = self.stage3[2](x)    # Conv
        x = self.stage3[3](x)    # CoordAtt

        # 分类
        x = self.classifier(x)

        return x

    def forward_with_attention(self, x, layer_idx=-1):
        """前向传播并返回指定层的注意力权重"""
        # Stage 1
        x = self.stage1[0](x)  # Conv
        if layer_idx == 0:
            feat, a_h, a_w = self.coord_att_layers[0].forward_with_attention(x)
            x = feat
        else:
            x = self.coord_att_layers[0](x)

        # Stage 2
        x = self.stage2[0](x)  # Conv
        if layer_idx == 1:
            feat, a_h, a_w = self.coord_att_layers[1].forward_with_attention(x)
            x = feat
        else:
            x = self.coord_att_layers[1](x)

        # Stage 3 - 第一部分
        x = self.stage3[0](x)  # Conv
        if layer_idx == 2:
            feat, a_h, a_w = self.coord_att_layers[2].forward_with_attention(x)
            x = feat
        else:
            x = self.coord_att_layers[2](x)

        # Stage 3 - 第二部分
        x = self.stage3[2](x)  # Conv
        if layer_idx == 3:
            feat, a_h, a_w = self.coord_att_layers[3].forward_with_attention(x)
            x = feat
        else:
            x = self.coord_att_layers[3](x)

        # 分类
        logits = self.classifier(x)

        if layer_idx >= 0:
            return logits, a_h, a_w
        return logits, None, None


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


def train_model(model, train_loader, val_loader, epochs=50, lr=0.001, device='cpu'):
    """训练模型 - 增强版训练策略"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    print("=" * 50)
    print("开始训练...")
    print("=" * 50)

    best_acc = 0
    patience = 15
    patience_counter = 0

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        scheduler.step()

        print(f"Epoch [{epoch+1}/{epochs}] "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% | "
              f"LR: {scheduler.get_last_lr()[0]:.6f}")

        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"早停: 验证准确率 {patience} 轮未提升")
            break

    print("=" * 50)
    print(f"训练完成! 最佳验证准确率: {best_acc:.2f}%")
    print("=" * 50)


def enhance_contrast(attention_map):
    """增强注意力图的对比度"""
    # 使用 percentile 范围来增强对比度
    p2, p98 = np.percentile(attention_map, (2, 98))
    enhanced = np.clip((attention_map - p2) / (p98 - p2 + 1e-8), 0, 1)
    return enhanced


def visualize_trained_attention_multi_layer(model, dataloader, device, save_path='attention_multi_layer.png'):
    """可视化多层的注意力效果"""

    model.eval()

    # 获取样本
    samples = []
    for imgs, labels, paths in dataloader:
        for i in range(min(4, len(imgs))):
            samples.append((imgs[i], labels[i], paths[i]))
        if len(samples) >= 4:
            break

    # 4张图片，每张展示4个注意力层，总共5列（原图+3个注意力层+叠加）
    fig, axes = plt.subplots(4, 5, figsize=(20, 16))

    layer_names = ['Layer 1 (Low)', 'Layer 2 (Mid)', 'Layer 3 (High)', 'Layer 4 (Top)']

    with torch.no_grad():
        for idx, (img, label, path) in enumerate(samples):
            img_input = img.unsqueeze(0).to(device)

            # 原始图像
            img_display = img.permute(1, 2, 0).numpy()
            axes[idx, 0].imshow(img_display)
            label_name = ["Circle", "Square"][label]
            axes[idx, 0].set_title(f'{label_name}', fontsize=14, fontweight='bold')
            axes[idx, 0].axis('off')

            # 获取每一层的注意力
            attention_maps = []

            for layer_idx in range(4):
                output, a_h, a_w = model.forward_with_attention(img_input, layer_idx=layer_idx)
                pred = output.argmax(1).item()

                # 合并注意力
                attention = (a_h * a_w).squeeze(0).mean(0).cpu().numpy()
                attention_maps.append(attention)

                # 上采样到原始图像大小
                attention_full = Image.fromarray((attention * 255).astype(np.uint8)).resize(
                    (224, 224), Image.BILINEAR
                )
                attention_full = np.array(attention_full) / 255.0

                # 增强对比度
                attention_enhanced = enhance_contrast(attention_full)

                # 显示注意力热力图
                im = axes[idx, layer_idx + 1].imshow(
                    attention_enhanced,
                    cmap='inferno',  # 使用 inferno 色彩映射，对比度更高
                    vmin=0, vmax=1
                )
                axes[idx, layer_idx + 1].set_title(layer_names[layer_idx], fontsize=11)
                axes[idx, layer_idx + 1].axis('off')

            # 在最后一列叠加最强的注意力
            strongest_attention = Image.fromarray((attention_maps[-1] * 255).astype(np.uint8)).resize(
                (224, 224), Image.BILINEAR
            )
            strongest_attention = np.array(strongest_attention) / 255.0
            strongest_attention = enhance_contrast(strongest_attention)

            axes[idx, 4].imshow(img_display)
            axes[idx, 4].imshow(strongest_attention, cmap='inferno', alpha=0.7)
            pred_name = ["Circle", "Square"][pred]
            axes[idx, 4].set_title(f'Pred: {pred_name}', fontsize=14, fontweight='bold')
            axes[idx, 4].axis('off')

    plt.suptitle('Multi-Layer Coordinate Attention Visualization', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"\n多层可视化结果已保存到: {save_path}")
    plt.close()

    # 单独创建对比图：训练前 vs 训练后
    visualize_comparison(model, dataloader, device, save_path.replace('.png', '_comparison.png'))


def visualize_comparison(model, dataloader, device, save_path='attention_comparison.png'):
    """创建训练前后对比图"""

    # 创建一个未训练的模型用于对比
    untrained_model = DeepCoordAttClassifier(num_classes=2).to(device)
    untrained_model.eval()

    model.eval()

    # 获取样本
    samples = []
    for imgs, labels, paths in dataloader:
        for i in range(min(2, len(imgs))):
            samples.append((imgs[i], labels[i], paths[i]))
        if len(samples) >= 2:
            break

    fig, axes = plt.subplots(2, 4, figsize=(18, 10))

    with torch.no_grad():
        for idx, (img, label, path) in enumerate(samples):
            img_input = img.unsqueeze(0).to(device)
            img_display = img.permute(1, 2, 0).numpy()

            # 未训练模型
            _, a_h_untrained, a_w_untrained = untrained_model.forward_with_attention(img_input, layer_idx=3)
            att_untrained = (a_h_untrained * a_w_untrained).squeeze(0).mean(0).cpu().numpy()
            att_untrained_full = np.array(Image.fromarray((att_untrained * 255).astype(np.uint8)).resize((224, 224), Image.BILINEAR)) / 255.0
            att_untrained_enhanced = enhance_contrast(att_untrained_full)

            # 训练后模型
            output, a_h_trained, a_w_trained = model.forward_with_attention(img_input, layer_idx=3)
            pred = output.argmax(1).item()
            att_trained = (a_h_trained * a_w_trained).squeeze(0).mean(0).cpu().numpy()
            att_trained_full = np.array(Image.fromarray((att_trained * 255).astype(np.uint8)).resize((224, 224), Image.BILINEAR)) / 255.0
            att_trained_enhanced = enhance_contrast(att_trained_full)

            # 显示
            axes[idx, 0].imshow(img_display)
            axes[idx, 0].set_title(f'Original ({["Circle", "Square"][label]})', fontsize=12)
            axes[idx, 0].axis('off')

            im1 = axes[idx, 1].imshow(att_untrained_enhanced, cmap='inferno', vmin=0, vmax=1)
            axes[idx, 1].set_title('Before Training', fontsize=12)
            axes[idx, 1].axis('off')

            im2 = axes[idx, 2].imshow(att_trained_enhanced, cmap='inferno', vmin=0, vmax=1)
            axes[idx, 2].set_title('After Training', fontsize=12)
            axes[idx, 2].axis('off')

            axes[idx, 3].imshow(img_display)
            axes[idx, 3].imshow(att_trained_enhanced, cmap='inferno', alpha=0.7)
            axes[idx, 3].set_title(f'Overlay (Pred: {["Circle", "Square"][pred]})', fontsize=12)
            axes[idx, 3].axis('off')

    plt.suptitle('Attention Before vs After Training (Layer 4 - Highest Level)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"对比可视化已保存到: {save_path}")
    plt.close()


if __name__ == '__main__':
    import matplotlib
    matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False

    # 配置
    data_root = 'datasets/MY_TEST_DATA'
    img_size = 224
    batch_size = 16
    epochs = 50  # 增加训练轮数
    lr = 0.001
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"使用设备: {device}")

    # 创建数据集
    print("\n加载数据集...")
    train_dataset = SimpleShapeDataset(
        img_dir=os.path.join(data_root, 'images/train'),
        img_size=img_size,
        augment=True
    )
    val_dataset = SimpleShapeDataset(
        img_dir=os.path.join(data_root, 'images/val'),
        img_size=img_size
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # 创建深度模型
    print("\n创建深度模型 (4层 CoordAtt)...")
    model = DeepCoordAttClassifier(num_classes=2).to(device)

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")

    # 训练前可视化
    print("\n" + "=" * 50)
    print("训练前注意力可视化 (随机初始化):")
    print("=" * 50)
    visualize_trained_attention_multi_layer(model, val_loader, device,
                                             save_path=os.path.join(OUTPUT_DIR, 'attention_before_training.png'))

    # 训练模型
    train_model(model, train_loader, val_loader, epochs=epochs, lr=lr, device=device)

    # 训练后可视化
    print("\n" + "=" * 50)
    print("训练后注意力可视化 (已学习):")
    print("=" * 50)
    visualize_trained_attention_multi_layer(model, val_loader, device,
                                             save_path=os.path.join(OUTPUT_DIR, 'attention_after_training.png'))

    print("\n完成!")
