"""
Coordinate Attention 可视化脚本

展示 CoordAtt 模块在图像上关注的重点位置，使用热力图形式
"""
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision import transforms

from models import CoordAtt


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


def load_image(image_path, img_size=224):
    """加载并预处理图像"""
    image = Image.open(image_path).convert('RGB')

    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    img_tensor = transform(image).unsqueeze(0)  # (1, 3, H, W)

    # 用于可视化的原始图像（反归一化）
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    img_display = inv_normalize(img_tensor.squeeze(0)).permute(1, 2, 0).numpy()
    img_display = np.clip(img_display, 0, 1)

    return img_tensor, img_display


def visualize_attention(image_path, save_path='attention_visualization.png'):
    """可视化注意力权重"""

    # 1. 加载图像
    img_tensor, img_display = load_image(image_path, img_size=224)

    # 2. 创建 CoordAtt 模块 (输入通道=3，输出通道=3，用于RGB图像)
    coord_att = CoordAttWithVisualization(inp=3, oup=3, reduction=32)
    coord_att.eval()

    # 3. 前向传播，获取注意力权重
    with torch.no_grad():
        output, a_h, a_w = coord_att.forward_with_attention(img_tensor)

    # 4. 处理注意力权重用于可视化
    # a_h: (1, 3, H, 1), a_w: (1, 3, 1, W)
    # 合并水平和垂直注意力，得到完整的注意力图
    attention_map = (a_h * a_w).squeeze(0)  # (3, H, W)

    # 对三个通道的注意力取平均
    attention_map = attention_map.mean(0).cpu().numpy()  # (H, W)

    # 5. 创建可视化
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 原始图像
    axes[0].imshow(img_display)
    axes[0].set_title('Original Image', fontsize=14)
    axes[0].axis('off')

    # 注意力热力图
    im = axes[1].imshow(attention_map, cmap='jet', vmin=0, vmax=1)
    axes[1].set_title('Attention Heatmap', fontsize=14)
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046)

    # 叠加显示
    axes[2].imshow(img_display)
    im2 = axes[2].imshow(attention_map, cmap='jet', alpha=0.5, vmin=0, vmax=1)
    axes[2].set_title('Overlay', fontsize=14)
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"可视化结果已保存到: {save_path}")

    plt.show()


def visualize_multiple_images(image_dir, num_samples=4, save_path='multi_attention.png'):
    """可视化多张图像的注意力效果"""

    # 获取图像文件
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]
    image_files = sorted(image_files)[:num_samples]

    # 创建 CoordAtt 模块
    coord_att = CoordAttWithVisualization(inp=3, oup=3, reduction=32)
    coord_att.eval()

    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))

    for idx, img_file in enumerate(image_files):
        img_path = os.path.join(image_dir, img_file)

        # 加载图像
        img_tensor, img_display = load_image(img_path, img_size=224)

        # 获取注意力
        with torch.no_grad():
            output, a_h, a_w = coord_att.forward_with_attention(img_tensor)

        # 处理注意力图
        attention_map = (a_h * a_w).squeeze(0).mean(0).cpu().numpy()

        # 显示
        # 原始图像
        axes[idx, 0].imshow(img_display)
        axes[idx, 0].set_title(f'{img_file} - Original', fontsize=10)
        axes[idx, 0].axis('off')

        # 水平注意力
        h_att = a_h.squeeze(0).mean(0).squeeze().cpu().numpy()
        axes[idx, 1].imshow(h_att.reshape(-1, 1), cmap='jet', aspect='auto', vmin=0, vmax=1)
        axes[idx, 1].set_title('Horizontal Attention', fontsize=10)
        axes[idx, 1].axis('off')

        # 垂直注意力
        w_att = a_w.squeeze(0).mean(0).squeeze().cpu().numpy()
        axes[idx, 2].imshow(w_att.reshape(1, -1), cmap='jet', aspect='auto', vmin=0, vmax=1)
        axes[idx, 2].set_title('Vertical Attention', fontsize=10)
        axes[idx, 2].axis('off')

        # 叠加
        axes[idx, 3].imshow(img_display)
        axes[idx, 3].imshow(attention_map, cmap='jet', alpha=0.5, vmin=0, vmax=1)
        axes[idx, 3].set_title('Attention Overlay', fontsize=10)
        axes[idx, 3].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"多图可视化已保存到: {save_path}")

    plt.show()


if __name__ == '__main__':
    import matplotlib
    matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False

    # 设置数据集路径
    data_root = 'datasets/MY_TEST_DATA/images/train'

    print("=" * 50)
    print("Coordinate Attention 可视化")
    print("=" * 50)

    # 1. 单张图像详细可视化
    print("\n[1] 单张图像可视化...")
    sample_image = os.path.join(data_root, 'circle_0001.jpg')
    if os.path.exists(sample_image):
        visualize_attention(sample_image, save_path='single_attention.png')
    else:
        print(f"图像不存在: {sample_image}")

    # 2. 多张图像对比可视化
    print("\n[2] 多张图像对比可视化...")
    visualize_multiple_images(data_root, num_samples=4, save_path='multi_attention.png')

    print("\n" + "=" * 50)
    print("可视化完成!")
    print("=" * 50)
