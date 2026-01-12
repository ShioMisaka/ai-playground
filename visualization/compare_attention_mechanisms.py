"""
注意力机制对比脚本 - 对比 CoordAtt 和 CoordCrossAtt (YOLO 检测版本)

使用 YOLO 数据集进行目标检测训练和可视化对比
"""
import os
import sys
import torch
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from PIL import Image
from pathlib import Path
from datetime import datetime

# 添加父目录到路径以导入 models 和 engine
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import YOLOCoordAttDetector, YOLOCoordCrossAttDetector
from engine.detector import train_detector
from engine.visualize import enhance_contrast
from utils import create_dataloaders
from utils.load import load_yaml_config

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False


# ==================== 配置 ====================
CONFIG = {
    'img_size': 256,          # 输入图像大小 (缩小以加快训练)
    'batch_size': 8,          # 批大小
    'epochs': 15,             # 训练轮数
    'lr': 0.001,              # 学习率
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'data_config': 'datasets/MY_TEST_DATA/data.yaml',
    'save_dir': 'outputs/attention_comparison',
}

# 创建输出目录
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = os.path.join(CONFIG['save_dir'], f'run_{TIMESTAMP}')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_coordatt_attention(model, img_tensor, layer_idx=1):
    """获取 CoordAtt 的注意力图"""
    model.eval()
    with torch.no_grad():
        _, a_h, a_w = model.forward_with_attention(img_tensor, layer_idx=layer_idx)
        attention = (a_h * a_w).squeeze(0).mean(0).cpu().numpy()
    return attention


def get_crossatt_attention(model, img_tensor, layer_idx=1):
    """获取 CoordCrossAtt 的注意力图"""
    model.eval()
    with torch.no_grad():
        _, attn, y_att = model.forward_with_attention(img_tensor, layer_idx=layer_idx)
        # 使用 y_att (门控权重) 进行可视化，因为它更直观
        attention = y_att.squeeze(0).mean(0).cpu().numpy()
        # 获取相关性矩阵用于额外可视化
        attn_map = attn.squeeze(0).mean(0).cpu().numpy()
    return attention, attn_map


def visualize_comparison(coordatt_model, crossatt_model, val_loader, device, img_size=256):
    """可视化对比两种注意力机制的检测效果"""
    print("\n生成对比可视化...")

    # 获取样本图片
    samples = []
    for imgs, targets, paths in val_loader:
        for i in range(min(4, len(imgs))):
            samples.append((imgs[i], targets[targets[:, 0] == i], paths[i]))
        if len(samples) >= 4:
            break

    fig, axes = plt.subplots(4, 4, figsize=(16, 16))

    layer_names = ['Layer 1\n(P3/4)', 'Layer 2\n(P4/8)', 'Layer 3\n(P5/16)', 'Layer 4\n(P6/32)']

    for idx, (img, target, path) in enumerate(samples):
        img_input = img.unsqueeze(0).to(device)
        img_display = img.permute(1, 2, 0).numpy()

        # 原图 + GT 框
        axes[idx, 0].imshow(img_display)
        if target.shape[0] > 0:
            for t in target:
                x, y, w, h = t[2:].cpu().numpy()
                x1 = (x - w/2) * img_size
                y1 = (y - h/2) * img_size
                rect = plt.Rectangle((x1, y1), w*img_size, h*img_size,
                                     linewidth=2, edgecolor='green', facecolor='none')
                axes[idx, 0].add_patch(rect)
        axes[idx, 0].set_title(f'Input + GT', fontsize=10)
        axes[idx, 0].axis('off')

        # 使用 Layer 1 (P3) 进行对比
        layer_idx = 0

        # CoordAtt 注意力
        coordatt_att = get_coordatt_attention(coordatt_model, img_input, layer_idx)
        coordatt_resized = np.array(Image.fromarray((coordatt_att * 255).astype(np.uint8)).resize(
            (img_size, img_size), Image.BILINEAR)) / 255.0
        coordatt_enhanced = enhance_contrast(coordatt_resized)

        axes[idx, 1].imshow(coordatt_enhanced, cmap='inferno', vmin=0, vmax=1)
        axes[idx, 1].set_title('CoordAtt\n(H × W Attention)', fontsize=10)
        axes[idx, 1].axis('off')

        # CoordCrossAtt - 门控权重
        crossatt_att, crossatt_attn = get_crossatt_attention(crossatt_model, img_input, layer_idx)
        crossatt_resized = np.array(Image.fromarray((crossatt_att * 255).astype(np.uint8)).resize(
            (img_size, img_size), Image.BILINEAR)) / 255.0
        crossatt_enhanced = enhance_contrast(crossatt_resized)

        axes[idx, 2].imshow(crossatt_enhanced, cmap='viridis', vmin=0, vmax=1)
        axes[idx, 2].set_title('CoordCrossAtt\n(Gate Weight)', fontsize=10)
        axes[idx, 2].axis('off')

        # 叠加对比
        axes[idx, 3].imshow(img_display)
        axes[idx, 3].imshow(coordatt_enhanced, cmap='inferno', alpha=0.4)
        if target.shape[0] > 0:
            for t in target:
                x, y, w, h = t[2:].cpu().numpy()
                x1 = (x - w/2) * img_size
                y1 = (y - h/2) * img_size
                rect = plt.Rectangle((x1, y1), w*img_size, h*img_size,
                                     linewidth=2, edgecolor='green', facecolor='none')
                axes[idx, 3].add_patch(rect)
        axes[idx, 3].set_title('CoordAtt Overlay + GT', fontsize=10)
        axes[idx, 3].axis('off')

    plt.suptitle('CoordAtt vs CoordCrossAtt Attention Comparison (Layer 1 - P3)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'attention_comparison.png'), dpi=150, bbox_inches='tight')
    print(f"对比图已保存: {os.path.join(OUTPUT_DIR, 'attention_comparison.png')}")
    plt.close()


def visualize_cross_attention_matrix(crossatt_model, val_loader, device, img_size=256):
    """可视化 Cross-Attention 的相关性矩阵"""
    print("\n生成 Cross-Attention 相关性矩阵可视化...")

    # 获取样本
    for imgs, targets, paths in val_loader:
        img = imgs[0]
        target = targets[targets[:, 0] == 0]
        break

    img_input = img.unsqueeze(0).to(device)
    img_display = img.permute(1, 2, 0).numpy()

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    axes[0].imshow(img_display)
    if target.shape[0] > 0:
        for t in target:
            x, y, w, h = t[2:].cpu().numpy()
            x1 = (x - w/2) * img_size
            y1 = (y - h/2) * img_size
            rect = plt.Rectangle((x1, y1), w*img_size, h*img_size,
                                 linewidth=2, edgecolor='green', facecolor='none')
            axes[0].add_patch(rect)
    axes[0].set_title('Input + GT', fontsize=12)
    axes[0].axis('off')

    crossatt_model.eval()
    with torch.no_grad():
        for layer_idx in range(min(3, len(crossatt_model.coord_att_layers))):
            _, attn, _ = crossatt_model.forward_with_attention(img_input, layer_idx=layer_idx)
            attn_map = attn.squeeze(0).mean(0).cpu().numpy()

            im = axes[layer_idx + 1].imshow(attn_map, cmap='RdYlBu_r', aspect='auto')
            axes[layer_idx + 1].set_title(f'Layer {layer_idx + 1}\nCross-Attention Matrix',
                                          fontsize=12)
            axes[layer_idx + 1].set_xlabel('Vertical Position', fontsize=10)
            axes[layer_idx + 1].set_ylabel('Horizontal Position', fontsize=10)
            plt.colorbar(im, ax=axes[layer_idx + 1])

    plt.suptitle('CoordCrossAtt: Cross-Attention Correlation Matrix\n'
                 '(Shows correlation between horizontal and vertical positions)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'cross_attention_matrix.png'), dpi=150, bbox_inches='tight')
    print(f"相关性矩阵已保存: {os.path.join(OUTPUT_DIR, 'cross_attention_matrix.png')}")
    plt.close()


def visualize_training_progress(coordatt_history, crossatt_history, img_size=256):
    """可视化训练进度对比"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(coordatt_history['train_loss']) + 1)

    # Loss
    axes[0].plot(epochs, coordatt_history['train_loss'], 'b-o', label='CoordAtt', linewidth=2)
    axes[0].plot(epochs, crossatt_history['train_loss'], 'r-s', label='CoordCrossAtt', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Validation Loss
    axes[1].plot(epochs, coordatt_history['val_loss'], 'b-o', label='CoordAtt', linewidth=2)
    axes[1].plot(epochs, crossatt_history['val_loss'], 'r-s', label='CoordCrossAtt', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Validation Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('Training Progress Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'training_progress.png'), dpi=150, bbox_inches='tight')
    print(f"训练进度图已保存: {os.path.join(OUTPUT_DIR, 'training_progress.png')}")
    plt.close()


# ==================== 主函数 ====================
def main():
    print("=" * 60)
    print("CoordAtt vs CoordCrossAtt 注意力机制对比 (YOLO 检测)")
    print("=" * 60)
    print(f"\n配置:")
    print(f"  - 设备: {CONFIG['device'].upper()}")
    print(f"  - 图像尺寸: {CONFIG['img_size']}x{CONFIG['img_size']}")
    print(f"  - 批大小: {CONFIG['batch_size']}")
    print(f"  - 训练轮数: {CONFIG['epochs']}")
    print(f"  - 学习率: {CONFIG['lr']}")
    print(f"  - 输出目录: {OUTPUT_DIR}")

    # 加载数据集配置
    config = load_yaml_config(CONFIG['data_config'])
    print(f"\n数据集配置:")
    print(f"  - 类别数: {config['nc']}")
    print(f"  - 类别名称: {config['names']}")

    # 加载数据
    print("\n加载数据集...")
    train_loader, val_loader, _ = create_dataloaders(
        config_path=CONFIG['data_config'],
        batch_size=CONFIG['batch_size'],
        img_size=CONFIG['img_size'],
        workers=0
    )

    # 创建模型
    print("\n创建模型...")

    # CoordAtt 模型
    coordatt_dir = os.path.join(OUTPUT_DIR, 'coordatt')
    os.makedirs(coordatt_dir, exist_ok=True)
    coordatt_model = YOLOCoordAttDetector(nc=config['nc']).to(CONFIG['device'])

    # CoordCrossAtt 模型
    crossatt_dir = os.path.join(OUTPUT_DIR, 'crossatt')
    os.makedirs(crossatt_dir, exist_ok=True)
    crossatt_model = YOLOCoordCrossAttDetector(nc=config['nc'], num_heads=1).to(CONFIG['device'])

    # 训练 CoordAtt 模型
    print("\n" + "=" * 50)
    print("训练 CoordAtt 模型")
    print("=" * 50)
    coordatt_params = sum(p.numel() for p in coordatt_model.parameters())
    print(f"参数量: {coordatt_params:,}")

    coordatt_history = train_detector(
        coordatt_model, train_loader, val_loader,
        epochs=CONFIG['epochs'], lr=CONFIG['lr'],
        device=CONFIG['device'], save_dir=coordatt_dir, patience=10
    )

    # 训练 CoordCrossAtt 模型
    print("\n" + "=" * 50)
    print("训练 CoordCrossAtt 模型")
    print("=" * 50)
    crossatt_params = sum(p.numel() for p in crossatt_model.parameters())
    print(f"参数量: {crossatt_params:,}")

    crossatt_history = train_detector(
        crossatt_model, train_loader, val_loader,
        epochs=CONFIG['epochs'], lr=CONFIG['lr'],
        device=CONFIG['device'], save_dir=crossatt_dir, patience=10
    )

    # 加载最佳模型进行可视化
    coordatt_checkpoint = torch.load(os.path.join(coordatt_dir, 'best_model.pth'), weights_only=True)
    coordatt_model.load_state_dict(coordatt_checkpoint['model_state_dict'])

    crossatt_checkpoint = torch.load(os.path.join(crossatt_dir, 'best_model.pth'), weights_only=True)
    crossatt_model.load_state_dict(crossatt_checkpoint['model_state_dict'])

    # 可视化对比
    visualize_comparison(coordatt_model, crossatt_model, val_loader,
                        CONFIG['device'], CONFIG['img_size'])

    # 可视化 Cross-Attention 相关性矩阵
    visualize_cross_attention_matrix(crossatt_model, val_loader, CONFIG['device'], CONFIG['img_size'])

    # 可视化训练进度
    visualize_training_progress(coordatt_history, crossatt_history, CONFIG['img_size'])

    # 打印对比结果
    print("\n" + "=" * 60)
    print("对比结果总结")
    print("=" * 60)

    coordatt_best = min(coordatt_history['val_loss'])
    crossatt_best = min(crossatt_history['val_loss'])

    print(f"\n最佳验证 Loss:")
    print(f"  CoordAtt:       {coordatt_best:.4f}")
    print(f"  CoordCrossAtt:  {crossatt_best:.4f}")
    print(f"  差异:           {crossatt_best - coordatt_best:+.4f}")

    print(f"\n训练时间:")
    print(f"  CoordAtt:       {coordatt_history['total_time_sec']:.1f}秒")
    print(f"  CoordCrossAtt:  {crossatt_history['total_time_sec']:.1f}秒")
    print(f"  比值:           {crossatt_history['total_time_sec'] / coordatt_history['total_time_sec']:.2f}x")

    print(f"\n参数量:")
    print(f"  CoordAtt:       {coordatt_params:,}")
    print(f"  CoordCrossAtt:  {crossatt_params:,}")
    print(f"  增加:           {crossatt_params - coordatt_params:+,} ({(crossatt_params / coordatt_params - 1) * 100:+.1f}%)")

    print(f"\n所有输出已保存到: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
