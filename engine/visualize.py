"""
注意力可视化模块 - 用于可视化 CoordAtt 等注意力机制
"""
import os
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
import numpy as np
from PIL import Image
from torchvision import transforms

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False


def enhance_contrast(attention_map):
    """增强注意力图的对比度"""
    p2, p98 = np.percentile(attention_map, (2, 98))
    enhanced = np.clip((attention_map - p2) / (p98 - p2 + 1e-8), 0, 1)
    return enhanced


def load_image(image_path, img_size=224):
    """
    加载并预处理图像

    Args:
        image_path: 图像路径
        img_size: 目标图像大小

    Returns:
        img_tensor: 预处理后的图像张量 [1, 3, H, W]
        img_display: 用于显示的图像数组 [H, W, 3]
    """
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


def get_coordatt_attention(model, img_tensor, layer_idx=0):
    """
    获取 CoordAtt 模型的注意力图

    Args:
        model: 带 forward_with_attention 方法的模型
        img_tensor: 输入图像张量
        layer_idx: 要获取的注意力层索引

    Returns:
        attention: 注意力图 numpy 数组
    """
    model.eval()
    with torch.no_grad():
        _, a_h, a_w = model.forward_with_attention(img_tensor, layer_idx=layer_idx)
        attention = (a_h * a_w).squeeze(0).mean(0).cpu().numpy()
    return attention


def get_crossatt_attention(model, img_tensor, layer_idx=0):
    """
    获取 CoordCrossAtt 模型的注意力图

    Args:
        model: 带 forward_with_attention 方法的模型
        img_tensor: 输入图像张量
        layer_idx: 要获取的注意力层索引

    Returns:
        attention: Cross-Attention 相关性矩阵 (H x W) - 用于可视化热力图
        attn_map: 原始相关性矩阵 (H x W)
    """
    model.eval()
    with torch.no_grad():
        _, attn, _ = model.forward_with_attention(img_tensor, layer_idx=layer_idx)
        # attn: [1, num_heads, H, W] -> [H, W]
        # 使用相关性矩阵作为注意力图，它包含完整的 H-W 位置信息
        attention = attn.squeeze(0).mean(0).cpu().numpy()
        attn_map = attn.squeeze(0).mean(0).cpu().numpy()
    return attention, attn_map


# ==================== 检测任务可视化 ====================

def visualize_detection_attention(model, dataloader, device,
                                   save_path='detection_attention.png',
                                   img_size=640, conf_threshold=0.5):
    """
    可视化检测任务的注意力效果（显示检测框 + 注意力热力图）

    Args:
        model: 训练好的检测模型（需有 forward_with_attention 方法）
        dataloader: 数据加载器
        device: 设备
        save_path: 保存路径
        img_size: 图像大小
        conf_threshold: 置信度阈值
    """
    model.eval()
    model.train()  # 切换到训练模式以获取原始预测

    # 获取样本
    samples = []
    for imgs, targets, paths in dataloader:
        for i in range(min(4, len(imgs))):
            samples.append((imgs[i], targets[targets[:, 0] == i], paths[i]))
        if len(samples) >= 4:
            break

    # 4张图片，每张展示原图+注意力+叠加
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))

    layer_names = ['Layer 1 (P3)', 'Layer 2 (P4)', 'Layer 3 (P5)', 'Layer 4 (P6)']

    with torch.no_grad():
        for idx, (img, target, path) in enumerate(samples):
            img_input = img.unsqueeze(0).to(device)
            img_display = img.permute(1, 2, 0).numpy()

            # 显示原始图像 + GT 框
            axes[idx, 0].imshow(img_display)
            if target.shape[0] > 0:
                for t in target:
                    x, y, w, h = t[2:].cpu().numpy()
                    x1 = (x - w/2) * img_size
                    y1 = (y - h/2) * img_size
                    rect = patches.Rectangle((x1, y1), w*img_size, h*img_size,
                                             linewidth=2, edgecolor='green', facecolor='none')
                    axes[idx, 0].add_patch(rect)
            axes[idx, 0].set_title('Original + GT Boxes', fontsize=11)
            axes[idx, 0].axis('off')

            # 获取每一层的注意力（只显示第一层用于简化）
            for layer_idx in range(1):
                predictions, a_h, a_w = model.forward_with_attention(img_input, layer_idx=layer_idx)

                # 合并注意力
                attention = (a_h * a_w).squeeze(0).mean(0).cpu().numpy()

                # 上采样到原始图像大小
                attention_full = Image.fromarray((attention * 255).astype(np.uint8)).resize(
                    (img_size, img_size), Image.BILINEAR
                )
                attention_full = np.array(attention_full) / 255.0

                # 增强对比度
                attention_enhanced = enhance_contrast(attention_full)

                # 显示注意力热力图
                axes[idx, 1].imshow(attention_enhanced, cmap='inferno', vmin=0, vmax=1)
                axes[idx, 1].set_title(f'{layer_names[layer_idx]} Attention', fontsize=11)
                axes[idx, 1].axis('off')

                # 叠加注意力 + 检测框
                axes[idx, 2].imshow(img_display)
                axes[idx, 2].imshow(attention_enhanced, cmap='inferno', alpha=0.5)
                if target.shape[0] > 0:
                    for t in target:
                        x, y, w, h = t[2:].cpu().numpy()
                        x1 = (x - w/2) * img_size
                        y1 = (y - h/2) * img_size
                        rect = patches.Rectangle((x1, y1), w*img_size, h*img_size,
                                                 linewidth=2, edgecolor='green', facecolor='none')
                        axes[idx, 2].add_patch(rect)
                axes[idx, 2].set_title('Attention + GT Boxes', fontsize=11)
                axes[idx, 2].axis('off')

            # 推理模式下的预测
            model.eval()
            with torch.no_grad():
                predictions = model(img_input)
                if isinstance(predictions, tuple):
                    pred_boxes = predictions[0]
                else:
                    pred_boxes = predictions[0]

            axes[idx, 3].imshow(img_display)
            axes[idx, 3].set_title('Predictions', fontsize=11)
            axes[idx, 3].axis('off')

            model.train()  # 切回训练模式

    plt.suptitle('YOLO Detection with Coordinate Attention', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n检测结果可视化已保存到: {save_path}")
    plt.close()


def visualize_attention_comparison(model, dataloader, device,
                                    save_path='attention_comparison.png',
                                    img_size=640):
    """
    创建训练前后对比图

    Args:
        model: 训练后的模型
        dataloader: 数据加载器
        device: 设备
        save_path: 保存路径
        img_size: 图像大小
    """
    from models import YOLOCoordAttDetector

    # 创建一个未训练的模型用于对比
    untrained_model = YOLOCoordAttDetector(nc=1).to(device)
    untrained_model.train()

    model.train()

    # 获取样本
    samples = []
    for imgs, targets, paths in dataloader:
        for i in range(min(2, len(imgs))):
            samples.append((imgs[i], targets[targets[:, 0] == i], paths[i]))
        if len(samples) >= 2:
            break

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    with torch.no_grad():
        for idx, (img, target, path) in enumerate(samples):
            img_input = img.unsqueeze(0).to(device)
            img_display = img.permute(1, 2, 0).numpy()

            # 未训练模型 - 使用最后一层 (layer 3)
            _, a_h_untrained, a_w_untrained = untrained_model.forward_with_attention(img_input, layer_idx=3)
            att_untrained = (a_h_untrained * a_w_untrained).squeeze(0).mean(0).cpu().numpy()
            att_untrained_full = np.array(Image.fromarray((att_untrained * 255).astype(np.uint8)).resize((img_size, img_size), Image.BILINEAR)) / 255.0
            att_untrained_enhanced = enhance_contrast(att_untrained_full)

            # 训练后模型
            _, a_h_trained, a_w_trained = model.forward_with_attention(img_input, layer_idx=3)
            att_trained = (a_h_trained * a_w_trained).squeeze(0).mean(0).cpu().numpy()
            att_trained_full = np.array(Image.fromarray((att_trained * 255).astype(np.uint8)).resize((img_size, img_size), Image.BILINEAR)) / 255.0
            att_trained_enhanced = enhance_contrast(att_trained_full)

            # 显示
            axes[idx, 0].imshow(img_display)
            if target.shape[0] > 0:
                for t in target:
                    x, y, w, h = t[2:].cpu().numpy()
                    x1 = (x - w/2) * img_size
                    y1 = (y - h/2) * img_size
                    rect = patches.Rectangle((x1, y1), w*img_size, h*img_size,
                                             linewidth=2, edgecolor='green', facecolor='none')
                    axes[idx, 0].add_patch(rect)
            axes[idx, 0].set_title('Original + GT', fontsize=12)
            axes[idx, 0].axis('off')

            axes[idx, 1].imshow(att_untrained_enhanced, cmap='inferno', vmin=0, vmax=1)
            axes[idx, 1].set_title('Before Training (Layer 4)', fontsize=12)
            axes[idx, 1].axis('off')

            axes[idx, 2].imshow(img_display)
            axes[idx, 2].imshow(att_trained_enhanced, cmap='inferno', alpha=0.6)
            if target.shape[0] > 0:
                for t in target:
                    x, y, w, h = t[2:].cpu().numpy()
                    x1 = (x - w/2) * img_size
                    y1 = (y - h/2) * img_size
                    rect = patches.Rectangle((x1, y1), w*img_size, h*img_size,
                                             linewidth=2, edgecolor='green', facecolor='none')
                    axes[idx, 2].add_patch(rect)
            axes[idx, 2].set_title('After Training + GT', fontsize=12)
            axes[idx, 2].axis('off')

    plt.suptitle('Attention Before vs After Training (Layer 4 - Highest Level)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"对比可视化已保存到: {save_path}")
    plt.close()


# ==================== 单图可视化 ====================

def visualize_single_image_attention(model, img_tensor, img_display,
                                      save_path='single_attention.png'):
    """
    可视化单张图像的注意力效果

    Args:
        model: 带 forward_with_attention 方法的模型
        img_tensor: 输入图像张量
        img_display: 用于显示的图像数组
        save_path: 保存路径
    """
    model.eval()

    with torch.no_grad():
        output, a_h, a_w = model.forward_with_attention(img_tensor)

    # 处理注意力权重
    attention_map = (a_h * a_w).squeeze(0)  # (C, H, W)
    attention_map = attention_map.mean(0).cpu().numpy()  # (H, W)

    # 创建可视化
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
    print(f"单图可视化已保存到: {save_path}")
    plt.close()


# ==================== 多图可视化 ====================

def visualize_multiple_images_attention(model, image_dir, num_samples=4,
                                         img_size=224,
                                         save_path='multi_attention.png'):
    """
    可视化多张图像的注意力效果

    Args:
        model: 带 forward_with_attention 方法的模型
        image_dir: 图像目录
        num_samples: 采样数量
        img_size: 图像大小
        save_path: 保存路径
    """
    model.eval()

    # 获取图像文件
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    image_files = sorted(image_files)[:num_samples]

    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))

    for idx, img_file in enumerate(image_files):
        img_path = os.path.join(image_dir, img_file)

        # 加载图像
        img_tensor, img_display = load_image(img_path, img_size=img_size)

        # 获取注意力
        with torch.no_grad():
            output, a_h, a_w = model.forward_with_attention(img_tensor)

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
    plt.close()


# ==================== 模型对比可视化 ====================

def visualize_model_comparison(coordatt_model, crossatt_model, val_loader, device,
                                save_path='model_comparison.png', img_size=256):
    """
    可视化对比 CoordAtt 和 CoordCrossAtt 两种注意力机制的检测效果

    Args:
        coordatt_model: CoordAtt 模型
        crossatt_model: CoordCrossAtt 模型
        val_loader: 验证数据加载器
        device: 设备
        save_path: 保存路径
        img_size: 图像大小
    """
    print("\n生成模型对比可视化...")

    # 获取样本图片
    samples = []
    for imgs, targets, paths in val_loader:
        for i in range(min(4, len(imgs))):
            samples.append((imgs[i], targets[targets[:, 0] == i], paths[i]))
        if len(samples) >= 4:
            break

    fig, axes = plt.subplots(4, 4, figsize=(16, 16))

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
                rect = patches.Rectangle((x1, y1), w*img_size, h*img_size,
                                         linewidth=2, edgecolor='green', facecolor='none')
                axes[idx, 0].add_patch(rect)
        axes[idx, 0].set_title(f'Input + GT', fontsize=10)
        axes[idx, 0].axis('off')

        # 使用 Layer 0 (P3) 进行对比
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
        crossatt_att, _ = get_crossatt_attention(crossatt_model, img_input, layer_idx)
        crossatt_resized = np.array(Image.fromarray((crossatt_att * 255).astype(np.uint8)).resize(
            (img_size, img_size), Image.BILINEAR)) / 255.0
        crossatt_enhanced = enhance_contrast(crossatt_resized)

        axes[idx, 2].imshow(crossatt_enhanced, cmap='viridis', vmin=0, vmax=1)
        axes[idx, 2].set_title('CoordCrossAtt\n(H-W Correlation)', fontsize=10)
        axes[idx, 2].axis('off')

        # 叠加对比
        axes[idx, 3].imshow(img_display)
        axes[idx, 3].imshow(coordatt_enhanced, cmap='inferno', alpha=0.4)
        if target.shape[0] > 0:
            for t in target:
                x, y, w, h = t[2:].cpu().numpy()
                x1 = (x - w/2) * img_size
                y1 = (y - h/2) * img_size
                rect = patches.Rectangle((x1, y1), w*img_size, h*img_size,
                                         linewidth=2, edgecolor='green', facecolor='none')
                axes[idx, 3].add_patch(rect)
        axes[idx, 3].set_title('CoordAtt Overlay + GT', fontsize=10)
        axes[idx, 3].axis('off')

    plt.suptitle('CoordAtt vs CoordCrossAtt Attention Comparison (Layer 1 - P3)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"模型对比图已保存到: {save_path}")
    plt.close()


def visualize_cross_attention_matrix(crossatt_model, val_loader, device,
                                       save_path='cross_attention_matrix.png', img_size=256):
    """
    可视化 Cross-Attention 的相关性矩阵

    Args:
        crossatt_model: CoordCrossAtt 模型
        val_loader: 验证数据加载器
        device: 设备
        save_path: 保存路径
        img_size: 图像大小
    """
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
            rect = patches.Rectangle((x1, y1), w*img_size, h*img_size,
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
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"相关性矩阵已保存到: {save_path}")
    plt.close()


def visualize_training_progress(history_dict, save_path='training_progress.png'):
    """
    可视化训练进度对比

    Args:
        history_dict: 字典，key 为模型名称，value 为训练历史字典
        save_path: 保存路径
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # 获取第一个历史来确定 epochs
    first_history = list(history_dict.values())[0]
    epochs = range(1, len(first_history['train_loss']) + 1)

    # Loss
    for model_name, history in history_dict.items():
        axes[0].plot(epochs, history['train_loss'], '-o', label=model_name, linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Validation Loss
    for model_name, history in history_dict.items():
        axes[1].plot(epochs, history['val_loss'], '-o', label=model_name, linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Validation Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('Training Progress Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"训练进度图已保存到: {save_path}")
    plt.close()
