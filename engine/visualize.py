"""
注意力可视化模块 - 用于可视化 CoordAtt 等注意力机制（目标检测版本）
"""
import os
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
import numpy as np
from PIL import Image

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False


def enhance_contrast(attention_map):
    """增强注意力图的对比度"""
    # 使用 percentile 范围来增强对比度
    p2, p98 = np.percentile(attention_map, (2, 98))
    enhanced = np.clip((attention_map - p2) / (p98 - p2 + 1e-8), 0, 1)
    return enhanced


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
                    # t: [batch_idx, class_id, x, y, w, h] (归一化坐标)
                    x, y, w, h = t[2:].cpu().numpy()
                    # 转换为像素坐标
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
                    pred_boxes = predictions[0]  # (bs, num_boxes, no)
                else:
                    pred_boxes = predictions[0]

            # 解析预测框（简单示例）
            axes[idx, 3].imshow(img_display)
            # 这里可以添加预测框的显示逻辑
            # 由于 YOLOv3 输出格式复杂，这里简化处理
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
