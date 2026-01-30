#!/usr/bin/env python3
"""
Letterbox 坐标变换诊断脚本

用于验证 letterbox 预处理是否导致训练损失无法下降的根本原因。

诊断内容：
1. 对比 letterbox=True 和 letterbox=False 的 boxes 坐标范围
2. 检查坐标是否超出 [0, 1] 归一化范围
3. 可视化 boxes 在图像中的位置
4. 分析损失函数中的坐标变换
"""
import sys
import torch
import numpy as np
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import create_dataloaders
from models import YOLOv11
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def print_box_stats(boxes, name):
    """打印 boxes 的统计信息"""
    if boxes.shape[0] == 0:
        print(f"{name}: 空 (没有 boxes)")
        return

    print(f"\n{'='*60}")
    print(f"{name}")
    print(f"{'='*60}")
    print(f"Shape: {boxes.shape}")
    print(f"Format: [class_id, x_center, y_center, width, height] (归一化)")

    # 提取坐标（跳过 class_id）
    coords = boxes[:, 1:]  # (N, 4)

    print(f"\n坐标范围:")
    print(f"  x_center: min={coords[:, 0].min():.4f}, max={coords[:, 0].max():.4f}")
    print(f"  y_center: min={coords[:, 1].min():.4f}, max={coords[:, 1].max():.4f}")
    print(f"  width:    min={coords[:, 2].min():.4f}, max={coords[:, 2].max():.4f}")
    print(f"  height:   min={coords[:, 3].min():.4f}, max={coords[:, 3].max():.4f}")

    # 检查是否超出 [0, 1]
    out_of_bounds = (coords < 0).any() or (coords > 1).any()
    if out_of_bounds:
        print(f"\n⚠️  警告: 坐标超出 [0, 1] 归一化范围！")
        print(f"   坐标 < 0 的数量: {(coords < 0).sum().item()}")
        print(f"   坐标 > 1 的数量: {(coords > 1).sum().item()}")
    else:
        print(f"\n✓ 坐标在 [0, 1] 范围内")

    # 计算边界框的像素范围（假设 img_size=640）
    img_size = 640
    xy = coords[:, :2] * img_size  # 中心点像素坐标
    wh = coords[:, 2:] * img_size  # 宽高像素

    print(f"\n映射到 {img_size}x{img_size} 像素空间:")
    print(f"  中心点 x: min={xy[:, 0].min():.1f}, max={xy[:, 0].max():.1f}")
    print(f"  中心点 y: min={xy[:, 1].min():.1f}, max={xy[:, 1].max():.1f}")
    print(f"  宽度:     min={wh[:, 0].min():.1f}, max={wh[:, 0].max():.1f}")
    print(f"  高度:     min={wh[:, 1].min():.1f}, max={wh[:, 1].max():.1f}")

    # 检查边界框是否超出图像范围
    x1y1 = xy - wh / 2
    x2y2 = xy + wh / 2
    out_of_image = (x1y1 < 0).any() or (x2y2 > img_size).any()
    if out_of_image:
        print(f"\n⚠️  警告: 边界框超出图像范围！")
        print(f"   x1 < 0: {(x1y1[:, 0] < 0).sum().item()} 个")
        print(f"   y1 < 0: {(x1y1[:, 1] < 0).sum().item()} 个")
        print(f"   x2 > {img_size}: {(x2y2[:, 0] > img_size).sum().item()} 个")
        print(f"   y2 > {img_size}: {(x2y2[:, 1] > img_size).sum().item()} 个")
    else:
        print(f"\n✓ 边界框在图像范围内")


def visualize_comparison(img_lb, boxes_lb, img_resize, boxes_resize, idx=0):
    """可视化 letterbox vs resize 的对比"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Letterbox 图像
    ax1 = axes[0]
    ax1.imshow(img_lb.permute(1, 2, 0).cpu().numpy())
    ax1.set_title(f'Letterbox (保持宽高比)\n{boxes_lb.shape[0]} boxes', fontsize=12)
    ax1.axis('off')

    # 绘制 letterbox boxes
    for box in boxes_lb:
        cls_id, x_c, y_c, w, h = box
        # 转换为像素坐标
        x1 = (x_c - w/2) * 640
        y1 = (y_c - h/2) * 640
        rect = patches.Rectangle(
            (x1, y1), w*640, h*640,
            linewidth=2, edgecolor='r', facecolor='none'
        )
        ax1.add_patch(rect)

    # Resize 图像
    ax2 = axes[1]
    ax2.imshow(img_resize.permute(1, 2, 0).cpu().numpy())
    ax2.set_title(f'Simple Resize (强制拉伸)\n{boxes_resize.shape[0]} boxes', fontsize=12)
    ax2.axis('off')

    # 绘制 resize boxes
    for box in boxes_resize:
        cls_id, x_c, y_c, w, h = box
        # 转换为像素坐标
        x1 = (x_c - w/2) * 640
        y1 = (y_c - h/2) * 640
        rect = patches.Rectangle(
            (x1, y1), w*640, h*640,
            linewidth=2, edgecolor='g', facecolor='none'
        )
        ax2.add_patch(rect)

    plt.tight_layout()
    save_path = Path(__file__).parent.parent / 'runs/diagnosis/letterbox_comparison.png'
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n可视化结果已保存到: {save_path}")
    plt.close()


def test_loss_forward_pass(model, imgs, targets, device, name):
    """测试损失函数的前向传播"""
    print(f"\n{'='*60}")
    print(f"{name} - 损失函数前向传播测试")
    print(f"{'='*60}")

    model.eval()
    model.detect.train()  # 确保检测头在训练模式

    with torch.no_grad():
        try:
            outputs = model(imgs.to(device), targets.to(device))

            if isinstance(outputs, dict):
                loss = outputs['loss']
                loss_items = outputs['loss_items']
                print(f"✓ 前向传播成功")
                # loss 是 tensor [box, cls, dfl] * batch_size
                print(f"  Loss Tensor: {loss}")
                print(f"  Box Loss:   {loss_items[0].item():.4f}")
                print(f"  Cls Loss:   {loss_items[1].item():.4f}")
                print(f"  DFL Loss:   {loss_items[2].item():.4f}")
                print(f"  Total Sum:  {loss_items.sum().item():.4f}")

                # 检查损失值是否异常
                if loss_items[1].item() > 5.0:
                    print(f"  ⚠️  Cls Loss 异常高 (> 5.0)")

                return loss_items
            else:
                print(f"⚠️  意外的输出格式: {type(outputs)}")
                return None
        except Exception as e:
            print(f"✗ 前向传播失败: {e}")
            import traceback
            traceback.print_exc()
            return None


def diagnose_coordinate_mismatch():
    """诊断坐标不匹配问题"""

    print("\n" + "="*70)
    print("Letterbox 坐标变换诊断")
    print("="*70)

    # 配置
    config_path = "/home/shiomisaka/workplace/ai-playground/datasets/MY_TEST_DATA/data.yaml"
    batch_size = 4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n配置:")
    print(f"  配置文件: {config_path}")
    print(f"  批大小: {batch_size}")
    print(f"  设备: {device}")

    # 创建模型
    print(f"\n创建 YOLOv11 模型...")
    model = YOLOv11(nc=2, scale='n', img_size=640).to(device)

    # ==================== 测试 1: Letterbox ====================
    print(f"\n{'='*70}")
    print(f"测试 1: 使用 Letterbox 预处理 (letterbox=True)")
    print(f"{'='*70}")

    try:
        train_loader_lb, _, _ = create_dataloaders(
            config_path,
            batch_size=batch_size,
            img_size=640,
            workers=0,
            letterbox=True
        )

        # 获取一个 batch
        imgs_lb, targets_lb, paths_lb = next(iter(train_loader_lb))

        # 分离不同图片的 boxes
        boxes_list_lb = []
        for i in range(batch_size):
            mask = targets_lb[:, 0] == i
            if mask.sum() > 0:
                boxes_list_lb.append(targets_lb[mask][:, 1:])  # 去掉 batch_idx
            else:
                boxes_list_lb.append(torch.zeros((0, 5)))

        # 打印第一张图片的 boxes 统计
        print_box_stats(boxes_list_lb[0], "Letterbox - 第 1 张图片的 Boxes")

        # 测试损失函数
        loss_items_lb = test_loss_forward_pass(model, imgs_lb, targets_lb, device, "Letterbox")

    except Exception as e:
        print(f"✗ Letterbox 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # ==================== 测试 2: Simple Resize ====================
    print(f"\n{'='*70}")
    print(f"测试 2: 使用 Simple Resize 预处理 (letterbox=False)")
    print(f"{'='*70}")

    try:
        train_loader_resize, _, _ = create_dataloaders(
            config_path,
            batch_size=batch_size,
            img_size=640,
            workers=0,
            letterbox=False
        )

        # 获取一个 batch
        imgs_resize, targets_resize, paths_resize = next(iter(train_loader_resize))

        # 分离不同图片的 boxes
        boxes_list_resize = []
        for i in range(batch_size):
            mask = targets_resize[:, 0] == i
            if mask.sum() > 0:
                boxes_list_resize.append(targets_resize[mask][:, 1:])  # 去掉 batch_idx
            else:
                boxes_list_resize.append(torch.zeros((0, 5)))

        # 打印第一张图片的 boxes 统计
        print_box_stats(boxes_list_resize[0], "Simple Resize - 第 1 张图片的 Boxes")

        # 测试损失函数
        loss_items_resize = test_loss_forward_pass(model, imgs_resize, targets_resize, device, "Simple Resize")

    except Exception as e:
        print(f"✗ Simple Resize 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # ==================== 对比分析 ====================
    print(f"\n{'='*70}")
    print(f"对比分析")
    print(f"{'='*70}")

    if loss_items_lb is not None and loss_items_resize is not None:
        print(f"\n损失对比:")
        print(f"{'指标':<15} {'Letterbox':<15} {'Simple Resize':<15} {'差异':<15}")
        print(f"{'-'*60}")
        print(f"{'Total Loss':<15} {loss_items_lb.sum().item():<15.4f} {loss_items_resize.sum().item():<15.4f} {(loss_items_lb.sum() - loss_items_resize.sum()).item():<15.4f}")
        print(f"{'Box Loss':<15} {loss_items_lb[0].item():<15.4f} {loss_items_resize[0].item():<15.4f} {(loss_items_lb[0] - loss_items_resize[0]).item():<15.4f}")
        print(f"{'Cls Loss':<15} {loss_items_lb[1].item():<15.4f} {loss_items_resize[1].item():<15.4f} {(loss_items_lb[1] - loss_items_resize[1]).item():<15.4f}")
        print(f"{'DFL Loss':<15} {loss_items_lb[2].item():<15.4f} {loss_items_resize[2].item():<15.4f} {(loss_items_lb[2] - loss_items_resize[2]).item():<15.4f}")

        # 关键诊断
        cls_diff = (loss_items_lb[1] - loss_items_resize[1]).item()
        if cls_diff > 0.5:
            print(f"\n✗ 确认: Letterbox 导致 Cls Loss 增加 {cls_diff:.4f}")
            print(f"   这证实了 letterbox 坐标变换是根本原因！")
        else:
            print(f"\n? Cls Loss 差异较小 ({cls_diff:.4f})")
            print(f"   可能需要检查其他因素")

    # ==================== 可视化 ====================
    try:
        visualize_comparison(
            imgs_lb[0], boxes_list_lb[0],
            imgs_resize[0], boxes_list_resize[0]
        )
    except Exception as e:
        print(f"可视化失败: {e}")

    print(f"\n{'='*70}")
    print(f"诊断完成")
    print(f"{'='*70}")


if __name__ == "__main__":
    diagnose_coordinate_mismatch()
