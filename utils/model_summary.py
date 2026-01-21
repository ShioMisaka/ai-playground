"""
模型和训练信息输出模块

提供训练配置信息和模型摘要的输出功能。
"""
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional


def print_training_info(config_path, epochs, batch_size, img_size, lr, device, save_dir):
    """打印训练配置信息

    Args:
        config_path: 数据集配置文件路径
        epochs: 训练轮数
        batch_size: 批大小
        img_size: 图像尺寸
        lr: 学习率
        device: 设备
        save_dir: 保存目录
    """
    # 获取绝对路径
    config_path = Path(config_path).resolve()
    save_dir = Path(save_dir).resolve()

    print(f"\n{'='*80}")
    print(f"训练配置信息")
    print(f"{'='*80}")
    print(f"  data={config_path}, epochs={epochs}, batch={batch_size}, imgsz={img_size}, "
          f"lr={lr}, device={device}")
    print(f"  save_dir={save_dir}")
    print(f"{'='*80}\n")


def count_layers(model: nn.Module) -> int:
    """计算模型层数

    Args:
        model: PyTorch 模型

    Returns:
        层数
    """
    # 计算所有叶子模块（没有子模块的模块）
    layer_count = 0
    for module in model.modules():
        if module is not model and list(module.children()) == []:
            layer_count += 1
    return layer_count


def get_model_summary(model: nn.Module, img_size: int = 640) -> dict:
    """获取模型摘要信息

    Args:
        model: PyTorch 模型
        img_size: 输入图像尺寸

    Returns:
        包含层数、参数量、梯度数、FLOPs 的字典
    """
    # 计算层数
    num_layers = count_layers(model)

    # 计算参数量和梯度数
    total_params = sum(p.numel() for p in model.parameters())
    total_gradients = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 计算 FLOPs
    gflops = estimate_flops(model, img_size)

    return {
        'layers': num_layers,
        'parameters': total_params,
        'gradients': total_gradients,
        'gflops': gflops
    }


def estimate_flops(model: nn.Module, img_size: int) -> float:
    """粗略估计 FLOPs

    Args:
        model: PyTorch 模型
        img_size: 输入图像尺寸

    Returns:
        估计的 GFLOPs
    """
    # 尝试使用 thop 进行精确计算
    try:
        from thop import profile
        input_tensor = torch.randn(1, 3, img_size, img_size)
        flops, _ = profile(model, inputs=(input_tensor,), verbose=False)
        return flops / 1e9
    except ImportError:
        pass

    # Fallback: 粗略估计
    total_params = sum(p.numel() for p in model.parameters())
    feature_map_size = (img_size / 32) ** 2
    estimated_flops = total_params * 2 * feature_map_size * 0.1
    return estimated_flops / 1e9


def format_number(num: int) -> str:
    """格式化数字，添加千位分隔符

    Args:
        num: 数字

    Returns:
        格式化后的字符串
    """
    return f"{num:,}"


def print_model_summary(model: nn.Module, img_size: int = 640, nc: Optional[int] = None):
    """打印模型摘要信息

    Args:
        model: PyTorch 模型
        img_size: 输入图像尺寸
        nc: 类别数量（如果覆盖了模型默认值）
    """
    # 如果提供了类别数，检查是否需要覆盖
    if nc is not None:
        if hasattr(model, 'nc') and model.nc != nc:
            print(f"Overriding model nc={model.nc} with nc={nc}")
            model.nc = nc
            # 如果有 detect 层，也需要更新
            if hasattr(model, 'detect'):
                model.detect.nc = nc
                model.detect.no = nc + 5

    # 获取模型摘要
    summary = get_model_summary(model, img_size)

    # 打印摘要
    model_name = model.__class__.__name__
    print(f"{model_name} summary: "
          f"{summary['layers']} layers, "
          f"{format_number(summary['parameters'])} parameters, "
          f"{format_number(summary['gradients'])} gradients, "
          f"{summary['gflops']:.1f} GFLOPs")
    print()
