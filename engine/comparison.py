"""
模型对比训练模块 - 用于对比不同注意力机制的训练效果
"""
import os
import torch
from pathlib import Path
from .detector import train_detector


def train_and_compare_models(model_dict, train_loader, val_loader,
                              epochs=15, lr=0.001, device='cpu',
                              save_dir='outputs/comparison', patience=10):
    """
    训练并对比多个模型

    Args:
        model_dict: 模型字典，格式为 {model_name: (model_class, kwargs)}
                     例如: {'CoordAtt': (YOLOCoordAttDetector, {'nc': 2})}
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        epochs: 训练轮数
        lr: 学习率
        device: 设备
        save_dir: 保存目录
        patience: 早停耐心值

    Returns:
        results: 字典，包含模型、历史记录和参数量信息
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    for model_name, (model_class, model_kwargs) in model_dict.items():
        print(f"\n{'='*50}")
        print(f"训练 {model_name} 模型")
        print(f"{'='*50}")

        # 创建模型保存目录
        model_dir = save_dir / model_name.lower()
        model_dir.mkdir(exist_ok=True)

        # 创建模型
        model = model_class(**model_kwargs).to(device)
        params = sum(p.numel() for p in model.parameters())
        print(f"参数量: {params:,}")

        # 训练模型
        history = train_detector(
            model, train_loader, val_loader,
            epochs=epochs, lr=lr, device=device,
            save_dir=str(model_dir), patience=patience
        )

        # 保存结果
        results[model_name] = {
            'model': model,
            'history': history,
            'params': params,
            'save_dir': model_dir
        }

    return results


def load_best_model(model_class, checkpoint_path, device='cpu', **model_kwargs):
    """
    加载最佳模型

    Args:
        model_class: 模型类
        checkpoint_path: 检查点文件路径
        device: 设备
        **model_kwargs: 模型初始化参数

    Returns:
        model: 加载了最佳权重的模型
    """
    model = model_class(**model_kwargs).to(device)
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model


def print_comparison_results(results):
    """
    打印模型对比结果

    Args:
        results: train_and_compare_models 返回的结果字典
    """
    print("\n" + "=" * 60)
    print("模型对比结果总结")
    print("=" * 60)

    # 提取第一个模型作为基准
    first_model = list(results.values())[0]
    first_name = list(results.keys())[0]
    first_best = min(first_model['history']['val_loss'])
    first_time = first_model['history']['total_time_sec']
    first_params = first_model['params']

    print(f"\n基准模型: {first_name}")
    print(f"  最佳验证 Loss: {first_best:.4f}")
    print(f"  训练时间: {first_time:.1f}秒")
    print(f"  参数量: {first_params:,}")

    for model_name, result in results.items():
        if model_name == first_name:
            continue

        history = result['history']
        params = result['params']

        best_loss = min(history['val_loss'])
        train_time = history['total_time_sec']

        print(f"\n{model_name} vs {first_name}:")
        print(f"  Loss 差异: {best_loss - first_best:+.4f}")
        print(f"  时间比值: {train_time / first_time:.2f}x")
        print(f"  参数量差异: {params - first_params:+,} ({(params / first_params - 1) * 100:+.1f}%)")
