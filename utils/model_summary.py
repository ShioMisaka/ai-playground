"""
模型和训练信息输出模块

提供训练配置信息和模型摘要的输出功能。
使用 rich 库实现美观的终端界面展示。
"""
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional

# 尝试导入 rich 库，如果不可用则使用 fallback
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.columns import Columns
    from rich.text import Text
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

console = Console() if RICH_AVAILABLE else None


def truncate_path(path: Path, max_parts: int = 3) -> str:
    """截断路径，只保留最后几级目录

    Args:
        path: 文件路径
        max_parts: 保留的目录级数

    Returns:
        截断后的路径字符串
    """
    path_str = str(path)
    cwd = Path.cwd()

    try:
        # 尝试获取相对路径
        rel_path = path.relative_to(cwd)
        if len(str(rel_path)) < len(path_str):
            # 如果相对路径更短，使用相对路径
            parts = rel_path.parts
            if len(parts) > max_parts + 1:
                return f".../{'/'.join(parts[-max_parts:])}"
            return str(rel_path)
    except ValueError:
        # 无法获取相对路径，使用绝对路径
        pass

    # 对于绝对路径，截断显示
    parts = path.parts
    if len(parts) > max_parts + 1:
        return f".../{'/'.join(parts[-max_parts:])}"
    return path_str


def print_training_info(
    config_path,
    epochs,
    batch_size,
    img_size,
    lr,
    device,
    save_dir,
    num_train_samples: Optional[int] = None,
    num_val_samples: Optional[int] = None,
    nc: Optional[int] = None,
    use_mosaic: Optional[bool] = None,
    use_ema: Optional[bool] = None,
    close_mosaic: Optional[int] = None,
):
    """打印训练配置信息

    Args:
        config_path: 数据集配置文件路径
        epochs: 训练轮数
        batch_size: 批大小
        img_size: 图像尺寸
        lr: 学习率
        device: 设备
        save_dir: 保存目录
        num_train_samples: 训练集样本数量
        num_val_samples: 验证集样本数量
        nc: 类别数量
        use_mosaic: 是否启用 Mosaic 增强
        use_ema: 是否启用 EMA
        close_mosaic: 最后 N 个 epoch 关闭 Mosaic
    """
    # 获取绝对路径
    config_path = Path(config_path).resolve()
    save_dir = Path(save_dir).resolve()

    # 如果没有 rich 库，使用简单的 fallback
    if not RICH_AVAILABLE:
        print("\n" + "=" * 60)
        print("训练配置信息")
        print("=" * 60)
        print(f"  data: {config_path}")
        print(f"  epochs: {epochs}, batch_size: {batch_size}, img_size: {img_size}")
        print(f"  lr: {lr}, device: {device}")
        print(f"  save_dir: {save_dir}")
        if num_train_samples is not None:
            print(f"  train_samples: {num_train_samples:,}")
        if num_val_samples is not None:
            print(f"  val_samples: {num_val_samples:,}")
        if nc is not None:
            print(f"  num_classes: {nc}")
        if use_mosaic is not None:
            status = f"启用 (最后 {close_mosaic} 个 epoch 关闭)" if use_mosaic and close_mosaic else ("启用" if use_mosaic else "禁用")
            print(f"  mosaic: {status}")
        if use_ema is not None:
            print(f"  ema: {'启用 (decay=0.9999)' if use_ema else '关闭'}")
        print("=" * 60 + "\n")
        return

    # 使用 rich 库创建美观的输出
    console.print()

    # Environment 板块
    env_table = Table.grid(padding=(0, 2))
    env_table.add_column(style="cyan", width=12)
    env_table.add_column(style="green")
    env_table.add_row("设备", str(device))
    env_table.add_row("保存路径", truncate_path(save_dir))

    env_panel = Panel(
        env_table,
        title="[bold yellow]🚀 Environment[/bold yellow]",
        title_align="left",
        border_style="bright_blue",
        padding=(0, 1),
    )

    # Dataset 板块
    dataset_table = Table.grid(padding=(0, 2))
    dataset_table.add_column(style="cyan", width=12)
    dataset_table.add_column(style="green")
    dataset_table.add_row("配置文件", truncate_path(config_path))
    if nc is not None:
        dataset_table.add_row("类别数", str(nc))
    if num_train_samples is not None:
        dataset_table.add_row("训练样本", f"{num_train_samples:,}")
    if num_val_samples is not None:
        dataset_table.add_row("验证样本", f"{num_val_samples:,}")

    dataset_panel = Panel(
        dataset_table,
        title="[bold yellow]📊 Dataset[/bold yellow]",
        title_align="left",
        border_style="bright_magenta",
        padding=(0, 1),
    )

    # Hyperparameters 板块（包含 Mosaic 和 EMA）
    hyper_table = Table.grid(padding=(0, 2))
    hyper_table.add_column(style="cyan", width=12)
    hyper_table.add_column(style="green")
    hyper_table.add_row("学习率", f"[bold green]{lr}[/bold green]")
    hyper_table.add_row("Batch Size", f"[bold green]{batch_size}[/bold green]")
    hyper_table.add_row("Epochs", f"[bold green]{epochs}[/bold green]")
    hyper_table.add_row("图像尺寸", f"[bold green]{img_size}[/bold green]")

    # 添加 Mosaic 和 EMA 信息到 Hyperparameters
    if use_mosaic is not None or use_ema is not None:
        hyper_table.add_row("", "")  # 空行分隔

    if use_mosaic is not None:
        if use_mosaic and close_mosaic and close_mosaic > 0:
            mosaic_val = f"[bold green]启用[/bold green] (最后 {close_mosaic} epoch 关闭)"
        elif use_mosaic:
            mosaic_val = "[bold green]启用[/bold green]"
        else:
            mosaic_val = "[bold red]禁用[/bold red]"
        hyper_table.add_row("Mosaic 增强", mosaic_val)

    if use_ema is not None:
        ema_val = "[bold green]启用[/bold green]" if use_ema else "[bold red]关闭[/bold red]"
        hyper_table.add_row("EMA", ema_val)

    hyper_panel = Panel(
        hyper_table,
        title="[bold yellow]⚙️ Hyperparameters[/bold yellow]",
        title_align="left",
        border_style="bright_cyan",
        padding=(0, 1),
    )

    # 使用 Columns 布局展示三个面板
    panels = Columns([env_panel, dataset_panel, hyper_panel], equal=True)
    console.print(panels)
    console.print()


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
            if RICH_AVAILABLE:
                console.print(f"[yellow]Overriding model nc={model.nc} with nc={nc}[/yellow]")
            else:
                print(f"Overriding model nc={model.nc} with nc={nc}")
            model.nc = nc
            # 如果有 detect 层，也需要更新
            if hasattr(model, 'detect'):
                model.detect.nc = nc
                model.detect.no = nc + 5

    # 获取模型摘要
    summary = get_model_summary(model, img_size)
    model_name = model.__class__.__name__

    # 如果没有 rich 库，使用简单的 fallback
    if not RICH_AVAILABLE:
        print(f"\n{model_name} summary:")
        print(f"  Layers: {summary['layers']}")
        print(f"  Parameters: {format_number(summary['parameters'])}")
        print(f"  Gradients: {format_number(summary['gradients'])}")
        print(f"  GFLOPs: {summary['gflops']:.1f}")
        print()
        return

    # 使用 rich 库创建美观的输出
    # 创建模型信息表格
    model_table = Table.grid(padding=(0, 1))
    model_table.add_column(style="cyan", width=10)
    model_table.add_column()

    model_table.add_row("模型名称", f"[bold white]{model_name}[/bold white]")
    model_table.add_row("层数", f"[bold green]{summary['layers']}[/bold green]")
    model_table.add_row("参数量", f"[bold yellow]{format_number(summary['parameters'])}[/bold yellow]")
    model_table.add_row("梯度数", f"[bold green]{format_number(summary['gradients'])}[/bold green]")
    model_table.add_row("GFLOPs", f"[bold magenta]{summary['gflops']:.1f} GFLOPs[/bold magenta]")

    model_panel = Panel(
        model_table,
        title="[bold yellow]🧠 Model Summary[/bold yellow]",
        title_align="left",
        border_style="bright_yellow",
        padding=(0, 1),
        expand=False,
    )

    console.print(model_panel)
    console.print()


def print_training_setup(
    use_mosaic: bool,
    use_ema: bool,
    close_mosaic: int,
    num_train_samples: int,
    num_val_samples: int,
    nc: int,
    class_names: list,
    mosaic_enabled: bool = False,
):
    """打印训练设置信息

    Args:
        use_mosaic: 是否启用 Mosaic
        use_ema: 是否启用 EMA
        close_mosaic: 最后 N 个 epoch 关闭 Mosaic
        num_train_samples: 训练集样本数
        num_val_samples: 验证集样本数
        nc: 类别数
        class_names: 类别名称列表
        mosaic_enabled: Mosaic 当前是否已启用（根据 epochs 判断）
    """
    # 如果没有 rich 库，使用简单的 fallback
    if not RICH_AVAILABLE:
        print("\n训练设置:")
        print(f"  Mosaic: {'启用' if mosaic_enabled else '禁用'}")
        if mosaic_enabled and close_mosaic > 0:
            print(f"    (最后 {close_mosaic} 个 epoch 关闭)")
        print(f"  EMA: {'启用 (decay=0.9999)' if use_ema else '关闭'}")
        print(f"  类别数: {nc}")
        print(f"  类别名称: {class_names}")
        print(f"  训练集: {num_train_samples:,} 张图片")
        print(f"  验证集: {num_val_samples:,} 张图片")
        print()
        return

    # 使用 rich 库创建美观的输出
    console.print()

    # 创建设置表格
    setup_table = Table.grid(padding=(0, 2))
    setup_table.add_column(style="cyan", width=12)
    setup_table.add_column()

    # Mosaic 状态
    if mosaic_enabled:
        mosaic_status = "[bold green]启用[/bold green]"
        if close_mosaic > 0:
            mosaic_status += f" (最后 {close_mosaic} 个 epoch 关闭)"
    else:
        mosaic_status = "[bold red]禁用[/bold red]"
    setup_table.add_row("Mosaic 增强", mosaic_status)

    # EMA 状态
    ema_status = "[bold green]启用[/bold green]" if use_ema else "[bold red]关闭[/bold red]"
    if use_ema:
        ema_status += " (decay=0.9999)"
    setup_table.add_row("EMA", ema_status)

    # 空行分隔
    setup_table.add_row("", "")

    # 数据集信息
    setup_table.add_row("类别数", f"[bold yellow]{nc}[/bold yellow]")
    setup_table.add_row("类别名称", str(class_names))
    setup_table.add_row("训练样本", f"[bold green]{num_train_samples:,}[/bold green]")
    setup_table.add_row("验证样本", f"[bold green]{num_val_samples:,}[/bold green]")

    setup_panel = Panel(
        setup_table,
        title="[bold yellow]⚡ Training Setup[/bold yellow]",
        title_align="left",
        border_style="bright_green",
        padding=(0, 1),
    )

    console.print(setup_panel)
    console.print()


def print_training_completion(save_dir: Path, csv_path: Path, best_loss: float = None):
    """打印训练完成信息

    Args:
        save_dir: 保存目录
        csv_path: 训练日志 CSV 路径
        best_loss: 最佳验证损失
    """
    # 如果没有 rich 库，使用简单的 fallback
    if not RICH_AVAILABLE:
        print("\n" + "=" * 60)
        print("训练完成!")
        print("=" * 60)
        print(f"  保存目录: {save_dir}")
        print(f"  训练日志: {csv_path}")
        if best_loss is not None:
            print(f"  最佳损失: {best_loss:.4f}")
        print("=" * 60 + "\n")
        return

    # 使用 rich 库创建美观的输出
    console.print()

    # 创建完成信息表格
    completion_table = Table.grid(padding=(0, 1))
    completion_table.add_column(style="cyan", width=10)
    completion_table.add_column()

    completion_table.add_row("状态", "[bold green]✓ 训练完成[/bold green]")
    completion_table.add_row("保存目录", truncate_path(save_dir))
    completion_table.add_row("训练日志", truncate_path(csv_path))
    if best_loss is not None:
        completion_table.add_row("最佳损失", f"[bold yellow]{best_loss:.4f}[/bold yellow]")

    completion_panel = Panel(
        completion_table,
        title="[bold yellow]✅ Training Complete[/bold yellow]",
        title_align="left",
        border_style="bright_green",
        padding=(0, 1),
        expand=False,
    )

    console.print(completion_panel)
    console.print()


def print_mosaic_disabled(epoch: int):
    """打印 Mosaic 关闭通知

    Args:
        epoch: 当前 epoch
    """
    if RICH_AVAILABLE:
        console.print(f"\n[bold cyan][Epoch {epoch}][/bold cyan] [yellow]关闭 Mosaic 增强，使用原始数据精调[/yellow]")
    else:
        print(f"\n[Epoch {epoch}] 关闭 Mosaic 增强，使用原始数据精调")


def print_plotting_status(csv_path: Path, save_dir: Path):
    """打印训练曲线绘制状态

    Args:
        csv_path: CSV 日志路径
        save_dir: 保存目录
    """
    if RICH_AVAILABLE:
        console.print("\n[bold cyan]正在绘制训练曲线...[/bold cyan]")
    else:
        print("\n正在绘制训练曲线...")
