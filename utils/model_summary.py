"""
模型和训练信息输出模块

提供训练配置信息和模型摘要的输出功能。
使用 rich 库实现美观的终端界面展示。
"""
import platform
import sys
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Tuple

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


def get_device_info(device_str: str) -> str:
    """获取设备信息型号"""
    device = torch.device(device_str)

    if device.type == 'cuda':
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(device)
            try:
                gpu_props = torch.cuda.get_device_properties(device)
                total_memory_gb = gpu_props.total_memory / 1024**3
                return f"{gpu_name} ({total_memory_gb:.1f} GB)"
            except Exception:
                return gpu_name
        return "CUDA (unavailable)"
    else:
        cpu_info = platform.processor()
        if not cpu_info:
            cpu_info = platform.machine() or "Unknown CPU"
        return f"{cpu_info}"


def truncate_path(path: Path, max_parts: int = 3) -> str:
    """截断路径，只保留最后几级目录"""
    path_str = str(path)
    cwd = Path.cwd()

    try:
        rel_path = path.relative_to(cwd)
        if len(str(rel_path)) < len(path_str):
            parts = rel_path.parts
            if len(parts) > max_parts + 1:
                return f".../{'/'.join(parts[-max_parts:])}"
            return str(rel_path)
    except ValueError:
        pass

    parts = path.parts
    if len(parts) > max_parts + 1:
        return f".../{'/'.join(parts[-max_parts:])}"
    return path_str


def format_number(num: int) -> str:
    """格式化数字，添加千位分隔符"""
    return f"{num:,}"


def count_layers(model: nn.Module) -> int:
    """计算模型层数"""
    layer_count = 0
    for module in model.modules():
        if module is not model and list(module.children()) == []:
            layer_count += 1
    return layer_count


def estimate_flops(model: nn.Module, img_size: int) -> float:
    """粗略估计 FLOPs"""
    try:
        from thop import profile
        input_tensor = torch.randn(1, 3, img_size, img_size)
        # Suppress thop printing
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()):
             flops, _ = profile(model, inputs=(input_tensor,), verbose=False)
        return flops / 1e9
    except (ImportError, Exception):
        pass

    total_params = sum(p.numel() for p in model.parameters())
    feature_map_size = (img_size / 32) ** 2
    estimated_flops = total_params * 2 * feature_map_size * 0.1
    return estimated_flops / 1e9


def get_model_summary(model: nn.Module, img_size: int = 640) -> dict:
    """获取模型摘要信息"""
    num_layers = count_layers(model)
    total_params = sum(p.numel() for p in model.parameters())
    total_gradients = sum(p.numel() for p in model.parameters() if p.requires_grad)
    gflops = estimate_flops(model, img_size)

    return {
        'layers': num_layers,
        'parameters': total_params,
        'gradients': total_gradients,
        'gflops': gflops
    }


def _create_info_tables(
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
) -> Tuple[Table, Table, Table]:
    """仅创建内容表格，不创建 Panel。用于后续灵活布局。"""
    
    # --- Environment Table ---
    env_table = Table.grid(padding=(0, 2))
    env_table.add_column(style="cyan", width=12)
    env_table.add_column(style="green")
    env_table.add_row("设备", f"[bold white]{str(device)}[/bold white]")
    env_table.add_row("", f"[dim]{get_device_info(device)}[/dim]")
    env_table.add_row("Python", f"[bold white]{sys.version.split()[0]}[/bold white]")
    env_table.add_row("PyTorch", f"[bold white]{torch.__version__}[/bold white]")
    env_table.add_row("保存路径", truncate_path(Path(save_dir).resolve()))

    # --- Dataset Table ---
    dataset_table = Table.grid(padding=(0, 2))
    dataset_table.add_column(style="cyan", width=12)
    dataset_table.add_column(style="green")
    dataset_table.add_row("配置文件", truncate_path(Path(config_path).resolve()))
    if nc is not None:
        dataset_table.add_row("类别数", str(nc))
    if num_train_samples is not None:
        dataset_table.add_row("训练样本", f"{num_train_samples:,}")
    if num_val_samples is not None:
        dataset_table.add_row("验证样本", f"{num_val_samples:,}")

    # --- Hyperparameters Table ---
    hyper_table = Table.grid(padding=(0, 2))
    hyper_table.add_column(style="cyan", width=12)
    hyper_table.add_column(style="green")
    hyper_table.add_row("学习率", f"[bold green]{lr}[/bold green]")
    hyper_table.add_row("Batch Size", f"[bold green]{batch_size}[/bold green]")
    hyper_table.add_row("Epochs", f"[bold green]{epochs}[/bold green]")
    hyper_table.add_row("图像尺寸", f"[bold green]{img_size}[/bold green]")

    if use_mosaic is not None or use_ema is not None:
        hyper_table.add_row("", "")

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

    return env_table, dataset_table, hyper_table


def _create_model_table(model: nn.Module, img_size: int, nc: Optional[int] = None) -> Table:
    """仅创建模型信息表格"""
    # 覆盖 nc 逻辑
    if nc is not None:
        if hasattr(model, 'nc') and model.nc != nc:
            # 注意：副作用，修改了模型属性
            model.nc = nc
            if hasattr(model, 'detect'):
                model.detect.nc = nc
                model.detect.no = nc + 5
    
    summary = get_model_summary(model, img_size)
    model_name = model.__class__.__name__

    model_table = Table.grid(padding=(0, 1))
    model_table.add_column(style="cyan", width=10)
    model_table.add_column()
    model_table.add_row("模型名称", f"[bold white]{model_name}[/bold white]")
    model_table.add_row("层数", f"[bold green]{summary['layers']}[/bold green]")
    model_table.add_row("参数量", f"[bold yellow]{format_number(summary['parameters'])}[/bold yellow]")
    model_table.add_row("梯度数", f"[bold green]{format_number(summary['gradients'])}[/bold green]")
    model_table.add_row("GFLOPs", f"[bold magenta]{summary['gflops']:.1f} GFLOPs[/bold magenta]")
    
    return model_table


def create_training_info_panels(
    config_path, epochs, batch_size, img_size, lr, device, save_dir,
    num_train_samples=None, num_val_samples=None, nc=None, 
    use_mosaic=None, use_ema=None, close_mosaic=None,
    panel_height=None, panel_width=None,
):
    """(旧接口) 创建训练配置信息的 Panels，主要用于非 2x2 布局的场景"""
    
    env_table, dataset_table, hyper_table = _create_info_tables(
        config_path, epochs, batch_size, img_size, lr, device, save_dir,
        num_train_samples, num_val_samples, nc, use_mosaic, use_ema, close_mosaic
    )

    env_panel = Panel(
        env_table, title="[bold yellow]🚀 Environment[/bold yellow]",
        title_align="left", border_style="bright_blue", padding=(0, 1),
        height=panel_height, width=panel_width
    )
    dataset_panel = Panel(
        dataset_table, title="[bold yellow]📊 Dataset[/bold yellow]",
        title_align="left", border_style="bright_magenta", padding=(0, 1),
        height=panel_height, width=panel_width
    )
    hyper_panel = Panel(
        hyper_table, title="[bold yellow]⚙️ Hyperparameters[/bold yellow]",
        title_align="left", border_style="bright_cyan", padding=(0, 1),
        height=panel_height, width=panel_width
    )

    return env_panel, dataset_panel, hyper_panel


def print_training_start_2x2(
    config_path,
    epochs,
    batch_size,
    img_size,
    lr,
    device,
    save_dir,
    model: nn.Module,
    num_train_samples: Optional[int] = None,
    num_val_samples: Optional[int] = None,
    nc: Optional[int] = None,
    use_mosaic: Optional[bool] = None,
    use_ema: Optional[bool] = None,
    close_mosaic: Optional[int] = None,
):
    """
    打印训练开始信息（完美的 2x2 布局）
    布局：
    [ Environment ] [ Dataset   ]
    [ Hyperparams ] [ Model     ]
    保证：同行等高，同列等宽。
    """
    # Fallback for non-rich environments
    if not RICH_AVAILABLE:
        print("\n" + "=" * 60)
        print("Training Config (Rich not installed)")
        print(f"  Device: {device}")
        print(f"  Model: {model.__class__.__name__}")
        print("=" * 60 + "\n")
        return

    console.print()

    # 1. 生成所有内容表格
    t_env, t_data, t_hyper = _create_info_tables(
        config_path, epochs, batch_size, img_size, lr, device, save_dir,
        num_train_samples, num_val_samples, nc, use_mosaic, use_ema, close_mosaic
    )
    t_model = _create_model_table(model, img_size, nc)

    # 2. 计算每一行的最大高度
    # Panel 高度 = 内容行数 + 2 (Border) + 0 (Vertical Padding is 0 in (0,1))
    # 为防万一，可以额外 +1 防止紧凑，这里使用标准的 +2
    row1_height = max(t_env.row_count, t_data.row_count) + 2
    row2_height = max(t_hyper.row_count, t_model.row_count) + 2

    # 3. 创建 Panels，强制指定 height
    p_env = Panel(
        t_env, title="[bold yellow]🚀 Environment[/bold yellow]",
        title_align="left", border_style="bright_blue", padding=(0, 1),
        height=row1_height
    )
    p_data = Panel(
        t_data, title="[bold yellow]📊 Dataset[/bold yellow]",
        title_align="left", border_style="bright_magenta", padding=(0, 1),
        height=row1_height
    )
    p_hyper = Panel(
        t_hyper, title="[bold yellow]⚙️ Hyperparameters[/bold yellow]",
        title_align="left", border_style="bright_cyan", padding=(0, 1),
        height=row2_height
    )
    p_model = Panel(
        t_model, title="[bold yellow]🧠 Model Summary[/bold yellow]",
        title_align="left", border_style="bright_yellow", padding=(0, 1),
        height=row2_height
    )

    # 4. 使用主布局 Grid 实现 2x2 对齐
    # expand=True 确保占满宽度，ratio=1 确保两列等宽
    grid = Table.grid(padding=(0, 1), expand=True)
    grid.add_column(ratio=1)
    grid.add_column(ratio=1)

    # 添加行
    grid.add_row(p_env, p_data)
    grid.add_row(p_hyper, p_model)

    console.print(grid)
    console.print()


def print_training_info(
    config_path, epochs, batch_size, img_size, lr, device, save_dir,
    num_train_samples=None, num_val_samples=None, nc=None,
    use_mosaic=None, use_ema=None, close_mosaic=None,
):
    """打印训练配置信息（三面板布局）"""
    if not RICH_AVAILABLE:
        # Fallback simplified
        print(f"Training Info: Epochs={epochs}, Batch={batch_size}, Device={device}")
        return

    console.print()
    # 使用 create_training_info_panels 获取默认高度的 panels
    panels = create_training_info_panels(
        config_path, epochs, batch_size, img_size, lr, device, save_dir,
        num_train_samples, num_val_samples, nc, use_mosaic, use_ema, close_mosaic
    )
    # 使用 Columns 布局
    console.print(Columns(panels, equal=True))
    console.print()


def print_model_summary(model: nn.Module, img_size: int = 640, nc: Optional[int] = None):
    """单独打印模型摘要"""
    if not RICH_AVAILABLE:
        print(f"Model: {model}")
        return

    t_model = _create_model_table(model, img_size, nc)
    p_model = Panel(
        t_model,
        title="[bold yellow]🧠 Model Summary[/bold yellow]",
        title_align="left",
        border_style="bright_yellow",
        padding=(0, 1),
        expand=False,
    )
    console.print(p_model)
    console.print()


# 保持原有辅助函数不变
def print_training_setup(use_mosaic, use_ema, close_mosaic, num_train_samples, num_val_samples, nc, class_names, mosaic_enabled=False):
    if not RICH_AVAILABLE:
        return
    console.print()
    setup_table = Table.grid(padding=(0, 2))
    setup_table.add_column(style="cyan", width=12)
    setup_table.add_column()

    if mosaic_enabled:
        mosaic_status = f"[bold green]启用[/bold green] (最后 {close_mosaic} epoch 关闭)" if close_mosaic > 0 else "[bold green]启用[/bold green]"
    else:
        mosaic_status = "[bold red]禁用[/bold red]"
    
    setup_table.add_row("Mosaic 增强", mosaic_status)
    setup_table.add_row("EMA", "[bold green]启用[/bold green]" if use_ema else "[bold red]关闭[/bold red]")
    setup_table.add_row("", "")
    setup_table.add_row("类别数", f"[bold yellow]{nc}[/bold yellow]")
    setup_table.add_row("类别名称", str(class_names))
    setup_table.add_row("训练样本", f"[bold green]{num_train_samples:,}[/bold green]")
    setup_table.add_row("验证样本", f"[bold green]{num_val_samples:,}[/bold green]")

    console.print(Panel(setup_table, title="[bold yellow]⚡ Training Setup[/bold yellow]", title_align="left", border_style="bright_green", padding=(0, 1)))
    console.print()


def print_training_completion(save_dir: Path, csv_path: Path, best_loss: float = None):
    if not RICH_AVAILABLE:
        print(f"Done. Results at {save_dir}")
        return
        
    console.print()
    completion_table = Table.grid(padding=(0, 1))
    completion_table.add_column(style="cyan", width=10)
    completion_table.add_column()
    completion_table.add_row("状态", "[bold green]✓ 训练完成[/bold green]")
    completion_table.add_row("保存目录", truncate_path(Path(save_dir)))
    completion_table.add_row("训练日志", truncate_path(Path(csv_path)))
    if best_loss is not None:
        completion_table.add_row("最佳损失", f"[bold yellow]{best_loss:.4f}[/bold yellow]")

    console.print(Panel(completion_table, title="[bold yellow]✅ Training Complete[/bold yellow]", title_align="left", border_style="bright_green", padding=(0, 1), expand=False))
    console.print()


def print_mosaic_disabled(epoch: int):
    if RICH_AVAILABLE:
        console.print(f"\n[bold cyan][Epoch {epoch}][/bold cyan] [yellow]关闭 Mosaic 增强，使用原始数据精调[/yellow]")
    else:
        print(f"\n[Epoch {epoch}] 关闭 Mosaic 增强")


def print_plotting_status(csv_path: Path, save_dir: Path):
    if RICH_AVAILABLE:
        console.print("\n[bold cyan]正在绘制训练曲线...[/bold cyan]")
    else:
        print("\n正在绘制训练曲线...")