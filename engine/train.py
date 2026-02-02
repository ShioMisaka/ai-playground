"""
YOLO 训练 CLI 脚本

这是一个简洁的 CLI 包装器，用于启动 YOLO 模型训练。
实际的训练逻辑已移至 engine.trainer.DetectionTrainer。
"""
import sys
from pathlib import Path

from rich.console import Console

from models import YOLO
from utils.config import parse_args

console = Console()


def main():
    """主训练入口"""
    # 解析 CLI 参数
    args = parse_args()

    # 确定模型配置文件路径
    if args.model_config:
        model_config = args.model_config
    else:
        # 使用默认的 YOLOv11n 配置
        model_config = 'configs/models/yolov11n.yaml'

    # 创建 YOLO 模型
    try:
        model = YOLO(model_config)
    except FileNotFoundError as e:
        console.print(f"[red]错误: {e}[/red]")
        console.print(f"[yellow]请确保模型配置文件存在: {model_config}[/yellow]")
        sys.exit(1)

    # 构建训练参数
    train_kwargs = {}

    # 从参数构建训练配置
    if args.name:
        train_kwargs['name'] = args.name
    if args.data:
        train_kwargs['data'] = args.data
    if args.epochs:
        train_kwargs['epochs'] = args.epochs
    if args.batch_size:
        train_kwargs['batch'] = args.batch_size
    if args.lr:
        train_kwargs['lr'] = args.lr
    if args.device:
        train_kwargs['device'] = args.device
    if args.img_size:
        train_kwargs['imgsz'] = args.img_size
    if args.save_dir:
        train_kwargs['save_dir'] = args.save_dir

    # 添加其他覆盖参数
    for key, value in args.overrides.items():
        # 跳过已处理的参数
        skip_keys = ['train.name', 'train.data', 'train.epochs', 'train.batch_size',
                     'train.lr', 'train.device', 'train.img_size', 'train.save_dir']
        if key in skip_keys:
            continue
        # 将点号命名转换为下划线命名
        train_kwargs[key.replace('.', '_')] = value

    # 打印开始信息
    console.print()
    console.print("[cyan]Starting YOLO training...[/cyan]")
    console.print(f"[green]Model config:[/green] {model_config}")
    if args.data:
        console.print(f"[green]Data config:[/green] {args.data}")
    console.print(f"[green]Epochs:[/green] {train_kwargs.get('epochs', 'default')}")
    console.print(f"[green]Batch size:[/green] {train_kwargs.get('batch', 'default')}")
    console.print(f"[green]Image size:[/green] {train_kwargs.get('imgsz', 'default')}")
    console.print(f"[green]Device:[/green] {train_kwargs.get('device', 'auto')}")
    console.print()

    # 开始训练
    try:
        results = model.train(**train_kwargs)

        # 打印训练完成信息
        console.print()
        console.print("[green]Training completed![/green]")
        console.print(f"[green]Results saved to:[/green] {results.get('save_dir', 'unknown')}")
        if 'best_map' in results:
            console.print(f"[green]Best mAP@0.5:[/green] {results['best_map']:.4f}")

    except FileNotFoundError as e:
        console.print(f"[red]错误: {e}[/red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]训练失败: {e}[/red]")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
