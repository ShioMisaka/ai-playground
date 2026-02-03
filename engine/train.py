"""
CLI Training Script

This script provides a command-line interface for training YOLO models.
It uses the unified YOLO.train() API under the hood.

Usage:
    python -m engine.train --name exp001 --epochs 100 --batch_size 16
    python -m engine.train --config configs/experiments/my_exp.yaml
"""
if __name__ == '__main__':
    from utils.config import parse_args, get_config, print_config, _parse_value
    from models import YOLO

    # Parse CLI arguments
    args = parse_args()

    # Collect override parameters
    overrides = {}
    for item in args.overrides:
        if '=' in item:
            key, value = item.split('=', 1)
            overrides[key] = _parse_value(value)

    # Build CLI parameters dict
    kwargs = {
        'name': args.name,
        'data': args.data,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'device': args.device,
        **overrides
    }
    # Filter None values
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    # Get configuration
    if args.config:
        cfg = get_config(config_file=args.config, model_config=args.model_config)
    else:
        cfg = get_config(model_config=args.model_config, **kwargs)

    # Print configuration
    print_config(cfg)

    # Create model and train using unified YOLO.train() API
    model = YOLO(cfg)
    model.train(cfg=cfg)
