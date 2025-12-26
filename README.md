project/
├── data/                   # 数据相关
│   ├── dataset.py
│   └── dataloader.py
│
├── models/                 # 模型结构
│   ├── cnn.py
│   └── yolov5.py
│
├── engine/                 # ⭐ 训练 / 验证逻辑（推荐）
│   ├── train.py            # train_one_epoch
│   ├── validate.py         # validate / evaluate
│   └── __init__.py
│
├── losses/                 # 损失函数
│   └── loss.py
│
├── utils/                  # 工具函数
│   ├── metrics.py
│   ├── logger.py
│   └── checkpoint.py
│
├── configs/                # 配置文件
│   └── train.yaml
│
├── main.py                 # 入口脚本
└── requirements.txt
