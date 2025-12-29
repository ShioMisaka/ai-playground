```txt
project/
├── datasets/               # 数据集存放位置
│
├── models/                 # 模型结构
│   │──__init__.py
│   ├── bifpn_enhance.py
│   ├── bifpn.py
│   ├── cnn_t.py
│   ├── conv.py
│   └── mlp_t.py
│
├── engine/                 
│   ├── __init__.py         # 训练 / 验证逻辑
│   ├── train.py            # train_one_epoch
│   └── validate.py         # validate / evaluate
│
├── losses/                 # 损失函数(还没写)
│   └── loss.py
│
├── utils/                  # 工具函数
│   ├── metrics.py          # (还没写)
│   ├── logger.py           # (还没写)
│   ├── checkpoint.py       # (还没写)
│   └── load.py             # 读取数据集
│
├── configs/                # 配置文件(还没写)
│   └── train.yaml
│
├── BiFPN_test.ipynb        # BiFPN测试脚本
├── fpn_test.py             # BiFPN测试
├── mnist_demo.ipynb        # mnist测试脚本
├── mnist_demo.py           # mnist测试脚本
└── requirements.txt        # (还没写)
```
