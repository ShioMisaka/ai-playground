import torch
import torch.nn as nn

class CNN_2Ct(nn.Module):
    def __init__(self):
        super(CNN_2Ct, self).__init__()
        # 卷积层特征提取
        self.conv_layers = nn.Sequential(
            # 输出 [1, 28, 28] -> 输出 [32, 28, 28]
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), # 缩小图片 size -> [32, 14, 14]

            # 输入 [32, 14, 14] -> 输出 [64, 14, 14]
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)  # 再次缩小 -> [64, 7, 7]
        )

        # 全连接层分类
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*7*7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc_layers(self.conv_layers(x))
        return x