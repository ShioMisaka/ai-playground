import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# BiFPN
from models.BiFPN import BiFPN_Concat3


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # 卷积层特征提取
        self.conv_layers = nn.Sequential(
            # 输出 [1, 28, 28] -> 输出 [32, 28, 28]
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), # 缩小图片 size -> [32, 14, 14]

            # 输入 [32, 14, 14] -> 输出 [64, 14, 14]
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 再次缩小 -> [64, 7, 7]
            
            # 输入 [32, 14, 14] -> 输出 [64, 14, 14]
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 再次缩小 -> [64, 7, 7]

            nn.Upsample(scale_factor=2),
            BiFPN_Concat3(...)  # BiFPN
        )

        # 全连接层分类
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*7*7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.fc_layers(self.conv_layers(x))
        return x
    

if __name__ == "__main__":
    ...