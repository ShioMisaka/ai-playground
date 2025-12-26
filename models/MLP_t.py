import torch
import torch.nn as nn

class MLP_3Lt(nn.Module):
    def __init__(self):
        super(MLP_3Lt, self).__init__()
        self.flatten = nn.Flatten()
        self.fc_layers = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),     # 丢弃 20% 的神经元防止过拟合

            nn.Linear(512,256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(256,10)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc_layers(self.flatten(x))