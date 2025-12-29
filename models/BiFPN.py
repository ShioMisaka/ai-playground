import torch
import torch.nn as nn


class BiFPN_Concat2(nn.Module):
    """双输入特征融合模块 (Fast Normalized Fusion)"""
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension
        # 设置可学习权重，初始化为 1
        self.w = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.epsilon = 1e-4

    def forward(self, x: list[torch.Tensor]):
        # x 是一个包含两个特征图的列表: [feat1, feat2]
        w = torch.relu(self.w)  # 保证权重非负
        weight = w / (torch.sum(w, dim=0) + self.epsilon)  # 归一化
        
        # 加权融合后再 Concat
        return torch.cat([weight[0] * x[0], weight[1] * x[1]], self.d)

class BiFPN_Concat3(nn.Module):
    """三输入特征融合模块"""
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension
        self.w = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.epsilon = 1e-4

    def forward(self, x: list[torch.Tensor]):
        w = torch.relu(self.w)
        weight = w / (torch.sum(w, dim=0) + self.epsilon)
        return torch.cat([weight[0] * x[0], weight[1] * x[1], weight[2] * x[2]], self.d)
    