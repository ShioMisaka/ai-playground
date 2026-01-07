import torch
import torch.nn as nn
# 假设你在 ultralytics/nn/modules/block.py 中编写，可以直接引入 Conv
from .conv import Conv 

class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        """
        使用 Ultralytics Conv 组件实现的 Coordinate Attention
        :param inp: 输入通道数
        :param oup: 输出通道数
        :param reduction: 缩减比例
        """
        super().__init__()
        # 1. 空间池化层 (无需 Conv)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        # 2. 核心变换层：使用 Ultralytics 的 Conv
        # 默认 act=True 对应 SiLU，自动包含 Conv+BN+SiLU
        self.cv1 = Conv(inp, mip, k=1, s=1, p=0) 

        # 3. 输出层：通常最后一步不需要激活函数，或者是 Sigmoid
        # 这里我们手动处理 Sigmoid，所以 Conv 设置 act=False
        self.cv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.cv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):
        n, c, h, w = x.size()
        
        # 信息嵌入 (H, W 维度聚合)
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        # 拼接与融合 (利用 Ultralytics Conv 的 SiLU 激活)
        y = self.cv1(torch.cat([x_h, x_w], dim=2))
        
        # 拆分
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        # 加权输出
        a_h = self.cv_h(x_h).sigmoid()
        a_w = self.cv_w(x_w).sigmoid()

        return x * a_h * a_w