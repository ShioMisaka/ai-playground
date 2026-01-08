import torch
import torch.nn as nn
from .conv import Conv

class CoordAtt(nn.Module):
    def __init__(self, inp: int, oup: int, reduction: int = 32):
        """
        Coordinate Attention 模块 (改进版：细化交叉点注意力)
        :param inp: 输入通道数
        :param oup: 输出通道数
        :param reduction: 缩减比例，用于构建 Bottleneck 结构
        """
        super().__init__()
        # 1. 空间池化：分别聚合水平和垂直信息
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        # 计算中间层的压缩通道数 (Bottleneck)
        mip = max(8, inp // reduction)

        # 2. 核心融合层：将 H 和 W 信息拼接后进行通道压缩和非线性激活
        self.cv1 = Conv(inp, mip, k=1, s=1, p=0)

        # 3. 恢复层：将压缩的通道 mip 恢复到输出通道 oup
        self.cv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.cv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

        # 4. 注意力细化：用 3x3 卷积增强交叉点，削弱边缘
        self.att_refine = nn.Conv2d(oup, oup, kernel_size=3, padding=1)

        # 5. 如果输入输出通道不一致，需要一个 shortcut 变换
        self.identity = nn.Conv2d(inp, oup, 1) if inp != oup else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.size()

        # --- 信息嵌入 (Embedding) ---
        x_h = self.pool_h(x)  # [N, C, H, 1]
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # [N, C, W, 1]

        # --- 协同注意力生成 (Generation) ---
        y = self.cv1(torch.cat([x_h, x_w], dim=2))  # [N, mip, H+W, 1]

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)  # [N, mip, 1, W]

        # 生成 H 和 W 方向的权重 (Sigmoid 激活)
        a_h = self.cv_h(x_h).sigmoid()  # [N, oup, H, 1]
        a_w = self.cv_w(x_w).sigmoid()  # [N, oup, 1, W]

        # --- 重加权 (Reweight) ---
        # 先计算基础注意力（外积）
        att_base = a_h * a_w  # [N, oup, H, W]
        # 用 3x3 卷积细化：增强交叉点，削弱边缘
        att_refined = att_base * self.att_refine(att_base).sigmoid()
        return self.identity(x) * att_refined


import torch
import torch.nn as nn
from .conv import Conv 

class CoordAtt(nn.Module):
    def __init__(self, inp: int, oup: int, reduction: int = 32):
        """
        Coordinate Attention 模块 (标准版)
        :param inp: 输入通道数
        :param oup: 输出通道数
        :param reduction: 缩减比例，用于构建 Bottleneck 结构
        """
        super().__init__()
        # 1. 空间池化：分别聚合水平和垂直信息
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        # 计算中间层的压缩通道数 (Bottleneck)
        mip = max(8, inp // reduction)

        # 2. 核心融合层：将 H 和 W 信息拼接后进行通道压缩和非线性激活
        # 这里使用 Ultralytics 的 Conv，默认包含 BatchNorm2d 和 SiLU
        self.cv1 = Conv(inp, mip, k=1, s=1, p=0) 

        # 3. 恢复层：将压缩的通道 mip 恢复到输出通道 oup
        # 注意：这里不需要 SiLU，后面手动跟 Sigmoid
        self.cv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.cv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        
        # 4. 如果输入输出通道不一致，需要一个 shortcut 变换
        self.identity = nn.Conv2d(inp, oup, 1) if inp != oup else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.size()
        
        # --- 信息嵌入 (Embedding) ---
        x_h = self.pool_h(x) # [N, C, H, 1]
        x_w = self.pool_w(x).permute(0, 1, 3, 2) # [N, C, 1, W] -> [N, C, W, 1]

        # --- 协同注意力生成 (Generation) ---
        # 拼接后的尺寸: [N, C, H+W, 1]
        y = self.cv1(torch.cat([x_h, x_w], dim=2)) # 压缩到 [N, mip, H+W, 1]
        
        # 拆分回两个方向
        x_h, x_w = torch.split(y, [h, w], dim=2) # 分别为 [N, mip, H, 1] 和 [N, mip, W, 1]
        x_w = x_w.permute(0, 1, 3, 2) # 转回 [N, mip, 1, W]

        # 生成 H 和 W 方向的权重 (Sigmoid 激活)
        a_h = self.cv_h(x_h).sigmoid() # [N, oup, H, 1]
        a_w = self.cv_w(x_w).sigmoid() # [N, oup, 1, W]

        # --- 重加权 (Reweight) ---
        # 如果 inp != oup，先转换 x，否则直接相乘
        return self.identity(x) * a_h * a_w