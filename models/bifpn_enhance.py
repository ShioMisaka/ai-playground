import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
    """深度可分离卷积"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, 
                                   stride, padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.act(x)


class BiFPNBlock(nn.Module):
    """
    BiFPN模块 - 可替换YOLO中的Concat
    
    Args:
        in_channels: 输入特征图的通道数列表 [C3, C4, C5]
        out_channels: 输出通道数
        num_layers: BiFPN重复次数
        epsilon: 防止除零的小常数
    """
    def __init__(self, in_channels, out_channels, num_layers=1, epsilon=1e-4):
        super().__init__()
        self.epsilon = epsilon
        self.num_layers = num_layers
        
        # 通道对齐层
        self.channel_align = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels)
            ) for c in in_channels
        ])
        
        # 自顶向下路径的卷积
        self.td_convs = nn.ModuleList([
            DepthwiseSeparableConv(out_channels, out_channels)
            for _ in range(len(in_channels) - 1)
        ])
        
        # 自底向上路径的卷积
        self.bu_convs = nn.ModuleList([
            DepthwiseSeparableConv(out_channels, out_channels)
            for _ in range(len(in_channels) - 1)
        ])
        
        # 可学习的权重参数
        self.td_weights = nn.ParameterList([
            nn.Parameter(torch.ones(2, dtype=torch.float32))
            for _ in range(len(in_channels) - 1)
        ])
        
        self.bu_weights = nn.ParameterList([
            nn.Parameter(torch.ones(3, dtype=torch.float32))
            for _ in range(len(in_channels) - 1)
        ])
        
        # 输出层的权重
        self.out_weights = nn.ParameterList([
            nn.Parameter(torch.ones(2, dtype=torch.float32))
            for _ in range(len(in_channels))
        ])
    
    def _resize(self, x, target_size):
        """调整特征图尺寸"""
        if x.shape[-2:] != target_size:
            x = F.interpolate(x, size=target_size, mode='nearest')
        return x
    
    def forward(self, inputs: torch.Tensor):
        """
        Args:
            inputs: 特征图列表 [P3, P4, P5] 从小尺度到大尺度
        Returns:
            融合后的特征图列表
        """
        assert len(inputs) == len(self.channel_align), "输入特征图数量不匹配"
        
        # 通道对齐
        feats = [align(x) for align, x in zip(self.channel_align, inputs)]
        
        # 自顶向下路径 (Top-down pathway)
        td_feats = [feats[-1]]  # 从最高层开始
        
        for i in range(len(feats) - 2, -1, -1):
            # 加权融合
            w = F.relu(self.td_weights[len(feats) - 2 - i])
            w = w / (w.sum() + self.epsilon)
            
            # 上采样高层特征
            top_down = self._resize(td_feats[0], feats[i].shape[-2:])
            
            # 加权融合
            fused = w[0] * feats[i] + w[1] * top_down
            
            # 应用卷积
            fused = self.td_convs[len(feats) - 2 - i](fused)
            td_feats.insert(0, fused)
        
        # 自底向上路径 (Bottom-up pathway)
        bu_feats = [td_feats[0]]  # 从最低层开始
        
        for i in range(1, len(td_feats)):
            # 加权融合
            w = F.relu(self.bu_weights[i - 1])
            w = w / (w.sum() + self.epsilon)
            
            # 下采样低层特征
            if bu_feats[-1].shape[-2:] != td_feats[i].shape[-2:]:
                bottom_up = F.max_pool2d(bu_feats[-1], kernel_size=2, stride=2)
            else:
                bottom_up = bu_feats[-1]
            
            # 加权融合：原始输入 + 自顶向下 + 自底向上
            fused = (w[0] * feats[i] + 
                    w[1] * td_feats[i] + 
                    w[2] * bottom_up)
            
            # 应用卷积
            fused = self.bu_convs[i - 1](fused)
            bu_feats.append(fused)
        
        # 输出加权融合
        output = []
        for i in range(len(bu_feats)):
            w = F.relu(self.out_weights[i])
            w = w / (w.sum() + self.epsilon)
            out = w[0] * feats[i] + w[1] * bu_feats[i]
            output.append(out)
        
        return output


class BiFPN(nn.Module):
    """
    完整的BiFPN模块，可直接替换YOLO中的Concat
    
    使用示例:
        # 在YOLO的yaml配置中
        # 原来: [-1, 6], 1, Concat, [1]]
        # 替换: [[-1, 6, 4], 1, BiFPN, [256, 2]]  # 256是输出通道数，2是重复次数
    """
    def __init__(self, in_channels_list, out_channels, num_layers=2):
        super().__init__()
        self.bifpn_blocks = nn.ModuleList([
            BiFPNBlock(
                in_channels_list if i == 0 else [out_channels] * len(in_channels_list),
                out_channels,
                num_layers=1
            ) for i in range(num_layers)
        ])
    
    def forward(self, x):
        """
        Args:
            x: 输入特征图列表或单个tensor
        """
        if isinstance(x, torch.Tensor):
            return x
        
        # 如果输入是列表，执行BiFPN融合
        for block in self.bifpn_blocks:
            x = block(x)
        
        # 返回融合后的特征（通常返回中间尺度的特征）
        return x[len(x) // 2] if len(x) > 1 else x[0]

