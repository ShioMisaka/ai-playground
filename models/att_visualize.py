"""
带可视化功能的注意力模块

继承基础注意力模块，添加 forward_with_attention 方法用于可视化。
"""
import torch
from .att import CoordAtt, CoordCrossAtt


class CoordAttWithVisualization(CoordAtt):
    """
    带可视化功能的 Coordinate Attention

    继承 CoordAtt，添加 forward_with_attention 方法返回注意力权重用于可视化。
    """

    def forward_with_attention(self, x):
        """
        前向传播并返回注意力权重

        Args:
            x: 输入张量 [N, C, H, W]

        Returns:
            output: 输出张量 [N, oup, H, W]
            a_h: 水平方向注意力权重 [N, oup, H, 1]
            a_w: 垂直方向注意力权重 [N, oup, 1, W]
        """
        n, c, h, w = x.size()

        # --- 信息嵌入 (Embedding) ---
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        # --- 协同注意力生成 (Generation) ---
        y = self.cv1(torch.cat([x_h, x_w], dim=2))

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        # 生成 H 和 W 方向的权重
        a_h = self.cv_h(x_h).sigmoid()
        a_w = self.cv_w(x_w).sigmoid()

        # --- 重加权 (Reweight) ---
        output = self.identity(x) * a_h * a_w

        return output, a_h, a_w


class CoordCrossAttWithVisualization(CoordCrossAtt):
    """
    带可视化功能的 Coordinate Cross Attention

    继承 CoordCrossAtt，添加 forward_with_attention 方法返回注意力权重用于可视化。
    """

    def forward_with_attention(self, x):
        """
        前向传播并返回注意力权重

        Args:
            x: 输入张量 [N, C, H, W]

        Returns:
            output: 输出张量 [N, oup, H, W]
            attn: Cross-Attention 相关性矩阵 [N, num_heads, H, W]
            y_att: 门控权重 [N, oup, H, 1]
        """
        n, c, h, w = x.size()

        # 1. 嵌入与压缩
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = self.cv1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(y, [h, w], dim=2)

        # 2. Cross-Attention 核心逻辑
        q = self.q_conv(x_h).view(n, self.num_heads, -1, h).permute(0, 1, 3, 2)
        k = self.k_conv(x_w).view(n, self.num_heads, -1, w)
        v = self.v_conv(x_w).view(n, self.num_heads, -1, w).permute(0, 1, 3, 2)

        # 计算相关性矩阵 (保存用于可视化)
        attn = (q @ k) * self.scale
        attn = attn.softmax(dim=-1)

        # 聚合信息
        z = (attn @ v).permute(0, 1, 3, 2).contiguous().view(n, self.mip, h, 1)

        # 3. 施加注意力
        y_att = self.gate(self.proj(z))

        output = x * y_att

        return output, attn, y_att
