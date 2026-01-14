"""
带可视化功能的注意力模块

继承基础注意力模块，添加 forward_with_attention 方法用于可视化。
"""
import torch
from .att import CoordAtt, CoordCrossAtt, BiCoordCrossAtt


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


class BiCoordCrossAttWithVisualization(BiCoordCrossAtt):
    """
    带可视化功能的改进版 Coordinate Cross Attention

    继承 ImprovedCoordCrossAtt，添加 forward_with_attention 方法返回双向注意力权重用于可视化。
    """

    def forward_with_attention(self, x):
        """
        前向传播并返回双向注意力权重

        Args:
            x: 输入张量 [N, C, H, W]

        Returns:
            output: 输出张量 [N, oup, H, W]
            attn_h: H->W 方向注意力图 [N, num_heads, H, W]
            attn_w: W->H 方向注意力图 [N, num_heads, W, H]
            weight_h: 高度方向门控权重 [N, oup, H, 1]
            weight_w: 宽度方向门控权重 [N, oup, 1, W]
        """
        n, c, h, w = x.size()

        # 1. 池化获得方向向量
        x_h = self.pool_h(x)  # [N, C, H, 1]
        x_w = self.pool_w(x)  # [N, C, 1, W]

        # -----------------------
        # Branch 1: 增强 Height 方向 (利用 Width 信息)
        # -----------------------
        q_h = self.proj_q_h(x_h).view(n, self.num_heads, self.dim_head, h).permute(0, 1, 3, 2)
        k_h = self.proj_k_h(x_w).view(n, self.num_heads, self.dim_head, w)
        v_h = self.proj_v_h(x_w).view(n, self.num_heads, self.dim_head, w).permute(0, 1, 3, 2)

        # Attn Map: [N, heads, H, W]
        attn_h = (q_h @ k_h) * self.scale
        attn_h = attn_h.softmax(dim=-1)

        y_h = (attn_h @ v_h).permute(0, 1, 3, 2).reshape(n, self.mid_dim, h, 1)
        weight_h = self.activation(self.out_h(y_h))  # [N, oup, H, 1]

        # -----------------------
        # Branch 2: 增强 Width 方向 (利用 Height 信息)
        # -----------------------
        q_w = self.proj_q_w(x_w).view(n, self.num_heads, self.dim_head, w).permute(0, 1, 3, 2)
        k_w = self.proj_k_w(x_h).view(n, self.num_heads, self.dim_head, h)
        v_w = self.proj_v_w(x_h).view(n, self.num_heads, self.dim_head, h).permute(0, 1, 3, 2)

        # Attn Map: [N, heads, W, H]
        attn_w = (q_w @ k_w) * self.scale
        attn_w = attn_w.softmax(dim=-1)

        y_w = (attn_w @ v_w).permute(0, 1, 3, 2).reshape(n, self.mid_dim, 1, w)
        weight_w = self.activation(self.out_w(y_w))  # [N, oup, 1, W]

        # -----------------------
        # Final Fusion
        # -----------------------
        output = x * weight_h * weight_w

        return output, attn_h, attn_w, weight_h, weight_w
