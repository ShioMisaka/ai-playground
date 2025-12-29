import torch
import torch.nn as nn

from models import BiFPN_Cat

# 1. 模拟特征图输入 (Batch_size, Channels, Height, Width)
# 场景：两个 128 通道，一个 256 通道（测试自动对齐功能）
feat1 = torch.randn(1, 128, 40, 40)
feat2 = torch.randn(1, 256, 40, 40)
feat3 = torch.randn(1, 512, 40, 40)

input_list = [feat1, feat2, feat3]
input_channels = [f.shape[1] for f in input_list] # [128, 128, 256]

# 2. 实例化模块
# 注意：现在需要传入输入通道列表
bifpn = BiFPN_Cat(c1=input_channels)

# 3. 前向传播测试
out = bifpn(input_list)

print(f"输入通道列表: {input_channels}")
print(f"输出尺寸: {out.shape}")

# 修正你的预期逻辑：
# BiFPN 是融合，不是堆叠。输出通道 = max(input_channels) = 256
expected_channels = max(input_channels)
if out.shape[1] == expected_channels:
    print(f"✅ 尺寸验证通过：输出通道为 {out.shape[1]}")
else:
    print(f"❌ 尺寸验证失败：预期 {expected_channels}, 实际 {out.shape[1]}")

# 4. 反向传播测试 (验证权重是否可学习)
print("\n=== 开始可学习权重测试 ===")
optimizer = torch.optim.SGD(bifpn.parameters(), lr=0.1)

# 记录初始权重
# 注意：w 的长度现在是根据输入自动确定的
initial_weights = bifpn.w.detach().clone()
print(f"初始权重 (w): {initial_weights.numpy()}")

# 模拟一次损失计算和反向传播
# 增加一个目标值，让 loss 更有意义
target = torch.randn_like(out)
loss = nn.MSELoss()(out, target)

optimizer.zero_grad()
loss.backward()
optimizer.step()

updated_weights = bifpn.w.detach().clone()
print(f"更新后权重 (w): {updated_weights.numpy()}")

if not torch.allclose(initial_weights, updated_weights):
    print("✅ 权重已成功更新，模块具有学习能力。")
else:
    print("❌ 权重未变化，请检查 requires_grad 或梯度流。")

# 5. 额外验证：检查 1x1 卷积是否存在
print(f"\n对齐卷积层数量: {len(bifpn.realign_convs)}")
