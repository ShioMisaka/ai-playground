import torch
import torch.nn as nn

from models import CoordAtt

# 1. 模拟特征图输入 (Batch_size, Channels, Height, Width)
# 场景：单层特征图，测试坐标注意力机制
feat = torch.randn(1, 256, 40, 40)

inp, oup = feat.shape[1], feat.shape[1]  # 输入和输出通道数
reduction = 32  # 缩减比例

# 2. 实例化模块
coord_att = CoordAtt(inp=inp, oup=oup, reduction=reduction)

# 3. 前向传播测试
out = coord_att(feat)

print(f"输入尺寸: {feat.shape}")
print(f"输出尺寸: {out.shape}")

# 4. 尺寸验证
# Coordinate Attention 是逐元素加权，输出尺寸应与输入相同
expected_shape = feat.shape
if out.shape == expected_shape:
    print(f"✅ 尺寸验证通过：输出尺寸为 {out.shape}")
else:
    print(f"❌ 尺寸验证失败：预期 {expected_shape}, 实际 {out.shape}")

# 5. 输出范围验证 (注意力权重应在 [0, 1] 之间，输出应与输入相关但范围可能不同)
print(f"\n输入统计: mean={feat.mean().item():.4f}, std={feat.std().item():.4f}")
print(f"输出统计: mean={out.mean().item():.4f}, std={out.std().item():.4f}")

# 6. 反向传播测试 (验证权重是否可学习)
print("\n=== 开始可学习权重测试 ===")
optimizer = torch.optim.SGD(coord_att.parameters(), lr=0.1)

# 记录初始权重
initial_params = []
for name, p in coord_att.named_parameters():
    if p.requires_grad:
        initial_params.append(p.detach().clone())
print(f"初始参数数量: {len(initial_params)}")

# 模拟一次损失计算和反向传播
target = torch.randn_like(out)
loss = nn.MSELoss()(out, target)

optimizer.zero_grad()
loss.backward()
optimizer.step()

updated_params = []
for name, p in coord_att.named_parameters():
    if p.requires_grad:
        updated_params.append(p.detach().clone())

# 验证参数是否更新
params_updated = False
for i, (init, updated) in enumerate(zip(initial_params, updated_params)):
    if not torch.allclose(init, updated):
        params_updated = True
        break

if params_updated:
    print("✅ 权重已成功更新，模块具有学习能力。")
else:
    print("❌ 权重未变化，请检查 requires_grad 或梯度流。")

# 7. 额外验证：测试不同输入尺寸
print("\n=== 开始不同输入尺寸测试 ===")
test_cases = [
    (1, 128, 20, 20),
    (2, 512, 64, 64),
    (1, 64, 16, 32),
]

for shape in test_cases:
    test_feat = torch.randn(shape)
    test_inp, test_oup = shape[1], shape[1]
    test_module = CoordAtt(inp=test_inp, oup=test_oup, reduction=16)
    test_out = test_module(test_feat)
    if test_out.shape == test_feat.shape:
        print(f"✅ 测试通过 {shape} -> {test_out.shape}")
    else:
        print(f"❌ 测试失败 {shape} -> {test_out.shape}")
