#!/usr/bin/env python3
"""独立测试 letterbox 实现"""
import numpy as np
import cv2
import torch
from PIL import Image

# 加载图像
img_path = 'datasets/MY_TEST_DATA/images/train/square_0057.jpg'
img_pil = Image.open(img_path)
img = np.array(img_pil).astype(np.float32)
print(f"原始图像: {img.shape}")

# Letterbox 参数
img_size = 640
img_h, img_w = img.shape[:2]
print(f"img_h={img_h}, img_w={img_w}")

# 计算缩放因子
r = min(img_size / img_h, img_size / img_w)
print(f"r = min({img_size}/{img_h}, {img_size}/{img_w}) = {r}")

# 计算缩放后尺寸
scaled_h = int(round(img_h * r))
scaled_w = int(round(img_w * r))
print(f"scaled_h={scaled_h}, scaled_w={scaled_w}")

# 缩放
img_resized = cv2.resize(img, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)
print(f"缩放后: {img_resized.shape}")

# 计算填充
pad_h = img_size - scaled_h
pad_w = img_size - scaled_w
print(f"pad_h={pad_h}, pad_w={pad_w}")

top, bottom = pad_h // 2, pad_h - pad_h // 2
left, right = pad_w // 2, pad_w - pad_w // 2
print(f"top={top}, bottom={bottom}, left={left}, right={right}")

# 计算预期最终尺寸
expected_h = scaled_h + top + bottom
expected_w = scaled_w + left + right
print(f"预期最终尺寸: ({expected_h}, {expected_w}, 3)")

# 填充
img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
print(f"填充后: {img_padded.shape}")

# 转换为 tensor
img_tensor = torch.from_numpy(img_padded.astype(np.float32) / 255.0).permute(2, 0, 1)
print(f"最终 tensor: {img_tensor.shape}")
print(f"预期: torch.Size([3, 640, 640])")
