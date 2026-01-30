#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/shiomisaka/workplace/ai-playground')

import numpy as np
import cv2
from PIL import Image

# 模拟 YOLODataset.__getitem__ 中的 letterbox 代码
def test_letterbox(img_path, img_size=640):
    # 加载图像
    img = Image.open(img_path)
    img = np.array(img).astype(np.float32)
    print(f"1. Original image: {img.shape}")

    # Letterbox 处理
    img_h, img_w = img.shape[:2]
    r = min(img_size / img_h, img_size / img_w)
    scaled_h, scaled_w = int(round(img_h * r)), int(round(img_w * r))
    print(f"2. Scale: r={r}, scaled_h={scaled_h}, scaled_w={scaled_w}")

    # 缩放
    img = cv2.resize(img, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)
    print(f"3. After resize: {img.shape}")

    # 填充
    pad_h = img_size - scaled_h
    pad_w = img_size - scaled_w
    print(f"4. Padding: pad_h={pad_h}, pad_w={pad_w}")

    top, bottom = pad_h // 2, pad_h - pad_h // 2
    left, right = pad_w // 2, pad_w - pad_w // 2
    print(f"5. top={top}, bottom={bottom}, left={left}, right={right}")

    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    print(f"6. After padding: {img.shape}")
    print(f"   Expected: ({img_size}, {img_size}, 3)")

    # 归一化
    img = img.astype(np.float32) / 255.0

    # HWC -> CHW
    img = torch.from_numpy(img).permute(2, 0, 1)
    print(f"7. Final shape (CHW): {img.shape}")

    return img

# 测试
import torch
img_dir = 'datasets/MY_TEST_DATA/images/train'
from pathlib import Path
img_files = list(Path(img_dir).glob('*.jpg'))[:1]

for img_path in img_files:
    print(f"\nTesting {img_path.name}:")
    img = test_letterbox(str(img_path))
