#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/shiomisaka/workplace/ai-playground')

import numpy as np
import cv2
import torch
from PIL import Image

def debug_getitem(idx, img_size=640):
    """完全复制 YOLODataset.__getitem__ 的逻辑"""

    # 加载原始图片和标签（模拟 _load_raw_item）
    img_path = f'datasets/MY_TEST_DATA/images/train/square_{idx+57:04d}.jpg'
    try:
        img_pil = Image.open(img_path)
    except:
        img_path = 'datasets/MY_TEST_DATA/images/train/circle_0001.jpg'
        img_pil = Image.open(img_path)

    img = np.array(img_pil).astype(np.float32)
    print(f"[{idx}] Original: {img.shape}")

    # Letterbox 处理（完全复制代码）
    letterbox_params = None

    if True:  # self.letterbox
        img_h, img_w = img.shape[:2]
        r = min(img_size / img_h, img_size / img_w)
        scaled_h, scaled_w = int(round(img_h * r)), int(round(img_w * r))
        print(f"[{idx}] r={r}, scaled=({scaled_h}, {scaled_w})")

        # 缩放
        img = cv2.resize(img, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)
        print(f"[{idx}] After resize: {img.shape}")

        # 填充
        pad_h = img_size - scaled_h
        pad_w = img_size - scaled_w
        top, bottom = pad_h // 2, pad_h - pad_h // 2
        left, right = pad_w // 2, pad_w - pad_w // 2
        print(f"[{idx}] Padding: pad_h={pad_h}, top={top}, bottom={bottom}")

        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        print(f"[{idx}] After padding: {img.shape}")

        letterbox_params = (r, left, top)

    # 归一化
    img = img.astype(np.float32) / 255.0

    # HWC -> CHW
    img = torch.from_numpy(img).permute(2, 0, 1)
    print(f"[{idx}] Final (CHW): {img.shape}, expected=({3}, {img_size}, {img_size})")

    return img, letterbox_params

# 测试
for i in range(1):
    img, lb_params = debug_getitem(i)
    if img.shape != (3, 640, 640):
        print(f"ERROR: Shape mismatch!")
        break
