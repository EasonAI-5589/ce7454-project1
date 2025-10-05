#!/usr/bin/env python3
import os
import sys

# 检查可能的数据路径
paths = [
    'data/train/images',
    './data/train/images',
    '../data/train/images',
    'CelebAMask-HQ-mini/train/images',
    'train/images',
]

print("检查数据路径...")
for path in paths:
    exists = os.path.exists(path)
    symbol = "✓" if exists else "✗"
    print(f"{symbol} {path}")
    if exists:
        count = len(os.listdir(path))
        print(f"  → 找到 {count} 个文件")

print("\n当前工作目录:", os.getcwd())
print("\n列出当前目录内容:")
for item in os.listdir('.'):
    if os.path.isdir(item):
        print(f"📁 {item}/")
    else:
        print(f"📄 {item}")
