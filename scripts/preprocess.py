#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Preprocess SO32 dataset
- Resize all images to 224x224 (default)
- Save resized images to *_resized folders
- Keeps original train/val folders intact
"""

import os
from PIL import Image

# -----------------------------
# 配置参数
# -----------------------------
SRC_TRAIN = "data/SO32_preproc/train"
SRC_VAL   = "data/SO32_preproc/val"
DST_TRAIN = "data/SO32_preproc/train_resized"
DST_VAL   = "data/SO32_preproc/val_resized"
IMG_SIZE  = (224, 224)  # 可以根据需要修改

# -----------------------------
# 核心处理函数
# -----------------------------
def preprocess_folder(src_root, dst_root, size=IMG_SIZE):
    if not os.path.exists(src_root):
        print(f"Source folder does not exist: {src_root}")
        return

    os.makedirs(dst_root, exist_ok=True)

    classes = sorted([d for d in os.listdir(src_root) if os.path.isdir(os.path.join(src_root, d))])
    if not classes:
        print(f"No class folders found in {src_root}")
        return

    for cls in classes:
        cls_src = os.path.join(src_root, cls)
        cls_dst = os.path.join(dst_root, cls)
        os.makedirs(cls_dst, exist_ok=True)

        for img_name in os.listdir(cls_src):
            img_path = os.path.join(cls_src, img_name)
            img_dst_path = os.path.join(cls_dst, img_name)
            try:
                with Image.open(img_path) as img:
                    img = img.convert("RGB")
                    img = img.resize(size)
                    img.save(img_dst_path)
            except Exception as e:
                print(f"[Warning] Failed to process {img_path}: {e}")

# -----------------------------
# 执行预处理
# -----------------------------
if __name__ == "__main__":
    print("Start preprocessing...")
    preprocess_folder(SRC_TRAIN, DST_TRAIN)
    preprocess_folder(SRC_VAL, DST_VAL)
    print("Preprocessing done!")

