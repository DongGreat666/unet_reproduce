# 机构：人工智能研究所
# 人员：东
# 时间：2026/6/6 14:37

import os
import random
import shutil
import numpy as np
from PIL import Image
from torchvision.datasets import OxfordIIITPet
import config

def clear_dirs():
    dirs = [
        config.TRAIN_IMAGE_DIR,
        config.TRAIN_MASK_DIR,
        config.VAL_IMAGE_DIR,
        config.VAL_MASK_DIR,
        config.TEST_IMAGE_DIR,
        config.TEST_MASK_DIR
    ]

    for d in dirs:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)

def convert_mask(mask):
    """
    Oxford Pet mask 是 trimap：
    1 = pet, 2 = background, 3 = boundary
    UNet 二分类 mask：宠物+边界=1，背景=0
    """
    mask = np.array(mask)
    mask = np.where(mask == 2, 0, 1)
    mask = mask.astype(np.uint8) * 255
    return Image.fromarray(mask)

def save_dataset(dataset, indices, image_dir, mask_dir):
    for new_idx, old_idx in enumerate(indices):
        img, mask = dataset[old_idx]
        img = img.convert("RGB")
        mask = convert_mask(mask)
        img_name = f"{new_idx:05d}.jpg"
        mask_name = f"{new_idx:05d}.png"
        img.save(os.path.join(image_dir, img_name))
        mask.save(os.path.join(mask_dir, mask_name))

def main():
    clear_dirs()

    # 下载数据集
    dataset = OxfordIIITPet(root=config.DATA_DIR, split='trainval',
                            target_types="segmentation", download=True)
    total = len(dataset)
    indices = list(range(total))
    random.seed(42)
    random.shuffle(indices)

    train_ratio = config.TRAINSET_RATIO
    val_ratio = config.VALSET_RATIO
    test_ratio = config.TESTSET_RATIO
    train_end = int(total * train_ratio)
    val_end = int(total * (train_ratio + val_ratio))
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]

    save_dataset(dataset, train_indices, config.TRAIN_IMAGE_DIR, config.TRAIN_MASK_DIR)
    save_dataset(dataset, val_indices, config.VAL_IMAGE_DIR, config.VAL_MASK_DIR)
    save_dataset(dataset, test_indices, config.TEST_IMAGE_DIR, config.TEST_MASK_DIR)

    print("Oxford Pet 数据集整理完成")
    print(f"训练集：{len(train_indices)}张")
    print(f"验证集：{len(val_indices)}张")
    print(f"测试集：{len(test_indices)}张")
    print(f"训练图片路径：{config.TRAIN_IMAGE_DIR}")
    print(f"训练MASK路径：{config.TRAIN_IMAGE_DIR}")

if __name__ == "__main__":
    main()




