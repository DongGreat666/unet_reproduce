# 机构：人工智能研究所
# 人员：东
# 时间：2026/6/6 14:37
# 该脚本用于下载并整理 Oxford Pet 数据集，生成适用于 UNet 训练的图像和掩码文件夹结构。

# os：用于文件和目录操作
import os
import random
# shutil：用于删除和创建目录
import shutil
import numpy as np
# PIL：用于处理图像文件
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

    # 清空并重新创建目录
    for d in dirs:
        if os.path.exists(d):
            # 删除目录及其内容，rmtree是移除整个目录树
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)

def convert_mask(mask):
    """
    Oxford Pet mask 是 trimap，三维图，将三维图转换为二分类 mask：
    1 = pet, 2 = background, 3 = boundary
    UNet 二分类 mask：宠物+边界=1，背景=0
    """
    mask = np.array(mask)
    # where 函数：如果 mask 中的值为 2（背景），则设置为 0；否则设置为 1（宠物+边界）
    mask = np.where(mask == 2, 0, 1)
    # 将二值化的 mask 转换为 0 和 255 的图像格式，便于保存为 PNG 文件
    mask = mask.astype(np.uint8) * 255
    return Image.fromarray(mask)

def save_dataset(dataset, indices, image_dir, mask_dir):
    '''
    根据给定的索引列表，从数据集中提取图像和掩码，进行处理后保存到指定目录。
    '''
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




