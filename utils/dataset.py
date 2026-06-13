# 机构：人工智能研究所
# 人员：东
# 时间：2026/6/6 9:40
import random

from torch.utils.data import Dataset
from PIL import Image
import os
import torchvision.transforms.functional as TF
import torch
import config

class SegDataset(Dataset):
    def __init__(self, image_dir,  mask_dir, img_size=config.IMG_SIZE, train=True, binary=True):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.img_size = img_size
        self.train = train
        self.binary = binary

        self.images = sorted(os.listdir(image_dir))
        self.masks = sorted(os.listdir(mask_dir))

        assert len(self.images) == len(self.masks)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_name = self.images[index]
        mask_name = self.masks[index]

        image_path = os.path.join(self.image_dir, image_name)
        mask_path = os.path.join(self.mask_dir, mask_name)

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # Resize 最短边到 img_size*1.125（572）
        resize_size = int(self.img_size * 1.125)
        image = TF.resize(image, resize_size)
        mask = TF.resize(mask, resize_size, interpolation=Image.NEAREST)

        # CenterCrop 到 img_size
        image = TF.center_crop(image, self.img_size)
        mask = TF.center_crop(mask, self.img_size)

        # 数据增强：训练集随机水平翻转 + 随机旋转 + 轻微颜色扰动
        if self.train:
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

            angle = random.uniform(-10, 10)
            image = TF.rotate(image, angle)
            mask = TF.rotate(mask, angle, interpolation=Image.NEAREST)

            # 颜色扰动只作用于图像
            brightness = random.uniform(0.8, 1.2)
            contrast = random.uniform(0.8, 1.2)
            saturation = random.uniform(0.8, 1.2)
            image = TF.adjust_brightness(image, brightness)
            image = TF.adjust_contrast(image, contrast)
            image = TF.adjust_saturation(image, saturation)

        # ToTensor
        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)

        if self.binary:
            mask = (mask > 0).float()

        return image, mask













