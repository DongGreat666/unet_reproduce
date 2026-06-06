# 机构：人工智能研究所
# 人员：东
# 时间：2026/6/6 9:40

from torch.utils.data import Dataset
from PIL import Image
import os
import torchvision.transforms as transforms
import torch
import config

class SegDataset(Dataset):
    def __init__(self, image_dir,  mask_dir, img_size=config.IMG_SIZE, binary=True):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.img_size = img_size
        self.binary = binary

        self.images = sorted(os.listdir(image_dir))
        self.masks = sorted(os.listdir(mask_dir))

        assert len(self.images) == len(self.masks)

        self.image_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor()
        ])

        self.mask_transform = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=Image.NEAREST),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_name = self.images[index]
        mask_name = self.masks[index]

        image_path = os.path.join(self.image_dir, image_name)
        mask_path = os.path.join(self.mask_dir, mask_name)

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = self.image_transform(image)
        mask = self.mask_transform(mask)

        if self.binary:
            mask = (mask > 0).float()

        return image, mask













