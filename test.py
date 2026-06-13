# 机构：人工智能研究所
# 人员：东
# 时间：2026/6/6 9:02
import os

from models.resnet_unet import ResNetUNet

# 设置环境变量，允许重复加载库，避免因库冲突导致的错误。
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
from torch.utils.data import DataLoader

import config
from models.unet import UNet
from utils.dataset import SegDataset
from utils.metrics import dice_score, iou_score
from utils.plot import visualize_prediction


def test():
    device = torch.device(config.DEVICE)
    print("Using Device: ", device)

    test_dataset = SegDataset(config.TEST_IMAGE_DIR, config.TEST_MASK_DIR, img_size=config.IMG_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    # 模型
    # model = UNet(n_channels=config.IN_CHANNELS, n_classes=config.NUM_CLASSES).to(device)
    model = ResNetUNet(n_channels=config.IN_CHANNELS, n_classes=config.NUM_CLASSES).to(device)
    checkpoint = torch.load(config.BEST_MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    total_dice = 0.0
    total_iou = 0.0
    n = len(test_loader)

    with torch.no_grad():
        for idx, (images, masks) in enumerate(test_loader):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            preds = torch.sigmoid(outputs)
            preds_bin = (preds > 0.5).float()

            total_dice += dice_score(preds_bin, masks).item()
            total_iou += iou_score(preds_bin, masks).item()

            # 可视化前几个样本
            if idx < 5:
                visualize_prediction(images[0], masks[0], preds_bin[0], save_path=f"outputs/pred_{idx}.png")

    avg_dice = total_dice / n
    avg_iou = total_iou / n
    print(f"Test Dice: {avg_dice:.4f}")
    print(f"Test IoU : {avg_iou:.4f}")

if __name__ == "__main__":
    os.makedirs(config.OUTPUTS_DIR, exist_ok=True)
    test()










