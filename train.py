# 机构：人工智能研究所
# 人员：东
# 时间：2026/6/6 9:02

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from models.unet import UNet
from utils.dataset import SegDataset
from utils import metrics

import csv

# csv文件路径
csv_file = os.path.join(config.LOG_DIR, "train_history.csv")

def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    val_dice = 0.0

    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)

            val_loss += loss.item()
            val_dice += metrics.dice_score(outputs, masks).item()

    val_loss /= len(val_loader)
    val_dice /= len(val_loader)
    return val_loss, val_dice

def train():
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.LOG_DIR, exist_ok=True)
    os.makedirs(config.OUTPUTS_DIR, exist_ok=True)

    device = torch.device(config.DEVICE)
    print(f"Using Device: {device}")

    train_dataset = SegDataset(config.TRAIN_IMAGE_DIR, config.TRAIN_MASK_DIR, img_size=config.IMG_SIZE)
    val_dataset = SegDataset(config.VAL_IMAGE_DIR, config.VAL_MASK_DIR, img_size=config.IMG_SIZE)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)

    model = UNet(n_channels=config.IN_CHANNELS, n_classes=config.NUM_CLASSES).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LR)

    best_dice = 0.0

    for epoch in range(config.EPOCHS):
        model.train()
        train_loss = 0.0

        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{config.EPOCHS}]")

        for images, masks in loop:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            loop.set_postfix(loss=loss.item())

        train_loss /= len(train_loader)

        val_loss, val_dice = validate(model, val_loader, criterion, device)

        print(f"Epoch [{epoch+1}/{config.EPOCHS}] "
              f"Train Loss: {train_loss:.4f} "
              f"Val Loss: {val_loss:.4f} "
              f"Val Dice: {val_dice:.4f}"
              )

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_dice": best_dice,
            }, config.BEST_MODEL_PATH)
            print(f"Best model saved, Dice: {best_dice:.4f}")

        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_dice": best_dice,
        }, config.LAST_MODEL_PATH)

        os.makedirs(config.LOG_DIR, exist_ok=True)
        files_exists = os.path.exist(csv_file)
        with open(csv_file, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss", "val_dice"])
            if not files_exists:
                writer.writeheader()
            writer.writerow({
                "epoch": epoch+1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_dice": val_dice
            })

    print("训练完成")
    print(f"Best Dice: {best_dice:4f}")

if __name__ == "__main__":
    train()











