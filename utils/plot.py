# 机构：人工智能研究所
# 人员：东
# 时间：2026/6/7 10:25
import csv
import os
# 设置环境变量，允许重复加载库，避免因库冲突导致的错误。
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 14
import numpy as np

import config


def visualize_prediction(image, mask, pred_mask, save_path=None):
    image = image.permute(1, 2, 0).cpu().numpy()
    mask = mask.squeeze(0).cpu().numpy()
    pred_mask = pred_mask.squeeze(0).cpu().numpy()

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(image)
    axs[0].set_title("Image")
    axs[0].axis("off")

    axs[1].imshow(mask, cmap="gray")
    axs[1].set_title("Ground Truth")
    axs[1].axis("off")

    axs[2].imshow(pred_mask, cmap="gray")
    axs[2].set_title("Prediction")
    axs[2].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close()

def plot_training_history(csv_path, save_dir=None):
    epochs = []
    train_loss = []
    val_loss = []
    val_dice = []

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            train_loss.append(float(row["train_loss"]))
            val_loss.append(float(row["val_loss"]))
            val_dice.append(float(row["val_dice"]))


    # 绘制 Loss 曲线
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, label="Train Loss")
    plt.plot(epochs, val_loss, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "loss_curve.png"), dpi=300)
    plt.close()

    # 绘制 Dice 曲线
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, val_dice, label="Val Dice", color="green")
    plt.xlabel("Epoch")
    plt.ylabel("Dice")
    plt.title("Validation Dice Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "dice_curve.png"), dpi=300)
    plt.close()

def plot_compare_from_one_csv(csv_path, save_dir=None, epochs_per_exp=20):
    train_loss_all = []
    val_loss_all = []
    val_dice_all = []

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            train_loss_all.append(float(row["train_loss"]))
            val_loss_all.append(float(row["val_loss"]))
            val_dice_all.append(float(row["val_dice"]))

    # 拆成三组
    def split(arr):
        return (
            arr[0:epochs_per_exp],
            arr[epochs_per_exp:epochs_per_exp * 2],
            arr[epochs_per_exp * 2:epochs_per_exp * 3]
        )

    train1, train2, train3 = split(train_loss_all)
    val1, val2, val3 = split(val_loss_all)
    dice1, dice2, dice3 = split(val_dice_all)

    epochs = list(range(1, epochs_per_exp + 1))

    # ===================== Loss 图 =====================
    plt.figure(figsize=(10, 6))

    plt.plot(epochs, val2, linewidth=2, label="U-Net")
    plt.plot(epochs, val3, linewidth=2, label="ResNet34-UNet")

    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.title("Validation Loss Comparison")
    plt.legend()
    plt.grid(True)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, "compare_loss_curve.png"),
                    dpi=300, bbox_inches="tight")

    plt.show()
    plt.close()

    # ===================== Dice 图 =====================
    plt.figure(figsize=(10, 6))

    plt.plot(epochs, dice1, linewidth=2, label="U-Net + BCE")
    plt.plot(epochs, dice2, linewidth=2, label="U-Net + BCE + Dice + Edge")
    plt.plot(epochs, dice3, linewidth=2, label="ResNet34-UNet")

    plt.xlabel("Epoch")
    plt.ylabel("Validation Dice")
    plt.title("Validation Dice Comparison")
    plt.legend()
    plt.grid(True)

    if save_dir:
        plt.savefig(os.path.join(save_dir, "compare_dice_curve.png"),
                    dpi=300, bbox_inches="tight")

    plt.show()
    plt.close()

if __name__ == "__main__":
    csv_file = os.path.join(config.LOG_DIR, "train_history.csv")
    plot_compare_from_one_csv(csv_file, save_dir=config.OUTPUTS_DIR)



