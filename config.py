# 机构：人工智能研究所
# 人员：东
# 时间：2026/6/6 9:02

# 配置文件

import os
import torch

# ===========================文件路径设置=============================
# 获取项目所在基础路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
LOG_DIR = os.path.join(BASE_DIR, "logs")

# ============================设备================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================数据集==================================
DATASET_NAME = "OxfordPet"
IMG_SIZE = 256
IN_CHANNELS = 3
NUM_CLASSES = 1
TRAINSET_RATIO = 0.7
VALSET_RATIO = 0.2
TESTSET_RATIO = 0.1

# ===========================训练参数==================================
EPOCHS = 50
BATCH_SIZE = 4
LR = 1e-4
NUM_WORKERS = 0

# ============================路径==================================
TRAIN_IMAGE_DIR = os.path.join(DATA_DIR, "train", "images")
TRAIN_MASK_DIR = os.path.join(DATA_DIR, "train", "masks")
VAL_IMAGE_DIR = os.path.join(DATA_DIR, "val", "images")
VAL_MASK_DIR = os.path.join(DATA_DIR, "val", "masks")
TEST_IMAGE_DIR = os.path.join(DATA_DIR, "test", "images")
TEST_MASK_DIR = os.path.join(DATA_DIR, "test", "masks")

# ==========================模型保存================================
BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_unet.pth")
LAST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "last_unet.pth")




