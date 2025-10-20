import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from omegaconf import OmegaConf
from tqdm import tqdm
import numpy as np
from PIL import Image
from scipy import ndimage
import datetime

from dataset import ValidImageMaskDataset
from models.deeplab import DeepLabV3Plus


# ---------------- Color Map & Palette ---------------- #
COLOR_TO_LABEL = {
    (255, 0, 0): 0,      # Tumor (TUM)
    (0, 255, 0): 1,      # Stroma (STR)
    (0, 0, 255): 2,      # Lymphocyte (LYM)
    (153, 0, 255): 3     # Necrosis (NEC)
}

LABEL_TO_COLOR = {
    0: (255, 0, 0),
    1: (0, 255, 0),
    2: (0, 0, 255),
    3: (153, 0, 255)
}


def rgb_to_label(mask_rgb: np.ndarray, color_map: dict) -> np.ndarray:
    """Convert RGB mask to label mask."""
    h, w, _ = mask_rgb.shape
    mask_label = np.zeros((h, w), dtype=np.uint8)
    for color, label in color_map.items():
        mask_label[(mask_rgb == color).all(axis=-1)] = label
    return mask_label


def label_to_rgb(mask_label: np.ndarray, label_map: dict) -> np.ndarray:
    """Convert label mask to RGB mask."""
    h, w = mask_label.shape
    mask_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for label, color in label_map.items():
        mask_rgb[mask_label == label] = color
    return mask_rgb


# ---------------- Metrics ---------------- #
def mean_iou(pred, target, num_classes):
    ious = []
    for c in range(num_classes):
        pred_c = (pred == c)
        target_c = (target == c)
        inter = (pred_c & target_c).sum()
        union = (pred_c | target_c).sum()
        if union > 0:
            ious.append((inter / union).item())
    return np.mean(ious) if ious else 0.0


def freq_weighted_iou(pred, target, num_classes):
    total_pixels = target.size
    fw_iou = 0
    for c in range(num_classes):
        pred_c = (pred == c)
        target_c = (target == c)
        inter = (pred_c & target_c).sum()
        union = (pred_c | target_c).sum()
        fw_iou += (target_c.sum() / total_pixels) * (inter / (union + 1e-8))
    return fw_iou.item()


def dice_coefficient(pred, target, num_classes):
    dice = []
    for c in range(num_classes):
        pred_c = (pred == c)
        target_c = (target == c)
        inter = (pred_c & target_c).sum()
        union = pred_c.sum() + target_c.sum()
        dice.append((2. * inter / (union + 1e-8)).item())
    return np.mean(dice)


def boundary_iou(pred, target, num_classes, dilation_ratio=0.02):
    h, w = pred.shape
    diag_len = np.sqrt(h ** 2 + w ** 2)
    dilation = max(1, int(round(dilation_ratio * diag_len)))
    total_biou, count = 0, 0
    for c in range(num_classes):
        pred_c = (pred == c).astype(np.uint8)
        target_c = (target == c).astype(np.uint8)
        pred_boundary = pred_c - ndimage.binary_erosion(pred_c, iterations=dilation)
        target_boundary = target_c - ndimage.binary_erosion(target_c, iterations=dilation)
        inter = np.logical_and(pred_boundary, target_boundary).sum()
        union = np.logical_or(pred_boundary, target_boundary).sum()
        if union > 0:
            total_biou += inter / union
            count += 1
    return total_biou / (count + 1e-8)


# ---------------- Evaluation ---------------- #
@torch.no_grad()
def evaluate_segmentation(cfg, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Config ---
    paths = cfg.paths
    num_classes = cfg.num_classes
    img_dir, mask_dir = paths.valid, paths.mask_valid
    image_size = cfg.image_size

    result_dir = os.path.join(cfg.paths.results, "pred_masks_stage2")
    os.makedirs(result_dir, exist_ok=True)
    log_path = os.path.join(cfg.paths.results, "segmentation_log_stage2.txt")

    # --- Dataset ---
    val_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(cfg.mean, cfg.std)
    ])
    val_ds = ValidImageMaskDataset(img_dir, mask_dir, image_size=image_size, transform=val_tf)
    val_dl = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    # --- Model ---
    model = DeepLabV3Plus(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()

    # --- Evaluation ---
    mIoU, fwIoU, bIoU, dice = 0, 0, 0, 0

    for imgs, masks, fnames in tqdm(val_dl, desc="Evaluating"):
        imgs = imgs.to(device)
        outputs = model(imgs)
        preds = torch.argmax(F.softmax(outputs, dim=1), dim=1)

        preds_np = preds.squeeze().cpu().numpy()

        # --- Convert prediction to RGB color ---
        pred_color = label_to_rgb(preds_np, LABEL_TO_COLOR)

        # --- Save color image ---
        out_path = os.path.join(result_dir, fnames[0])
        Image.fromarray(pred_color).save(out_path)

        # --- Load GT mask & convert RGB -> label if needed ---
        # mask từ dataset đã được resize đúng kích thước image_size
        mask_np = masks.squeeze().cpu().numpy().astype(np.uint8)
        mask_np = masks.squeeze().cpu().numpy()
        print(f"{fnames[0]}  unique values: {np.unique(mask_np)}  dtype={mask_np.dtype}")

        # --- Compute metrics ---
        mIoU += mean_iou(preds_np, mask_np, num_classes)
        fwIoU += freq_weighted_iou(preds_np, mask_np, num_classes)
        dice += dice_coefficient(preds_np, mask_np, num_classes)
        bIoU += boundary_iou(preds_np, mask_np, num_classes)

    n = len(val_dl)
    result = {
        "mIoU": 100 * mIoU / n,
        "FwIoU": 100 * fwIoU / n,
        "bIoU": 100 * bIoU / n,
        "Dice": 100 * dice / n
    }

    # --- Log results ---
    with open(log_path, "a") as f:
        f.write(f"\n[{datetime.datetime.now()}]\n")
        for k, v in result.items():
            f.write(f"{k}: {v:.2f}\n")
        f.write("-" * 30 + "\n")

    print("\n✅ Evaluation Results:")
    for k, v in result.items():
        print(f"  {k}(%): {v:.2f}")
    print(f"\n🎨 Saved color predictions to: {result_dir}")
    print(f"🧾 Logged results to: {log_path}")
def test_mask_vs_mask(cfg):
    num_classes = cfg.num_classes
    img_dir, mask_dir = cfg.paths.valid, cfg.paths.mask_valid
    image_size = cfg.image_size

    val_ds = ValidImageMaskDataset(img_dir, mask_dir, image_size=image_size)
    val_dl = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    mIoU, fwIoU, bIoU, dice = 0, 0, 0, 0

    for _, masks, fnames in tqdm(val_dl, desc="Testing Mask vs Mask"):
        mask_np = masks.squeeze().cpu().numpy().astype(np.uint8)
        mIoU += mean_iou(mask_np, mask_np, num_classes)
        fwIoU += freq_weighted_iou(mask_np, mask_np, num_classes)
        dice += dice_coefficient(mask_np, mask_np, num_classes)
        bIoU += boundary_iou(mask_np, mask_np, num_classes)

    n = len(val_dl)
    print("\n✅ Mask vs Mask Results:")
    print(f"  mIoU: {100*mIoU/n:.2f}%")
    print(f"  FwIoU: {100*fwIoU/n:.2f}%")
    print(f"  Dice: {100*dice/n:.2f}%")
    print(f"  Boundary IoU: {100*bIoU/n:.2f}%")


if __name__ == "__main__":
    cfg = OmegaConf.load("segmentation/configuration_seg.yml")
    model_name = cfg.model_name
    model_path = os.path.join(cfg.paths.results, f"seg_model_{model_name}.pth")
    evaluate_segmentation(cfg, model_path)

