import os
from PIL import Image
import numpy as np
from multiprocessing import Array, Process, cpu_count
from utils.pyutils import chunks
import random
import torch
import torch.nn.functional as F
from pathlib import Path


# ============================================================
# 🧮 F1-score utility (nếu cần)
# ============================================================
def calculate_F1(pred_path, gt_path, numofclass):
    TPs = [0] * numofclass
    FPs = [0] * numofclass
    FNs = [0] * numofclass
    ims = os.listdir(pred_path)
    for im in ims:
        pred = np.asarray(Image.open(os.path.join(pred_path, im)))
        gt = np.asarray(Image.open(os.path.join(gt_path, im)))
        for k in range(numofclass):
            TPs[k] += np.sum(np.logical_and(pred == k, gt == k))
            FPs[k] += np.sum(np.logical_and(pred == k, gt != k))
            FNs[k] += np.sum(np.logical_and(pred != k, gt == k))

    f1_score = np.array(TPs) / (np.array(TPs) + (np.array(FPs) + np.array(FNs)) / 2 + 1e-7)
    return np.mean(f1_score)


# ============================================================
# 📈 mIoU evaluation with CPU/GPU handling
# ============================================================
def get_overall_valid_score(pred_image_path, groundtruth_path, num_workers=5, num_class=4):
    """
    Compute mean IoU score between predicted CAM maps (.npy) and ground truth masks (.png).

    Args:
        pred_image_path (str): directory of predicted CAM .npy files
        groundtruth_path (str): directory of ground truth .png masks
        num_workers (int): number of parallel processes (only used if GPU available)
        num_class (int): number of segmentation classes

    Returns:
        float: mean IoU across all classes
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_names = [Path(f).stem for f in os.listdir(pred_image_path) if f.endswith('.npy')]
    random.shuffle(image_names)

    # ========================================================
    # 🧠 CPU mode — sequential evaluation
    # ========================================================
    if device == "cpu":
        print("🧠 CPU mode detected → running validation sequentially (no multiprocessing).")
        intersection = np.zeros(num_class, dtype=np.float64)
        union = np.zeros(num_class, dtype=np.float64)

        for im_name in image_names:
            cam_path = os.path.join(pred_image_path, f"{im_name}.npy")
            mask_file = im_name if im_name.lower().endswith(".png") else f"{im_name}.png"
            mask_path = os.path.join(groundtruth_path, mask_file)

            if not os.path.exists(cam_path) or not os.path.exists(mask_path):
                print(f"⚠️ Skipping {im_name} (missing file).")
                continue

            cam = np.load(cam_path, allow_pickle=True)
            gt = np.asarray(Image.open(mask_path))

            # 🔧 Convert multi-channel CAM → single channel (argmax)
            if cam.ndim == 3:
                cam = np.argmax(cam, axis=0)

            # 🔧 Resize CAM to match ground truth if needed
            if cam.shape != gt.shape:
                cam_t = torch.tensor(cam, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                h, w = gt.shape[:2]
                cam = F.interpolate(cam_t, size=(h, w), mode="nearest").squeeze().numpy().astype(np.uint8)

            cam = cam.reshape(-1)
            gt = gt.reshape(-1)

            for i in range(num_class):
                inter = np.sum(np.logical_and(cam == i, gt == i))
                u = np.sum(np.logical_or(cam == i, gt == i))
                intersection[i] += inter
                union[i] += u

        eps = 1e-7
        ious = intersection / (union + eps)
        miou = np.mean(ious)
        print(f"✅ Validation mIoU (CPU): {miou:.4f}")
        return float(miou)

    # ========================================================
    # 🚀 GPU mode — multiprocessing evaluation
    # ========================================================
    else:
        print("🚀 GPU mode detected → running validation with multiprocessing.")
        num_workers = min(num_workers, cpu_count())
        image_list = chunks(image_names, num_workers)

        def f(intersection, union, image_list):
            for im_name in image_list:
                cam_path = os.path.join(pred_image_path, f"{im_name}.npy")
                mask_path = os.path.join(groundtruth_path, f"{im_name}.png")
                if not os.path.exists(cam_path) or not os.path.exists(mask_path):
                    continue

                cam = np.load(cam_path, allow_pickle=True)
                gt = np.asarray(Image.open(mask_path))

                if cam.ndim == 3:
                    cam = np.argmax(cam, axis=0)

                if cam.shape != gt.shape:
                    cam_t = torch.tensor(cam, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                    h, w = gt.shape[:2]
                    cam = F.interpolate(cam_t, size=(h, w), mode="nearest").squeeze().numpy().astype(np.uint8)

                cam = cam.reshape(-1)
                gt = gt.reshape(-1)

                for i in range(num_class):
                    inter = np.sum(np.logical_and(cam == i, gt == i))
                    u = np.sum(np.logical_or(cam == i, gt == i))
                    intersection[i] += inter
                    union[i] += u

        intersection = Array("d", [0] * num_class)
        union = Array("d", [0] * num_class)
        p_list = []

        for subset in image_list:
            p = Process(target=f, args=(intersection, union, subset))
            p.start()
            p_list.append(p)
        for p in p_list:
            p.join()

        eps = 1e-7
        ious = np.array([intersection[i] / (union[i] + eps) for i in range(num_class)])
        miou = np.mean(ious)
        print(f"✅ Validation mIoU (GPU): {miou:.4f}")
        return float(miou)
