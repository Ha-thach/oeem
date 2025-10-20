import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from omegaconf import OmegaConf


def generate_pseudo_masks_from_cam(cfg, model_name, threshold=0.0):
    """
    Generate pseudo segmentation masks from CAM .npy files (Stage 1 → Stage 2)
    """
    # --- Paths from config ---
    paths = cfg.paths
    cam_dir = os.path.join(paths.valid_result, model_name, "cam")
    save_dir = os.path.join(paths.valid_result, model_name, "pseudo_mask")
    os.makedirs(save_dir, exist_ok=True)

    num_classes = cfg.num_classes

    print(f"📂 Loading CAMs from: {cam_dir}")
    print(f"🖼️ Saving pseudo masks to: {save_dir}")

    cam_files = [f for f in os.listdir(cam_dir) if f.endswith(".npy")]
    print(f"Found {len(cam_files)} CAM files.")

    for cam_file in tqdm(cam_files, desc="Generating pseudo masks"):
        name = os.path.splitext(cam_file)[0]
        if name.endswith(".png"): 
            name = os.path.splitext(name)[0]
        cam_path = os.path.join(cam_dir, cam_file)

        # Load CAM (C,H,W)
        cam = np.load(cam_path, allow_pickle=True)
        if cam.ndim == 2:
            cam = np.expand_dims(cam, axis=0)

        # Normalize each channel to [0,1]
        cam_min = cam.min(axis=(1, 2), keepdims=True)
        cam_max = cam.max(axis=(1, 2), keepdims=True)
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

        # Optional threshold: remove weak activation
        cam[cam < threshold] = 0

        # Choose argmax label per pixel
        pred_label = np.argmax(cam, axis=0).astype(np.uint8)

        # Convert to PIL grayscale image
        mask_img = Image.fromarray(pred_label)

        # Save to pseudo_mask folder
        save_path = os.path.join(save_dir, f"{name}.png")
        mask_img.save(save_path)

    print(f"✅ Done. Pseudo masks saved at: {save_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate pseudo masks from CAMs")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--model_name", type=str, required=True, help="Model name (folder under valid_result/)")
    parser.add_argument("--threshold", type=float, default=0.0, help="Activation threshold (default=0)")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    generate_pseudo_masks_from_cam(cfg, args.model_name, args.threshold)


if __name__ == "__main__":
    main()
