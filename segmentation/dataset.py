import torch
import re
import glob
from torch.utils.data import Dataset
import os
import numpy as np
from PIL import Image
from torchvision import transforms


class PseudoSegDataset(Dataset):
    def __init__(self, img_dir, mask_dir, image_size, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.image_size = image_size
        self.images = sorted(os.listdir(img_dir))
        self.masks = sorted(os.listdir(mask_dir))
        self.transform = transform
        self.img_files = sorted([
            f for f in os.listdir(img_dir)
            if not f.startswith('.') and f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        fname = self.img_files[idx]
        img = Image.open(os.path.join(self.img_dir, fname)).convert("RGB")
        mask = Image.open(os.path.join(self.mask_dir, fname)).convert("L")

        # Resize mask để trùng kích thước model output
        mask = mask.resize((self.image_size, self.image_size), Image.NEAREST)

        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)

        mask = torch.from_numpy(np.array(mask)).long()
        return img, mask

COLOR_TO_LABEL = {
    (255, 0, 0): 0,      # Tumor
    (0, 255, 0): 1,      # Stroma
    (0, 0, 255): 2,      # Lymphocyte
    (153, 0, 255): 3     # Necrosis
}

def rgb_to_label(mask_rgb: np.ndarray, color_map: dict) -> np.ndarray:
    h, w, _ = mask_rgb.shape
    mask_label = np.zeros((h, w), dtype=np.uint8)
    for color, label in color_map.items():
        mask_label[(mask_rgb == color).all(axis=-1)] = label
    return mask_label

class ValidImageMaskDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, mask_dir, image_size, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.image_size = image_size
        self.imgs = sorted(os.listdir(img_dir))
        self.masks = sorted(os.listdir(mask_dir))
        self.transform = transform

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.imgs[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])

        image = Image.open(img_path).convert("RGB")
        mask_rgb = np.array(Image.open(mask_path).convert("RGB"))
        mask_label = rgb_to_label(mask_rgb, COLOR_TO_LABEL)

        # resize cả ảnh và mask
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        mask_label = Image.fromarray(mask_label).resize((self.image_size, self.image_size), Image.NEAREST)

        img = transforms.ToTensor()(image)
        mask = torch.tensor(np.array(mask_label), dtype=torch.long)

        return img, mask, self.imgs[idx]

    def __len__(self):
        return len(self.imgs)