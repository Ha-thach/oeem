import torch
import re
import glob
from torch.utils.data import Dataset
import os
import numpy as np
from PIL import Image
from torchvision import transforms


# ============================================================
# 🧩 1️⃣ Hàm đọc label từ tên file [xxxxx]
# ============================================================
def get_file_label(filename: str, num_class: int = 5) -> np.ndarray:
    """
    Trích one-hot label từ tên file BCSS-WSSS, ví dụ:
    - 'patch_001_[10000].png'  → [1,0,0,0,0]
    - 'patch_057_[00101].png'  → [0,0,1,0,1]
    """
    term_split = re.split(r"\[|\]", filename)
    if len(term_split) < 2:
        raise ValueError(f"Tên file không chứa nhãn trong dấu []: {filename}")
    label_str = term_split[1]
    cls_label = np.array([int(x) for x in label_str[:num_class]], dtype=np.int64)
    return cls_label


# ============================================================
# 🧠 2️⃣ Dataset cho huấn luyện (ảnh + label từ filename)
# ============================================================
class TrainPatchesDataset(Dataset):
    """
    Dataset dùng cho giai đoạn classification training (Stage 1).
    - Ảnh lấy từ folder `train`
    - Label lấy từ tên file (one-hot [xxxxx])
    """
    def __init__(self, data_path, transform=None, num_class=5):
        self.path = data_path
        self.files = sorted([
            f for f in os.listdir(data_path)
            if not f.startswith('.') and f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        self.transform = transform
        self.num_class = num_class

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        img_path = os.path.join(self.path, fname)
        img = Image.open(img_path).convert("RGB")
        label = get_file_label(fname, num_class=self.num_class)

        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)

        return img, torch.tensor(label, dtype=torch.float32)


# ============================================================
# 🧩 3️⃣ Dataset cho validation (ảnh + mask thực tế)
# ============================================================
class ValidImageMaskDataset(Dataset):
    """
    Dataset dùng cho validation segmentation.
    - Ảnh đọc từ `valid/img`
    - Mask đọc từ `valid/mask`
    - Validation dataset: có image + mask.Dùng để sinh CAM và tính IoU.
    """
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_files = sorted([
            f for f in os.listdir(img_dir)
            if not f.startswith('.') and f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        self.transform = transform

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        fname = self.img_files[idx]
        img_path = os.path.join(self.img_dir, fname)
        mask_path = os.path.join(self.mask_dir, fname)

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # mask grayscale (0–C−1)
        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)

        mask = torch.tensor(np.array(mask), dtype=torch.long)

        return img, mask, fname

# ============================================================
# 🔬 4️⃣ Dataset dùng cho multi-scale CAM (Stage 2)
# ============================================================
class TrainingSetCAM(Dataset):
    """
    Dataset sinh patch đa tỉ lệ để train segmentation với CAM.
    - Sử dụng multiscale_online_crop() để tạo patch tại nhiều scale
    """
    def __init__(self, data_path, transform, patch_size, stride, scales, num_class=5):
        self.path = data_path
        self.files = sorted([
            f for f in os.listdir(data_path)
            if not f.startswith('.') and f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        self.transform = transform
        self.patch_size = patch_size
        self.stride = stride
        self.scales = scales
        self.num_class = num_class

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        img_path = os.path.join(self.path, fname)
        img = np.asarray(Image.open(img_path).convert("RGB"))
        scaled_im_list, scaled_pos_list = multiscale_online_crop(img, self.patch_size, self.stride, self.scales)

        if self.transform:
            for im_list in scaled_im_list:
                for i in range(len(im_list)):
                    im_list[i] = self.transform(im_list[i])

        label = get_file_label(fname, num_class=self.num_class)
        return fname, scaled_im_list, scaled_pos_list, self.scales, label

class PseudoSegDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.images = sorted(os.listdir(img_dir))
        self.masks = sorted(os.listdir(mask_dir))
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.img_dir, self.images[idx])).convert('RGB')
        mask = Image.open(os.path.join(self.mask_dir, self.masks[idx]))
        mask = np.array(mask, dtype=np.int64)

        if self.transform:
            img = self.transform(img)
        mask = torch.tensor(mask, dtype=torch.long)
        return img, mask
