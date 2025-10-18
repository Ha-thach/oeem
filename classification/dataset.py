import torch
import re
import glob
from torch.utils.data import Dataset
import os
import numpy as np
from PIL import Image
from utils.pyutils import multiscale_online_crop
from torchvision import transforms

# ============================================================
# Hàm đọc label từ tên file: [01010]
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
# 2️⃣ Dataset gốc cho phân loại patch-level (Stage 1)
# ============================================================
class OriginPatchesDataset(Dataset):
    """
    Dataset đọc patch ảnh BCSS-WSSS có label nằm trong tên file dạng [xxxxx].
    """
    def __init__(self, data_path_name=None, transform=None, num_class=5):
        self.path = data_path_name
        # ⚙️ Lọc file hợp lệ, tránh .DS_Store hoặc file ẩn
        self.files = sorted(
            [f for f in os.listdir(data_path_name)
             if not f.startswith('.') and f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        )
        self.transform = transform
        self.num_class = num_class

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.path, self.files[idx])
        im = Image.open(image_path).convert("RGB")
        label = get_file_label(self.files[idx], num_class=self.num_class)

        # ⚠️ Không cần ToTensor() ở đây nếu transform đã có ToTensor()
        if self.transform:
            im = self.transform(im)
        else:
            im = transforms.ToTensor()(im)

        return im, label


# ============================================================
# 3️⃣ Dataset cho Validation hoặc Offline patch extraction
# ============================================================
class OfflineDataset(Dataset):
    """
    Dataset dùng cho validation / CAM generation.
    """
    def __init__(self, dataset_path, transform=None):
        self.path = dataset_path
        self.files = sorted(
            [f for f in os.listdir(dataset_path)
             if not f.startswith('.') and f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        )
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.path, self.files[idx])
        im = Image.open(image_path).convert("RGB")
        positions = list(map(int, re.findall(r'\d+', self.files[idx])))
        if self.transform:
            im = self.transform(im)
        else:
            im = transforms.ToTensor()(im)
        return im, np.array(positions)


# ============================================================
# 4️⃣ Dataset cho sinh CAM (Stage 2)
# ============================================================
class TrainingSetCAM(Dataset):
    """
    Dataset sinh patch đa tỉ lệ để train segmentation với CAM.
    """
    def __init__(self, data_path_name, transform, patch_size, stride, scales, num_class=5):
        self.path = data_path_name
        self.files = sorted(
            [f for f in os.listdir(data_path_name)
             if not f.startswith('.') and f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        )
        self.transform = transform
        self.patch_size = patch_size
        self.stride = stride
        self.scales = scales
        self.num_class = num_class

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.path, self.files[idx])
        im = np.asarray(Image.open(image_path).convert("RGB"))
        scaled_im_list, scaled_position_list = multiscale_online_crop(im, self.patch_size, self.stride, self.scales)

        # Áp transform cho từng patch
        if self.transform:
            for im_list in scaled_im_list:
                for patch_id in range(len(im_list)):
                    im_list[patch_id] = self.transform(im_list[patch_id])

        label = get_file_label(self.files[idx], num_class=self.num_class)
        return self.files[idx], scaled_im_list, scaled_position_list, self.scales, label


# ============================================================
# 5️⃣ Dataset chính cho BCSS-WSSS segmentation
# ============================================================
class BCSS_WSSS_Dataset(Dataset):
    """
    Dataset chính cho BCSS-WSSS segmentation (weakly-supervised).
    Mỗi file ảnh có label trong tên, ví dụ: patch_[01010].png.
    """
    CLASSES = ["TUM", "STR", "LYM", "NEC"]

    def __init__(self, data_path_name=None, transform=None, num_class=4):
        self.path_name = data_path_name
        # ⚙️ Lọc file hợp lệ, bỏ .DS_Store
        self.files = sorted(
            [f for f in os.listdir(data_path_name)
             if not f.startswith('.') and f.lower().endswith('.png')]
        )
        self.transform = transform
        self.num_class = num_class

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.path_name, self.files[idx])
        im = Image.open(image_path).convert("RGB")
        label = get_file_label(self.files[idx], num_class=self.num_class)

        if self.transform:
            im = self.transform(im)
        else:
            im = transforms.ToTensor()(im)

        return im, label
