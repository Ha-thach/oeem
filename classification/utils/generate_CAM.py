import os
import torch
import numpy as np
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
import dataset
import torch.nn.functional as F


def generate_validation_cam(net, config, batch_size, model_name, elimate_noise=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🧠 Using device for CAM generation: {device}")

    paths = config.paths
    num_classes = config.num_classes
    mean = config.mean
    std = config.std
    image_size = config.network_image_size

    save_dir = os.path.join(paths.valid_result, model_name, "cam")
    os.makedirs(save_dir, exist_ok=True)

    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    valid_dataset = dataset.ValidImageMaskDataset(
        img_dir=paths.valid,
        mask_dir=paths.mask_valid,
        transform=val_transform
    )
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    net.eval()
    net = net.to(device)

    print(f"📸 Generating CAMs for {len(valid_dataset)} validation samples...")

    for imgs, masks, img_names in tqdm(valid_loader, desc="Generating CAMs"):
        imgs = imgs.to(device)

        with torch.no_grad():
            # Forward pass for CAMs
            if hasattr(net, "module"):
                cams = net.module.forward_cam(imgs)
            else:
                cams = net.forward_cam(imgs)

            # 🔧 Resize CAMs về kích thước ảnh gốc (ví dụ mask 224×224)
            cams_resized = F.interpolate(
                cams, size=(image_size, image_size),
                mode="bilinear", align_corners=False
            )
            cams_resized = cams_resized.cpu().numpy()

            # Save mỗi CAM ra .npy
            for b_idx, name in enumerate(img_names):
                cam_map = np.maximum(cams_resized[b_idx], 0)
                np.save(os.path.join(save_dir, f"{name}.npy"), cam_map)

    print(f"✅ CAM generation completed. Results saved to: {save_dir}")