import json
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
import dataset
import torch
import os


def generate_validation_cam(net, config, batch_size, dataset_path, validation_folder_name, model_name,
                            elimate_noise=False, label_path=None):
    """
    Generate class activation maps for validation set.

    Args:
        net (torch.nn.Module): trained classification model
        config (dict): model config (mean, std, image size, scales)
        batch_size (int): batch size for inference
        dataset_path (str): validation image directory
        validation_folder_name (str): output folder for CAM results
        model_name (str): name for saving results
        elimate_noise (bool): optionally filter out impossible classes
        label_path (str): path to label json (only if elimate_noise=True)
    """
    # --- device auto detection ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🧠 Using device for CAM generation: {device}")

    side_length = config['network_image_size']
    mean = config['mean']
    std = config['std']
    network_image_size = config['network_image_size']
    scales = config['scales']

    # --- move model to device ---
    net = net.to(device)
    net.eval()

    crop_image_path = f'{validation_folder_name}/crop_images/'
    image_name_list = os.listdir(crop_image_path)
    extension_name = os.listdir(dataset_path)[0].split('.')[-1]
    
    for image_name in tqdm(image_name_list, desc="Generating CAMs"):
        orig_img = np.asarray(Image.open(f'{dataset_path}/{image_name}.{extension_name}'))
        w, h, _ = orig_img.shape
        num_classes = getattr(net.module.fc_cls, "out_features", net.fc_cls.out_features)
        ensemble_cam = np.zeros((num_classes, w, h))
        
        for scale in scales:
            image_per_scale_path = os.path.join(crop_image_path, image_name, str(scale))
            scale = float(scale)
            offlineDataset = dataset.OfflineDataset(
                image_per_scale_path,
                transform=transforms.Compose([
                    transforms.Resize((network_image_size, network_image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std)
                ])
            )
            offlineDataloader = DataLoader(offlineDataset, batch_size=batch_size, drop_last=False)

            w_ = int(w * scale)
            h_ = int(h * scale)
            interpolatex = min(w_, side_length)
            interpolatey = min(h_, side_length)

            with torch.no_grad():
                cam_list, position_list = [], []
                for ims, positions in offlineDataloader:
                    ims = ims.to(device)
                    cam_scores = net.module.forward_cam(ims) if hasattr(net, "module") else net.forward_cam(ims)
                    cam_scores = F.interpolate(cam_scores, (interpolatex, interpolatey), mode='bilinear', align_corners=False)
                    cam_list.append(cam_scores.detach().cpu().numpy())
                    position_list.append(positions.numpy())

                cam_list = np.concatenate(cam_list)
                position_list = np.concatenate(position_list)

                sum_cam = np.zeros((num_classes, w_, h_))
                sum_counter = np.zeros_like(sum_cam)
                
                for k in range(cam_list.shape[0]):
                    y, x = position_list[k][0], position_list[k][1]
                    crop = cam_list[k]
                    sum_cam[:, y:y+side_length, x:x+side_length] += crop
                    sum_counter[:, y:y+side_length, x:x+side_length] += 1

                sum_counter[sum_counter < 1] = 1
                norm_cam = sum_cam / sum_counter
                norm_cam = F.interpolate(
                    torch.unsqueeze(torch.tensor(norm_cam), 0),
                    (w, h),
                    mode='bilinear',
                    align_corners=False
                ).detach().cpu().numpy()[0]

                ensemble_cam += norm_cam                

        # --- optional noise elimination ---
        if elimate_noise and label_path is not None:
            with open(os.path.join(validation_folder_name, label_path)) as f:
                big_labels = json.load(f)
            big_label = big_labels.get(f'{image_name}.png', None)
            if big_label is not None:
                for k in range(num_classes):
                    if big_label[k] == 0:
                        ensemble_cam[k, :, :] = -np.inf
                    
        # --- save results ---
        result_label = ensemble_cam.argmax(axis=0)
        save_dir = os.path.join(validation_folder_name, model_name)
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, f'{image_name}.npy'), result_label)

    print("✅ Validation CAM generation completed.")
