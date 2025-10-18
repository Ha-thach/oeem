import argparse
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from torchvision import transforms
import dataset
from torch.utils.data import DataLoader
from utils.metric import get_overall_valid_score
from utils.generate_CAM import generate_validation_cam
from utils.pyutils import crop_validation_images
from utils.torchutils import PolyOptimizer
import yaml
import importlib


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-batch', default=20, type=int)
    parser.add_argument('-epoch', default=2, type=int)
    parser.add_argument('-lr', default=0.01, type=float)
    parser.add_argument('-test_every', default=5, type=int, help="how often to test a model while training")
    parser.add_argument('-d', '--device', nargs='+', help='GPU id to use parallel (e.g. -d 0 1)', type=int, default=None)
    parser.add_argument('-m', type=str, required=True, help='the save model name')
    args = parser.parse_args()

    # ====================================================
    # 1️⃣ Thiết lập thiết bị
    # ====================================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🧠 Using device: {device}")

    batch_size = args.batch
    epochs = args.epoch
    base_lr = args.lr
    test_every = args.test_every
    devices = args.device
    model_name = args.m

    # ====================================================
    # 2️⃣ Đọc file cấu hình
    # ====================================================
    with open('classification/configuration.yml') as f:
        config = yaml.safe_load(f)

    mean = config['mean']
    std = config['std']
    network_image_size = config['network_image_size']
    scales = config['scales']

    os.makedirs('./classification/weights', exist_ok=True)
    os.makedirs('./classification/result', exist_ok=True)

    # ====================================================
    # 3️⃣ Chuẩn bị validation folder
    # ====================================================
    validation_folder_name = 'classification/valid_result'
    validation_dataset_path = '/Users/thachha/Desktop/AIO2025-official/AIMA/CP- WSSS/PBIP/data/BCSS-WSSS/sub/valid/img'
    validation_mask_path = '/Users/thachha/Desktop/AIO2025-official/AIMA/CP- WSSS/PBIP/data/BCSS-WSSS/sub/valid/mask'

    if not os.path.exists(validation_folder_name):
        os.mkdir(validation_folder_name)
        print('🪄 Cropping validation set images ...')
        crop_validation_images(validation_dataset_path, network_image_size, network_image_size, scales, validation_folder_name)
        print('✅ Cropping finishes!')

    # ====================================================
    # 4️⃣ Load model
    # ====================================================
    resnet38_path = "classification/weights/res38d.pth.tar"

    net = getattr(importlib.import_module("network.wide_resnet"), 'wideResNet')()
    checkpoint = torch.load(resnet38_path, map_location=device)

    # Nếu checkpoint có chứa key 'state_dict', load chuẩn
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        net.load_state_dict(checkpoint["state_dict"], strict=False)
    else:
        net.load_state_dict(checkpoint, strict=False)

    if torch.cuda.is_available() and devices:
        net = torch.nn.DataParallel(net, device_ids=devices)
    net = net.to(device)

    # ====================================================
    # 5️⃣ Data augmentation
    # ====================================================
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=network_image_size, scale=(0.7, 1)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),  # ⚠️ cần thiết cho Normalize
        transforms.Normalize(mean=mean, std=std)
    ])

    # ====================================================
    # 6️⃣ Dataset và DataLoader
    # ====================================================
    data_path_name = '/Users/thachha/Desktop/AIO2025-official/AIMA/CP- WSSS/PBIP/data/BCSS-WSSS/sub/train'
    TrainDataset = dataset.OriginPatchesDataset(data_path_name=data_path_name, transform=train_transform)
    print(f"📂 Training dataset loaded: {len(TrainDataset)} samples")

    TrainDataLoader = DataLoader(
        TrainDataset,
        batch_size=batch_size,
        num_workers=0 if device == "cpu" else 4,
        shuffle=True,
        drop_last=True
    )

    # ====================================================
    # 7️⃣ Optimizer và Loss
    # ====================================================
    optimizer = PolyOptimizer(net.parameters(), base_lr, weight_decay=1e-4, max_step=epochs, momentum=0.9)
    criteria = torch.nn.BCEWithLogitsLoss(reduction='mean').to(device)

    # ====================================================
    # 8️⃣ Training loop
    # ====================================================
    loss_t, iou_v = [], []
    best_val = 0.0

    for epoch in range(epochs):
        net.train()
        running_loss = 0.0

        for img, label in tqdm(TrainDataLoader, desc=f"Epoch {epoch+1}/{epochs}"):
            img, label = img.to(device), label.to(device).float()

            scores = net(img)
            loss = criteria(scores, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / len(TrainDataLoader)
        loss_t.append(train_loss)

        print(f"📘 Epoch {epoch+1}: Train Loss = {train_loss:.4f}")

        # ====================================================
        # Validation mỗi test_every epoch
        # ====================================================
        if test_every != 0 and ((epoch + 1) % test_every == 0 or (epoch + 1) == epochs):
            net_cam = getattr(importlib.import_module("network.wide_resnet"), 'wideResNet')()
            pretrained = net.state_dict()
            pretrained = {k.replace("module.", ""): v for k, v in pretrained.items()}
            pretrained['fc_cam.weight'] = pretrained['fc_cls.weight'].unsqueeze(-1).unsqueeze(-1).to(torch.float64)
            pretrained['fc_cam.bias'] = pretrained['fc_cls.bias']

            net_cam.load_state_dict(pretrained, strict=False)
            net_cam = net_cam.to(device)

            valid_image_path = os.path.join(validation_folder_name, model_name)
            generate_validation_cam(net_cam, config, batch_size, validation_dataset_path, validation_folder_name, model_name)
            valid_iou = get_overall_valid_score(valid_image_path, validation_mask_path, num_workers=4)
            iou_v.append(valid_iou)

            if valid_iou > best_val:
                best_val = valid_iou
                print("🔥 New best model found — saving checkpoint...")
                torch.save({
                    "model": net.state_dict(),
                    "optimizer": optimizer.state_dict()
                }, f"./classification/weights/{model_name}_best.pth")

            print(f"📈 Valid mIOU: {valid_iou:.4f}, Dice: {2 * valid_iou / (1 + valid_iou):.4f}")

    # ====================================================
    # 9️⃣ Save last model + plots
    # ====================================================
    torch.save({"model": net.state_dict(), "optimizer": optimizer.state_dict()},
               f"./classification/weights/{model_name}_last.pth")

    plt.figure()
    plt.plot(loss_t)
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.title('Training Loss')
    plt.savefig('./classification/result/train_loss.png')

    plt.figure()
    plt.plot(list(range(test_every, epochs + 1, test_every)), iou_v)
    plt.ylabel('mIoU')
    plt.xlabel('Epochs')
    plt.title('Validation mIoU')
    plt.savefig('./classification/result/valid_iou.png')

    print("✅ Training finished successfully!")
