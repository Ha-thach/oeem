import argparse
import os
import sys
import importlib
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from omegaconf import OmegaConf

import dataset
from utils.metric import get_overall_valid_score
from utils.generate_CAM import generate_validation_cam
from utils.torchutils import PolyOptimizer


def safe_load_checkpoint(path, device):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    try:
        ckpt = torch.load(path, map_location=device)
    except Exception as e:
        print(f"Default torch.load failed: {e}")
        ckpt = torch.load(path, map_location=device, weights_only=False)

    if isinstance(ckpt, dict):
        for key in ["state_dict", "model", "module", "weights", "net"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt
        if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            return {"model": ckpt}
    return {"model": ckpt}


def strip_module_prefix(state_dict):
    return {k[len("module."):] if k.startswith("module.") else k: v for k, v in state_dict.items()}


def main():
    parser = argparse.ArgumentParser(description="Train or validate classifier using config YAML")
    parser.add_argument("--config", type=str, required=True, help="path to YAML config")
    parser.add_argument("-m", "--model_name", type=str, required=True, help="model name to save")
    parser.add_argument("--batch", type=int, default=None, help="override batch size")
    parser.add_argument("--epoch", type=int, default=None, help="override epochs")
    parser.add_argument("--lr", type=float, default=None, help="override learning rate")
    parser.add_argument("--resume", type=str, default=None, help="path to checkpoint to resume training")
    parser.add_argument("--eval_only", action="store_true", help="run validation only without training")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    print("Loaded config from:", args.config)
    print(OmegaConf.to_yaml(cfg))

    if args.batch:
        cfg.batch_size = args.batch
    if args.epoch:
        cfg.epochs = args.epoch
    if args.lr:
        cfg.learning_rate = args.lr

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    paths = cfg.paths
    os.makedirs(paths.weights, exist_ok=True)
    os.makedirs(paths.results, exist_ok=True)
    os.makedirs(paths.valid_result, exist_ok=True)

    # Logging setup
    log_dir = os.path.join(paths.results, args.model_name)
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "logging.txt")

    sys.stdout = open(log_path, "a", buffering=1)
    sys.stderr = sys.stdout

    print("=" * 80)
    print(f"Logging started for model: {args.model_name}")
    print(f"Config file: {args.config}")
    print(f"Logs will be saved to: {log_path}")
    print("=" * 80)

    # Build model
    NetClass = getattr(importlib.import_module("network.wide_resnet"), "wideResNet")
    net = NetClass(num_class=cfg.num_classes)
    optimizer = PolyOptimizer(
        net.parameters(),
        lr=cfg.learning_rate,
        weight_decay=1e-4,
        max_step=cfg.epochs,
        momentum=0.9
    )
    start_epoch = 0
    best_val = 0.0

    # Load pretrained or resume
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from checkpoint: {args.resume}")
        ckpt = safe_load_checkpoint(args.resume, device)
        if "model" in ckpt:
            state_dict = strip_module_prefix(ckpt["model"])
            net.load_state_dict(state_dict, strict=False)
            print("Model weights restored.")
        if "optimizer" in ckpt and hasattr(optimizer, "load_state_dict"):
            optimizer.load_state_dict(ckpt["optimizer"])
            print("Optimizer state restored.")
        start_epoch = ckpt.get("epoch", 0)
        print(f"Resumed from epoch {start_epoch}")
    elif paths.resnet38 and os.path.exists(paths.resnet38):
        print(f"Loading pretrained backbone: {paths.resnet38}")
        state_dict = safe_load_checkpoint(paths.resnet38, device)
        state_dict = strip_module_prefix(state_dict.get("model", state_dict))
        net.load_state_dict(state_dict, strict=False)
        print("Pretrained weights loaded.")
    else:
        print("No pretrained or resume checkpoint found. Training from scratch.")

    if torch.cuda.is_available() and getattr(cfg, "devices", None):
        net = torch.nn.DataParallel(net, device_ids=cfg.devices)
    net = net.to(device)

    # Dataset setup
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=cfg.network_image_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.mean, std=cfg.std)
    ])
    val_transform = transforms.Compose([
        transforms.Resize((cfg.network_image_size, cfg.network_image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.mean, std=cfg.std)
    ])

    TrainDataset = dataset.TrainPatchesDataset(paths.train, transform=train_transform, num_class=cfg.num_classes)
    ValidDataset = dataset.ValidImageMaskDataset(paths.valid, paths.mask_valid, transform=val_transform)
    num_workers = 0 if device.type == "cpu" else 4
    TrainLoader = DataLoader(TrainDataset, batch_size=cfg.batch_size, num_workers=num_workers,
                             shuffle=True, drop_last=True)
    ValidLoader = DataLoader(ValidDataset, batch_size=1, num_workers=num_workers, shuffle=False)
    print(f"Train samples: {len(TrainDataset)} | Valid samples: {len(ValidDataset)}")

    criterion = torch.nn.BCEWithLogitsLoss().to(device)
    loss_t, iou_v = [], []

    # Evaluation only
    if args.eval_only:
        print("Running evaluation only...")
        cam_dir = os.path.join(paths.valid_result, args.model_name, "cam")
        os.makedirs(cam_dir, exist_ok=True)
        generate_validation_cam(
            net,
            cfg,
            batch_size=cfg.batch_size,
            model_name=os.path.join(args.model_name, "cam")
        )
        valid_iou = get_overall_valid_score(
            os.path.join(paths.valid_result, args.model_name, "cam"),
            paths.mask_valid,
            num_workers=num_workers
        )
        print(f"Validation mIoU: {valid_iou:.4f}")
        return

    # Training loop
    for epoch in range(start_epoch, cfg.epochs):
        net.train()
        running_loss = 0.0

        for imgs, labels in tqdm(TrainLoader, desc=f"Epoch {epoch+1}/{cfg.epochs}"):
            imgs, labels = imgs.to(device), labels.to(device).float()
            preds = net(imgs)
            loss = criterion(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / max(1, len(TrainLoader))
        loss_t.append(avg_loss)
        print(f"Epoch {epoch+1}: Train Loss = {avg_loss:.4f}")

        # Validation
        if (epoch + 1) % cfg.test_every == 0 or (epoch + 1) == cfg.epochs:
            pre_val_path = os.path.join(paths.weights, f"{args.model_name}_epoch{epoch+1:02d}_preval.pth")
            torch.save({
                "model": net.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch + 1
            }, pre_val_path)
            print(f"Saved checkpoint before validation → {pre_val_path}")

            cam_dir = os.path.join(paths.valid_result, args.model_name, "cam")
            os.makedirs(cam_dir, exist_ok=True)
            print("Generating CAMs for validation...")
            generate_validation_cam(
                net,
                cfg,
                batch_size=cfg.batch_size,
                model_name=os.path.join(args.model_name, "cam")
            )
            valid_iou = get_overall_valid_score(
                os.path.join(paths.valid_result, args.model_name, "cam"),
                paths.mask_valid,
                num_workers=num_workers
            )
            iou_v.append(valid_iou)

            if valid_iou > best_val:
                best_val = valid_iou
                best_path = os.path.join(paths.weights, f"{args.model_name}_best.pth")
                torch.save({
                    "model": net.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch + 1
                }, best_path)
                print(f"New best model saved to {best_path}")

            print(f"Validation mIoU: {valid_iou:.4f}")

    last_path = os.path.join(paths.weights, f"{args.model_name}_last.pth")
    torch.save({
        "model": net.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": cfg.epochs
    }, last_path)
    print(f"Final model saved → {last_path}")
    print("Training completed successfully!")


if __name__ == "__main__":
    main()
