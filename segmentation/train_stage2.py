import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from omegaconf import OmegaConf
from tqdm import tqdm

from dataset import PseudoSegDataset
from models.deeplab import DeepLabV3Plus


def parse_args():
    parser = argparse.ArgumentParser(description="Train Stage 2 (Segmentation) with pseudo masks")
    parser.add_argument("--config", type=str, default="segmentation/configuration_seg.yml", help="Path to YAML config")
    parser.add_argument("--epochs", type=int, help="Override number of epochs")
    parser.add_argument("--batch", type=int, help="Override batch size")
    parser.add_argument("--lr", type=float, help="Override learning rate")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], help="Force device (cpu or cuda)")
    parser.add_argument("--save_dir", type=str, help="Directory to save checkpoints")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    paths = cfg.paths

    # --- Override CLI nếu có ---
    if args.epochs:
        cfg.train.epochs = args.epochs
    if args.batch:
        cfg.train.batch_size = args.batch
    if args.lr:
        cfg.train.learning_rate = args.lr
    if args.device:
        cfg.device = args.device
    if args.save_dir:
        cfg.save_dir = args.save_dir

    # --- Device setup ---
    if cfg.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("⚙️ Using CPU")

    # --- Transform ---
    train_tf = transforms.Compose([
        transforms.Resize((cfg.image_size, cfg.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(cfg.mean, cfg.std)
    ])

    # --- Dataset & Dataloader ---
    train_ds = PseudoSegDataset(
        img_dir=paths.valid, image_size=cfg.image_size,
        mask_dir=cfg.pseudo_mask_dir,
        transform=train_tf
    )

    # --- Multiprocessing fix (macOS/Windows) ---
    num_workers = 0 if os.name != 'posix' else cfg.num_workers

    train_dl = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    # --- Model ---
    model = DeepLabV3Plus(num_classes=cfg.num_classes).to(device)
    print(f"🧠 Model initialized: DeepLabV3Plus ({cfg.num_classes} classes)")

    # --- Optimizer & Loss ---
    if cfg.train.optimizer.lower() == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg.train.learning_rate,
            weight_decay=cfg.train.weight_decay
        )
    else:
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=cfg.train.learning_rate,
            momentum=0.9,
            weight_decay=cfg.train.weight_decay
        )

    criterion = nn.CrossEntropyLoss()

    # --- Training loop ---
    print(f"🚀 Start training for {cfg.train.epochs} epochs...")
    for epoch in range(cfg.train.epochs):
        model.train()
        running_loss = 0.0
        progress = tqdm(train_dl, desc=f"Epoch [{epoch+1}/{cfg.train.epochs}]", leave=False)
        for imgs, masks in progress:
            imgs, masks = imgs.to(device), masks.to(device)

            outputs = model(imgs)
            loss = criterion(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            progress.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = running_loss / len(train_dl)
        print(f"✅ Epoch {epoch+1}/{cfg.train.epochs} - Avg loss: {avg_loss:.4f}")

    # --- Save model ---
    os.makedirs(cfg.save_dir, exist_ok=True)
    save_path = os.path.join(cfg.save_dir, f"seg_model_{cfg.model_name}.pth")
    torch.save(model.state_dict(), save_path)
    print(f"💾 Model saved to {save_path}")


if __name__ == "__main__":
    try:
        torch.multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
