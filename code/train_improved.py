# code/train_improved.py
import os
import argparse
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm


# ------------------------------------------------------------------------------
#  Custom dataset that reads the updated train.txt / test.txt you generated
# ------------------------------------------------------------------------------

class CustomFoodDataset(Dataset):
    """
    Reads <class_name>/<filename-without-ext> lines from a txt file and loads
    the corresponding JPEG from the images folder.
    """
    def __init__(self, txt_file: Path, images_root: Path, transform=None):
        self.images_root = images_root
        self.transform = transform

        with open(txt_file, "r") as f:
            self.samples = [ln.strip() for ln in f.readlines() if "/" in ln]

        # Build class-to-index mapping
        classes = sorted({s.split("/")[0] for s in self.samples})
        self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rel_path = self.samples[idx]
        cls_name, file_stem = rel_path.split("/")
        img_path = self.images_root / cls_name / f"{file_stem}.jpg"

        # Fallback: try png if jpg missing
        if not img_path.exists():
            img_path = img_path.with_suffix(".png")
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")

        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        label = self.class_to_idx[cls_name]
        return img, label


# ------------------------------------------------------------------------------
#  Dataloaders
# ------------------------------------------------------------------------------

def get_dataloaders(data_root: Path, batch_size: int, img_size: int = 300):
    meta_dir   = data_root / "food-101" / "meta"
    images_dir = data_root / "food-101" / "images"
    train_txt  = meta_dir / "train.txt"
    test_txt   = meta_dir / "test.txt"

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size),
        transforms.RandAugment(),           # stronger augmentation
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(int(img_size * 1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    train_ds = CustomFoodDataset(train_txt, images_dir, transform=train_tf)
    val_ds   = CustomFoodDataset(test_txt,  images_dir, transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=4, pin_memory=False)

    return train_loader, val_loader, len(train_ds.class_to_idx)


# ------------------------------------------------------------------------------
#  Model
# ------------------------------------------------------------------------------

def build_model(num_classes: int, device: torch.device):
    weights = models.EfficientNet_B3_Weights.DEFAULT
    model   = models.efficientnet_b3(weights=weights)
    in_f    = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_f, num_classes)
    return model.to(device)


# ------------------------------------------------------------------------------
#  Train / Validate loops
# ------------------------------------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = correct = total = 0

    for imgs, labels in tqdm(loader, desc="Train", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        _, preds = outputs.max(1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)

    return running_loss / total, correct / total


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = correct = total = 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * imgs.size(0)
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)

    return running_loss / total, correct / total


# ------------------------------------------------------------------------------
#  Main
# ------------------------------------------------------------------------------

def main(args):
    project_root = Path(__file__).resolve().parent.parent
    data_root    = project_root / "data"
    models_dir   = project_root / "models"
    models_dir.mkdir(exist_ok=True)

    # Prefer Apple-Silicon MPS → CUDA → CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("Device:", device)

    # Data
    print("Loading data …")
    train_loader, val_loader, n_classes = get_dataloaders(
        data_root, args.batch_size, img_size=300
    )
    print(f"Train images: {len(train_loader.dataset)} | "
          f"Val images: {len(val_loader.dataset)} | "
          f"Classes: {n_classes}")

    # Model / loss / optim / sched
    model     = build_model(num_classes=n_classes, device=device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        print(f"\n=== Epoch {epoch}/{args.epochs} ===")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion,
                                                optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            ckpt = models_dir / "efficientnetb3_food_extended.pth"
            torch.save(model.state_dict(), ckpt)
            print(f"→ Saved best model (acc={best_acc:.4f})")

    print(f"\nFinished! Best val accuracy: {best_acc:.4f}")


# ------------------------------------------------------------------------------
#  Entry
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int,   default=25,
                        help="total training epochs")
    parser.add_argument("--batch_size", type=int,   default=32,
                        help="mini-batch size")
    parser.add_argument("--lr",         type=float, default=3e-4,
                        help="initial learning rate")
    args = parser.parse_args()
    main(args)
