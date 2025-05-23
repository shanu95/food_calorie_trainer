# code/train.py

import os
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torchvision.datasets import Food101
from torchvision import transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm

def get_dataloaders(data_root, batch_size):
    # transforms for training & validation
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    # Food101 expects the *parent* of "food-101" here:
    train_ds = Food101(root=data_root, split="train", transform=train_tf, download=False)
    val_ds   = Food101(root=data_root, split="test",  transform=val_tf,    download=False)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True
    )
    val_loader   = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )
    return train_loader, val_loader

def build_model(num_classes, device):
    # Load pretrained MobileNetV3 Small
    model = models.mobilenet_v3_small(pretrained=True)
    # Replace the final classifier: it's in model.classifier[3]
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    return model.to(device)

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for imgs, labels in tqdm(loader, desc="Training", leave=False):
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

    epoch_loss = running_loss / total
    epoch_acc  = correct / total
    return epoch_loss, epoch_acc

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * imgs.size(0)
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)

    val_loss = running_loss / total
    val_acc  = correct / total
    return val_loss, val_acc

def main(args):
    # Paths
    project_root = Path(__file__).resolve().parent.parent
    data_root    = project_root / "data"
    models_dir   = project_root / "models"
    models_dir.mkdir(exist_ok=True)

    # Device: prefer Apple MPS, then CUDA, else CPU
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Data
    print("Loading data…")
    train_loader, val_loader = get_dataloaders(data_root, args.batch_size)
    print(f"→ Train imgs: {len(train_loader.dataset)}")
    print(f"→ Val   imgs: {len(val_loader.dataset)}")

    # Model, loss, optimizer, scheduler
    model     = build_model(num_classes=101, device=device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=3, gamma=0.1)

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        print(f"\n--- Starting epoch {epoch}/{args.epochs} ---")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc     = validate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"Epoch {epoch}/{args.epochs} "
              f"| Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} "
              f"| Val   Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

        # save best
        if val_acc > best_acc:
            best_acc = val_acc
            ckpt_path = models_dir / "mobilenet_food101.pth"
            torch.save(model.state_dict(), ckpt_path)
            print(f"→ Saved best model (acc={best_acc:.4f}) to {ckpt_path}")

    print("Training complete. Best val acc: {:.4f}".format(best_acc))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",     type=int,   default=5,    help="number of training epochs")
    p.add_argument("--batch_size", type=int,   default=32,   help="batch size")
    p.add_argument("--lr",         type=float, default=1e-3, help="learning rate")
    args = p.parse_args()
    main(args)
