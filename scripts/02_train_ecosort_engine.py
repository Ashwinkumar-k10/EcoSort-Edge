"""
EcoSort-Edge: Two-Stage Transfer Learning Engine
Optimized for NVIDIA GeForce RTX 4050 Laptop GPU (Ada Lovelace Architecture).
High-Resolution Progressive Input: 256 x 256 RGB (+30.6% spatial fidelity).
Features:
- Automatic Mixed Precision (AMP / FP16) via Tensor Cores
- WeightedRandomSampler (Class Imbalance Mitigation)
- Label Smoothing (alpha = 0.05)
- Differential Learning Rates (Backbone: 1e-5, Head: 1e-4)
- Live Real-Time Progress Bar & VRAM Telemetry
"""

import sys
import os
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import models, datasets, transforms
from tqdm import tqdm

# Ensure UTF-8 output
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

CLASSES = [
    "battery",
    "cardboard",
    "food_waste",
    "glass_bottle",
    "metal_can",
    "multi_layer_packaging",
    "paper",
    "plastic_bottle",
    "vegetable_scrap"
]

DATA_DIR = Path("data/processed")
CHECKPOINT_DIR = Path("models/checkpoints")
CHECKPOINT_PATH = CHECKPOINT_DIR / "mobilenetv3_ecosort.pth"

# Training Hyperparameters
BATCH_SIZE = 32
EPOCHS = 15
INPUT_SIZE = 256  # 256 x 256 High-Resolution Mode
LR_BACKBONE = 1e-5
LR_HEAD = 1e-4
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.05

def print_header(gpu_name: str, vram_gb: float):
    print("\n" + "=" * 80)
    print("           ECOSORT-EDGE: CHAMPION MODEL TRAINING ENGINE (RTX 4050)           ")
    print("=" * 80)
    print(f" • Target Hardware : {gpu_name} ({vram_gb:.1f} GB GDDR6 VRAM)")
    print(f" • Architecture    : MobileNetV3-Large (5.48M parameters)")
    print(f" • Classes Count   : {len(CLASSES)} Canonical Statutory Classes")
    print(f" • Input Geometry  : 3 x {INPUT_SIZE} x {INPUT_SIZE} RGB (+30.6% Area)")
    print(f" • Batch Size      : {BATCH_SIZE} samples / batch")
    print(f" • Total Epochs    : {EPOCHS}")
    print(f" • Learning Rates  : Backbone = {LR_BACKBONE:.1e} | Head = {LR_HEAD:.1e}")
    print(f" • Loss Function   : CrossEntropyLoss (Label Smoothing alpha = {LABEL_SMOOTHING})")
    print(f" • Acceleration    : PyTorch Automatic Mixed Precision (AMP / FP16)")
    print(f" • Sampler Policy  : WeightedRandomSampler (Inverse Class Frequency)")
    print("=" * 80 + "\n")

def train_ecosort():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3) if torch.cuda.is_available() else 0.0

    print_header(gpu_name, vram_gb)

    # 1. 256x256 Enhanced Optical Transforms (Resolves Transparent Bottles & Foil Wrappers)
    train_transforms = transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.35),
        transforms.ColorJitter(brightness=0.3, contrast=0.35, saturation=0.25, hue=0.06),
        transforms.RandomAutocontrast(p=0.25),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_path = DATA_DIR / "train"
    val_path = DATA_DIR / "val"

    if not train_path.exists():
        print(f"[!] Error: {train_path} not found. Run scripts/01_curate_and_dedup.py first.")
        return

    print("[*] Ingesting Clean Datasets...")
    train_dataset = datasets.ImageFolder(train_path, transform=train_transforms)
    val_dataset = datasets.ImageFolder(val_path, transform=val_transforms)

    class_counts = np.bincount(train_dataset.targets)
    print("\n" + "-" * 45)
    print(f"{'Class Name':<25} | {'Training Images':<15}")
    print("-" * 45)
    for idx, cls_name in enumerate(train_dataset.classes):
        print(f"{cls_name:<25} | {class_counts[idx]:<15}")
    print("-" * 45)
    print(f"Total Training Samples  : {len(train_dataset)}")
    print(f"Total Validation Samples: {len(val_dataset)}\n")

    class_weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = [class_weights[t] for t in train_dataset.targets]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    print("[*] Loading MobileNetV3-Large Pretrained Weights...")
    model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, len(CLASSES))
    model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    optimizer = torch.optim.AdamW([
        {'params': model.features.parameters(), 'lr': LR_BACKBONE},
        {'params': model.classifier.parameters(), 'lr': LR_HEAD}
    ], weight_decay=WEIGHT_DECAY)

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    print("\n" + "=" * 80)
    print("            STARTING RTX 4050 256x256 HIGH-RES TRAINING PIPELINE             ")
    print("=" * 80)

    best_val_acc = 0.0
    start_total_time = time.perf_counter()

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:02d}/{EPOCHS:02d} [Train]", unit="batch", leave=False)
        for images, labels in pbar:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad()

            with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            vram_used = torch.cuda.memory_allocated(0) / (1024**2) if torch.cuda.is_available() else 0.0
            pbar.set_postfix({
                "Loss": f"{loss.item():.3f}",
                "Acc": f"{(correct/total)*100:.1f}%",
                "VRAM": f"{vram_used:.0f}MB"
            })

        train_loss = running_loss / total
        train_acc = (correct / total) * 100

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        epoch_val_loss = val_loss / val_total
        epoch_val_acc = (val_correct / val_total) * 100

        saved_flag = ""
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            saved_flag = "  ★ BEST MODEL SAVED"

        print(f"Epoch [{epoch+1:02d}/{EPOCHS:02d}] | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:5.2f}% | Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:5.2f}%{saved_flag}")

    total_duration = time.perf_counter() - start_total_time
    print("\n" + "=" * 80)
    print("                         TRAINING SUMMARY                                    ")
    print("=" * 80)
    print(f" • Input Resolution    : 256 x 256 pixels")
    print(f" • Total Training Time : {total_duration:.1f} seconds ({total_duration/60:.2f} mins)")
    print(f" • Best Validation Acc : {best_val_acc:.2f}%")
    print(f" • Saved Checkpoint    : {CHECKPOINT_PATH.resolve()}")
    print("=" * 80)
    print("\n[NEXT STEP] Run `python scripts/export_executorch.py` to compile to INT8 .pte for Raspberry Pi 5!\n")

if __name__ == "__main__":
    train_ecosort()
