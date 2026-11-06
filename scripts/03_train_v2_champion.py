"""
EcoSort-Edge V2: Champion Training Engine
==========================================
Strategy 1 Upgrades over V1:
  1. Focal Loss (γ=2.0) — hammers hard/confused samples
  2. Mixup (α=0.4) + CutMix (α=1.0) — better generalization
  3. Cosine Annealing LR with Warm Restarts (T0=10)
  4. 60 Epochs with Early Stopping (patience=12)
  5. Progressive Resizing: 224 → 256 → 288
  6. Stronger augmentation pipeline (RandAugment + RandomErasing)
  7. EMA (Exponential Moving Average) of weights
  8. Gradient Clipping (max_norm=1.0)

Target: Push accuracy from 90.4% → 95%+
Hardware: NVIDIA GeForce RTX 4050 Laptop GPU (6 GB VRAM)
"""

import sys
import os
import time
import copy
import math
import random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Register dummy torchvision meta ops to resolve C++ DLL symbol binding on Windows
for op_sig in [
    ('torchvision::nms', '(Tensor corners, Tensor scores, float iou_threshold) -> Tensor'),
    ('torchvision::qnms', '(Tensor corners, Tensor scores, float iou_threshold) -> Tensor'),
    ('torchvision::roi_align', '(Tensor input, Tensor rois, float spatial_scale, int pooled_height, int pooled_width, int sampling_ratio, bool aligned) -> Tensor'),
    ('torchvision::roi_pool', '(Tensor input, Tensor rois, float spatial_scale, int pooled_height, int pooled_width) -> (Tensor, Tensor)')
]:
    try:
        torch.library.define(op_sig[0], op_sig[1])
    except Exception:
        pass

from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import models, datasets, transforms
from tqdm import tqdm

# Ensure UTF-8 output
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# ─── Seed for Reproducibility ───
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

CLASSES = [
    "battery", "cardboard", "food_waste", "glass_bottle",
    "metal_can", "multi_layer_packaging", "paper",
    "plastic_bottle", "vegetable_scrap"
]

DATA_DIR = Path("data/processed")
CHECKPOINT_DIR = Path("models/checkpoints")
CHECKPOINT_V2 = CHECKPOINT_DIR / "mobilenetv3_ecosort_v2.pth"
CHECKPOINT_EMA = CHECKPOINT_DIR / "mobilenetv3_ecosort_v2_ema.pth"

# ─── Hyperparameters ───
BATCH_SIZE = 32
EPOCHS = 60
LR_BACKBONE = 1e-5
LR_HEAD = 5e-4  # Higher head LR for faster convergence
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.1  # Slightly higher than v1
FOCAL_GAMMA = 2.0
MIXUP_ALPHA = 0.4
CUTMIX_ALPHA = 1.0
MIXUP_PROB = 0.5  # 50% chance mixup, 50% cutmix per batch
EMA_DECAY = 0.999
EARLY_STOP_PATIENCE = 12
GRADIENT_CLIP = 1.0

# Progressive resize schedule
PROGRESSIVE_SCHEDULE = {
    0: 224,    # Epochs 0-14: 224x224 (fast warmup)
    15: 256,   # Epochs 15-34: 256x256 (standard)
    35: 288,   # Epochs 35-59: 288x288 (high-res fine-tune)
}


# ═══════════════════════════════════════════════════════════════
#  FOCAL LOSS
# ═══════════════════════════════════════════════════════════════
class FocalLoss(nn.Module):
    """Focal Loss with label smoothing support."""
    def __init__(self, gamma=2.0, label_smoothing=0.1, weight=None):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.weight = weight

    def forward(self, logits, targets):
        num_classes = logits.size(1)
        # Apply label smoothing
        if targets.dim() == 1:
            # Hard labels → smooth one-hot
            smooth = self.label_smoothing / num_classes
            one_hot = torch.zeros_like(logits).scatter_(1, targets.unsqueeze(1), 1.0)
            one_hot = one_hot * (1 - self.label_smoothing) + smooth
        else:
            # Already soft labels (from mixup/cutmix)
            one_hot = targets

        log_probs = F.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)

        # Focal modulation: (1 - p_t)^gamma
        focal_weight = (1 - probs) ** self.gamma
        loss = -focal_weight * one_hot * log_probs

        if self.weight is not None:
            w = self.weight.to(logits.device)
            loss = loss * w.unsqueeze(0)

        return loss.sum(dim=1).mean()


# ═══════════════════════════════════════════════════════════════
#  MIXUP & CUTMIX
# ═══════════════════════════════════════════════════════════════
def mixup_data(x, y, alpha=0.4):
    """Mixup: blend two images and their labels."""
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def cutmix_data(x, y, alpha=1.0):
    """CutMix: cut and paste rectangular regions between images."""
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    _, _, H, W = x.shape
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    x1 = np.clip(cx - cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y2 = np.clip(cy + cut_h // 2, 0, H)

    x[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]
    lam = 1 - ((x2 - x1) * (y2 - y1) / (W * H))  # Adjust lambda
    y_a, y_b = y, y[index]
    return x, y_a, y_b, lam


def mixup_criterion(criterion_fn, logits, y_a, y_b, lam, num_classes):
    """Compute loss for mixed labels."""
    # Convert to one-hot soft labels
    one_hot_a = torch.zeros(y_a.size(0), num_classes, device=logits.device).scatter_(1, y_a.unsqueeze(1), 1.0)
    one_hot_b = torch.zeros(y_b.size(0), num_classes, device=logits.device).scatter_(1, y_b.unsqueeze(1), 1.0)
    mixed_targets = lam * one_hot_a + (1 - lam) * one_hot_b
    return criterion_fn(logits, mixed_targets)


# ═══════════════════════════════════════════════════════════════
#  EMA (Exponential Moving Average)
# ═══════════════════════════════════════════════════════════════
class EMAModel:
    """Maintains an exponential moving average of model parameters."""
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1 - self.decay)

    def apply(self, model):
        """Apply EMA weights to model (for evaluation)."""
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model):
        """Restore original weights after evaluation."""
        for name, param in model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}


# ═══════════════════════════════════════════════════════════════
#  DATA AUGMENTATION PIPELINES
# ═══════════════════════════════════════════════════════════════
def to_rgb(img):
    return img.convert("RGB")

def get_train_transforms(input_size):
    return transforms.Compose([
        transforms.Lambda(to_rgb),
        transforms.Resize((input_size, input_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.RandomRotation(20),
        transforms.RandomPerspective(distortion_scale=0.25, p=0.4),
        transforms.ColorJitter(brightness=0.35, contrast=0.4, saturation=0.3, hue=0.08),
        transforms.RandomAutocontrast(p=0.3),
        transforms.RandAugment(num_ops=2, magnitude=9),  # NEW: Auto-augmentation
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),  # NEW: Random erasing
    ])


def get_val_transforms(input_size):
    return transforms.Compose([
        transforms.Lambda(to_rgb),
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def get_current_size(epoch):
    """Get input size for progressive resizing."""
    current_size = 224
    for start_epoch, size in sorted(PROGRESSIVE_SCHEDULE.items()):
        if epoch >= start_epoch:
            current_size = size
    return current_size


# ═══════════════════════════════════════════════════════════════
#  MAIN TRAINING LOOP
# ═══════════════════════════════════════════════════════════════
def train_v2():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3) if torch.cuda.is_available() else 0.0

    print("\n" + "=" * 80)
    print("        ECOSORT-EDGE V2: CHAMPION TRAINING ENGINE (STRATEGY 1)")
    print("=" * 80)
    print(f" • GPU               : {gpu_name} ({vram_gb:.1f} GB VRAM)")
    print(f" • Architecture      : MobileNetV3-Large (Fine-Tuned)")
    print(f" • Classes           : {len(CLASSES)}")
    print(f" • Epochs            : {EPOCHS} (Early Stop patience={EARLY_STOP_PATIENCE})")
    print(f" • Progressive Resize: 224 → 256 → 288")
    print(f" • Loss              : Focal Loss (γ={FOCAL_GAMMA}) + Label Smoothing ({LABEL_SMOOTHING})")
    print(f" • Augmentation      : Mixup (α={MIXUP_ALPHA}) + CutMix (α={CUTMIX_ALPHA}) + RandAugment")
    print(f" • LR Schedule       : CosineAnnealingWarmRestarts (T0=10)")
    print(f" • EMA Decay         : {EMA_DECAY}")
    print(f" • Gradient Clip     : {GRADIENT_CLIP}")
    print("=" * 80 + "\n")

    train_path = DATA_DIR / "train"
    val_path = DATA_DIR / "val"

    if not train_path.exists():
        print(f"[!] Error: {train_path} not found. Run scripts/01_curate_and_dedup.py first.")
        return

    # ─── Initial dataset load (for class counts + sampler) ───
    initial_dataset = datasets.ImageFolder(train_path)
    class_counts = np.bincount(initial_dataset.targets)
    targets_array = initial_dataset.targets

    print("-" * 50)
    print(f"{'Class Name':<28} | {'Train':>6} | {'Weight':>8}")
    print("-" * 50)
    class_weights = 1.0 / np.maximum(class_counts, 1)
    class_weights_norm = class_weights / class_weights.sum() * len(CLASSES)
    for idx, cls_name in enumerate(initial_dataset.classes):
        print(f"{cls_name:<28} | {class_counts[idx]:>6} | {class_weights_norm[idx]:>8.3f}")
    print("-" * 50)
    print(f"Total Training: {len(initial_dataset)} | Validation: {len(os.listdir(val_path))} classes\n")

    # ─── Sampler weights ───
    sample_weights = [class_weights[t] for t in targets_array]

    # ─── Class weights tensor for Focal Loss ───
    focal_class_weights = torch.FloatTensor(class_weights_norm).to(device)

    # ─── Model ───
    print("[*] Loading MobileNetV3-Large Pretrained Weights...")
    model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)

    # Enhanced classifier head with dropout
    in_features = model.classifier[3].in_features
    model.classifier = nn.Sequential(
        nn.Linear(model.classifier[0].in_features, 1280),
        nn.Hardswish(),
        nn.Dropout(p=0.3),  # Increased from 0.2
        nn.Linear(1280, len(CLASSES))
    )
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[*] Parameters: {total_params:,} total | {trainable_params:,} trainable\n")

    # ─── Loss ───
    criterion = FocalLoss(gamma=FOCAL_GAMMA, label_smoothing=LABEL_SMOOTHING, weight=focal_class_weights)

    # ─── Optimizer ───
    optimizer = torch.optim.AdamW([
        {'params': model.features.parameters(), 'lr': LR_BACKBONE},
        {'params': model.classifier.parameters(), 'lr': LR_HEAD}
    ], weight_decay=WEIGHT_DECAY)

    # ─── Cosine Annealing with Warm Restarts ───
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-7
    )

    # ─── AMP Scaler ───
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    # ─── EMA ───
    ema = EMAModel(model, decay=EMA_DECAY)

    # ─── Training State ───
    best_val_acc = 0.0
    best_ema_acc = 0.0
    epochs_no_improve = 0
    current_input_size = 224
    start_total = time.perf_counter()

    print("=" * 80)
    print("                    STARTING V2 CHAMPION TRAINING")
    print("=" * 80 + "\n")

    for epoch in range(EPOCHS):
        # ─── Progressive Resizing: rebuild dataloaders if size changes ───
        new_size = get_current_size(epoch)
        if new_size != current_input_size or epoch == 0:
            current_input_size = new_size
            print(f"\n[→] Resolution change: {current_input_size}x{current_input_size}")

            train_dataset = datasets.ImageFolder(train_path, transform=get_train_transforms(current_input_size))
            val_dataset = datasets.ImageFolder(val_path, transform=get_val_transforms(current_input_size))

            sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler,
                                      num_workers=0, pin_memory=(device.type == "cuda"), drop_last=True)
            val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                    num_workers=0, pin_memory=(device.type == "cuda"))

        # ─── TRAIN ───
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:02d}/{EPOCHS:02d} [{current_input_size}px]",
                     unit="batch", leave=False)

        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # ─── Apply Mixup or CutMix (50/50 chance) ───
            use_mix = random.random() < 0.8  # 80% of batches get mixing
            if use_mix:
                if random.random() < MIXUP_PROB:
                    images, targets_a, targets_b, lam = mixup_data(images, labels, MIXUP_ALPHA)
                else:
                    images, targets_a, targets_b, lam = cutmix_data(images, labels, CUTMIX_ALPHA)

            optimizer.zero_grad()

            with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
                outputs = model(images)
                if use_mix:
                    loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam, len(CLASSES))
                else:
                    loss = criterion(outputs, labels)

            scaler.scale(loss).backward()

            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)

            scaler.step(optimizer)
            scaler.update()

            # Update EMA
            ema.update(model)

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            if use_mix:
                correct += (lam * (preds == targets_a).float() + (1 - lam) * (preds == targets_b).float()).sum().item()
            else:
                correct += (preds == labels).sum().item()
            total += labels.size(0)

            vram_used = torch.cuda.memory_allocated(0) / (1024**2) if torch.cuda.is_available() else 0.0
            pbar.set_postfix({
                "Loss": f"{loss.item():.3f}",
                "Acc": f"{(correct/total)*100:.1f}%",
                "VRAM": f"{vram_used:.0f}MB"
            })

        scheduler.step(epoch)
        train_loss = running_loss / total
        train_acc = (correct / total) * 100

        # ─── VALIDATE (Regular Model) ───
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        epoch_val_loss = val_loss / val_total
        epoch_val_acc = (val_correct / val_total) * 100

        # ─── VALIDATE (EMA Model) ───
        ema.apply(model)
        ema_correct = 0
        ema_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
                    outputs = model(images)
                preds = outputs.argmax(dim=1)
                ema_correct += (preds == labels).sum().item()
                ema_total += labels.size(0)
        ema_val_acc = (ema_correct / ema_total) * 100
        ema.restore(model)

        # ─── Save Best ───
        saved_flag = ""
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), CHECKPOINT_V2)
            saved_flag = " ★ BEST"
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if ema_val_acc > best_ema_acc:
            best_ema_acc = ema_val_acc
            ema.apply(model)
            torch.save(model.state_dict(), CHECKPOINT_EMA)
            ema.restore(model)
            saved_flag += " ★ EMA-BEST"

        current_lr = scheduler.get_last_lr()[0]
        print(f"E{epoch+1:02d}/{EPOCHS:02d} | {current_input_size}px | "
              f"TrL:{train_loss:.3f} TrA:{train_acc:5.1f}% | "
              f"VaL:{epoch_val_loss:.3f} VaA:{epoch_val_acc:5.1f}% | "
              f"EMA:{ema_val_acc:5.1f}% | LR:{current_lr:.1e}{saved_flag}")

        # ─── Early Stopping ───
        if epochs_no_improve >= EARLY_STOP_PATIENCE:
            print(f"\n[!] Early stopping triggered after {EARLY_STOP_PATIENCE} epochs without improvement.")
            break

    # ═══════════════════════════════════════════════════════════════
    #  TRAINING SUMMARY
    # ═══════════════════════════════════════════════════════════════
    total_duration = time.perf_counter() - start_total
    print("\n" + "=" * 80)
    print("                   V2 CHAMPION TRAINING SUMMARY")
    print("=" * 80)
    print(f" • Total Epochs Run    : {epoch + 1}")
    print(f" • Total Time          : {total_duration:.1f}s ({total_duration/60:.1f} min)")
    print(f" • Best Val Accuracy   : {best_val_acc:.2f}%")
    print(f" • Best EMA Accuracy   : {best_ema_acc:.2f}%")
    print(f" • Regular Checkpoint  : {CHECKPOINT_V2}")
    print(f" • EMA Checkpoint      : {CHECKPOINT_EMA}")
    print(f" • Progressive Sizes   : 224 → 256 → 288")
    print("=" * 80)

    # ─── Pick the better checkpoint ───
    if best_ema_acc > best_val_acc:
        print(f"\n[★] EMA model is better ({best_ema_acc:.2f}% vs {best_val_acc:.2f}%)")
        print(f"    Use: {CHECKPOINT_EMA}")
        winner = CHECKPOINT_EMA
    else:
        print(f"\n[★] Regular model is better ({best_val_acc:.2f}% vs {best_ema_acc:.2f}%)")
        print(f"    Use: {CHECKPOINT_V2}")
        winner = CHECKPOINT_V2

    # Copy winner as the main checkpoint
    import shutil
    final_path = CHECKPOINT_DIR / "mobilenetv3_ecosort_v2_best.pth"
    shutil.copy2(winner, final_path)
    print(f"    Copied to: {final_path}")

    print(f"\n[NEXT] Run: python scripts/export_executorch.py")
    print(f"       (Update CHECKPOINT_PATH in export script to point to {final_path})\n")


if __name__ == "__main__":
    train_v2()
