"""
scripts/train_progressive_resizing_256.py
EcoSort-Edge: Progressive Multi-Scale Resizing (256x256)
Fine-tunes the verified 224x224 baseline model at 256x256 resolution.
"""

import os
import sys
import time
import json
import random
import argparse
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms, datasets, models
from torch.amp import GradScaler, autocast
import numpy as np
from tqdm import tqdm

# Fix Windows console UTF-8 encoding and unbuffered line printing
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TRAIN_DIR = PROCESSED_DIR / "train"
VAL_DIR = PROCESSED_DIR / "val"
TEST_DIR = PROCESSED_DIR / "test"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')

def get_transforms_256():
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(256, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.20, scale=(0.02, 0.2), value="random")
    ])

    eval_transform = transforms.Compose([
        transforms.Resize(288),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return train_transform, eval_transform

def make_balanced_sampler(dataset):
    targets = [s[1] for s in dataset.samples]
    class_counts = Counter(targets)
    num_samples = len(targets)
    class_weights = {cls: num_samples / count for cls, count in class_counts.items()}
    weights = [class_weights[t] for t in targets]
    sampler = WeightedRandomSampler(torch.DoubleTensor(weights), num_samples=num_samples, replacement=True)
    return sampler, class_counts

def build_mobilenet_v3_small(num_classes=9):
    model = models.mobilenet_v3_small(weights=None)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model

def compute_metrics(y_true, y_pred, num_classes=9):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    acc = np.mean(y_true == y_pred)
    recalls = []
    precisions = []
    f1s = []
    
    for c in range(num_classes):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
        
    macro_f1 = np.mean(f1s)
    macro_recall = np.mean(recalls)
    macro_precision = np.mean(precisions)
    cm = [[int(np.sum((y_true == r) & (y_pred == c))) for c in range(num_classes)] for r in range(num_classes)]
    
    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "macro_recall": float(macro_recall),
        "macro_precision": float(macro_precision),
        "per_class_recall": [float(r) for r in recalls],
        "per_class_precision": [float(p) for p in precisions],
        "per_class_f1": [float(f) for f in f1s],
        "confusion_matrix": cm
    }

def run_progressive_256(epochs=10, batch_size=32, lr_head=5e-5, patience=4, seed=42, resume=False):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*75}")
    print(f"[ECOSORT-EDGE] PROGRESSIVE MULTI-SCALE RESIZING: 224x224 -> 256x256")
    print(f"[*] Target Architecture: MobileNetV3-Small (2.54M params)")
    print(f"[*] Device: {device} | Mixed Precision: FP16 AMP + cuDNN Benchmark")
    print(f"[*] Fine-Tuning Budget: {epochs} Epochs | Batch Size: {batch_size} | EarlyStop Patience: {patience}")
    print(f"{'='*75}")
    
    train_tf, eval_tf = get_transforms_256()
    train_dataset = datasets.ImageFolder(str(TRAIN_DIR), transform=train_tf)
    val_dataset = datasets.ImageFolder(str(VAL_DIR), transform=eval_tf)
    test_dataset = datasets.ImageFolder(str(TEST_DIR), transform=eval_tf)
    
    sampler, class_counts = make_balanced_sampler(train_dataset)
    num_classes = len(train_dataset.classes)
    
    nw = 0 if sys.platform == "win32" else 2
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=nw, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=nw, pin_memory=(device.type == "cuda"))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=nw, pin_memory=(device.type == "cuda"))
    
    model = build_mobilenet_v3_small(num_classes).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    scaler = GradScaler(enabled=(device.type == "cuda"))
    
    baseline_ckpt_path = MODELS_DIR / "ecosort_mobilenetv3_small_224_baseline.pth"
    best_256_path = MODELS_DIR / "ecosort_mobilenetv3_small_256_progressive.pth"
    last_256_path = MODELS_DIR / "ecosort_mobilenetv3_small_256_checkpoint_last.pth"
    
    best_macro_f1 = -1.0
    epochs_no_improve = 0
    history = []
    start_epoch = 1
    
    if resume and last_256_path.exists():
        print(f"\n[+] Resuming from existing 256 checkpoint: {last_256_path.name}")
        ckpt = torch.load(last_256_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        scaler.load_state_dict(ckpt["scaler_state_dict"])
        best_macro_f1 = ckpt.get("best_macro_f1", -1.0)
        epochs_no_improve = ckpt.get("epochs_no_improve", 0)
        history = ckpt.get("history", [])
        start_epoch = ckpt.get("epoch", 1) + 1
        print(f"[+] Resuming from Epoch {start_epoch}/{epochs} (Best Macro-F1: {best_macro_f1:.4f})...")
    elif baseline_ckpt_path.exists():
        print(f"\n[+] Warm-starting from 224x224 baseline checkpoint: {baseline_ckpt_path.name}")
        ckpt = torch.load(baseline_ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        prev_f1 = ckpt.get("metrics", {}).get("macro_f1", 0.8864)
        print(f"    Baseline 224x224 Val Macro-F1: {prev_f1:.4f}")
    else:
        print(f"[!] Warning: Baseline checkpoint not found at {baseline_ckpt_path}. Using ImageNet weights.")
        weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        model = models.mobilenet_v3_small(weights=weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes).to(device)

    # Differential fine-tuning optimizer
    param_groups = [
        {"params": model.features.parameters(), "lr": lr_head * 0.2},
        {"params": model.classifier.parameters(), "lr": lr_head}
    ]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    if resume and last_256_path.exists():
        ckpt = torch.load(last_256_path, map_location=device)
        if "optimizer_state_dict" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            except Exception:
                pass
        for _ in range(start_epoch - 1):
            scheduler.step()

    try:
        print(f"\n[*] Commencing Progressive 256x256 Adaptation ({epochs} Epochs)...")
        for epoch in range(start_epoch, epochs + 1):
            model.train()
            running_loss = 0.0
            curr_lr = optimizer.param_groups[1]['lr']
            pbar = tqdm(train_loader, desc=f"Prog 256 [Ep {epoch:02d}/{epochs:02d}]", unit="batch", dynamic_ncols=True, leave=False)
            
            for x, y in pbar:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with autocast(device_type=device.type, dtype=torch.float16):
                    out = model(x)
                    loss = criterion(out, y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                running_loss += loss.item()
                pbar.set_postfix({"loss": f"{loss.item():.4f}", "avg": f"{running_loss/(pbar.n+1):.4f}", "lr": f"{curr_lr:.1e}"})
                
            scheduler.step()
            
            # Validation
            model.eval()
            v_preds, v_trues = [], []
            val_pbar = tqdm(val_loader, desc="  ↳ Validating 256", unit="batch", dynamic_ncols=True, leave=False)
            with torch.no_grad():
                for x, y in val_pbar:
                    x = x.to(device, non_blocking=True)
                    with autocast(device_type=device.type, dtype=torch.float16):
                        out = model(x)
                    v_preds.extend(out.argmax(dim=1).cpu().numpy())
                    v_trues.extend(y.numpy())
                    
            metrics = compute_metrics(v_trues, v_preds, num_classes)
            train_loss = running_loss / len(train_loader)
            
            save_msg = ""
            if metrics['macro_f1'] > best_macro_f1:
                best_macro_f1 = metrics['macro_f1']
                epochs_no_improve = 0
                torch.save({
                    "seed": seed,
                    "img_size": 256,
                    "model_state_dict": model.state_dict(),
                    "classes": train_dataset.classes,
                    "metrics": metrics,
                    "epoch": epoch
                }, best_256_path)
                save_msg = " -> [BEST 256 SAVED]"
            else:
                epochs_no_improve += 1
                save_msg = f" -> [patience {epochs_no_improve}/{patience}]"
                
            print(f"Prog 256 Epoch {epoch:02d}/{epochs:02d} (lr: {curr_lr:.1e}) | Loss: {train_loss:.4f} | Val Acc: {metrics['accuracy']*100:.2f}% | Val Macro-F1: {metrics['macro_f1']:.4f} | Recall: {metrics['macro_recall']*100:.2f}%{save_msg}", flush=True)
            
            history.append({
                "epoch": epoch,
                "loss": train_loss,
                "val_acc": metrics['accuracy'],
                "macro_f1": metrics['macro_f1'],
                "macro_recall": metrics['macro_recall']
            })
            
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "best_macro_f1": best_macro_f1,
                "epochs_no_improve": epochs_no_improve,
                "history": history
            }, last_256_path)
            
            if epochs_no_improve >= patience:
                print(f"\n[!] Early Stopping Triggered: Macro-F1 did not improve for {patience} consecutive epochs.")
                break
                
    except KeyboardInterrupt:
        print(f"\n[!] Paused by user. Checkpoint saved to: {last_256_path}")
        sys.exit(0)
        
    print(f"\n{'='*75}")
    print(f"[EVALUATION] Benchmarking Best 256x256 Model on Untouched Test Set (1,210 Images)...")
    print(f"{'='*75}", flush=True)
    
    ckpt = torch.load(best_256_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    
    t_preds, t_trues = [], []
    test_pbar = tqdm(test_loader, desc="[*] Holdout Test Eval 256", unit="batch", dynamic_ncols=True, leave=False)
    with torch.no_grad():
        for x, y in test_pbar:
            x = x.to(device, non_blocking=True)
            with autocast(device_type=device.type, dtype=torch.float16):
                out = model(x)
            t_preds.extend(out.argmax(dim=1).cpu().numpy())
            t_trues.extend(y.numpy())
            
    test_metrics = compute_metrics(t_trues, t_preds, num_classes)
    
    # Calculate Statutory 4-Stream Accuracy
    stream_map = {0:'Red', 1:'Blue', 2:'Green', 3:'Blue', 4:'Blue', 5:'Grey', 6:'Blue', 7:'Blue', 8:'Green'}
    cm = test_metrics['confusion_matrix']
    correct_stream = sum(count for a_idx, row in enumerate(cm) for p_idx, count in enumerate(row) if stream_map[a_idx] == stream_map[p_idx])
    total = sum(sum(row) for row in cm)
    statutory_acc = (correct_stream / total) * 100.0
    
    print(f"\n{'='*75}")
    print(f"[FINAL TEST RESULTS - PROGRESSIVE 256x256 MODEL]")
    print(f"{'='*75}")
    print(f"[*] Test Accuracy                   : {test_metrics['accuracy']*100:.2f}%")
    print(f"[*] Test Macro-F1                   : {test_metrics['macro_f1']:.4f}")
    print(f"[*] Test Macro-Recall               : {test_metrics['macro_recall']*100:.2f}%")
    print(f"[*] STATUTORY 4-STREAM BIN ACCURACY : {statutory_acc:.2f}% ({correct_stream}/{total})")
    print(f"{'-'*75}")
    print(f"{'Class Name':<26} {'Precision':>12} {'Recall':>12} {'F1-Score':>12}")
    print(f"{'-'*75}")
    for idx, cname in enumerate(test_dataset.classes):
        p_val = test_metrics['per_class_precision'][idx] * 100
        r_val = test_metrics['per_class_recall'][idx] * 100
        f_val = test_metrics['per_class_f1'][idx]
        print(f"{cname:<26} {p_val:>11.2f}% {r_val:>11.2f}% {f_val:>12.4f}")
    print(f"{'='*75}\n")
    
    # Save standalone ablation report
    report_data = {
        "model_name": "MobileNetV3-Small",
        "resolution": "256x256",
        "training_strategy": "Progressive Resizing from 224x224 Baseline",
        "best_val_macro_f1": best_macro_f1,
        "test_metrics": test_metrics,
        "statutory_4_stream_accuracy": statutory_acc,
        "checkpoint": str(best_256_path),
        "history": history
    }
    with open(RESULTS_DIR / "ablation_mobilenetv3_small_256.json", "w") as f:
        json.dump(report_data, f, indent=2)
    print(f"[+] Saved progressive ablation report to: results/ablation_mobilenetv3_small_256.json")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EcoSort-Edge: Progressive Resizing 256x256")
    parser.add_argument("--epochs", type=int, default=10, help="Fine-tuning epochs (default: 10)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate (default: 5e-5)")
    parser.add_argument("--patience", type=int, default=4, help="Early stopping patience (default: 4)")
    parser.add_argument("--resume", action="store_true", help="Resume from last 256 checkpoint")
    args = parser.parse_args()
    
    run_progressive_256(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr_head=args.lr,
        patience=args.patience,
        resume=args.resume
    )
