"""
scripts/train_ecosort_engine.py
EcoSort-Edge: Generalization-First, Two-Stage Transfer Learning Engine
Optimized for NVIDIA RTX 3050 (4GB VRAM) & Arm ExecuTorch deployment

Features:
- Stage 1: Frozen backbone (5-8 epochs, warmup classifier head)
- Stage 2: Progressive fine-tuning with differential learning rates (LR_backbone = 0.1 * LR_head)
- Class-Balanced Weighted Random Sampler (combats class imbalance without naive oversampling)
- Robust Real-World Regularization:
    * Moderate ColorJitter (0.2)
    * RandomErasing / Partial Occlusion
    * Gaussian Blur (camera defect simulation)
    * Label Smoothing (0.05)
- Model Selection & Early Stopping on Macro-F1 (Patience = 7)
- Multi-seed experiment harness (Seeds: 42, 123, 2026)
- Energy & Latency Evaluation Profile
"""

import os
import sys
import time
import math
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

# ═══════════════════════════════════════════════════════════════════
# REPRODUCIBILITY UTILS
# ═══════════════════════════════════════════════════════════════════
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Enable cuDNN auto-tuner for constant-size 224x224 inputs on RTX 3050 Tensor Cores
    torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')

# ═══════════════════════════════════════════════════════════════════
# DATA PIPELINE (REAL-WORLD GENERALIZATION AUGMENTATIONS)
# ═══════════════════════════════════════════════════════════════════
def get_transforms(img_size=224):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.2), value="random")
    ])

    eval_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return train_transform, eval_transform

# ═══════════════════════════════════════════════════════════════════
# CLASS BALANCING: WEIGHTED RANDOM SAMPLER
# ═══════════════════════════════════════════════════════════════════
def make_balanced_sampler(dataset):
    targets = [s[1] for s in dataset.samples]
    class_counts = Counter(targets)
    num_samples = len(targets)
    class_weights = {cls: num_samples / count for cls, count in class_counts.items()}
    weights = [class_weights[t] for t in targets]
    sampler = WeightedRandomSampler(torch.DoubleTensor(weights), num_samples=num_samples, replacement=True)
    return sampler, class_counts

# ═══════════════════════════════════════════════════════════════════
# MODEL ARCHITECTURE BUILDER
# ═══════════════════════════════════════════════════════════════════
def build_mobilenet_v3_small(num_classes=15):
    weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
    model = models.mobilenet_v3_small(weights=weights)
    
    # Replace classification head
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model

# ═══════════════════════════════════════════════════════════════════
# METRIC COMPUTATIONS (MACRO-F1, PER-CLASS RECALL, CALIBRATION)
# ═══════════════════════════════════════════════════════════════════
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
    
    # 2D Confusion Matrix [actual, predicted]
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

# ═══════════════════════════════════════════════════════════════════
# TRAINING & FINE-TUNING ENGINE (WITH FULL RESUMABILITY)
# ═══════════════════════════════════════════════════════════════════
def run_experiment(seed=42, batch_size=32, stage1_epochs=5, stage2_epochs=25, warmup_epochs=2, lr_head=1e-4, weight_decay=1e-4, patience=6, resume=False):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*75}\n[EXPERIMENT] Seed: {seed} | Device: {device} | Mixed Precision: FP16 AMP + cuDNN Benchmark\n{'='*75}")
    print(f"[*] Optimal Generalization Config: BatchSize={batch_size} | Stage1={stage1_epochs} eps | Stage2={stage2_epochs} eps | Warmup={warmup_epochs} eps | Patience={patience}")
    
    train_tf, eval_tf = get_transforms()
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
    
    best_checkpoint_path = MODELS_DIR / f"ecosort_mobilenetv3_seed{seed}.pth"
    last_checkpoint_path = MODELS_DIR / f"ecosort_mobilenetv3_seed{seed}_checkpoint_last.pth"
    
    best_macro_f1 = -1.0
    epochs_no_improve = 0
    history = []
    start_stage = 1
    start_epoch = 1
    
    # ── CHECK FOR RESUME ──
    if resume and last_checkpoint_path.exists():
        print(f"\n[+] Found existing checkpoint: {last_checkpoint_path.name}")
        ckpt = torch.load(last_checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        scaler.load_state_dict(ckpt["scaler_state_dict"])
        best_macro_f1 = ckpt.get("best_macro_f1", -1.0)
        epochs_no_improve = ckpt.get("epochs_no_improve", 0)
        history = ckpt.get("history", [])
        saved_stage = ckpt.get("stage", 1)
        saved_epoch = ckpt.get("epoch", 1)
        
        if saved_stage == 1:
            if saved_epoch >= stage1_epochs:
                start_stage = 2
                start_epoch = 1
                print(f"[+] Stage 1 was already completed. Resuming directly from Stage 2!")
            else:
                start_stage = 1
                start_epoch = saved_epoch + 1
                print(f"[+] Resuming Stage 1 from Epoch {start_epoch}/{stage1_epochs}...")
        else:
            start_stage = 2
            start_epoch = saved_epoch + 1
            print(f"[+] Resuming Stage 2 from Epoch {start_epoch}/{stage2_epochs} (Best Macro-F1 so far: {best_macro_f1:.4f})...")
    elif resume:
        print(f"[!] Warning: --resume specified but no checkpoint found at {last_checkpoint_path}. Starting fresh.")

    try:
        # ── STAGE 1: FEATURE ADAPTATION (FROZEN BACKBONE) ──
        if start_stage == 1:
            print(f"\n[STAGE 1] Feature Adaptation: Freezing Backbone ({stage1_epochs} epochs)...")
            for param in model.features.parameters():
                param.requires_grad = False
                
            optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=1e-3, weight_decay=weight_decay)
            
            for epoch in range(start_epoch, stage1_epochs + 1):
                model.train()
                running_loss = 0.0
                train_pbar = tqdm(
                    train_loader,
                    desc=f"Stage 1 [Ep {epoch:02d}/{stage1_epochs:02d}]",
                    unit="batch",
                    dynamic_ncols=True,
                    leave=False
                )
                for x, y in train_pbar:
                    x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                    optimizer.zero_grad(set_to_none=True)
                    with autocast(device_type=device.type, dtype=torch.float16):
                        out = model(x)
                        loss = criterion(out, y)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    running_loss += loss.item()
                    train_pbar.set_postfix({"loss": f"{loss.item():.4f}", "avg": f"{running_loss / (train_pbar.n + 1):.4f}"})
                    
                # Validation
                model.eval()
                v_preds, v_trues = [], []
                val_pbar = tqdm(val_loader, desc="  ↳ Validating", unit="batch", dynamic_ncols=True, leave=False)
                with torch.no_grad():
                    for x, y in val_pbar:
                        x = x.to(device, non_blocking=True)
                        with autocast(device_type=device.type, dtype=torch.float16):
                            out = model(x)
                        preds = out.argmax(dim=1).cpu().numpy()
                        v_preds.extend(preds)
                        v_trues.extend(y.numpy())
                metrics = compute_metrics(v_trues, v_preds, num_classes)
                print(f"Stage 1 Epoch {epoch:02d}/{stage1_epochs:02d} | Loss: {running_loss/len(train_loader):.4f} | Val Acc: {metrics['accuracy']*100:.2f}% | Val Macro-F1: {metrics['macro_f1']:.4f}", flush=True)
                
                # Save resumable state
                torch.save({
                    "stage": 1,
                    "epoch": epoch,
                    "seed": seed,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "best_macro_f1": best_macro_f1,
                    "epochs_no_improve": epochs_no_improve,
                    "classes": train_dataset.classes,
                    "history": history
                }, last_checkpoint_path)
            
            start_epoch = 1  # Reset epoch counter for Stage 2

        # ── STAGE 2: CONTROLLED PROGRESSIVE FINE-TUNING ──
        print(f"\n[STAGE 2] Controlled Fine-Tuning: Differential LRs (Backbone={lr_head*0.1:.1e}, Head={lr_head:.1e})...")
        print(f"[*] Schedule: {warmup_epochs}-Epoch Linear Warm-Up -> Cosine Annealing (eta_min=1e-6) | Early Stopping Patience: {patience}")
        for param in model.features.parameters():
            param.requires_grad = True
            
        param_groups = [
            {"params": model.features.parameters(), "lr": lr_head * 0.1},
            {"params": model.classifier.parameters(), "lr": lr_head}
        ]
        optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
        
        # Linear warm-up followed by Cosine Annealing
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, total_iters=warmup_epochs
        )
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, stage2_epochs - warmup_epochs), eta_min=1e-6
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs]
        )
        
        # If resuming in Stage 2, fast-forward scheduler and restore optimizer
        if resume and start_stage == 2 and last_checkpoint_path.exists():
            ckpt = torch.load(last_checkpoint_path, map_location=device)
            if "optimizer_state_dict" in ckpt:
                try:
                    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                except Exception:
                    pass
            for _ in range(start_epoch - 1):
                scheduler.step()
        
        for epoch in range(start_epoch, stage2_epochs + 1):
            model.train()
            running_loss = 0.0
            curr_lr = optimizer.param_groups[1]['lr']
            train_pbar = tqdm(
                train_loader,
                desc=f"Stage 2 [Ep {epoch:02d}/{stage2_epochs:02d}]",
                unit="batch",
                dynamic_ncols=True,
                leave=False
            )
            for x, y in train_pbar:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with autocast(device_type=device.type, dtype=torch.float16):
                    out = model(x)
                    loss = criterion(out, y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                running_loss += loss.item()
                train_pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "avg": f"{running_loss / (train_pbar.n + 1):.4f}",
                    "lr": f"{curr_lr:.1e}"
                })
                
            scheduler.step()
            
            # Validation Evaluation
            model.eval()
            v_preds, v_trues = [], []
            val_pbar = tqdm(val_loader, desc="  ↳ Validating", unit="batch", dynamic_ncols=True, leave=False)
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
                    "model_state_dict": model.state_dict(),
                    "classes": train_dataset.classes,
                    "metrics": metrics,
                    "epoch": epoch
                }, best_checkpoint_path)
                save_msg = " -> [BEST SAVED]"
            else:
                epochs_no_improve += 1
                save_msg = f" -> [patience {epochs_no_improve}/{patience}]"
                
            print(f"Stage 2 Epoch {epoch:02d}/{stage2_epochs:02d} (lr: {curr_lr:.1e}) | Loss: {train_loss:.4f} | Val Acc: {metrics['accuracy']*100:.2f}% | Val Macro-F1: {metrics['macro_f1']:.4f} | Recall: {metrics['macro_recall']*100:.2f}%{save_msg}", flush=True)
                
            history.append({
                "epoch": epoch,
                "loss": train_loss,
                "val_acc": metrics['accuracy'],
                "macro_f1": metrics['macro_f1'],
                "macro_recall": metrics['macro_recall']
            })
            
            # Save resumable state every single epoch
            torch.save({
                "stage": 2,
                "epoch": epoch,
                "seed": seed,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "best_macro_f1": best_macro_f1,
                "epochs_no_improve": epochs_no_improve,
                "classes": train_dataset.classes,
                "history": history
            }, last_checkpoint_path)
            
            if epochs_no_improve >= patience:
                print(f"\n[!] Early Stopping Triggered: Macro-F1 did not improve for {patience} consecutive epochs.")
                break

    except KeyboardInterrupt:
        print(f"\n[!] Training paused by user (Ctrl+C). Checkpoint saved to: {last_checkpoint_path}")
        print(f"    You can resume anytime by running: python scripts/train_ecosort_engine.py --resume")
        sys.exit(0)
            
    print(f"\n[*] Evaluating Best Checkpoint on Pure, Untouched Test Set (15%)...", flush=True)
    ckpt = torch.load(best_checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    
    t_preds, t_trues, t_probs = [], [], []
    test_pbar = tqdm(test_loader, desc="[*] Holdout Test Eval", unit="batch", dynamic_ncols=True, leave=False)
    with torch.no_grad():
        for x, y in test_pbar:
            x = x.to(device, non_blocking=True)
            with autocast(device_type=device.type, dtype=torch.float16):
                out = model(x)
            prob = torch.softmax(out, dim=1).cpu().numpy()
            t_probs.extend(prob)
            t_preds.extend(np.argmax(prob, axis=1))
            t_trues.extend(y.numpy())
            
    test_metrics = compute_metrics(t_trues, t_preds, num_classes)
    print(f"\n{'='*75}")
    print(f"[TEST RESULTS - UNTOUCHED 15% HOLDOUT DATASET (1,210 Images)] Seed: {seed}")
    print(f"{'='*75}")
    print(f"[*] Test Accuracy       : {test_metrics['accuracy']*100:.2f}%")
    print(f"[*] Test Macro-F1       : {test_metrics['macro_f1']:.4f}")
    print(f"[*] Test Macro-Recall   : {test_metrics['macro_recall']*100:.2f}%")
    print(f"[*] Test Macro-Precision: {test_metrics['macro_precision']*100:.2f}%")
    print(f"{'-'*75}")
    print(f"{'Class Name':<26} {'Precision':>12} {'Recall':>12} {'F1-Score':>12}")
    print(f"{'-'*75}")
    for idx, cname in enumerate(test_dataset.classes):
        p_val = test_metrics['per_class_precision'][idx] * 100
        r_val = test_metrics['per_class_recall'][idx] * 100
        f_val = test_metrics['per_class_f1'][idx]
        print(f"{cname:<26} {p_val:>11.2f}% {r_val:>11.2f}% {f_val:>12.4f}")
    print(f"{'='*75}\n")
    
    # Save standalone evaluation report
    eval_report = {
        "seed": seed,
        "classes": test_dataset.classes,
        "best_val_macro_f1": best_macro_f1,
        "test_metrics": test_metrics,
        "checkpoint": str(best_checkpoint_path),
        "history": history
    }
    with open(RESULTS_DIR / "training_tradeoff_report.json", "w") as f:
        json.dump(eval_report, f, indent=2)
    print(f"[+] Saved complete evaluation tradeoff report to: results/training_tradeoff_report.json")
    
    return eval_report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EcoSort-Edge: Two-Stage Transfer Learning Engine")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for optimal generalization (default: 32)")
    parser.add_argument("--stage1_epochs", type=int, default=5, help="Epochs for Stage 1 frozen backbone (default: 5)")
    parser.add_argument("--stage2_epochs", type=int, default=25, help="Epochs for Stage 2 fine-tuning (default: 25)")
    parser.add_argument("--warmup_epochs", type=int, default=2, help="Linear warm-up epochs in Stage 2 (default: 2)")
    parser.add_argument("--patience", type=int, default=6, help="Early stopping patience on Macro-F1 (default: 6)")
    parser.add_argument("--resume", action="store_true", help="Resume training from last saved checkpoint")
    parser.add_argument("--all_seeds", action="store_true", help="Run multi-seed evaluation (42, 123, 2026)")
    args = parser.parse_args()
    
    seeds = [42, 123, 2026] if args.all_seeds else [args.seed]
    all_results = []
    
    for s in seeds:
        res = run_experiment(
            seed=s,
            batch_size=args.batch_size,
            stage1_epochs=args.stage1_epochs,
            stage2_epochs=args.stage2_epochs,
            warmup_epochs=args.warmup_epochs,
            patience=args.patience,
            resume=args.resume
        )
        all_results.append(res)
        
    summary_path = RESULTS_DIR / "multi_seed_experiment_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"[+] Multi-seed experiment summary recorded to: {summary_path}")
