"""
scripts/train_mobilenetv3_large_256.py
EcoSort-Edge: Raspberry Pi 5 High-Capacity Champion Model
Trains MobileNetV3-Large at 256x256 resolution with Squeeze-and-Excitation attention.
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

def build_mobilenet_v3_large(num_classes=9):
    weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V1
    model = models.mobilenet_v3_large(weights=weights)
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

def run_experiment_v3_large(stage1_epochs=3, stage2_epochs=20, warmup_epochs=2, batch_size=32, lr_head=1e-4, patience=5, seed=42, resume=False):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*75}")
    print(f"[ECOSORT-EDGE] TRAINING PI 5 CHAMPION: MobileNetV3-Large @ 256x256")
    print(f"[*] Model Capacity: 5.48M params (Full Squeeze-and-Excitation Attention)")
    print(f"[*] Target Hardware: Raspberry Pi 5 (Quad Cortex-A76 @ 2.4 GHz)")
    print(f"[*] Device: {device} | Mixed Precision: FP16 AMP + cuDNN Benchmark")
    print(f"[*] Config: Stage1={stage1_epochs} eps | Stage2={stage2_epochs} eps | BatchSize={batch_size} | Patience={patience}")
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
    
    model = build_mobilenet_v3_large(num_classes).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    scaler = GradScaler(enabled=(device.type == "cuda"))
    
    best_path = MODELS_DIR / "ecosort_mobilenetv3_large_256_champion.pth"
    last_path = MODELS_DIR / "ecosort_mobilenetv3_large_256_checkpoint_last.pth"
    
    best_macro_f1 = -1.0
    epochs_no_improve = 0
    history = []
    start_stage = 1
    start_epoch = 1
    
    if resume and last_path.exists():
        print(f"\n[+] Found existing checkpoint: {last_path.name}")
        ckpt = torch.load(last_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        scaler.load_state_dict(ckpt["scaler_state_dict"])
        best_macro_f1 = ckpt.get("best_macro_f1", -1.0)
        epochs_no_improve = ckpt.get("epochs_no_improve", 0)
        history = ckpt.get("history", [])
        saved_stage = ckpt.get("stage", 1)
        saved_epoch = ckpt.get("epoch", 1)
        
        if saved_stage == 1 and saved_epoch >= stage1_epochs:
            start_stage = 2
            start_epoch = 1
            print(f"[+] Stage 1 already complete. Resuming directly from Stage 2!")
        elif saved_stage == 1:
            start_stage = 1
            start_epoch = saved_epoch + 1
            print(f"[+] Resuming Stage 1 from Epoch {start_epoch}/{stage1_epochs}...")
        else:
            start_stage = 2
            start_epoch = saved_epoch + 1
            print(f"[+] Resuming Stage 2 from Epoch {start_epoch}/{stage2_epochs} (Best Macro-F1: {best_macro_f1:.4f})...")

    try:
        # ── STAGE 1: FEATURE ADAPTATION (FROZEN BACKBONE) ──
        if start_stage == 1:
            print(f"\n[STAGE 1] Feature Adaptation: Freezing Backbone ({stage1_epochs} epochs)...")
            for param in model.features.parameters():
                param.requires_grad = False
                
            optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=1e-3, weight_decay=1e-4)
            
            for epoch in range(start_epoch, stage1_epochs + 1):
                model.train()
                running_loss = 0.0
                pbar = tqdm(train_loader, desc=f"Large Stg1 [Ep {epoch:02d}/{stage1_epochs:02d}]", unit="batch", dynamic_ncols=True, leave=False)
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
                    pbar.set_postfix({"loss": f"{loss.item():.4f}", "avg": f"{running_loss/(pbar.n+1):.4f}"})
                    
                # Validation
                model.eval()
                v_preds, v_trues = [], []
                val_pbar = tqdm(val_loader, desc="  ↳ Validating Stg1", unit="batch", dynamic_ncols=True, leave=False)
                with torch.no_grad():
                    for x, y in val_pbar:
                        x = x.to(device, non_blocking=True)
                        with autocast(device_type=device.type, dtype=torch.float16):
                            out = model(x)
                        v_preds.extend(out.argmax(dim=1).cpu().numpy())
                        v_trues.extend(y.numpy())
                metrics = compute_metrics(v_trues, v_preds, num_classes)
                print(f"Large Stage 1 Epoch {epoch:02d}/{stage1_epochs:02d} | Loss: {running_loss/len(train_loader):.4f} | Val Acc: {metrics['accuracy']*100:.2f}% | Val Macro-F1: {metrics['macro_f1']:.4f}", flush=True)
                
                torch.save({
                    "stage": 1,
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "best_macro_f1": best_macro_f1,
                    "epochs_no_improve": epochs_no_improve,
                    "history": history
                }, last_path)
            
            start_epoch = 1

        # ── STAGE 2: CONTROLLED PROGRESSIVE FINE-TUNING ──
        print(f"\n[STAGE 2] Controlled Fine-Tuning: Differential LRs (Backbone={lr_head*0.1:.1e}, Head={lr_head:.1e})...")
        print(f"[*] Schedule: {warmup_epochs}-Epoch Linear Warm-Up -> Cosine Annealing (eta_min=1e-6) | Early Stopping Patience: {patience}")
        for param in model.features.parameters():
            param.requires_grad = True
            
        param_groups = [
            {"params": model.features.parameters(), "lr": lr_head * 0.1},
            {"params": model.classifier.parameters(), "lr": lr_head}
        ]
        optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, stage2_epochs - warmup_epochs), eta_min=1e-6)
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])
        
        if resume and start_stage == 2 and last_path.exists():
            ckpt = torch.load(last_path, map_location=device)
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
            pbar = tqdm(train_loader, desc=f"Large Stg2 [Ep {epoch:02d}/{stage2_epochs:02d}]", unit="batch", dynamic_ncols=True, leave=False)
            
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
            val_pbar = tqdm(val_loader, desc="  ↳ Validating Stg2", unit="batch", dynamic_ncols=True, leave=False)
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
                }, best_path)
                save_msg = " -> [BEST CHAMPION SAVED]"
            else:
                epochs_no_improve += 1
                save_msg = f" -> [patience {epochs_no_improve}/{patience}]"
                
            print(f"Large Stg2 Epoch {epoch:02d}/{stage2_epochs:02d} (lr: {curr_lr:.1e}) | Loss: {train_loss:.4f} | Val Acc: {metrics['accuracy']*100:.2f}% | Val Macro-F1: {metrics['macro_f1']:.4f} | Recall: {metrics['macro_recall']*100:.2f}%{save_msg}", flush=True)
            
            history.append({
                "epoch": epoch,
                "loss": train_loss,
                "val_acc": metrics['accuracy'],
                "macro_f1": metrics['macro_f1'],
                "macro_recall": metrics['macro_recall']
            })
            
            torch.save({
                "stage": 2,
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "best_macro_f1": best_macro_f1,
                "epochs_no_improve": epochs_no_improve,
                "history": history
            }, last_path)
            
            if epochs_no_improve >= patience:
                print(f"\n[!] Early Stopping Triggered: Macro-F1 did not improve for {patience} consecutive epochs.")
                break
                
    except KeyboardInterrupt:
        print(f"\n[!] Paused by user. Checkpoint saved to: {last_path}")
        sys.exit(0)
        
    print(f"\n{'='*75}")
    print(f"[EVALUATION] Benchmarking Best MobileNetV3-Large on Untouched Test Set (1,210 Images)...")
    print(f"{'='*75}", flush=True)
    
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    
    t_preds, t_trues = [], []
    test_pbar = tqdm(test_loader, desc="[*] Holdout Test Eval Large", unit="batch", dynamic_ncols=True, leave=False)
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
    print(f"[FINAL TEST RESULTS - PI 5 CHAMPION: MOBILENETV3-LARGE @ 256x256]")
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
        "model_name": "MobileNetV3-Large",
        "resolution": "256x256",
        "parameters": "5.48M",
        "target_hardware": "Raspberry Pi 5 (Quad Cortex-A76 @ 2.4 GHz)",
        "best_val_macro_f1": best_macro_f1,
        "test_metrics": test_metrics,
        "statutory_4_stream_accuracy": statutory_acc,
        "checkpoint": str(best_path),
        "history": history
    }
    with open(RESULTS_DIR / "ablation_mobilenetv3_large_256.json", "w") as f:
        json.dump(report_data, f, indent=2)
    print(f"[+] Saved Pi 5 Champion ablation report to: results/ablation_mobilenetv3_large_256.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EcoSort-Edge: MobileNetV3-Large @ 256x256 (Pi 5 Champion)")
    parser.add_argument("--stage1_epochs", type=int, default=3, help="Stage 1 frozen epochs (default: 3)")
    parser.add_argument("--stage2_epochs", type=int, default=20, help="Stage 2 fine-tuning epochs (default: 20)")
    parser.add_argument("--warmup_epochs", type=int, default=2, help="Linear warm-up epochs (default: 2)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Classifier head learning rate (default: 1e-4)")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience (default: 5)")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    args = parser.parse_args()
    
    run_experiment_v3_large(
        stage1_epochs=args.stage1_epochs,
        stage2_epochs=args.stage2_epochs,
        warmup_epochs=args.warmup_epochs,
        batch_size=args.batch_size,
        lr_head=args.lr,
        patience=args.patience,
        resume=args.resume
    )
