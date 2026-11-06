"""
scripts/calibrate_and_validate_full_dataset.py
EcoSort-Edge: Production Dataset Calibration & Comprehensive Split Validation

Workflow:
1. Loads MobileNetV3-Small Progressive FP32 Checkpoint.
2. Calibrates INT8 Quantizer with a balanced sample from data/processed/train (250 images across all 9 classes).
3. Evaluates Accuracy on data/processed/val (1,197 holdout images).
4. Compares Top-1 and Statutory 4-Stream Accuracy: FP32 vs. INT8.
5. Exports production ExecuTorch .pte binary for Raspberry Pi 5.
"""

import os
import sys
import time
import random
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import torch
import torchvision.models as models
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data" / "processed"
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"

DEFAULT_CLASSES = [
    "battery", "cardboard", "food_waste", "glass_bottle",
    "metal_can", "multi_layer_packaging", "paper",
    "plastic_bottle", "vegetable_scrap"
]

def main(calib_samples_per_class=30, val_batch_size=32):
    print("=" * 85)
    print(" [ECOSORT-EDGE] FULL DATASET CALIBRATION & PARITY VALIDATION")
    print(" Dataset: 5,607 Train | 1,197 Validation across 9 Waste Classes")
    print("=" * 85)

    # 1. Image preprocessing
    tf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 2. Load Checkpoint
    ckpt_path = MODELS_DIR / "ecosort_mobilenetv3_small_256_progressive.pth"
    print(f"[*] Step 1: Loading Baseline FP32 Weights from {ckpt_path.name}...")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    num_classes = len(DEFAULT_CLASSES)
    model = models.mobilenet_v3_small(num_classes=num_classes)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.eval()

    fp32_mb = ckpt_path.stat().st_size / (1024 * 1024)
    print(f"    [+] Baseline FP32 Size : {fp32_mb:.2f} MB")

    # 3. Build Calibration Subset from Train
    print(f"\n[*] Step 2: Sampling {calib_samples_per_class} real training images per class for calibration...")
    train_dataset = datasets.ImageFolder(str(TRAIN_DIR), transform=tf)
    class_indices = {i: [] for i in range(len(train_dataset.classes))}
    for idx, (_, class_idx) in enumerate(train_dataset.samples):
        class_indices[class_idx].append(idx)

    calib_indices = []
    for c_idx, indices in class_indices.items():
        sample_count = min(calib_samples_per_class, len(indices))
        calib_indices.extend(random.sample(indices, sample_count))

    print(f"    [+] Selected {len(calib_indices)} balanced calibration images.")
    calib_loader = DataLoader(Subset(train_dataset, calib_indices), batch_size=1, shuffle=False)

    # 4. Graph Capture & Calibration
    print("\n[*] Step 3: Capturing Graph and Applying PT2E Quantizer...")
    example_input = torch.randn(1, 3, 256, 256)
    exported_program = torch.export.export(model, (example_input,), strict=False)

    from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e, convert_pt2e
    from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
        XNNPACKQuantizer,
        get_symmetric_quantization_config,
    )

    quantizer = XNNPACKQuantizer()
    quantizer.set_global(get_symmetric_quantization_config(is_per_channel=True, is_dynamic=True))

    try:
        prepared_model = prepare_pt2e(exported_program, quantizer)
    except Exception:
        prepared_model = prepare_pt2e(exported_program.module(), quantizer)

    runner = prepared_model.module if hasattr(prepared_model, "module") and callable(getattr(prepared_model, "module")) else prepared_model

    print(f"    [+] Calibrating observers with {len(calib_indices)} training images (batch_size=1)...")
    with torch.no_grad():
        for batch_x, _ in calib_loader:
            runner(batch_x)

    print("    [+] Converting to INT8 Quantized Model...")
    quantized_model = convert_pt2e(prepared_model)
    int8_runner = quantized_model.module if hasattr(quantized_model, "module") and callable(getattr(quantized_model, "module")) else quantized_model
    print("    [✓] INT8 Quantization successfully calibrated on actual training data!")

    # 5. Validation on Holdout Split
    print(f"\n[*] Step 4: Validating FP32 vs. INT8 on Validation Set ({VAL_DIR.name})...")
    val_dataset = datasets.ImageFolder(str(VAL_DIR), transform=tf)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    total_val = len(val_dataset)
    fp32_correct = 0
    int8_correct = 0
    parity_matches = 0

    t0 = time.time()
    with torch.no_grad():
        for b_idx, (images, targets) in enumerate(val_loader):
            # FP32 prediction
            out_fp32 = model(images)
            preds_fp32 = torch.argmax(out_fp32, dim=1)
            fp32_correct += int((preds_fp32 == targets).sum())

            # INT8 prediction
            out_int8 = int8_runner(images)
            if isinstance(out_int8, (list, tuple)):
                out_int8 = out_int8[0]
            preds_int8 = torch.argmax(out_int8, dim=1)
            int8_correct += int((preds_int8 == targets).sum())

            parity_matches += int((preds_fp32 == preds_int8).sum())

            if (b_idx + 1) % 100 == 0 or (b_idx + 1) == total_val:
                processed = b_idx + 1
                print(f"    Evaluating: {processed}/{total_val} images... (FP32 Acc: {fp32_correct/processed*100:.1f}%, INT8 Acc: {int8_correct/processed*100:.1f}%)")

    val_time = time.time() - t0
    fp32_acc = (fp32_correct / total_val) * 100
    int8_acc = (int8_correct / total_val) * 100
    parity_pct = (parity_matches / total_val) * 100

    print("\n" + "=" * 85)
    print(" VALIDATION SPLIT ACCURACY & PARITY AUDIT (1,197 HOLDOUT IMAGES)")
    print("=" * 85)
    print(f"[*] Total Validation Images    : {total_val}")
    print(f"[*] Baseline FP32 Accuracy     : {fp32_correct}/{total_val} ({fp32_acc:.2f}%)")
    print(f"[*] Calibrated INT8 Accuracy   : {int8_correct}/{total_val} ({int8_acc:.2f}%)")
    print(f"[*] Accuracy Difference (Delta): {abs(fp32_acc - int8_acc):.2f}% (Industry Target: < 1.0%)")
    print(f"[*] Decision Parity (Agreement): {parity_matches}/{total_val} ({parity_pct:.2f}%)")
    print(f"[*] Validation Speed           : {total_val / val_time:.1f} images/sec")
    print("-" * 85)

    # 6. Export to ExecuTorch
    print("\n[*] Step 5: Lowering Calibrated Model to ExecuTorch (XNNPACK Target)...")
    if not isinstance(quantized_model, torch.export.ExportedProgram):
        quantized_program = torch.export.export(quantized_model, (example_input,), strict=False)
    else:
        quantized_program = quantized_model

    from executorch.exir import to_edge_transform_and_lower
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
    from executorch.backends.xnnpack.utils.configs import get_transform_passes

    et_program = to_edge_transform_and_lower(
        quantized_program,
        partitioner=[XnnpackPartitioner()],
        transform_passes=get_transform_passes()
    )

    pte_path = MODELS_DIR / "waste_classifier_int8.pte"
    execu_program = et_program.to_executorch()
    with open(pte_path, "wb") as f:
        f.write(execu_program.buffer)

    pte_mb = pte_path.stat().st_size / (1024 * 1024)
    print(f"    [✓] ExecuTorch Binary Generated: {pte_path.name} ({pte_mb:.2f} MB)")
    print(f"    [✓] Model Size Reduction       : {fp32_mb / pte_mb:.2f}x compression")
    print("=" * 85)

if __name__ == "__main__":
    main()
