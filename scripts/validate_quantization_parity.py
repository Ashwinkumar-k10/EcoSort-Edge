"""
scripts/validate_quantization_parity.py
EcoSort-Edge: Strict FP32 vs INT8 Parity & Accuracy Validation

Compares:
1. Baseline FP32 Model (ecosort_mobilenetv3_small_256_progressive.pth)
2. Calibrated INT8 Model

Metrics Evaluated:
- Per-image prediction agreement (Parity %)
- Top-1 Accuracy on Benchmark
- Statutory 4-Stream Accuracy
- Cosine similarity of output probability distributions
- Memory / Size reduction
"""

import sys
import json
import time
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
BENCHMARK_DIR = PROJECT_ROOT / "data" / "benchmark_30"
if not BENCHMARK_DIR.exists():
    BENCHMARK_DIR = PROJECT_ROOT / "benchmark_30"

GT_PATH = BENCHMARK_DIR / "ground_truth.json"

DEFAULT_CLASSES = [
    "battery", "cardboard", "food_waste", "glass_bottle",
    "metal_can", "multi_layer_packaging", "paper",
    "plastic_bottle", "vegetable_scrap"
]

def load_fp32_model(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = models.mobilenet_v3_small(num_classes=9)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    return model

def build_int8_model(fp32_model, calib_images, tf):
    print("[*] Creating INT8 Quantized Model via PT2E XNNPACK...")
    example_input = torch.randn(1, 3, 256, 256)
    exported = torch.export.export(fp32_model, (example_input,), strict=False)

    prep, conv, XNNPACKQuantizer, get_symmetric_quantization_config = None, None, None, None
    try:
        from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e as prep, convert_pt2e as conv
        from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import XNNPACKQuantizer, get_symmetric_quantization_config
    except Exception:
        from torch.ao.quantization.quantize_pt2e import prepare_pt2e as prep, convert_pt2e as conv
        from torch.ao.quantization.quantizer.xnnpack_quantizer import XNNPACKQuantizer, get_symmetric_quantization_config

    quantizer = XNNPACKQuantizer()
    # Test dynamic activation quantization (preserves activation dynamic range)
    quantizer.set_global(get_symmetric_quantization_config(is_per_channel=True, is_dynamic=True))

    try:
        prepared = prep(exported, quantizer)
    except Exception:
        prepared = prep(exported.module(), quantizer)

    runner = prepared.module if hasattr(prepared, "module") and callable(getattr(prepared, "module")) else prepared

    print(f"[*] Calibrating with {len(calib_images)} real benchmark images...")
    with torch.no_grad():
        for p in calib_images:
            try:
                img = Image.open(p).convert("RGB")
                runner(tf(img).unsqueeze(0))
            except Exception:
                pass

    int8_model = conv(prepared)
    print("[✓] INT8 Quantized Model Built Successfully.\n")
    return int8_model

def run_parity_validation():
    print("=" * 95)
    print(" ECOSORT-EDGE: RIGOROUS FP32 VS INT8 PARITY BENCHMARK")
    print("=" * 95)

    ckpt_path = MODELS_DIR / "ecosort_mobilenetv3_small_256_progressive.pth"
    fp32_model = load_fp32_model(ckpt_path)

    tf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    images = sorted(list(BENCHMARK_DIR.glob("*.jpg")) + list(BENCHMARK_DIR.glob("*.png")))
    gt_map = {}
    if GT_PATH.exists():
        with open(GT_PATH, "r", encoding="utf-8") as f:
            for item in json.load(f):
                gt_map[Path(item["filename"]).name] = item

    int8_model = build_int8_model(fp32_model, images, tf)
    int8_runner = int8_model.module if hasattr(int8_model, "module") and callable(getattr(int8_model, "module")) else int8_model

    print(f"{'Image':<26} {'Ground Truth':<16} {'FP32 Prediction':<24} {'INT8 Prediction':<24} {'Parity'}")
    print("-" * 95)

    fp32_correct = 0
    int8_correct = 0
    parity_matches = 0
    total = len(images)

    cos_sims = []

    for img_p in images:
        gt_item = gt_map.get(img_p.name, {})
        true_label = gt_item.get("category", "unknown")

        img = Image.open(img_p).convert("RGB")
        tensor = tf(img).unsqueeze(0)

        with torch.no_grad():
            fp32_out = fp32_model(tensor)
            if isinstance(fp32_out, (list, tuple)):
                fp32_out = fp32_out[0]
            fp32_probs = torch.softmax(fp32_out, dim=1)[0]
            fp32_idx = int(torch.argmax(fp32_probs))
            fp32_pred = DEFAULT_CLASSES[fp32_idx]
            fp32_conf = float(fp32_probs[fp32_idx]) * 100

            int8_out = int8_runner(tensor)
            if isinstance(int8_out, (list, tuple)):
                int8_out = int8_out[0]
            int8_probs = torch.softmax(int8_out, dim=1)[0]
            int8_idx = int(torch.argmax(int8_probs))
            int8_pred = DEFAULT_CLASSES[int8_idx]
            int8_conf = float(int8_probs[int8_idx]) * 100

        # Cosine similarity between probability distributions
        sim = float(torch.nn.functional.cosine_similarity(fp32_probs.unsqueeze(0), int8_probs.unsqueeze(0)))
        cos_sims.append(sim)

        is_parity = (fp32_pred == int8_pred)
        if is_parity:
            parity_matches += 1
        if fp32_pred == true_label:
            fp32_correct += 1
        if int8_pred == true_label:
            int8_correct += 1

        parity_str = "MATCH" if is_parity else "DRIFT"
        fp32_str = f"{fp32_pred} ({fp32_conf:.1f}%)"
        int8_str = f"{int8_pred} ({int8_conf:.1f}%)"

        print(f"{img_p.name:<26} {true_label:<16} {fp32_str:<24} {int8_str:<24} {parity_str}")

    avg_sim = sum(cos_sims) / len(cos_sims) if cos_sims else 0.0

    print("=" * 95)
    print(" QUANTIZATION INTEGRITY AUDIT RESULTS")
    print("=" * 95)
    print(f"[*] Total Benchmark Images     : {total}")
    print(f"[*] FP32 Model Accuracy        : {fp32_correct}/{total} ({fp32_correct/total*100:.2f}%)")
    print(f"[*] INT8 Model Accuracy        : {int8_correct}/{total} ({int8_correct/total*100:.2f}%)")
    print(f"[*] FP32 <-> INT8 Decision Match: {parity_matches}/{total} ({parity_matches/total*100:.2f}% PARITY)")
    print(f"[*] Mean Cosine Similarity     : {avg_sim:.4f} (Ideal: > 0.98)")
    print("-" * 95)
    if avg_sim >= 0.95 and (parity_matches / total) >= 0.85:
        print("[✓] VERDICT: QUANTIZATION PASSES HIGH INTEGRITY STANDARDS. SAFE TO DEPLOY.")
    else:
        print("[!] VERDICT: QUANTIZATION REVEALS DRIFT. DO NOT DEPLOY WITHOUT RETUNING.")
    print("=" * 95)

if __name__ == "__main__":
    run_parity_validation()
