"""
EcoSort-Edge: Terminal Prediction & Live Classifier
Run single-image inference, interactive drag-and-drop, or live webcam classification.
"""

import os
import sys
import time
import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# Ensure UTF-8 output
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import torch
import torch.nn as nn

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

from torchvision import models, transforms
from scripts.decision_engine import ConfidenceAwareDecisionEngine
from scripts.llm_explainer import CognitiveExplanationEngine

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

def load_model(device, checkpoint_path: str = "models/checkpoints/mobilenetv3_ecosort.pth"):
    p = Path(checkpoint_path)
    if p.suffix == ".pte":
        try:
            from executorch.runtime import Runtime, Verification
            runtime = Runtime.get()
            program = runtime.load_program(p.read_bytes(), verification=Verification.Minimal)
            method = program.load_method("forward")
            print(f"[*] Successfully loaded authentic ExecuTorch FlatBuffer: {p.resolve()}")
            from evaluate_pipeline import ExecuTorchModelWrapper
            return ExecuTorchModelWrapper(program, method)
        except Exception as e:
            print(f"[!] ExecuTorch load note ({e}). Falling back to PyTorch checkpoint...")
            p = Path("models/checkpoints/mobilenetv3_ecosort.pth")

    if p.suffix == ".pt" and p.exists():
        print(f"[*] Loading TorchScript model: {p.resolve()}")
        m = torch.jit.load(str(p), map_location=device)
        m.eval()
        return m

    model = models.mobilenet_v3_large(weights=None)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, len(CLASSES))

    ckpt = Path(p)
    if ckpt.exists():
        state_dict = torch.load(ckpt, map_location=device, weights_only=False)
        model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    return model

def classify_image(img_path: Path, model, engine, device, preprocess, out_dir: Path):
    if not img_path.exists():
        print(f"[!] Error: File '{img_path}' does not exist.\n")
        return

    try:
        raw_img = Image.open(img_path).convert("RGB")
    except Exception as e:
        print(f"[!] Error opening image: {e}\n")
        return

    tensor_in = preprocess(raw_img).unsqueeze(0).to(device)

    # Warmup / inference
    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = model(tensor_in)
        probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()
    lat_ms = (time.perf_counter() - t0) * 1000.0

    prob_dict = {CLASSES[i]: float(probs[i]) for i in range(len(CLASSES))}
    decision = engine.evaluate(prob_dict)
    explainer = CognitiveExplanationEngine()
    edu_info = explainer.explain(decision.predicted_category, decision.confidence_score, decision.suggested_stream)

    # Top-3 display
    top3_indices = probs.argsort()[-3:][::-1]
    top3 = [(CLASSES[i], probs[i] * 100.0) for i in top3_indices]

    # Save annotated overlay
    out_dir.mkdir(parents=True, exist_ok=True)
    annotated = raw_img.copy()
    draw = ImageDraw.Draw(annotated)
    draw.rectangle([0, 0, annotated.width, 36], fill=(20, 20, 20))
    banner = f"[{decision.suggested_stream}] {decision.predicted_category.upper()} ({decision.confidence_score*100:.1f}%)"
    color = (255, 60, 60) if decision.is_hazard else ((0, 255, 128) if decision.confidence_tier == "HIGH" else (255, 180, 0))
    draw.text((10, 8), banner, fill=color)
    out_file = out_dir / f"pred_{img_path.name}"
    annotated.save(out_file)

    # Terminal Dashboard Display
    print("\n" + "=" * 76)
    print(f"  ECOSORT-EDGE INFERENCE DASHBOARD | {img_path.name}")
    print("=" * 76)
    print(f" • Predicted Category : {decision.predicted_category.upper()} ({decision.confidence_score*100:.2f}%)")
    print(f" • Triage Tier        : {decision.confidence_tier}")
    print(f" • Target Stream      : {decision.suggested_stream.upper()}")
    print(f" • Action Guidance    : {decision.action_guidance}")
    print(f" • Eco Impact (CO2e)  : {edu_info['environmental_impact']}")
    print(f" • Inference Latency  : {lat_ms:.2f} ms ({1000.0/lat_ms:.1f} FPS)")
    print(f" • Hardware Device    : {device.type.upper()}")
    print("-" * 76)
    print(" • Top-3 Probabilities:")
    for rank, (cls_name, prob_pct) in enumerate(top3, 1):
        bar = "█" * int(prob_pct / 4)
        print(f"    {rank}. {cls_name:<22} : {prob_pct:5.1f}%  {bar}")
    print("-" * 76)
    print(f" • Annotated Output   : {out_file.resolve()}")
    print("=" * 76 + "\n")

def run_webcam(model, engine, device, preprocess):
    try:
        import cv2
    except ImportError:
        print("[!] OpenCV is required for webcam mode. Install it via: pip install opencv-python")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[!] Error: Could not open default camera (index 0).")
        return

    print("\n" + "=" * 70)
    print(" ECOSORT-EDGE: LIVE WEBCAM SORTING MODE ACTIVE")
    print(" Hold waste items in front of the camera.")
    print(" Press 'q' or 'ESC' in the camera window to exit.")
    print("=" * 70 + "\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to PIL
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)
        tensor_in = preprocess(pil_img).unsqueeze(0).to(device)

        t0 = time.perf_counter()
        with torch.no_grad():
            outputs = model(tensor_in)
            probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()
        lat_ms = (time.perf_counter() - t0) * 1000.0

        prob_dict = {CLASSES[i]: float(probs[i]) for i in range(len(CLASSES))}
        decision = engine.evaluate(prob_dict)

        # Draw overlay on OpenCV frame
        color_bgr = (0, 0, 255) if decision.is_hazard else ((0, 255, 0) if decision.confidence_tier == "HIGH" else (0, 165, 255))
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 70), (20, 20, 20), -1)
        text1 = f"[{decision.suggested_stream}] {decision.predicted_category.upper()} ({decision.confidence_score*100:.1f}%)"
        text2 = f"Latency: {lat_ms:.1f}ms ({1000.0/lat_ms:.0f} FPS) | Tier: {decision.confidence_tier}"
        cv2.putText(frame, text1, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_bgr, 2)
        cv2.putText(frame, text2, (15, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        cv2.imshow("EcoSort-Edge Live Vision (Press Q to quit)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[*] Live camera scanner closed.\n")

def main():
    parser = argparse.ArgumentParser(description="EcoSort-Edge Prediction CLI")
    parser.add_argument("image", nargs="?", type=str, default=None, help="Path to waste image file")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cpu or cuda)")
    parser.add_argument("--checkpoint", type=str, default="models/checkpoints/mobilenetv3_ecosort.pth", help="Checkpoint path")
    parser.add_argument("--output_dir", type=str, default="results/annotated", help="Directory for output visuals")
    parser.add_argument("--camera", action="store_true", help="Launch live webcam scanning mode")
    args = parser.parse_args()

    device = torch.device(args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    print(f"\n[*] Initializing EcoSort Engine on {device.type.upper()}...")
    model = load_model(device, args.checkpoint)
    engine = ConfidenceAwareDecisionEngine()

    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    out_dir = Path(args.output_dir)

    if args.camera:
        run_webcam(model, engine, device, preprocess)
        return

    if args.image:
        classify_image(Path(args.image), model, engine, device, preprocess, out_dir)
        return

    # Interactive Drag-and-Drop Loop
    print("\n" + "=" * 76)
    print("   ECOSORT-EDGE: INTERACTIVE CLI TERMINAL (DRAG-AND-DROP READY)")
    print("=" * 76)
    print(" Tips:")
    print("   • Drag and drop any image into this terminal and press ENTER")
    print("   • Or type 'sample' to run an instant benchmark test")
    print("   • Or type 'exit' or 'q' to quit")
    print("=" * 76 + "\n")

    sample_images = list(Path("data/smoke_test").glob("*.jpg"))

    while True:
        try:
            user_input = input("EcoSort > Enter image path (or 'sample' / 'q'): ").strip().strip('"').strip("'")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if not user_input:
            continue
        if user_input.lower() in ("q", "quit", "exit"):
            print("Goodbye!")
            break
        if user_input.lower() == "sample":
            if sample_images:
                import random
                chosen = random.choice(sample_images)
                print(f"[*] Running on random canonical sample: {chosen}")
                classify_image(chosen, model, engine, device, preprocess, out_dir)
            else:
                print("[!] No sample images found in data/smoke_test/")
            continue

        classify_image(Path(user_input), model, engine, device, preprocess, out_dir)

if __name__ == "__main__":
    main()
