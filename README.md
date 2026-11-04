# 🌿 EcoSort-Edge: Offline Edge AI Waste Triage & Fire-Safety Pipeline

[![Hardware](https://img.shields.io/badge/Target-Raspberry%20Pi%205-C51A4A?logo=raspberry-pi)](https://www.raspberrypi.com/products/raspberry-pi-5/)
[![Runtime](https://img.shields.io/badge/Runtime-ExecuTorch%20INT8-EE4C2C?logo=pytorch)](https://pytorch.org/executorch/)
[![Model Size](https://img.shields.io/badge/Model%20Size-4.31%20MB-brightgreen)](#)
[![Inference Speed](https://img.shields.io/badge/Pi%205%20Latency-6.64%20ms%20(150.5%20FPS)-blue)](#)
[![Fire Hazard Recall](https://img.shields.io/badge/Battery%20Safety-100%25%20Recall-red)](#)
[![Physical Accuracy](https://img.shields.io/badge/4--Stream%20Accuracy-96.30%25-success)](#)

> **Bharat AI-SoC Challenge / Namma AI Hackathon (Problem Statement 3: AI for Sustainability)**  
> Engineered by **Ashwin Kumar K** (`Ashwinkumar-k10`)

---

## 🏛️ Project Directory Structure

```plaintext
EcoSort-Edge/
├── 📁 models/                                # Model binaries & checkpoints
│   ├── ⚡ mobilenetv3_large_int8.pte         # 4.31 MB Authentic ExecuTorch FlatBuffer (ET12) [D1]
│   ├── 📦 mobilenetv3_large_optimized.pt     # 16.60 MB Optimized TorchScript fallback
│   └── 📂 checkpoints/                       # Fine-tuned PyTorch state dicts (.pth)
│
├── 📁 data/                                  # Test imagery & ground truth
│   ├── 📂 test_27_samples/                   # 27 authentic holdout photos (3/class across 9 classes)
│   │   ├── battery_01.jpeg ... battery_03.jpg
│   │   ├── cardboard_01.jpg ... vegetable_scrap_03.jpg
│   │   └── ground_truth.json                 # Canonical labels & statutory disposal streams
│   └── 📂 smoke_test/                        # 18 curated real test photos for fast verification
│
├── 📁 scripts/                               # Core logic, export & deployment utilities
│   ├── 🧠 decision_engine.py                 # Multi-tier confidence gating & cross-class calibration
│   ├── ⚙️ export_executorch.py               # AOT EXIR INT8 per-channel quantization compiler
│   ├── 🍓 run_d4_on_pi.py                    # Automated on-device telemetry runner for Raspberry Pi 5
│   ├── 📡 test_pi_27_samples.bat             # 1-click network deployment & test script for Pi 5
│   ├── 📊 compute_full_metrics.py            # 9x9 confusion matrix & per-class F1 metric calculator
│   └── 🖼️ prepare_27_test_samples.py         # Test split sampling utility
│
├── 📁 results/                               # Benchmark outputs & diagnostic artifacts
│   ├── 📊 deliverable_d3_results.json        # Official Deliverable D3 metrics JSON
│   ├── 📊 deliverable_d3_27_samples_benchmark.json # 27-image benchmark JSON
│   ├── 📂 annotated/                         # Visual bounding & triage classification overlays [D2]
│   └── 📂 test_27_eval/                      # Pi 5 evaluated outputs and diagnostic images
│
├── 🚀 evaluate_pipeline.py                   # Automated CLI evaluation runner for Deliverables D1–D4
├── 🔍 predict.py                             # Single-image & webcam live prediction terminal
├── 🎬 demo_modes.py                          # 2-minute operational scenario demo (MRF & Kiosk) [D5]
│
├── 📋 DELIVERABLE_D4_HARDWARE_REPORT.md      # On-device Pi 5 hardware performance & telemetry report [D4]
├── 📖 ECOSORT_EDGE_MASTER_REPORT.md          # Exhaustive engineering specification & benchmark report
├── 🍓 PI_DEPLOYMENT_GUIDE.md                 # Step-by-step Raspberry Pi 5 setup & execution guide
├── 📑 PIPELINE.pdf                           # Official hackathon presentation document
├── 📦 requirements.txt                       # Python runtime dependencies
└── 🛡️ .gitignore                             # Excludes raw datasets, cache, and temp files
```

---

## 🎯 Deliverables Verification Matrix

| Deliverable | Technical Requirement | On-Device Raspberry Pi 5 Result | Status |
|---|---|:---:|:---:|
| **D1: Edge Binary** | ExecuTorch `.pte` model `< 7.5 MB` ceiling | **4.31 MB** (`ET12` FlatBuffer header, Shannon entropy `7.0281`) | 🟢 **PASS** |
| **D2: Diagnostic Imagery** | Predicted categories with bounding overlays | 27 authentic holdout visuals with confidence tiers & triage guidance | 🟢 **PASS** |
| **D3: Results & Metrics** | Top-1 accuracy, confusion matrix, stream breakdown | **90.68%** 9-class top-1, **96.30%** 4-stream physical bin accuracy | 🟢 **PASS** |
| **D4: On-Device Telemetry** | Real Pi 5 latency, throughput, RAM, temperature | **6.64 ms (150.5 FPS)**, **45.5 °C peak load**, **415.3 MB RAM** | 🟢 **PASS** |
| **D5: 2-Minute Demo** | Worker diversion + Citizen Guidance Kiosk scenarios | Complete dual-mode demonstration (`demo_modes.py`) with zero-RAM explainer | 🟢 **PASS** |
| **Safety Standard** | Zero missed battery hazards (SWM Rules 2016) | **100.0% Battery Hazard Recall** (Zero missed fire hazards, $\tau_{\text{hazard}}=0.28$) | 🚨 **PERFECT** |

---

## ⚡ Quick Start: Running on Raspberry Pi 5

```bash
# 1. Clone repository on Pi 5
git clone https://github.com/Ashwinkumar-k10/EcoSort-Edge.git
cd EcoSort-Edge

# 2. Install lightweight edge dependencies
pip3 install torch torchvision executorch pillow numpy psutil

# 3. Run the automated D4 hardware telemetry benchmark
python3 scripts/run_d4_on_pi.py

# 4. Run the 2-minute operational demo (MRF worker & Citizen Kiosk)
python3 demo_modes.py

# 5. Run single-image terminal prediction
python3 predict.py data/test_27_samples/battery_01.jpeg --checkpoint models/mobilenetv3_large_int8.pte --device cpu
```

---

## 🔬 Core Architectural Highlights

1. **Ahead-of-Time INT8 Quantization**:
   - Quantized with `XNNPACKQuantizer` (Static Per-Channel) calibrated against 225 real waste photos.
   - Lowers directly to ExecuTorch EXIR edge dialect for native ARM NEON / Cortex-A76 micro-kernel acceleration.

2. **Confidence-Aware Triage Gating ([`scripts/decision_engine.py`](file:///A:/Projects/ARM/scripts/decision_engine.py))**:
   - `HAZARD_ALERT`: Triggers instant red strobe and compactor halt upon detecting battery features ($\tau_{\text{hazard}} = 0.28$).
   - `HIGH`: Confident autonomous pneumatic sorting.
   - `CAUTION` / `MANUAL_CHECK`: Prompts worker secondary inspection to prevent facility batch contamination.

3. **100% Offline & Air-Gapped**:
   - Zero HTTP/cloud calls; runs self-contained on edge hardware in rural or unconditioned waste transfer stations.
