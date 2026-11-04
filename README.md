# 🌿 EcoSort-Edge: Offline Edge AI Waste Triage & Fire-Safety Pipeline

[![Hardware](https://img.shields.io/badge/Hardware-Raspberry%20Pi%205-C51A4A?logo=raspberry-pi)](https://www.raspberrypi.com/products/raspberry-pi-5/)
[![Runtime](https://img.shields.io/badge/Runtime-ExecuTorch%20INT8-EE4C2C?logo=pytorch)](https://pytorch.org/executorch/)
[![Model Size](https://img.shields.io/badge/Binary%20Size-4.31%20MB-brightgreen)](#)
[![Inference Speed](https://img.shields.io/badge/Pi%205%20Latency-6.64%20ms%20(150.5%20FPS)-blue)](#)
[![Fire Hazard Recall](https://img.shields.io/badge/Battery%20Safety-100%25%20Recall-red)](#)
[![Physical Accuracy](https://img.shields.io/badge/4--Stream%20Accuracy-96.30%25-success)](#)

> **Bharat AI-SoC Challenge / Namma AI Hackathon (Problem Statement 3: AI for Sustainability)**  
> Author: **Ashwin Kumar K** (`Ashwinkumar-k10`)

---

## 🏛️ Repository Architecture

```plaintext
EcoSort-Edge/
│
├── 📁 models/                                # Optimized neural network binaries & checkpoints
│   ├── ⚡ mobilenetv3_large_int8.pte         # 4.31 MB Authentic ExecuTorch FlatBuffer (ET12) [D1]
│   ├── 📦 mobilenetv3_large_optimized.pt     # 16.60 MB Optimized TorchScript runtime fallback
│   └── 📂 checkpoints/                       # Model checkpoint storage
│
├── 📁 data/                                  # Canonical evaluation test imagery & annotations
│   ├── 📂 test_27_samples/                   # 27 authentic holdout photos (3/class across 9 classes)
│   │   ├── battery_01.jpeg ... battery_03.jpg
│   │   ├── cardboard_01.jpg ... vegetable_scrap_03.jpg
│   │   └── ground_truth.json                 # Statutory class labels & target physical streams
│   └── 📂 smoke_test/                        # 18 curated real test photos for fast verification
│
├── 📁 scripts/                               # Core compilation, triage & deployment engines
│   ├── 🧠 decision_engine.py                 # Multi-tier confidence gating & cross-class calibration
│   ├── ⚙️ export_executorch.py               # AOT EXIR INT8 per-channel quantization compiler
│   ├── 🍓 run_d4_on_pi.py                    # Automated on-device telemetry runner for Raspberry Pi 5
│   ├── 📡 test_pi_27_samples.bat             # 1-click deployment & test script for Pi 5
│   ├── 📊 compute_full_metrics.py            # 9x9 confusion matrix & per-class F1 metric calculator
│   └── 🖼️ prepare_27_test_samples.py         # Test split sampling utility
│
├── 📁 results/                               # Verified benchmark outputs & diagnostic artifacts
│   ├── 📊 deliverable_d3_results.json        # Official Deliverable D3 metrics JSON (1,095 holdout)
│   ├── 📊 deliverable_d3_27_samples_benchmark.json # 27-image on-device benchmark JSON
│   ├── 📂 annotated/                         # Visual bounding & triage classification overlays [D2]
│   └── 📂 test_27_eval/                      # Pi 5 evaluated outputs and diagnostic images
│
├── 📁 docs/                                  # Technical documentation & engineering reports
│   ├── 📋 HARDWARE_BENCHMARK.md              # Raspberry Pi 5 hardware performance & telemetry report [D4]
│   ├── 📖 TECHNICAL_SPECIFICATION.md         # Exhaustive engineering specification & full holdout audit
│   └── 🍓 DEPLOYMENT.md                      # Step-by-step Raspberry Pi 5 setup & execution guide
│
├── 🚀 evaluate_pipeline.py                   # Automated CLI evaluation runner for Deliverables D1–D4
├── 🔍 predict.py                             # Single-image & webcam live prediction terminal
├── 🎬 demo_modes.py                          # 2-minute operational scenario demo (MRF & Kiosk) [D5]
│
├── 📦 requirements.txt                       # Edge & host Python runtime dependencies
└── 🛡️ .gitignore                             # Clean exclusion of raw/processed datasets & cache
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

## 📚 Technical Documentation Index

Detailed technical reports and operational specifications are organized in the [`docs/`](docs/) directory:

- 📊 **[Hardware Benchmark Report (`docs/HARDWARE_BENCHMARK.md`)](docs/HARDWARE_BENCHMARK.md)**: Full on-device telemetry logs, thermal curve (+3.8 °C delta), RAM allocation (415 MB), and 150.5 FPS throughput analysis on Raspberry Pi 5 (Deliverable D4).
- 📖 **[Technical Specification & Master Audit (`docs/TECHNICAL_SPECIFICATION.md`)](docs/TECHNICAL_SPECIFICATION.md)**: Complete mathematical formulation, AOT EXIR INT8 compilation pipeline, 1,095-sample full holdout confusion matrix, and statutory alignment with SWM Rules 2016.
- 🛠️ **[Edge Deployment Guide (`docs/DEPLOYMENT.md`)](docs/DEPLOYMENT.md)**: Production deployment instructions, system requirements, power guidelines, and hardware monitoring commands.
