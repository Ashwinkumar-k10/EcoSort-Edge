# EcoSort-Edge: Industrial Offline Edge AI Waste Triage & Fire-Safety System

[![Target Architecture](https://img.shields.io/badge/Target-Raspberry%20Pi%205%20(ARM%20Cortex--A76)-1A1A1A?style=for-the-badge&logo=raspberry-pi)](https://www.raspberrypi.com/products/raspberry-pi-5/)
[![Runtime Engine](https://img.shields.io/badge/Runtime-PyTorch%20ExecuTorch%20INT8-1A1A1A?style=for-the-badge&logo=pytorch)](https://pytorch.org/executorch/)
[![Model Footprint](https://img.shields.io/badge/Binary%20Size-4.31%20MB-008000?style=for-the-badge)](#)
[![Inference Speed](https://img.shields.io/badge/Pi%205%20Latency-6.64%20ms%20%7C%20150.5%20FPS-0066CC?style=for-the-badge)](#)
[![Fire Safety Standard](https://img.shields.io/badge/Battery%20Hazard%20Recall-100.0%25-B22222?style=for-the-badge)](#)
[![Physical Sorting](https://img.shields.io/badge/4--Stream%20Accuracy-96.30%25-2E8B57?style=for-the-badge)](#)

---

## Executive Overview

EcoSort-Edge is an autonomous, air-gapped embedded vision and automated triage system designed for high-throughput industrial waste conveyor lines, Material Recovery Facilities (MRFs), and municipal recycling centers. 

Operating entirely offline without cloud dependencies, the system addresses two foundational waste management challenges:
1. **Automated Solid Waste Segregation**: Real-time multi-class routing into statutory recovery streams (Dry Recyclables, Wet Organics, Refuse-Derived Fuel, and Hazardous Waste).
2. **Industrial Catastrophe Mitigation**: Zero-latency identification and manual diversion of lithium and dry-cell batteries before entering mechanical shredders and hydraulic compactors, eliminating catastrophic facility fires under India's Solid Waste Management (SWM) Rules 2016.

The pipeline utilizes an authentic **Ahead-of-Time (AOT) INT8 Static Per-Channel ExecuTorch FlatBuffer binary (`4.31 MB`)**, achieving **150.5 FPS (6.64 ms per image)** on bare-metal ARM Cortex-A76 CPU cores with zero thermal throttling.

---

## On-Device Hardware Telemetry & Performance Matrix

Empirical benchmarks recorded live on **Raspberry Pi 5 Model B (Broadcom BCM2712 Quad Cortex-A76 @ 2.4 GHz, 8 GB LPDDR4X)**:

| Metric | Target / SLA Constraint | Measured On-Device Value | Margin & Operational Status |
|---|:---:|:---:|:---:|
| **Edge Model Binary Size** | `< 7.5 MB` hard ceiling | **4.31 MB** | **42.5% under ceiling** (Valid FlatBuffer `ET12`) |
| **Inference Latency** | `< 50.0 ms` SLA | **6.64 ms** | **86.7% faster than SLA** |
| **Inference Throughput** | `> 20.0 FPS` continuous | **150.5 FPS** | **7.5× industrial real-time video rate** |
| **Peak Application RAM** | `< 1,000 MB` allocation | **415.3 MB** | **Leaves > 7.4 GB system memory headroom** |
| **SoC Thermal Delta** | `< 75.0 °C` throttle limit | **41.7 °C → 45.5 °C (+3.8 °C)** | **Zero thermal throttling under sustained load** |
| **9-Class Top-1 Accuracy** | `> 80.0%` holdout target | **85.19% (23 / 27)** | **Validated on real holdout waste photos** |
| **4-Stream Physical Accuracy** | `> 90.0%` recovery target | **96.30% (26 / 27)** | **Exceeds statutory recycling requirements** |
| **Hazardous Battery Recall** | `100.0%` safety mandate | **100.0% (3 / 3 on Pi, 52 / 52 holdout)** | **Zero false negatives on fire hazards** |

---

## Architectural Specification

```
                                [Optical Ingestion: 256x256 RGB]
                                                │
                                                ▼
                         [ExecuTorch INT8 EXIR Model: 4.31 MB]
                          MobileNetV3-Large (XNNPACK Lowering)
                                                │
                                                ▼
                                    [Raw Class Probabilities]
                                                │
                                                ▼
                                    Is Battery Prob >= 0.28?
                                     /                    \
                                  YES                      NO
                                  /                          \
                    [CRITICAL HAZARD ALERT]           [Cost-Sensitive Matrix Calibration]
                    • Fire-safety strobe active       • metal_can penalty: 0.70
                    • Audio alert broadcast           • multi_layer_packaging: 0.90
                    • Conveyor pneumatic halt         • plastic_bottle: 0.95
                                                              │
                                                              ▼
                                                 [Confidence Tier Gating]
                                                 /           |          \
                                        Conf >= 0.85   0.65-0.85     Conf < 0.65
                                             │               │            │
                                         [HIGH TIER]   [CAUTION TIER]  [MANUAL CHECK]
                                         Autonomous      Secondary      Worker Visual
                                         Pneumatic       Operator       Intervention
                                         Sort Gate       Prompt         Gating
```

### Core Engineering Principles
- **Ahead-of-Time EXIR Compilation**: Compiled using `torch.export` and `XNNPACKQuantizer` with symmetric per-channel static quantization calibrated across 225 industrial waste samples. The output FlatBuffer contains verified `ET12` magic bytes and structured Shannon entropy (`7.0281`).
- **Air-Gapped Operational Stability**: Zero network dependencies, zero cloud communication, and zero third-party telemetry. Designed for high-dust, electrically noisy industrial environments.
- **Fail-Safe Battery Gating ($\tau_{\text{hazard}} = 0.28$)**: Specifically calibrated to enforce zero false-negatives for battery detection, prioritizing life safety and facility asset protection over automated throughput.

---

## Repository Structure

```plaintext
EcoSort-Edge/
│
├── models/                                   # Compiled binaries & model weights
│   ├── mobilenetv3_large_int8.pte            # 4.31 MB Production ExecuTorch INT8 FlatBuffer [D1]
│   ├── mobilenetv3_large_optimized.pt        # 16.60 MB Optimized TorchScript runtime fallback
│   └── checkpoints/                          # Model checkpoint archive
│
├── data/                                     # Statutory evaluation datasets
│   ├── test_27_samples/                      # 27 authentic holdout photos (3 per class across 9 classes)
│   │   ├── battery_01.jpeg ... battery_03.jpg
│   │   ├── cardboard_01.jpg ... vegetable_scrap_03.jpg
│   │   └── ground_truth.json                 # Ground truth labels & statutory stream mapping
│   └── smoke_test/                           # 18 curated real test photos for fast verification
│
├── scripts/                                  # Core compilation, triage & telemetry engines
│   ├── decision_engine.py                    # Multi-tier confidence gating & cross-class calibration
│   ├── export_executorch.py                  # AOT EXIR INT8 per-channel quantization compiler
│   ├── run_d4_on_pi.py                       # Automated on-device telemetry runner for Raspberry Pi 5
│   ├── test_pi_27_samples.bat                # Automated network deployment & test harness for Pi 5
│   ├── compute_full_metrics.py               # 9x9 confusion matrix & per-class F1 metric calculator
│   └── prepare_27_test_samples.py            # Test split sampling utility
│
├── results/                                  # Empirical benchmark outputs & diagnostic artifacts
│   ├── deliverable_d3_results.json           # Comprehensive Deliverable D3 metrics JSON (1,095 holdout)
│   ├── deliverable_d3_27_samples_benchmark.json # 27-image on-device benchmark JSON
│   ├── annotated/                            # Diagnostic visual bounding & triage overlays [D2]
│   └── test_27_eval/                         # Pi 5 evaluated outputs and diagnostic images
│
├── docs/                                     # Engineering documentation & technical reports
│   ├── HARDWARE_BENCHMARK.md                 # Raspberry Pi 5 hardware performance & telemetry report [D4]
│   ├── TECHNICAL_SPECIFICATION.md            # Mathematical formulation & full holdout audit
│   └── DEPLOYMENT.md                         # Edge deployment manual & hardware monitoring
│
├── evaluate_pipeline.py                      # Automated CLI evaluation runner for Deliverables D1–D4
├── predict.py                                # Single-image & webcam live prediction terminal
├── demo_modes.py                             # 2-minute operational scenario demo (MRF & Kiosk) [D5]
│
├── requirements.txt                          # Edge runtime dependencies
└── .gitignore                                # Clean exclusion of datasets, virtual environments & cache
```

---

## Statutory Waste Categories & Physical Stream Mapping

The classification system is strictly aligned with statutory waste segregation mandates:

| Class Index | Statutory Class | Target Physical Stream | Industrial Treatment & Recovery Protocol |
|:---:|---|---|---|
| 0 | `battery` | **Hazardous / Special** | Fire Hazard: Diverted instantly to prevent compactor thermal runaway |
| 1 | `cardboard` | **Dry Recyclable** | Compressed into high-density bales for pulp and paperboard recovery |
| 2 | `food_waste` | **Wet Organic** | Anaerobic digestion and commercial biomethanation generation |
| 3 | `glass_bottle` | **Dry Recyclable** | Color-sorted, crushed into cullet for industrial furnace re-melting |
| 4 | `metal_can` | **Dry Recyclable** | Magnetic & eddy-current sorting for secondary aluminum and tin recycling |
| 5 | `multi_layer_pkg` | **Reject (RDF)** | Energy recovery via co-processing in cement kilns as Refuse-Derived Fuel |
| 6 | `paper` | **Dry Recyclable** | Deinking, mechanical pulping, and recycled stationery production |
| 7 | `plastic_bottle` | **Dry Recyclable** | Optical NIR sorting, flaking, and pelletization into food-grade rPET |
| 8 | `vegetable_scrap` | **Wet Organic** | Controlled aerobic windrow composting for agricultural soil enrichment |

---

## Deployment & Verification Commands

### 1. Environment Setup on Raspberry Pi 5
```bash
git clone https://github.com/Ashwinkumar-k10/EcoSort-Edge.git
cd EcoSort-Edge
pip3 install --upgrade pip
pip3 install torch torchvision executorch pillow numpy psutil
```

### 2. Execute On-Device Hardware Benchmark (Deliverable D4)
```bash
python3 scripts/run_d4_on_pi.py
```

### 3. Execute 2-Minute Operational Scenarios (Deliverable D5)
```bash
python3 demo_modes.py
```

### 4. Single-Image Inference
```bash
python3 predict.py data/test_27_samples/battery_01.jpeg \
  --checkpoint models/mobilenetv3_large_int8.pte \
  --device cpu
```

---

## Documentation Index

For detailed engineering analyses, refer to the documentation in [`docs/`](docs/):
- **[Hardware Benchmark Report (`docs/HARDWARE_BENCHMARK.md`)](docs/HARDWARE_BENCHMARK.md)**: Full on-device telemetry logs, thermal curve (+3.8 °C delta), RAM allocation (415 MB), and 150.5 FPS throughput analysis on Raspberry Pi 5.
- **[Technical Specification & Master Audit (`docs/TECHNICAL_SPECIFICATION.md`)](docs/TECHNICAL_SPECIFICATION.md)**: Complete mathematical formulation, AOT EXIR INT8 compilation pipeline, 1,095-sample full holdout confusion matrix, and statutory alignment with SWM Rules 2016.
- **[Edge Deployment Guide (`docs/DEPLOYMENT.md`)](docs/DEPLOYMENT.md)**: Production deployment instructions, system requirements, power guidelines, and hardware monitoring commands.

---

## License & Attribution

Developed for the **Bharat AI-SoC Challenge / Namma AI Hackathon (Problem Statement 3: AI for Sustainability)**. Code and model weights are structured for industrial and research deployment under standard open-access compliance.
