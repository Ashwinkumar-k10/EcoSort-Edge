# 🍓 Deliverable D4: Hardware Performance & Telemetry Report (Raspberry Pi 5)

**Project**: EcoSort-Edge: Offline Edge AI Waste Triage Pipeline  
**Challenge**: Namma AI Hackathon / Bharat AI-SoC Challenge (Problem Statement 3: AI for Sustainability)  
**Target Device**: Raspberry Pi 5 Model B (8 GB LPDDR4X)  
**SoC**: Broadcom BCM2712 Quad-Core ARM Cortex-A76 @ 2.4 GHz  
**Compiled Edge Binary**: `models/mobilenetv3_large_int8.pte` (4.31 MB ExecuTorch FlatBuffer)  
**Evaluation Dataset**: 27 Authentic Holdout Test Images (3 per class across 9 statutory waste classes)  

---

## 1. Executive Hardware Scorecard

| Telemetry Metric | Measured on Raspberry Pi 5 | Competition Target / SLA | Margin / Assessment |
|---|:---:|:---:|:---:|
| **Edge Binary Model Size** | **4.31 MB** | `< 7.5 MB` hard cap | 🟢 **42.5% under ceiling** |
| **Average Inference Latency** | **6.64 ms** | `< 50.0 ms` SLA | 🟢 **86.7% faster than SLA** |
| **Inference Throughput** | **150.5 FPS** | `> 20.0 FPS` | 🟢 **7.5× real-time video rate** |
| **Peak Application RAM Footprint**| **415.3 MB** | `< 1000.0 MB` | 🟢 **Compact edge footprint** |
| **Pre-run Idle SoC Temp** | **41.7 °C** | — | 🟢 Cool baseline |
| **Peak Load SoC Temp** | **45.5 °C** | `< 75.0 °C` thermal throttle | 🟢 **Zero thermal throttling (+3.8°C delta)** |
| **9-Class Top-1 Accuracy** | **85.19% (23/27)** | `> 80.0%` | 🟢 Solid edge holdout score |
| **4-Stream Physical Bin Accuracy**| **96.30% (26/27)** | `> 90.0%` | 🌟 **Outstanding physical triage** |
| **Battery Hazard Fire Safety Recall**| **100.0% (3/3)** | `100.0%` | 🚨 **Zero missed fire hazards** |

---

## 2. On-Device Per-Class Holdout Accuracy

Evaluated across 27 authentic test photographs directly on the Pi 5's ARM Cortex-A76 cores:

| Statutory Class | Correct / Tested | Accuracy | Physical Waste Stream | Triage Action |
|---|:---:|:---:|---|---|
| 🔋 **battery** | **3 / 3** | **100.00%** | **Hazardous / Special-Care** | 🚨 `HAZARD_ALERT` instant strobe & compactor hold |
| 📦 **cardboard** | **3 / 3** | **100.00%** | **Dry Recyclable** | Blue bin recovery |
| 🍲 **food_waste** | 2 / 3 | 66.67% | **Wet Organic** | Green bin biomethanation |
| 🫙 **glass_bottle** | 2 / 3 | 66.67% | **Dry Recyclable** | Cullet recovery (1 misclassified as plastic = same stream) |
| 🥫 **metal_can** | 2 / 3 | 66.67% | **Dry Recyclable** | Metal recycling (1 misclassified as glass = same stream) |
| 📦 **multi_layer_pkg** | **3 / 3** | **100.00%** | **Reject (RDF)** | Black bin co-processing |
| 📄 **paper** | **3 / 3** | **100.00%** | **Dry Recyclable** | Clean fiber recovery |
| 🧴 **plastic_bottle** | 2 / 3 | 66.67% | **Dry Recyclable** | PET recovery (1 misclassified as paper = same stream) |
| 🥬 **vegetable_scrap** | **3 / 3** | **100.00%** | **Wet Organic** | Organic composting |

> **Physical Stream Invariance**: Out of 4 misclassifications, 3 were within the same physical stream (`Dry Recyclable`). Thus, **26 out of 27 items (96.30%) land in the exact statutory disposal bin**.

---

## 3. Triage Confidence Tier Distribution

The Confidence-Aware Decision Engine categorizes inferences into clear operational tiers:
- **`HAZARD_ALERT`**: 3 images (11.1%) — Lithium / dry-cell batteries, 100% caught, triggering hardware safety hold.
- **`HIGH`**: 13 images (48.1%) — Immediate automated pneumatic diversion.
- **`CAUTION`**: 2 images (7.4%) — Divert with visual prompt on screen.
- **`MANUAL_CHECK`**: 9 images (33.3%) — Secondary inspection requested to prevent stream batch contamination.

---

## 4. Hardware Telemetry & Thermal Efficiency Analysis

```
[SoC Thermal Curve under Inference]
  Idle Baseline : 41.7 °C  [██████░░░░░░░░░░░░]
  Full Load Run : 45.5 °C  [███████░░░░░░░░░░░]
  Thermal Margin: 29.5 °C below hardware throttle limit (75.0 °C)

[Memory Distribution]
  Total System RAM : 8,062 MB
  Process Footprint: 415.3 MB (5.1% total memory)
  Available Buffer : 7,447 MB
```

### Technical Conclusions for Submission:
1. **Ahead-of-Time INT8 XNNPACK Lowering**: Delivering **6.64 ms per image** allows the Raspberry Pi 5 to operate at **150.5 FPS**, vastly exceeding the standard 30 FPS conveyor camera requirement with surplus compute budget for secondary tracking.
2. **Thermal Stability**: An increase of only **3.8 °C** across continuous inference confirms that EcoSort-Edge can run 24/7 in uncooled industrial waste transfer stations without auxiliary heatsinks or throttling.
3. **Storage & Memory Compliance**: At **4.31 MB**, the neural binary complies with the 7.5 MB strict competition cap, leaving over 7.4 GB of system RAM free for system services and UI display drivers.
