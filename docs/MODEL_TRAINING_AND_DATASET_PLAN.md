# EcoSort-Edge: Generalization-First Training & Dataset Plan
**Architecture:** ImageNet-Pretrained MobileNetV3-Small (Head: 15 Classes)  
**Execution Environment:** NVIDIA GeForce RTX 3050 Laptop GPU (4GB VRAM)  
**Target Hardware:** Raspberry Pi 4B (Arm Cortex-A72) & Pi 5 (Arm Cortex-A76) via ExecuTorch (XNNPACK INT8)  

---

## 1. Executive Strategy & Guiding Principles

The objective is not achieving 99% training accuracy on familiar images. The primary objective is **real-world generalization**:
- Recognizing crushed, soiled, deformed, and torn packaging.
- Handling varying lighting conditions and camera angles without internet access.
- Correctly routing Indian FMCG packaging never seen during training.
- **A 92% accurate model on genuinely unseen waste is fundamentally superior to a 99% model that fails in field deployment.**

```
┌────────────────────────────────────────────────────────────────────────┐
│                   GENERALIZATION-FIRST ARCHITECTURE                    │
├────────────────────────────────┬───────────────────────────────────────┤
│ Stage 1: Feature Adaptation    │ Backbone Frozen (5-8 epochs, LR=1e-3) │
│ Stage 2: Controlled Fine-Tune  │ Differential LR (Backbone=1e-5, Head) │
│ Loss Function                  │ CrossEntropy with Label Smoothing 0.05│
│ Class Balancing                │ WeightedRandomSampler (no duplicates) │
│ Early Stopping                 │ Monitored on Validation Macro-F1      │
│ Safety Gate                    │ Calibrated Unknown Rejection Threshold│
└────────────────────────────────┴───────────────────────────────────────┘
```

---

## 2. Dataset Blueprint & Leakage-Free Splitting

### 2.1 The 70 / 15 / 15 Partitioning Strategy
Unlike standard 85:15 splits, we reserve **15% as a pristine, untouched Test Set**:

| Dataset Split | Percentage | Approx. Volume | Primary Objective |
| :--- | :---: | :---: | :--- |
| **Training** | **70%** | ~3,360 images | Feature adaptation and progressive representation learning. |
| **Validation** | **15%** | ~720 images | Checkpoint selection, hyperparameter tuning, early stopping. |
| **Final Test** | **15%** | ~720 images | **Untouched hold-out.** Evaluated only ONCE for the final report. |
| **Smoke Test** | Separate | 15 canonical | Deployment smoke test & latency benchmarking. |

### 2.2 Leakage Prevention Measures
1. **Exact Duplicate Purge:** SHA-256 byte hashing drops identical image files scraped across web mirrors.
2. **Perceptual Near-Duplicate Purge:** Difference Hashing (`dHash` via Pillow) with a Hamming distance threshold $\le 4$ purges near-identical photos of the same item taken seconds apart.
3. **Partition Before Augmentation:** Data augmentation is strictly applied *on-the-fly* inside PyTorch `DataLoader` during training. Validation and Test splits remain completely raw and unaugmented.

---

## 3. The 15-Class Statutory Taxonomy

```
MODEL OUTPUT (15 Fine Classes)              STATUTORY STREAM (SWM Rules)
──────────────────────────────              ────────────────────────────
0.  food_waste                        ──►   🟢 GREEN BIN: Wet Organic
1.  fruit_peel                        ──►   🟢 GREEN BIN: Wet Organic
2.  vegetable_scrap                   ──►   🟢 GREEN BIN: Wet Organic
3.  cooked_leftover                   ──►   🟢 GREEN BIN: Wet Organic

4.  plastic_bottle (clean)            ──►   🔵 BLUE BIN: Dry Recyclable
5.  metal_can (clean)                 ──►   🔵 BLUE BIN: Dry Recyclable
6.  cardboard (clean)                 ──►   🔵 BLUE BIN: Dry Recyclable
7.  paper (clean)                     ──►   🔵 BLUE BIN: Dry Recyclable
8.  glass_bottle (clean)              ──►   🔵 BLUE BIN: Dry Recyclable
9.  tetra_pak                         ──►   🔵 BLUE BIN: Dry Recyclable

10. plastic_soiled                    ──►   ⚫ GREY BIN: Non-Recyclable Reject
11. paper_soiled                      ──►   ⚫ GREY BIN: Non-Recyclable Reject
12. multi_layer_packaging             ──►   ⚫ GREY BIN: Non-Recyclable Reject

13. battery                           ──►   🔴 RED BIN: Hazardous / E-Waste
14. electronic_scrap                  ──►   🔴 RED BIN: Hazardous / E-Waste
```

---

## 4. Dataset Acquisition & Execution Guide

To collect the raw datasets onto your machine without manual camera work:

### Source A: RealWaste (4,752 images from an authentic Material Recovery Facility)
* **URL:** `https://www.kaggle.com/datasets/joebeachcapital/realwaste`
* **Download Command:**
  ```bash
  kaggle datasets download -d joebeachcapital/realwaste -p data/raw/
  tar -xf data/raw/realwaste.zip -C data/raw/realwaste/
  ```

### Source B: Garbage Classification v2 (30,000+ categorized images)
* **URL:** `https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2`
* **Download Command:**
  ```bash
  kaggle datasets download -d sumn2u/garbage-classification-v2 -p data/raw/
  tar -xf data/raw/garbage-classification-v2.zip -C data/raw/garbage_v2/
  ```

### Source C: Automated Indian Packaging Web Scraper
Runs automated query batches for Parle-G, Frooti, Bisleri, gutkha sachets, and lithium cells:
```bash
python download_indian_waste_images.py
```

### Source D: Dataset Deduplication & Curation Execution
Once raw folders are populated, execute the leakage-resistant curator:
```bash
python scripts/dataset_curator.py
```
This automatically deduplicates via SHA-256 and dHash, splits into `70% train`, `15% val`, and `15% test`, and saves a full integrity report to `results/dataset_curation_report.json`.

---

## 5. Two-Stage Transfer Learning Protocol (RTX 3050 Optimized)

### Stage 1: Feature Adaptation (Backbone Frozen)
* **Epochs:** 6
* **Backbone State:** Frozen (ImageNet weights locked)
* **Trainable Parameters:** Classifier Head only (`Linear(1024, 15)`)
* **Learning Rate:** `1e-3` (AdamW, Weight Decay `1e-4`)
* **Objective:** Align the output projection head to the 15 waste categories without destabilizing pretrained convolutional representations.

### Stage 2: Progressive Fine-Tuning with Differential Learning Rates
* **Epochs:** 25 (with Early Stopping, Patience = 7)
* **Backbone State:** Unfrozen
* **Differential Learning Rates:**
  $$\text{LR}_{\text{backbone}} = 1\times 10^{-5} \quad (\text{0.1} \times \text{LR}_{\text{head}})$$
  $$\text{LR}_{\text{head}} = 1\times 10^{-4}$$
* **Learning Rate Scheduler:** Cosine Annealing ($\eta_{\text{min}} = 1\times 10^{-6}$)
* **Regularization:**
  * Label Smoothing: `0.05`
  * Weight Decay: `1e-4`
  * Partial Occlusion (`RandomErasing(p=0.2)`)
  * ColorJitter (brightness=0.2, contrast=0.2, saturation=0.2)
* **Primary Checkpoint Criterion:** **Validation Macro-F1** (not raw accuracy).

---

## 6. Multi-Seed Robustness & Repeatability

To verify stability and eliminate lucky random splits, training runs across 3 distinct seeds:

| Run | Random Seed | Purpose |
| :---: | :---: | :--- |
| **Run 1** | `42` | Primary Baseline & Checkpoint Generation |
| **Run 2** | `123` | Stability & Variance Verification |
| **Run 3** | `2026` | Final Reproducibility & Out-of-Domain Generalization Check |

**Execution Command:**
```bash
python scripts/train_ecosort_engine.py --all_seeds
```

---

## 7. Unknown-Waste Rejection Gate (Deterministic Policy Guard)

In real municipal deployment, the model must know when it does **not** know.
* Softmax confidence threshold: $\tau = 0.72$.
* If confidence $< \tau$: The item is automatically routed to **Grey Bin (Manual Review)**.
* **Safety Invariant:** If an uncertain item shares feature proximity with batteries or electronics, it is **never** sent to dry recyclables; it triggers mandatory safety holding.
* **Contamination Detection:** When food residue is flagged on clean packaging, the policy engine diverts the item to prevent batch contamination.

---

## 8. Export & Pi 4B Deployment Protocol

1. **Model Selection:** Select best checkpoint based on Validation Macro-F1.
2. **Quantization:** Apply Post-Training Quantization (PT2E INT8) via `XNNPACKQuantizer` with 100 representative calibration samples.
3. **ExecuTorch Compilation:**
   ```bash
   python scripts/export_quantized_executorch.py
   ```
   * Output: `models/waste_classifier_int8.pte` ($< 7.5\text{ MB}$).
4. **Validation Smoke Test:**
   ```bash
   python scripts/smoke_test_runner.py
   ```
   * Measures per-image inference latency on CPU ($< 80\text{ ms}$ target).
   * Generates Hackathon Deliverables D2 and D3 (`results/smoke_test_deliverable_d3.json`).
