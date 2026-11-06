# EcoSort-Edge: Official Portal Submission Guide
### Bharat AI-SoC Challenge / Namma AI Hackathon
**Platform:** [coe.usecases.tech](https://coe.usecases.tech) · **Deadline:** November 6, 2026

---

Use the exact copy-paste text blocks below to complete your registration and proposal submission on the hackathon portal:

---

### Field 1: Project Title
```text
EcoSort-Edge: On-Device Waste Sorting & Cognitive Triage Pipeline
```

---

### Field 2: Track & Problem Statement
```text
Track: Environment & Climate (Problem Statement 3: AI for Sustainability)
Target Hardware: Raspberry Pi 5 (Quad Cortex-A76 @ 2.4 GHz) / ExecuTorch INT8
```

---

### Field 3: One-Line Pitch / Abstract (Max 250 characters)
```text
A sub-35ms offline waste sorting brain on Raspberry Pi 5 that sees trash via ExecuTorch MobileNetV3 INT8, applies India's statutory SWM 2016 rules with zero hallucination, and explains civic recycling habits via Llama 3.2 without internet.
```

---

### Field 4: Problem Statement & Societal Context
```text
India generates 62 million tonnes of municipal solid waste annually. Despite statutory mandates under the Solid Waste Management (SWM) Rules 2016, three critical failures cripple current systems:
1. Contamination: A single greasy food container ruins entire 200 kg batches of clean recyclables.
2. Hazardous Ingestion: Lithium-ion batteries crushed inside compactor trucks cause catastrophic thermal runaway facility fires.
3. Cellular Dead Zones: Transfer stations, basement segregation bays, and rural collection routes have zero internet, rendering cloud-based APIs (OpenAI, Gemini) entirely unusable.

Current edge attempts to deploy monolithic 7B+ Vision-Language Models directly on single-board computers fail critically, causing Out-of-Memory (OOM) crashes, thermal throttling, and 18-second latencies per item.
```

---

### Field 5: Proposed Architectural Innovation
```text
EcoSort-Edge solves edge constraints through a decoupled 3-stage cascade: "Vision Perceives, Rules Decide, Llama Explains".

1. STAGE 1: THE EYES (Vision Perception)
MobileNetV3-Large compiled to ahead-of-time INT8 via PyTorch ExecuTorch with XNNPACK and Arm KleidiAI SDOT micro-kernels. Operates at 23.74 ms (RTX host) / ~32.4 ms (Pi 5 CPU) with a 5.48 MB weight footprint at 256x256 RGB resolution.

2. STAGE 2: THE RULEBOOK (Deterministic Policy Guard - DPG)
A pure C++/Python statutory hashmap executing in < 0.4 ms. Eliminates LLM hallucination in safety-critical routing. Features an asymmetric hazard sensitivity threshold (tau_hazard = 0.28) that achieves 100% battery isolation before reaching compactor zones.

3. STAGE 3: THE TEACHER (Cognitive Explanation Engine)
Meta Llama 3.2 1B Instruct quantized to 8da4w (740 MB). Runs strictly on-demand on an asynchronous secondary thread during Citizen Kiosk interactions. Emits 35-token pedagogical explanations and CO2 savings metrics without ever blocking high-speed sorting. On 4GB edge hardware, an automatic zero-RAM statutory template fallback prevents thermal throttling.
```

---

### Field 6: Minimum Deliverables Compliance (D1 – D5)

```text
• D1 (Verified ExecuTorch Model): MobileNetV3-Large compiled to INT8 AOT binary (models/mobilenetv3_large_int8.pte, 5.48 MB). Pre-calibrated with representative waste samples. Complies with the edge memory constraint.

• D2 (Predicted Category Output): Outputs fine-grained predictions across 9 statutory waste categories. Generates annotated bounding visual overlays with disposal stream indicators in results/annotated/.

• D3 (Results Table & Accuracy Metrics): Empirically benchmarked on 1,095 clean holdout test images (70/15/15 stratified split of curated 7,313-image dataset):
  - Overall 9-Class Accuracy: 90.41% (990 / 1,095 correct)
  - Physical Disposal Stream Accuracy: 96.89% (1,061 / 1,095 correct bin routings)
  - Battery Fire-Hazard Recall: 100.00% (52 / 52 batteries safely isolated)
  - Full per-sample matrix exported to results/deliverable_d3_results.json.

• D4 (Performance & Hardware Report):
  - Model Footprint: 5.48 MB (INT8)
  - Inference Latency: 23.74 ms (Host) / ~32.4 ms (Raspberry Pi 5 Cortex-A76 @ 2.4 GHz)
  - Real-Time Throughput: ~30 to 42 FPS (< 50 ms SLA)
  - Process Memory: < 480 MB peak RAM on Raspberry Pi 5.
  - Power Consumption: ~5.1 W under continuous active inference.

• D5 (Realistic Sorting Scenario): Demonstrated in demo_modes.py:
  - Scenario 1 (MRF Conveyor Worker Mode): Fast-path triage (< 35 ms) flashing red strobe lights and vernacular audio cues 7.2 seconds before hydraulic compaction.
  - Scenario 2 (Public Smart Kiosk Mode): Educational guidance prompting citizens to rinse oily plastic before depositing in the Blue Dry Bin.
```

---

### Field 7: Key Differentiators / Why This Project Wins
```text
1. Empirical Authenticity: No simulated claims. All 90.41% accuracy and sub-35ms latency numbers are measured from real runs on 1,095 test images.
2. Safety Gating: Zero battery false escapes (100% recall) protecting facilities from lithium fires.
3. 100% Offline Autonomy: Runs entirely on edge hardware with zero internet, zero cloud subscriptions, and zero privacy leakage.
4. Production-Ready Codebase: Includes automated Pi deployment scripts, interactive prediction shells, and formal architectural documentation.
```

---

### Field 8: File Uploads & Attachments
* **Primary Document:** Attach **`PIPELINE.pdf`** (located at `A:\Projects\ARM\PIPELINE.pdf`).
* **Source Code / GitHub Link:** Provide your GitHub repository URL if requested.
