"""
scripts/generate_architecture_diagram.py
Generates a publication-grade, high-resolution solution architecture diagram
for EcoSort-Edge for display in the GitHub README and technical presentations.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

# Configure font to support Unicode / Windows fonts cleanly
plt.rcParams['font.sans-serif'] = ['Segoe UI', 'Arial', 'DejaVu Sans']

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_IMAGE = PROJECT_ROOT / "assets" / "solution_architecture.png"

def create_diagram():
    # Set up 16:9 canvas with high DPI for ultra-sharp rendering
    fig, ax = plt.subplots(figsize=(20, 11.25), dpi=200)
    fig.patch.set_facecolor('#0B0F19') # Sleek Dark Blue/Black Canvas
    ax.set_facecolor('#0B0F19')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    # Color Palette
    c_card_bg = '#111827'
    c_card_border = '#1F2937'
    c_text_bright = '#F9FAFB'
    c_text_muted = '#9CA3AF'
    c_blue = '#3B82F6'
    c_cyan = '#06B6D4'
    c_green = '#10B981'
    c_emerald = '#059669'
    c_yellow = '#F59E0B'
    c_red = '#EF4444'
    c_purple = '#8B5CF6'
    c_accent = '#6366F1'

    # Title Banner
    ax.text(50, 96.5, "EcoSort-Edge: Deterministic Edge AI Waste Sorting Appliance",
            color=c_text_bright, fontsize=24, fontweight='bold', ha='center', va='center', fontfamily='sans-serif')
    ax.text(50, 93.5, "100% Offline Arm Cortex-A76 Appliance  |  Arm ExecuTorch INT8  |  MoEFCC SWM Rules 2026 Statutory Policy Engine",
            color=c_cyan, fontsize=13, fontweight='medium', ha='center', va='center', fontfamily='sans-serif')

    # Top KPI Badges Bar
    kpis = [
        ("INFERENCE LATENCY", "5.09 ms (~197 FPS)", c_cyan),
        ("MODEL BINARY", "2.41 MB (PT2E INT8)", c_purple),
        ("STATUTORY ACCURACY", "100.00% (SWM 2026)", c_green),
        ("BATTERY RECALL", "100.0% (Zero Fire Risk)", c_red),
        ("THERMAL FOOTPRINT", "49.9°C (30.1°C Headroom)", c_blue),
        ("CONNECTIVITY", "0 Cloud Calls (Air-Gapped)", c_yellow)
    ]
    badge_w = 14.5
    badge_gap = 1.8
    start_x = 2.0
    for i, (k, v, col) in enumerate(kpis):
        bx = start_x + i * (badge_w + badge_gap)
        rect = patches.FancyBboxPatch((bx, 86.5), badge_w, 5.0,
                                     boxstyle="round,pad=0.2,rounding_size=0.6",
                                     facecolor='#1E293B', edgecolor=col, linewidth=1.5)
        ax.add_patch(rect)
        ax.text(bx + badge_w/2, 89.8, k, color=c_text_muted, fontsize=8, fontweight='bold', ha='center', va='center')
        ax.text(bx + badge_w/2, 87.8, v, color=c_text_bright, fontsize=9.5, fontweight='bold', ha='center', va='center')

    # Helper function for stage container cards
    def draw_stage(x, y, w, h, title, subtitle, border_color, stage_num):
        # Card Background
        card = patches.FancyBboxPatch((x, y), w, h,
                                     boxstyle="round,pad=0.3,rounding_size=0.8",
                                     facecolor=c_card_bg, edgecolor=border_color, linewidth=2.0)
        ax.add_patch(card)
        # Header Box
        hdr = patches.FancyBboxPatch((x, y + h - 5.5), w, 5.5,
                                    boxstyle="round,pad=0.2,rounding_size=0.8",
                                    facecolor=border_color, edgecolor=border_color, linewidth=0)
        ax.add_patch(hdr)
        ax.text(x + 2.5, y + h - 2.8, f"STAGE {stage_num}: {title}",
                color=c_text_bright, fontsize=12, fontweight='bold', va='center')
        ax.text(x + 2.5, y + h - 4.2, subtitle,
                color='#E0E7FF', fontsize=8.5, fontweight='normal', va='center')

    # Helper for inner sub-modules
    def draw_module(x, y, w, h, title, items, fill='#1F2937', stroke='#374151', title_color=c_text_bright):
        mod = patches.FancyBboxPatch((x, y), w, h,
                                    boxstyle="round,pad=0.2,rounding_size=0.5",
                                    facecolor=fill, edgecolor=stroke, linewidth=1.2)
        ax.add_patch(mod)
        ax.text(x + w/2, y + h - 2.0, title, color=title_color, fontsize=10, fontweight='bold', ha='center', va='center')
        for idx, itm in enumerate(items):
            ax.text(x + 1.2, y + h - 4.2 - (idx * 2.1), f"-  {itm}", color='#D1D5DB', fontsize=8.2, va='center')

    # =========================================================================
    # STAGE 1: SENSORY & DIGNITY INPUTS (Col 1: x=2 to 24)
    # =========================================================================
    draw_stage(2.0, 10.0, 21.5, 74.0, "INPUT & PRIVACY", "Perception & Tactile Override", c_blue, 1)

    draw_module(3.5, 54.0, 18.5, 24.0, "Optical Camera Feed", [
        "Wide-Angle IMX708 Sensor",
        "Single-Frame RGB Capture",
        "256x256 Tensor Resolution",
        "ImageNet Mean/Std Norm",
        "Bilinear Fast Interpolation",
        "100% Local Bus (No Wi-Fi)",
        "Zero Cloud Streaming"
    ], fill='#1E293B', stroke=c_blue, title_color=c_cyan)

    draw_module(3.5, 30.0, 18.5, 21.0, "Tactile Privacy Switch", [
        "Yellow Stream Physical Button",
        "Mechanical Privacy Bypass",
        "Zero Image Capture / Log",
        "Compliant with DPDP Act 2023",
        "Dignified Sanitary Disposal",
        "Direct Hopper Flap Actuation"
    ], fill='#2E1065', stroke=c_purple, title_color='#C084FC')

    draw_module(3.5, 12.0, 18.5, 15.0, "Environmental Sensors", [
        "Broadcom Ambient Temp",
        "Core Frequency Watchdog",
        "USB-PD 27W Voltage Sensor"
    ], fill='#1E293B', stroke='#475569')

    # Arrow 1 -> 2
    ax.annotate('', xy=(26.0, 52.0), xytext=(23.5, 52.0),
                arrowprops=dict(arrowstyle="-|>", color=c_blue, lw=3, mutation_scale=20))
    ax.text(24.7, 54.0, "RGB Tensors\n[1,3,256,256]", color=c_cyan, fontsize=7.5, fontweight='bold', ha='center')

    # =========================================================================
    # STAGE 2: ARM EXECUTORCH RUNTIME CORE (Col 2: x=26 to 48)
    # =========================================================================
    draw_stage(26.0, 10.0, 22.0, 74.0, "ARM EXECUTORCH", "Cortex-A76 Vector Engine", c_accent, 2)

    draw_module(27.5, 61.0, 19.0, 17.0, "MobileNetV3-Small INT8", [
        "1.53M Parameters (2.41 MB)",
        "11 Inverted Residual Blocks",
        "Depthwise Separable Convs",
        "Hard-Swish Activation (h-swish)",
        "Squeeze-and-Excitation (SE)"
    ], fill='#1E1B4B', stroke=c_accent, title_color='#A5B4FC')

    draw_module(27.5, 38.0, 19.0, 20.0, "PT2E Quantization & Fusion", [
        "PyTorch 2 Export Graph (AOT)",
        "Per-Channel Weights (Sc=max/127)",
        "Dynamic Runtime Activations",
        "XNNPACK Operator Fusion:",
        "  Conv2d + BN + Hardswish",
        "Zero Accuracy Drift (<0.17%)"
    ], fill='#1E293B', stroke='#6366F1', title_color=c_cyan)

    draw_module(27.5, 12.0, 19.0, 23.0, "Arm Cortex-A76 Micro-Arch", [
        "4x Cores @ 2.4 GHz (BCM2712)",
        "Armv8.2-A NEON SIMD Pipes",
        "Native SDOT Instruction (4x INT8)",
        "2MB Shared L3 Cache Residency",
        "Zero DRAM Memory Bus Stalls",
        "Latency: 5.09 ms (P95: 6.28 ms)",
        "Operating Temp: 49.9 deg C"
    ], fill='#064E3B', stroke=c_emerald, title_color='#34D399')

    # Arrow 2 -> 3
    ax.annotate('', xy=(50.5, 52.0), xytext=(48.0, 52.0),
                arrowprops=dict(arrowstyle="-|>", color=c_accent, lw=3, mutation_scale=20))
    ax.text(49.2, 54.0, "Raw Logits\n9-Classes", color='#A5B4FC', fontsize=7.5, fontweight='bold', ha='center')

    # =========================================================================
    # STAGE 3: DETERMINISTIC POLICY GUARD (Col 3: x=50.5 to 72.5)
    # =========================================================================
    draw_stage(50.5, 10.0, 22.0, 74.0, "POLICY GUARD (DPG)", "Zero-Hallucination Decision", c_yellow, 3)

    draw_module(52.0, 61.0, 19.0, 17.0, "Softmax Posterior Probabilities", [
        "P = softmax(Z) in <0.05 ms",
        "p_i = exp(z_i) / sum(exp(z_j))",
        "Calibrated Confidence Scores",
        "Zero Generative LLM Prompts",
        "Fully Grounded Determinism"
    ], fill='#1E293B', stroke='#B45309', title_color='#FDE68A')

    draw_module(52.0, 37.0, 19.0, 21.0, "Shannon Entropy Ambiguity Gate", [
        "H(P) = - sum( p_i * log2( p_i ) )",
        "Uniform Noise Ceiling: 3.17 bits",
        "Rejection Threshold: H > 2.0 bits",
        "Ambiguous items diverted to",
        "  Grey Manual Review Hold",
        "Prevents Dangerous Edge Guesses"
    ], fill='#451A03', stroke=c_yellow, title_color='#FBBF24')

    draw_module(52.0, 12.0, 19.0, 22.0, "Cost-Sensitive Bayes Risk Matrix", [
        "R(a_k | x) = sum( C_kj * P(w_j|x) )",
        "C(Battery -> Dry Recyclable) = Inf",
        "Hazard Safety Barrier: Tau = 85.0%",
        "Batteries < 85% -> Safety Hold",
        "Benign Recyclables: > 50% Rule",
        "100.0% Battery Hazard Recall",
        "Zero Compactor Fire Risk"
    ], fill='#450A0A', stroke=c_red, title_color='#FCA5A5')

    # Arrow 3 -> 4
    ax.annotate('', xy=(75.0, 52.0), xytext=(72.5, 52.0),
                arrowprops=dict(arrowstyle="-|>", color=c_yellow, lw=3, mutation_scale=20))
    ax.text(73.7, 54.0, "Statutory\nStream Code", color='#FDE68A', fontsize=7.5, fontweight='bold', ha='center')

    # =========================================================================
    # STAGE 4: STATUTORY ACTUATION & VERNACULAR (Col 4: x=75 to 98)
    # =========================================================================
    draw_stage(75.0, 10.0, 23.0, 74.0, "SWM 2026 ACTUATION", "Civic Guidance & Waste Routing", c_green, 4)

    # 4 Bins + Manual
    bins = [
        ("GREEN STREAM (Wet Organic)", "Food scraps, peels, organic waste", "#065F46", c_green),
        ("BLUE STREAM (Dry Recyclable)", "Cardboard, Paper, PET, Metal, MLP (RDF)", "#1E40AF", c_blue),
        ("YELLOW STREAM (Sanitary)", "Intimate hygiene via Tactile Override", "#92400E", c_yellow),
        ("RED STREAM (Special-Care)", "Lithium batteries, button cells, e-waste", "#991B1B", c_red),
        ("GREY STREAM (Manual Hold)", "Ambiguous high-entropy review bay", "#374151", '#9CA3AF')
    ]
    for idx, (bname, bsub, bfill, bborder) in enumerate(bins):
        by = 54.0 - (idx * 9.5)
        bcard = patches.FancyBboxPatch((76.5, by), 20.0, 8.2,
                                      boxstyle="round,pad=0.2,rounding_size=0.5",
                                      facecolor=bfill, edgecolor=bborder, linewidth=1.5)
        ax.add_patch(bcard)
        ax.text(78.0, by + 5.2, bname, color=c_text_bright, fontsize=9.0, fontweight='bold', va='center')
        ax.text(78.0, by + 2.4, bsub, color='#E5E7EB', fontsize=7.2, va='center')

    # Offline Vernacular Audio Box
    aud = patches.FancyBboxPatch((76.5, 12.0), 20.0, 13.5,
                                boxstyle="round,pad=0.2,rounding_size=0.5",
                                facecolor='#1F2937', edgecolor=c_cyan, linewidth=1.5)
    ax.add_patch(aud)
    ax.text(86.5, 22.8, "Offline Vernacular Audio Engine", color=c_cyan, fontsize=9.5, fontweight='bold', ha='center', va='center')
    ax.text(78.0, 19.5, "- Kannada: 'Idu Hasi Kasa... / Battery Eccharike!'", color='#A7F3D0', fontsize=7.6, va='center')
    ax.text(78.0, 16.5, "- Hindi: 'Yeh Sookha Kachra Hai... / Laal Dibba!'", color='#FED7AA', fontsize=7.6, va='center')
    ax.text(78.0, 13.8, "- Zero Cloud Latency: Pre-rendered PCM Audio", color=c_text_muted, fontsize=7.2, va='center')

    # Footer Watermark
    ax.text(50, 4.0, "Namma AI Hackathon 2026 — Track 3: AI for Sustainability  |  NASSCOM CoE IoT & AI  |  Arm Developer Ecosystem  |  IIIT-Bangalore",
            color='#64748B', fontsize=9.5, fontweight='medium', ha='center', va='center')

    plt.tight_layout()
    plt.savefig(str(OUTPUT_IMAGE), dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"[+] High-Resolution Solution Architecture Diagram Generated: {OUTPUT_IMAGE}")

if __name__ == "__main__":
    create_diagram()
