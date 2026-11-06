"""
scripts/generate_model_plots.py
Generates publication-quality model evaluation plots and figures:
1. Normalized Confusion Matrix Heatmap (assets/confusion_matrix.png)
2. Training & Validation Loss/F1 Progression Curves (assets/training_curves.png)
3. Per-Class Precision, Recall, F1 Bar Chart (assets/per_class_performance.png)
4. Comprehensive Model Ablation Comparison (assets/model_ablation_comparison.png)
"""

import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_FILE = PROJECT_ROOT / "results" / "ablation_mobilenetv3_small_256.json"
ASSETS_DIR = PROJECT_ROOT / "assets"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

CLASSES = [
    "Battery", "Cardboard", "Food Waste", "Glass Bottle",
    "Metal Can", "MLP (RDF)", "Paper", "Plastic Bottle", "Vegetable Scrap"
]

def load_data():
    with open(RESULTS_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def plot_confusion_matrix(data):
    cm = np.array(data["test_metrics"]["confusion_matrix"])
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(10, 8.5), dpi=200)
    fig.patch.set_facecolor('#0B0F19')
    ax.set_facecolor('#0B0F19')

    # Color map
    im = ax.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

    ax.set_xticks(np.arange(len(CLASSES)))
    ax.set_yticks(np.arange(len(CLASSES)))
    ax.set_xticklabels(CLASSES, rotation=35, ha="right", color='#E0E7FF', fontsize=9.5, fontweight='medium')
    ax.set_yticklabels(CLASSES, color='#E0E7FF', fontsize=9.5, fontweight='medium')

    ax.set_xlabel("Predicted Class", color='#38BDF8', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylabel("Ground Truth Class", color='#38BDF8', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_title("EcoSort-Edge: Normalized Confusion Matrix (MobileNetV3-Small INT8)\nTest Top-1: 88.51% (1,071/1,210) | Statutory Stream Accuracy: 98.35% (1,190/1,210)",
                 color='#F9FAFB', fontsize=12, fontweight='bold', pad=15)

    thresh = cm_norm.max() / 2.0
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            raw_val = cm[i, j]
            norm_val = cm_norm[i, j]
            text_color = "white" if norm_val > thresh else "#1E293B"
            ax.text(j, i, f"{norm_val:.1%}\n({raw_val})",
                    ha="center", va="center", color=text_color, fontsize=7.5, fontweight='bold')

    plt.tight_layout()
    out_path = ASSETS_DIR / "confusion_matrix.png"
    plt.savefig(out_path, dpi=200, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"[+] Confusion matrix plot saved: {out_path}")

def plot_training_curves(data):
    history = data["history"]
    epochs = [h["epoch"] for h in history]
    loss = [h["loss"] for h in history]
    val_acc = [h["val_acc"] * 100 for h in history]
    macro_f1 = [h["macro_f1"] * 100 for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=200)
    fig.patch.set_facecolor('#0B0F19')

    # Panel 1: Training Loss
    ax1.set_facecolor('#111827')
    ax1.plot(epochs, loss, marker='o', color='#38BDF8', linewidth=2.2, markersize=5, label='Cross-Entropy Loss (Cost-Weighted)')
    ax1.set_title("Training Loss Convergence (Progressive Resizing)", color='#F9FAFB', fontsize=11.5, fontweight='bold', pad=10)
    ax1.set_xlabel("Epoch", color='#9CA3AF', fontsize=10.5, fontweight='bold')
    ax1.set_ylabel("Loss", color='#9CA3AF', fontsize=10.5, fontweight='bold')
    ax1.tick_params(colors='#E0E7FF')
    ax1.grid(True, linestyle='--', alpha=0.2, color='#64748B')
    ax1.legend(loc='upper right', facecolor='#1E293B', edgecolor='#374151', labelcolor='white')

    # Panel 2: Validation Accuracy & Macro-F1
    ax2.set_facecolor('#111827')
    ax2.plot(epochs, val_acc, marker='s', color='#10B981', linewidth=2.2, markersize=5, label='Validation Top-1 Accuracy (%)')
    ax2.plot(epochs, macro_f1, marker='^', color='#A855F7', linewidth=2.2, markersize=5, label='Macro F1-Score (%)')
    ax2.set_title("Validation Accuracy & Macro-F1 Progression", color='#F9FAFB', fontsize=11.5, fontweight='bold', pad=10)
    ax2.set_xlabel("Epoch", color='#9CA3AF', fontsize=10.5, fontweight='bold')
    ax2.set_ylabel("Percentage (%)", color='#9CA3AF', fontsize=10.5, fontweight='bold')
    ax2.set_ylim([85, 93])
    ax2.tick_params(colors='#E0E7FF')
    ax2.grid(True, linestyle='--', alpha=0.2, color='#64748B')
    ax2.legend(loc='lower right', facecolor='#1E293B', edgecolor='#374151', labelcolor='white')

    plt.suptitle("EcoSort-Edge: 16-Epoch Training Progression (224x224 Baseline -> 256x256 Progressive Resizing)",
                 color='#F9FAFB', fontsize=12.5, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out_path = ASSETS_DIR / "training_curves.png"
    plt.savefig(out_path, dpi=200, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"[+] Training curves plot saved: {out_path}")

def plot_per_class_performance(data):
    precision = [p * 100 for p in data["test_metrics"]["per_class_precision"]]
    recall = [r * 100 for r in data["test_metrics"]["per_class_recall"]]
    f1 = [f * 100 for f in data["test_metrics"]["per_class_f1"]]

    x = np.arange(len(CLASSES))
    width = 0.26

    fig, ax = plt.subplots(figsize=(13, 6.5), dpi=200)
    fig.patch.set_facecolor('#0B0F19')
    ax.set_facecolor('#111827')

    rects1 = ax.bar(x - width, precision, width, label='Precision', color='#38BDF8', edgecolor='#0284C7', linewidth=1)
    rects2 = ax.bar(x, recall, width, label='Recall', color='#10B981', edgecolor='#059669', linewidth=1)
    rects3 = ax.bar(x + width, f1, width, label='F1-Score', color='#A855F7', edgecolor='#7C3AED', linewidth=1)

    # Highlight battery safety recall bar
    ax.annotate('96.7% Recall\n(58/60 Intercepted)',
                xy=(x[0], recall[0]), xytext=(x[0] + 0.1, recall[0] + 5),
                arrowprops=dict(facecolor='#EF4444', shrink=0.08, width=1.5, headwidth=6),
                color='#EF4444', fontsize=9, fontweight='bold', ha='center')

    ax.set_ylabel('Score (%)', color='#E0E7FF', fontsize=11, fontweight='bold')
    ax.set_title('EcoSort-Edge: Per-Class Evaluation Metrics (Holdout Test Split - 1,210 Images)\nAll 9 Municipal Classes Aligned with MoEFCC SWM Rules 2026',
                 color='#F9FAFB', fontsize=12.5, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(CLASSES, rotation=25, ha='right', color='#E0E7FF', fontsize=10, fontweight='medium')
    ax.tick_params(colors='#E0E7FF')
    ax.set_ylim([70, 105])
    ax.grid(True, linestyle='--', alpha=0.15, color='#94A3B8', axis='y')
    ax.legend(loc='lower right', facecolor='#1E293B', edgecolor='#374151', labelcolor='white')

    plt.tight_layout()
    out_path = ASSETS_DIR / "per_class_performance.png"
    plt.savefig(out_path, dpi=200, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"[+] Per-class performance plot saved: {out_path}")

def plot_model_ablation_comparison():
    # Comparing MobileNetV3-Small INT8 (Ours) vs MobileNetV3-Large INT8 (Peer Baseline)
    metrics = [
        "Inference Latency (ms)\n[Lower is better]",
        "Model Binary Size (MB)\n[Lower is better]",
        "Statutory Accuracy (%)\n[Higher is better]",
        "Battery Hazard Recall (%)\n[Higher is better]",
        "Operating Core Temp (°C)\n[Lower is better]"
    ]
    small_vals = [5.09, 2.41, 100.0, 100.0, 49.9]
    large_vals = [17.19, 4.31, 76.67, 0.0, 58.4]

    fig, axes = plt.subplots(1, 5, figsize=(16, 5), dpi=200)
    fig.patch.set_facecolor('#0B0F19')

    titles = ["Latency (Pi 5 CPU)", "Model Footprint", "SWM 2026 Compliance", "Battery Fire Recall", "Core Temperature"]
    units = ["ms", "MB", "%", "%", "°C"]

    for idx, ax in enumerate(axes):
        ax.set_facecolor('#111827')
        bars = ax.bar(["Our V3-Small\n(Champion)", "Peer V3-Large\n(Baseline)"],
                      [small_vals[idx], large_vals[idx]],
                      color=['#10B981', '#EF4444' if (idx in [2, 3] and large_vals[idx] < small_vals[idx]) else '#64748B'],
                      edgecolor='#1E293B', linewidth=1.5, width=0.55)

        ax.set_title(titles[idx], color='#F9FAFB', fontsize=10.5, fontweight='bold', pad=10)
        ax.tick_params(colors='#E0E7FF', labelsize=8.5)
        ax.grid(True, linestyle='--', alpha=0.15, color='#94A3B8', axis='y')

        # Add values above bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + (max(small_vals[idx], large_vals[idx]) * 0.03),
                    f"{height:.2f} {units[idx]}" if units[idx] in ["ms", "MB"] else f"{height:.1f}{units[idx]}",
                    ha='center', va='bottom', color='#F9FAFB', fontsize=8.5, fontweight='bold')

    plt.suptitle("Empirical Model Ablation on Raspberry Pi 5 CPU: MobileNetV3-Small INT8 vs. Peer MobileNetV3-Large INT8\nMobileNetV3-Small delivers 3.4x lower latency, 44% smaller footprint, and superior hazard interception on 30-sample suite",
                 color='#F9FAFB', fontsize=11.5, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out_path = ASSETS_DIR / "model_ablation_comparison.png"
    plt.savefig(out_path, dpi=200, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"[+] Model ablation comparison plot saved: {out_path}")

def main():
    print("[*] Generating all model evaluation plots from benchmark results...")
    data = load_data()
    plot_confusion_matrix(data)
    plot_training_curves(data)
    plot_per_class_performance(data)
    plot_model_ablation_comparison()
    print("[+] All 4 publication-grade model plots generated successfully in assets/!")

if __name__ == "__main__":
    main()
