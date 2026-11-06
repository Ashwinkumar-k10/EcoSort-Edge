"""
EcoSort-Edge: Full Comprehensive Metrics Computation
Computes:
- Full 9x9 Confusion Matrix
- Per-Class TP, FP, FN, TN, Precision, Recall, Specificity, F1-Score, Support
- Macro-Average & Weighted-Average Precision, Recall, F1-Score
- Overall Micro Accuracy
- 4-Stream Physical Disposal Confusion Matrix & Metrics
"""

import json
from pathlib import Path
from collections import defaultdict
import numpy as np

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

STREAM_MAP = {
    "food_waste": "Wet Organic",
    "vegetable_scrap": "Wet Organic",
    "plastic_bottle": "Dry Recyclable",
    "metal_can": "Dry Recyclable",
    "cardboard": "Dry Recyclable",
    "paper": "Dry Recyclable",
    "glass_bottle": "Dry Recyclable",
    "multi_layer_packaging": "Reject (RDF)",
    "battery": "Hazardous / Special"
}

STREAMS = [
    "Dry Recyclable",
    "Wet Organic",
    "Reject (RDF)",
    "Hazardous / Special"
]

def main():
    json_path = Path("results/deliverable_d3_results.json")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    evals = data["evaluations"]
    n_samples = len(evals)

    # 1. Build 9x9 Confusion Matrix
    class_to_idx = {c: i for i, c in enumerate(CLASSES)}
    cm = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)

    for item in evals:
        gt = item["ground_truth"]
        pred = item["predicted_category"]
        if gt in class_to_idx and pred in class_to_idx:
            cm[class_to_idx[gt], class_to_idx[pred]] += 1

    # 2. Per-Class Metrics
    per_class_metrics = {}
    precisions = []
    recalls = []
    f1s = []
    supports = []

    for i, c in enumerate(CLASSES):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum() - tp)
        fn = int(cm[i, :].sum() - tp)
        tn = int(cm.sum() - (tp + fp + fn))
        support = int(cm[i, :].sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        supports.append(support)

        per_class_metrics[c] = {
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TN": tn,
            "Support": support,
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "Specificity": round(specificity, 4),
            "F1_Score": round(f1, 4)
        }

    # Macro and Weighted Averages
    macro_precision = np.mean(precisions)
    macro_recall = np.mean(recalls)
    macro_f1 = np.mean(f1s)

    total_support = sum(supports)
    weighted_precision = sum(p * s for p, s in zip(precisions, supports)) / total_support
    weighted_recall = sum(r * s for r, s in zip(recalls, supports)) / total_support
    weighted_f1 = sum(f * s for f, s in zip(f1s, supports)) / total_support
    overall_accuracy = cm.diagonal().sum() / total_support

    # 3. Stream Confusion Matrix
    stream_to_idx = {s: i for i, s in enumerate(STREAMS)}
    stream_cm = np.zeros((len(STREAMS), len(STREAMS)), dtype=int)

    for item in evals:
        gt_stream = STREAM_MAP.get(item["ground_truth"])
        pred_stream = STREAM_MAP.get(item["predicted_category"])
        if gt_stream in stream_to_idx and pred_stream in stream_to_idx:
            stream_cm[stream_to_idx[gt_stream], stream_to_idx[pred_stream]] += 1

    stream_metrics = {}
    for i, s in enumerate(STREAMS):
        tp = int(stream_cm[i, i])
        fp = int(stream_cm[:, i].sum() - tp)
        fn = int(stream_cm[i, :].sum() - tp)
        tn = int(stream_cm.sum() - (tp + fp + fn))
        supp = int(stream_cm[i, :].sum())

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        stream_metrics[s] = {
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TN": tn,
            "Support": supp,
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1_Score": round(f1, 4)
        }
    stream_accuracy = stream_cm.diagonal().sum() / total_support

    # Output full report
    output_data = {
        "overall_summary": {
            "total_samples": total_support,
            "overall_accuracy": round(overall_accuracy, 4),
            "macro_precision": round(macro_precision, 4),
            "macro_recall": round(macro_recall, 4),
            "macro_f1": round(macro_f1, 4),
            "weighted_precision": round(weighted_precision, 4),
            "weighted_recall": round(weighted_recall, 4),
            "weighted_f1": round(weighted_f1, 4),
            "physical_stream_accuracy": round(stream_accuracy, 4),
            "battery_recall": round(per_class_metrics["battery"]["Recall"], 4),
            "battery_precision": round(per_class_metrics["battery"]["Precision"], 4)
        },
        "classes": CLASSES,
        "confusion_matrix_9x9": cm.tolist(),
        "per_class_metrics": per_class_metrics,
        "streams": STREAMS,
        "stream_confusion_matrix_4x4": stream_cm.tolist(),
        "stream_metrics": stream_metrics
    }

    out_json = Path("results/full_detailed_model_metrics.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    print("Successfully generated full_detailed_model_metrics.json")
    print(f"Overall Accuracy: {overall_accuracy*100:.2f}%")
    print(f"Macro Precision: {macro_precision*100:.2f}% | Macro Recall: {macro_recall*100:.2f}% | Macro F1: {macro_f1*100:.2f}%")
    print(f"Weighted Precision: {weighted_precision*100:.2f}% | Weighted Recall: {weighted_recall*100:.2f}% | Weighted F1: {weighted_f1*100:.2f}%")
    print(f"Physical Stream Bin Accuracy: {stream_accuracy*100:.2f}%")

if __name__ == "__main__":
    main()
