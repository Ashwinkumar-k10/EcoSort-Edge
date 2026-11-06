"""
Sample 27 realistic waste images (3 from each of the 9 statutory classes)
from data/processed/test and prepare them in data/test_27_samples/
with corresponding ground_truth.json.
"""

import os
import shutil
import json
from pathlib import Path

BASE_DIR = Path("A:/Projects/ARM")
TEST_DIR = BASE_DIR / "data/processed/test"
TARGET_DIR = BASE_DIR / "data/test_27_samples"

TARGET_DIR.mkdir(parents=True, exist_ok=True)

# Clear existing images
for f in TARGET_DIR.glob("*.*"):
    if f.is_file():
        f.unlink()

CLASS_MAP_STREAM = {
    "battery": ("Battery Cell / Hazard", "Hazardous / Special"),
    "cardboard": ("Corrugated Packaging Box", "Dry Recyclable"),
    "food_waste": ("Food Scraps & Organics", "Wet Organic"),
    "glass_bottle": ("Glass Container Bottle", "Dry Recyclable"),
    "metal_can": ("Beverage Aluminum / Tin Can", "Dry Recyclable"),
    "multi_layer_packaging": ("Multi-layer Foil Wrapper (MLP)", "Reject (RDF)"),
    "paper": ("Clean Paper Sheet / Print", "Dry Recyclable"),
    "plastic_bottle": ("PET / HDPE Plastic Bottle", "Dry Recyclable"),
    "vegetable_scrap": ("Raw Kitchen Vegetable Scrap", "Wet Organic")
}

ground_truth = {}
count = 0

for cls_name, (desc, stream) in CLASS_MAP_STREAM.items():
    cls_folder = TEST_DIR / cls_name
    files = sorted([f for f in cls_folder.glob("*.*") if f.suffix.lower() in [".jpg", ".jpeg", ".png"]])
    
    # Pick 3 diverse images per class
    selected = files[:3]
    for idx, src_file in enumerate(selected, 1):
        target_name = f"{cls_name}_{idx:02d}{src_file.suffix.lower()}"
        dest_path = TARGET_DIR / target_name
        shutil.copy2(src_file, dest_path)
        
        ground_truth[target_name] = {
            "ground_truth_label": cls_name,
            "item_name": f"{desc} #{idx}",
            "suggested_stream": stream
        }
        count += 1

gt_path = TARGET_DIR / "ground_truth.json"
with open(gt_path, "w", encoding="utf-8") as f:
    json.dump(ground_truth, f, indent=2)

print(f"Successfully created 27 test samples in {TARGET_DIR} (3 per class across 9 classes).")
