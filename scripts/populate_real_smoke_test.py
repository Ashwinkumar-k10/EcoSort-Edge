"""
Replace synthetic smoke test placeholders with real, distinct test set photos.
Also update ground_truth.json to reflect the real sample names.
"""

import os
import shutil
import json
from pathlib import Path

BASE_DIR = Path("A:/Projects/ARM")
TEST_DIR = BASE_DIR / "data/processed/test"
SMOKE_DIR = BASE_DIR / "data/smoke_test"

CLASS_MAP_STREAM = {
    "battery": ("Battery Cell / Pack", "Hazardous / Special"),
    "cardboard": ("Corrugated Cardboard Box", "Dry Recyclable"),
    "food_waste": ("Food Scraps / Leftovers", "Wet Organic"),
    "glass_bottle": ("Glass Container Bottle", "Dry Recyclable"),
    "metal_can": ("Beverage Metal Can", "Dry Recyclable"),
    "multi_layer_packaging": ("Multi-layer Foil Snack Wrapper", "Reject (RDF)"),
    "paper": ("Clean Paper Document", "Dry Recyclable"),
    "plastic_bottle": ("Rigid Plastic Bottle", "Dry Recyclable"),
    "vegetable_scrap": ("Vegetable Kitchen Scrap", "Wet Organic")
}

# Clear old smoke test images (keep ground_truth backup)
for f in SMOKE_DIR.glob("*.jpg"):
    f.unlink()
for f in SMOKE_DIR.glob("*.jpeg"):
    f.unlink()
for f in SMOKE_DIR.glob("*.png"):
    f.unlink()

new_ground_truth = {}

# Select 2 distinct real samples for each class (except battery where 1 or 2 is fine)
for cls_name, (item_desc, stream_name) in CLASS_MAP_STREAM.items():
    cls_files = sorted(list((TEST_DIR / cls_name).glob("*.*")))
    # Take 2 samples per class
    samples = cls_files[:2] if len(cls_files) >= 2 else cls_files[:1]
    
    for idx, sample_path in enumerate(samples, 1):
        target_name = f"{cls_name}_{idx:02d}.jpg"
        dest_path = SMOKE_DIR / target_name
        shutil.copy2(sample_path, dest_path)
        
        new_ground_truth[target_name] = {
            "ground_truth_label": cls_name,
            "item_name": f"{item_desc} #{idx}",
            "suggested_stream": stream_name
        }

with open(SMOKE_DIR / "ground_truth.json", "w", encoding="utf-8") as f:
    json.dump(new_ground_truth, f, indent=2)

print(f"Successfully populated {len(new_ground_truth)} real images into {SMOKE_DIR}")
for k, v in new_ground_truth.items():
    print(f"  {k:<28} -> {v['ground_truth_label']} ({v['suggested_stream']})")
