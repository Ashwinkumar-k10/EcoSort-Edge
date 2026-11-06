import os
import json
from pathlib import Path
from PIL import Image, ImageDraw

CANONICAL_ITEMS = [
    # 🟢 Wet Organic
    {"id": "food_waste_01", "name": "Apple Core & Peels", "label": "food_waste", "stream": "Wet Organic", "color": (139, 69, 19)},
    {"id": "food_waste_02", "name": "Leftover Cooked Rice", "label": "food_waste", "stream": "Wet Organic", "color": (245, 245, 220)},
    {"id": "vegetable_scrap_01", "name": "Tomato Trimmings", "label": "vegetable_scrap", "stream": "Wet Organic", "color": (220, 20, 60)},
    {"id": "vegetable_scrap_02", "name": "Banana Skin", "label": "vegetable_scrap", "stream": "Wet Organic", "color": (218, 165, 32)},

    # 🔵 Dry Recyclable
    {"id": "plastic_bottle_01", "name": "Clear PET Water Bottle", "label": "plastic_bottle", "stream": "Dry Recyclable", "color": (100, 149, 237)},
    {"id": "plastic_bottle_02", "name": "Crushed Milk Jug (HDPE)", "label": "plastic_bottle", "stream": "Dry Recyclable", "color": (240, 248, 255)},
    {"id": "metal_can_01", "name": "Aluminium Soda Can", "label": "metal_can", "stream": "Dry Recyclable", "color": (192, 192, 192)},
    {"id": "metal_can_02", "name": "Tin Food Can", "label": "metal_can", "stream": "Dry Recyclable", "color": (176, 196, 222)},
    {"id": "cardboard_01", "name": "Corrugated Delivery Box", "label": "cardboard", "stream": "Dry Recyclable", "color": (205, 133, 63)},
    {"id": "cardboard_02", "name": "Cereal Box Flap", "label": "cardboard", "stream": "Dry Recyclable", "color": (210, 180, 140)},
    {"id": "paper_01", "name": "Newspaper Sheet", "label": "paper", "stream": "Dry Recyclable", "color": (220, 220, 220)},
    {"id": "glass_bottle_01", "name": "Green Beverage Glass Bottle", "label": "glass_bottle", "stream": "Dry Recyclable", "color": (46, 139, 87)},

    # 🟤 Non-Recyclable Reject (RDF)
    {"id": "multi_layer_packaging_01", "name": "Metallized Chips Pouch", "label": "multi_layer_packaging", "stream": "Non-Recyclable Reject", "color": (255, 215, 0)},
    {"id": "multi_layer_packaging_02", "name": "Foil Biscuit Wrapper", "label": "multi_layer_packaging", "stream": "Non-Recyclable Reject", "color": (169, 169, 169)},

    # 🔴 Hazardous / Special-Care
    {"id": "battery_01", "name": "Lithium-Ion Cylindrical Cell (18650)", "label": "battery", "stream": "Hazardous / Special-Care", "color": (255, 69, 0)},
]

def generate_canonical_dataset(output_dir="data/smoke_test"):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    metadata = {}
    print(f"Generating 15 canonical test images in {out_path.resolve()}...")

    for item in CANONICAL_ITEMS:
        filename = f"{item['id']}.jpg"
        img_path = out_path / filename

        img = Image.new("RGB", (224, 224), color=(240, 240, 240))
        draw = ImageDraw.Draw(img)
        draw.rounded_rectangle([35, 45, 189, 179], radius=16, fill=item["color"], outline=(50, 50, 50), width=2)
        draw.rectangle([0, 0, 224, 26], fill=(30, 30, 30))
        draw.text((8, 6), f"EcoSort: {item['label']}", fill=(255, 255, 255))
        draw.text((40, 105), item["name"][:20], fill=(0, 0, 0))

        img.save(img_path, quality=95)
        metadata[filename] = {
            "ground_truth_label": item["label"],
            "item_name": item["name"],
            "suggested_stream": item["stream"]
        }

    gt_path = out_path / "ground_truth.json"
    with open(gt_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[SUCCESS] Created 15 canonical images + ground_truth.json at {gt_path.resolve()}")

if __name__ == "__main__":
    generate_canonical_dataset()
