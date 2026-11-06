"""
EcoSort-Edge: Dataset Synthesis & Stratification Script
Ingests raw Stanford TrashNet, UCI RealWaste, and Singapore Battery Dataset.
Maps them into the canonical 9-class statutory taxonomy.
Performs SHA-256 deduplication and creates clean 70% Train / 15% Val / 15% Test splits.
"""

import os
import sys
import shutil
import hashlib
from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split

# Ensure UTF-8 output
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
SEED = 42

def compute_sha256(file_path: Path):
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def gather_raw_samples():
    samples = [] # (file_path, canonical_class)

    # 1. Stanford TrashNet
    trashnet_base = RAW_DIR / "trashnet" / "dataset-resized"
    if trashnet_base.exists():
        tn_mapping = {
            "cardboard": "cardboard",
            "glass": "glass_bottle",
            "metal": "metal_can",
            "paper": "paper",
            "plastic": "plastic_bottle",
            "trash": "multi_layer_packaging"
        }
        for sub, cls in tn_mapping.items():
            folder = trashnet_base / sub
            if folder.exists():
                for p in folder.glob("*.jpg"):
                    samples.append((p, cls))

    # 2. UCI RealWaste
    rw_base = RAW_DIR / "realwaste" / "realwaste-main" / "RealWaste"
    if rw_base.exists():
        rw_mapping = {
            "Cardboard": "cardboard",
            "Food Organics": "food_waste",
            "Glass": "glass_bottle",
            "Metal": "metal_can",
            "Miscellaneous Trash": "multi_layer_packaging",
            "Paper": "paper",
            "Plastic": "plastic_bottle",
            "Vegetation": "vegetable_scrap"
        }
        for sub, cls in rw_mapping.items():
            folder = rw_base / sub
            if folder.exists():
                for p in folder.glob("*.jpg"):
                    samples.append((p, cls))

    # 3. Singapore Battery Dataset
    battery_base = RAW_DIR / "battery"
    if battery_base.exists():
        for p in battery_base.rglob("*.*"):
            if p.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                samples.append((p, "battery"))

    return samples

def curate_and_split():
    print("=" * 75)
    print(" EcoSort-Edge: Dataset Synthesis & Deduplication Pipeline")
    print(" Target Output: A:\\Projects\\ARM\\data\\processed")
    print("=" * 75)

    raw_samples = gather_raw_samples()
    print(f"[*] Total raw samples gathered across all 3 datasets: {len(raw_samples)}")

    # Deduplicate via SHA-256 and PIL integrity check
    seen_hashes = set()
    clean_samples = []
    corrupted_count = 0
    duplicate_count = 0

    print("[*] Validating image integrity and checking byte collisions...")
    for path, cls in raw_samples:
        try:
            with Image.open(path) as img:
                img.verify()
        except Exception:
            corrupted_count += 1
            continue

        sha = compute_sha256(path)
        if sha in seen_hashes:
            duplicate_count += 1
            continue
        seen_hashes.add(sha)
        clean_samples.append((path, cls))

    print(f"[+] Total Clean Verified Images: {len(clean_samples)}")
    print(f"[+] Exact byte duplicates purged: {duplicate_count}")

    # Stratified Split (70% Train, 15% Val, 15% Test)
    paths = [s[0] for s in clean_samples]
    labels = [s[1] for s in clean_samples]

    X_train, X_temp, y_train, y_temp = train_test_split(
        paths, labels, test_size=0.30, stratify=labels, random_state=SEED
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=SEED
    )

    splits = {
        "train": (X_train, y_train),
        "val": (X_val, y_val),
        "test": (X_test, y_test)
    }

    print("\n[*] Writing stratified splits to data/processed/...")
    for split_name, (x_data, y_data) in splits.items():
        split_dir = PROCESSED_DIR / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        for src_path, cls in zip(x_data, y_data):
            dst = split_dir / cls / src_path.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_path, dst)
        print(f"    • {split_name.upper():<5} split: {len(x_data)} images")

    print("=" * 75)
    print("[SUCCESS] Dataset synthesis and 70/15/15 stratification complete!")
    print("Ready for RTX 4050 training: `python scripts/02_train_ecosort_engine.py`")
    print("=" * 75)

if __name__ == "__main__":
    curate_and_split()
